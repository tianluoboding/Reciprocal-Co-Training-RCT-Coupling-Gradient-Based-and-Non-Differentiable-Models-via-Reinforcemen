#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative Training Loop V2: PPO + RF Auto-Convergence (Correct Reward Function)

PURPOSE:
========
Automatically run iterative LLM-RF training until convergence.
Uses the CORRECT reward function from 58_phase2 and 60_phase4:
  R = R_RF + λ * R_acc
where R_RF = reward_scale * clip(Q(x,a) - V(x), -0.45, 0.45)

Each iteration:
  1. Train LLM using PPO with current RF as teacher
  2. Update RF using LLM's best checkpoint embeddings
  3. Check if RF test AUC AND LLM test AUC improved
  4. If BOTH don't improve for 5 consecutive iterations → STOP

EARLY STOPPING (NEW):
=====================
Stop when BOTH RF test AUC AND LLM test AUC do not improve for 5 consecutive iterations.
If either improves, reset the counter.

REWARD FUNCTION (from 58/60):
=============================
1. RF Advantage: A(x,a) = Q(x,a) - V(x), where V(x) = mean(Q(x,0), Q(x,1))
2. Clip: clip(A, -0.45, 0.45)
3. RF reward: r_rf = reward_scale * clipped_A
4. Accuracy reward (fn_heavy): TP=+1.0, FN=-1.5, TN=+0.2, FP=-0.2
5. Combined: r_total = r_rf + lambda_acc * r_acc

USAGE:
======
python 71_iterative_training_v2.py                    # Start from scratch
python 71_iterative_training_v2.py --resume 5         # Resume from iteration 5

OUTPUT:
=======
- checkpoints/iterative_v2_ppo_iter{N}/best.pt
- models/rf_v2_iter{N}_pca5.pkl
- models/pca_v2_iter{N}_pca5.pkl
- results/iterative_training_v2_history.json
"""

import sys
import argparse
import json
import time
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score, accuracy_score,
    confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib

# Force unbuffered output
import os
os.environ['PYTHONUNBUFFERED'] = '1'

# ---------------------------------------------------------------------------
# Project paths & imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent  # exp1_clean/
sys.path.insert(0, str(BASE_DIR / "src"))

from data.loader import DataLoader
from models.llm_wrapper import BioClinicalBERTWithLoRA

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Configuration (matching 58/60 scripts)
# ---------------------------------------------------------------------------

@dataclass
class IterativeConfigV2:
    """Configuration for iterative training V2 - matching 58/60 scripts"""
    # Paths
    base_dir: Path = BASE_DIR
    
    # PPO Training Config (from 58/60)
    ppo_num_epochs: int = 25
    ppo_samples_per_epoch: int = 1000
    ppo_batch_size: int = 32
    ppo_lr: float = 3e-5
    ppo_clip_epsilon: float = 0.2
    ppo_epochs_per_batch: int = 4
    ppo_patience: int = 8
    ppo_val_ratio: float = 0.15
    normalize_advantages: bool = True
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Reward Config (from 58/60 - CORRECT VALUES)
    reward_mode: str = "fn_heavy"  # or "rf_plus_acc"
    reward_scale: float = 2.0
    reward_clip_advantage: tuple = (-0.45, 0.45)
    lambda_acc: float = 0.5
    entropy_coef: float = 0.05  # from 58/60, not 0.1
    
    # Sampling
    y1_weight: float = 1.5
    
    # RF Config
    rf_n_estimators: int = 500
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 2
    rf_min_samples_split: int = 5
    rf_class_weight: str = "balanced_subsample"
    rf_calibration_split: float = 0.2
    
    # PCA Config
    pca_dim: int = 5
    
    # Early Stopping (NEW: both RF AND LLM must not improve)
    convergence_patience: int = 5
    
    # LLM Config
    max_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Random seed
    seed: int = 42


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(cfg: IterativeConfigV2):
    """Load all data components"""
    print("\n" + "=" * 80)
    print("📦 Loading Data")
    print("=" * 80)
    
    loader = DataLoader(cfg.base_dir / "data")
    all_data = loader.load_all()
    
    texts = all_data['texts']
    y_all = all_data['y']
    train_idx = all_data['train_idx']
    test_idx = all_data['test_idx']
    
    # Load tabular features (15 selected features)
    X_train_tab = np.load(cfg.base_dir / 'data/processed/X_train_selected.npy')
    X_test_tab = np.load(cfg.base_dir / 'data/processed/X_test_selected.npy')
    
    # Reconstruct full tabular array
    X_tabular_full = np.zeros((len(y_all), X_train_tab.shape[1]), dtype=np.float32)
    X_tabular_full[train_idx] = X_train_tab
    X_tabular_full[test_idx] = X_test_tab
    
    # Split train into train_sub + val
    train_sub_idx, val_idx = train_test_split(
        train_idx, test_size=cfg.ppo_val_ratio, 
        random_state=cfg.seed, stratify=y_all[train_idx]
    )
    
    print(f"  ✓ Train: {len(train_sub_idx)} samples")
    print(f"  ✓ Val: {len(val_idx)} samples")
    print(f"  ✓ Test: {len(test_idx)} samples")
    print(f"  ✓ Tabular features: {X_train_tab.shape[1]}")
    print(f"  ✓ Positive rate (train): {y_all[train_sub_idx].mean():.2%}")
    
    return {
        'texts': texts,
        'y_all': y_all,
        'train_idx': train_sub_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'full_train_idx': train_idx,  # For RF training
        'X_tabular_full': X_tabular_full
    }


# ---------------------------------------------------------------------------
# LLM Model
# ---------------------------------------------------------------------------

def create_llm_model(cfg: IterativeConfigV2):
    """Create fresh BioClinicalBERT + LoRA model"""
    print("\n📦 Creating LLM Model...")
    
    lora_config = {
        'lora_r': cfg.lora_r,
        'lora_alpha': cfg.lora_alpha,
        'lora_dropout': cfg.lora_dropout,
        'target_modules': ['query', 'value'],
    }
    
    model = BioClinicalBERTWithLoRA(
        model_name_or_path='emilyalsentzer/Bio_ClinicalBERT',
        lora_config=lora_config,
        num_labels=2,
        device=DEVICE
    )
    
    print(f"  ✓ Model created on {DEVICE}")
    return model


def load_llm_checkpoint(model, checkpoint_path):
    """Load LLM checkpoint"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', 'unknown')
    else:
        model.load_state_dict(ckpt)
        epoch = 'unknown'
    
    model.to(DEVICE)
    return model, epoch


# ---------------------------------------------------------------------------
# RF Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, tokenizer, texts, cfg: IterativeConfigV2):
    """Extract LLM embeddings"""
    model.eval()
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), cfg.ppo_batch_size), desc="  Extracting embeddings"):
        batch_texts = texts[i:i + cfg.ppo_batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors='pt'
        ).to(DEVICE)
        
        logits, embeddings = model(
            inputs['input_ids'],
            inputs['attention_mask'],
            return_embeddings=True
        )
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def train_rf(embeddings_train, y_train, embeddings_test, y_test, 
             X_train_tab, X_test_tab, cfg: IterativeConfigV2, iteration: int):
    """Train RF with PCA embeddings"""
    print(f"\n🌲 Training RF (Iteration {iteration})")
    
    # Fit PCA on train
    pca = PCA(n_components=cfg.pca_dim, random_state=cfg.seed)
    pca.fit(embeddings_train)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA explained variance: {explained_var:.4f}")
    
    # Transform
    phi_train = pca.transform(embeddings_train)
    phi_test = pca.transform(embeddings_test)
    
    # Combine features
    X_train = np.concatenate([X_train_tab, phi_train], axis=1)
    X_test = np.concatenate([X_test_tab, phi_test], axis=1)
    
    print(f"  Combined features: {X_train.shape[1]} (tab={X_train_tab.shape[1]} + pca={cfg.pca_dim})")
    
    # Split for calibration
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train,
        test_size=cfg.rf_calibration_split,
        random_state=cfg.seed,
        stratify=y_train
    )
    
    # Train RF
    rf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        min_samples_split=cfg.rf_min_samples_split,
        class_weight=cfg.rf_class_weight,
        random_state=cfg.seed,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    
    # Calibrate
    calibrator = CalibratedClassifierCV(
        estimator=rf,
        method='sigmoid',
        cv='prefit'
    )
    calibrator.fit(X_cal, y_cal)
    
    # Evaluate
    proba_train = calibrator.predict_proba(X_train)[:, 1]
    proba_test = calibrator.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, proba_train)
    test_auc = roc_auc_score(y_test, proba_test)
    test_pr_auc = average_precision_score(y_test, proba_test)
    
    # More metrics
    y_pred = (proba_test >= 0.5).astype(int)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Test AUC:  {test_auc:.4f}")
    print(f"  Test PR AUC: {test_pr_auc:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    
    # Save models
    rf_path = cfg.base_dir / f"models/rf_v2_iter{iteration:02d}_pca{cfg.pca_dim}.pkl"
    pca_path = cfg.base_dir / f"models/pca_v2_iter{iteration:02d}_pca{cfg.pca_dim}.pkl"
    
    joblib.dump({
        'rf': rf,
        'calibrator': calibrator,
        'config': {
            'iteration': iteration,
            'pca_dim': cfg.pca_dim,
            'n_estimators': cfg.rf_n_estimators,
            'max_depth': cfg.rf_max_depth
        },
        'metrics': {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'test_pr_auc': test_pr_auc,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
    }, rf_path)
    
    joblib.dump(pca, pca_path)
    
    print(f"  💾 Saved: {rf_path.name}")
    print(f"  💾 Saved: {pca_path.name}")
    
    return {
        'rf': rf,
        'calibrator': calibrator,
        'pca': pca,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'test_pr_auc': test_pr_auc,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'rf_path': str(rf_path),
        'pca_path': str(pca_path)
    }


# ---------------------------------------------------------------------------
# Reward Function (CORRECT - from 58/60)
# ---------------------------------------------------------------------------

def compute_rf_advantage(rf_calibrator, X_combined: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """
    Compute RF advantage: A(x,a) = Q(x,a) - V(x)
    where V(x) = mean(Q(x,0), Q(x,1))
    
    This is the CORRECT implementation from 58/60 scripts.
    """
    probs = rf_calibrator.predict_proba(X_combined)  # shape: (batch, 2)
    # Q(x, a=0) = probs[:, 0], Q(x, a=1) = probs[:, 1]
    q_values = probs[np.arange(len(actions)), actions]  # Q(x, a_chosen)
    v_values = probs.mean(axis=1)  # V(x) = (Q(x,0) + Q(x,1)) / 2
    return q_values - v_values


def compute_rewards(
    rf_calibrator,
    X_combined: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    cfg: IterativeConfigV2,
) -> tuple:
    """
    Compute rewards based on reward_mode.
    CORRECT implementation from 58/60 scripts:
      r_total = r_rf + lambda_acc * r_acc
    
    where r_rf = reward_scale * clip(advantage, -0.45, 0.45)
    """
    # 1. RF advantage component
    adv = compute_rf_advantage(rf_calibrator, X_combined, actions)
    clip_low, clip_high = cfg.reward_clip_advantage
    adv_clipped = np.clip(adv, clip_low, clip_high)
    r_rf = cfg.reward_scale * adv_clipped
    
    # 2. Accuracy reward component
    if cfg.reward_mode == "rf_plus_acc":
        correct = (actions == labels).astype(float)
        r_acc = correct - 0.5  # +0.5 if correct, -0.5 if wrong
        
    elif cfg.reward_mode == "fn_heavy":
        r_acc = np.zeros(len(labels), dtype=np.float32)
        for i in range(len(labels)):
            y, a = labels[i], actions[i]
            if y == 1 and a == 1:      # TP
                r_acc[i] = +1.0
            elif y == 1 and a == 0:    # FN (heavy penalty)
                r_acc[i] = -1.5
            elif y == 0 and a == 0:    # TN
                r_acc[i] = +0.2
            elif y == 0 and a == 1:    # FP
                r_acc[i] = -0.2
        correct = (actions == labels).astype(float)
    else:
        raise ValueError(f"Unknown reward_mode: {cfg.reward_mode}")
    
    # 3. Combined reward
    r_total = r_rf + cfg.lambda_acc * r_acc
    
    diagnostics = {
        "r_rf_mean": float(r_rf.mean()),
        "r_acc_mean": float(r_acc.mean()),
        "r_total_mean": float(r_total.mean()),
        "accuracy": float(correct.mean()),
        "adv_mean": float(adv.mean()),
        "adv_clipped_mean": float(adv_clipped.mean()),
    }
    
    return r_total, diagnostics


# ---------------------------------------------------------------------------
# PPO Training (CORRECT - matching 58/60)
# ---------------------------------------------------------------------------

def train_ppo_epoch(model, tokenizer, optimizer, data, rf_calibrator, pca, cfg, epoch):
    """
    Train one PPO epoch with CORRECT reward function.
    """
    model.train()
    
    texts = data['texts']
    y_all = data['y_all']
    train_idx = data['train_idx']
    X_tabular_full = data['X_tabular_full']
    
    # Weighted sampling (favor positive class)
    train_labels = y_all[train_idx]
    weights = np.where(train_labels == 1, cfg.y1_weight, 1.0)
    weights = weights / weights.sum()
    
    n_samples = min(cfg.ppo_samples_per_epoch, len(train_idx))
    sample_idx = np.random.choice(train_idx, size=n_samples, replace=True, p=weights)
    
    epoch_ppo_losses = []
    epoch_policy_losses = []
    epoch_value_losses = []
    epoch_entropies = []
    epoch_rewards = []
    epoch_actions = []
    epoch_ratios = []
    epoch_clip_fractions = []
    epoch_reward_diag = []
    
    for i in range(0, n_samples, cfg.ppo_batch_size):
        batch_idx = sample_idx[i:i + cfg.ppo_batch_size]
        batch_texts = [texts[j] for j in batch_idx]
        batch_y = y_all[batch_idx]
        batch_tab = X_tabular_full[batch_idx]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors='pt'
        ).to(DEVICE)
        
        # Forward pass to get old policy
        with torch.no_grad():
            old_logits, old_embeddings = model(
                inputs['input_ids'],
                inputs['attention_mask'],
                return_embeddings=True
            )
            old_probs = F.softmax(old_logits, dim=-1)
            
            # Sample actions from old policy
            actions_dist = torch.distributions.Categorical(old_probs)
            actions = actions_dist.sample()
            old_log_probs = actions_dist.log_prob(actions)
            
            # Get old value estimates
            old_values = model.value_head(old_embeddings).squeeze(-1)
        
        # Compute rewards using CORRECT function
        with torch.no_grad():
            phi = pca.transform(old_embeddings.cpu().numpy())
            X_combined = np.concatenate([batch_tab, phi], axis=1)
            
            rewards, reward_diag = compute_rewards(
                rf_calibrator, X_combined, 
                actions.cpu().numpy(), batch_y, cfg
            )
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
            epoch_reward_diag.append(reward_diag)
        
        # Compute advantages
        advantages = rewards_tensor - old_values.detach()
        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(cfg.ppo_epochs_per_batch):
            logits, embeddings = model(
                inputs['input_ids'],
                inputs['attention_mask'],
                return_embeddings=True
            )
            probs = F.softmax(logits, dim=-1)
            
            # Current policy
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Value prediction
            values = model.value_head(embeddings).squeeze(-1)
            
            # Ratio
            ratio = torch.exp(log_probs - old_log_probs.detach())
            
            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - cfg.ppo_clip_epsilon, 1 + cfg.ppo_clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, rewards_tensor)
            
            # Total loss
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
        
        # Record stats
        epoch_ppo_losses.append(loss.item())
        epoch_policy_losses.append(policy_loss.item())
        epoch_value_losses.append(value_loss.item())
        epoch_entropies.append(entropy.item())
        epoch_rewards.append(rewards.mean())
        epoch_actions.extend(actions.cpu().numpy())
        epoch_ratios.append(ratio.mean().item())
        epoch_clip_fractions.append(((ratio - 1).abs() > cfg.ppo_clip_epsilon).float().mean().item())
    
    # Aggregate reward diagnostics
    agg_diag = {}
    for key in epoch_reward_diag[0].keys():
        agg_diag[key] = np.mean([d[key] for d in epoch_reward_diag])
    
    return {
        'ppo_loss': np.mean(epoch_ppo_losses),
        'policy_loss': np.mean(epoch_policy_losses),
        'value_loss': np.mean(epoch_value_losses),
        'entropy': np.mean(epoch_entropies),
        'reward': np.mean(epoch_rewards),
        'action_mean': np.mean(epoch_actions),
        'ratio_mean': np.mean(epoch_ratios),
        'clip_fraction': np.mean(epoch_clip_fractions),
        **agg_diag
    }


@torch.no_grad()
def evaluate_llm(model, tokenizer, data, split_idx, cfg):
    """Evaluate LLM on a data split"""
    model.eval()
    
    texts = data['texts']
    y_all = data['y_all']
    
    all_probs = []
    all_labels = []
    
    for i in range(0, len(split_idx), cfg.ppo_batch_size):
        batch_idx = split_idx[i:i + cfg.ppo_batch_size]
        batch_texts = [texts[j] for j in batch_idx]
        batch_y = y_all[batch_idx]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors='pt'
        ).to(DEVICE)
        
        logits, _ = model(
            inputs['input_ids'],
            inputs['attention_mask'],
            return_embeddings=True
        )
        probs = F.softmax(logits, dim=-1)[:, 1]
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch_y)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_probs)
    
    y_pred = (all_probs >= 0.5).astype(int)
    recall = recall_score(all_labels, y_pred, zero_division=0)
    precision = precision_score(all_labels, y_pred, zero_division=0)
    f1 = f1_score(all_labels, y_pred, zero_division=0)
    
    # P separation
    pos_mean = all_probs[all_labels == 1].mean()
    neg_mean = all_probs[all_labels == 0].mean()
    p_sep = pos_mean - neg_mean
    
    return {
        'auc': auc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'p_separation': p_sep
    }


def train_ppo(model, tokenizer, data, rf_calibrator, pca, cfg, iteration):
    """Full PPO training for one iteration"""
    print(f"\n🚀 PPO Training (Iteration {iteration})")
    print("=" * 60)
    print(f"  reward_mode: {cfg.reward_mode}")
    print(f"  reward_scale: {cfg.reward_scale}")
    print(f"  lambda_acc: {cfg.lambda_acc}")
    print(f"  entropy_coef: {cfg.entropy_coef}")
    print(f"  clip_epsilon: {cfg.ppo_clip_epsilon}")
    print("=" * 60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ppo_lr)
    
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    
    checkpoint_dir = cfg.base_dir / f"checkpoints/iterative_v2_ppo_iter{iteration:02d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    history = []
    
    for epoch in range(1, cfg.ppo_num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_stats = train_ppo_epoch(
            model, tokenizer, optimizer, data, rf_calibrator, pca, cfg, epoch
        )
        
        # Evaluate on val
        val_metrics = evaluate_llm(model, tokenizer, data, data['val_idx'], cfg)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch}/{cfg.ppo_num_epochs} ({elapsed:.1f}s):")
        print(f"  PPO Loss: {train_stats['ppo_loss']:.4f} | Policy: {train_stats['policy_loss']:.4f} | Value: {train_stats['value_loss']:.4f}")
        print(f"  Reward: {train_stats['reward']:.4f} (r_rf={train_stats['r_rf_mean']:.3f}, r_acc={train_stats['r_acc_mean']:.3f})")
        print(f"  Val AUC: {val_metrics['auc']:.4f} | p_sep: {val_metrics['p_separation']:+.4f}")
        print(f"  Entropy: {train_stats['entropy']:.4f} | Ratio: {train_stats['ratio_mean']:.3f} | Clip%: {train_stats['clip_fraction']*100:.1f}%")
        
        history.append({
            'epoch': epoch,
            **train_stats,
            'val_auc': val_metrics['auc'],
            'val_p_separation': val_metrics['p_separation']
        })
        
        # Check for improvement
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['auc'],
                'iteration': iteration
            }, checkpoint_dir / 'best.pt')
            
            print(f"  🌟 NEW BEST (saved)")
        else:
            patience_counter += 1
            if patience_counter >= cfg.ppo_patience:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break
    
    # Final test evaluation
    model, _ = load_llm_checkpoint(model, checkpoint_dir / 'best.pt')
    test_metrics = evaluate_llm(model, tokenizer, data, data['test_idx'], cfg)
    
    print(f"\n📊 Best Model (epoch {best_epoch}):")
    print(f"  Val AUC: {best_val_auc:.4f}")
    print(f"  Test AUC: {test_metrics['auc']:.4f}")
    print(f"  Test Recall: {test_metrics['recall']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    
    return {
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_auc': test_metrics['auc'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'checkpoint_path': str(checkpoint_dir / 'best.pt'),
        'history': history
    }


# ---------------------------------------------------------------------------
# Main Iterative Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Iterative PPO + RF Training Loop V2")
    parser.add_argument('--resume', type=int, default=0, help='Resume from iteration N')
    parser.add_argument('--reward_mode', type=str, default='fn_heavy', 
                        choices=['fn_heavy', 'rf_plus_acc'], help='Reward mode')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔄 ITERATIVE TRAINING LOOP V2: PPO + RF Auto-Convergence")
    print("   With CORRECT Reward Function (from 58/60 scripts)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print(f"Device: {DEVICE}")
    print(f"Early stopping: RF AND LLM AUC both no improvement for 5 iterations")
    print("=" * 80)
    
    cfg = IterativeConfigV2()
    cfg.reward_mode = args.reward_mode
    set_seed(cfg.seed)
    
    print(f"\n📋 Configuration:")
    print(f"  reward_mode: {cfg.reward_mode}")
    print(f"  reward_scale: {cfg.reward_scale}")
    print(f"  lambda_acc: {cfg.lambda_acc}")
    print(f"  entropy_coef: {cfg.entropy_coef}")
    print(f"  reward_clip: {cfg.reward_clip_advantage}")
    print(f"  pca_dim: {cfg.pca_dim}")
    
    # Load data
    data = load_data(cfg)
    
    # History tracking
    history_path = cfg.base_dir / "results/iterative_training_v2_history.json"
    
    if args.resume > 0 and history_path.exists():
        with open(history_path, 'r') as f:
            full_history = json.load(f)
        start_iteration = args.resume
        print(f"\n📂 Resuming from iteration {start_iteration}")
    else:
        full_history = {
            'config': {k: str(v) if isinstance(v, Path) else v 
                      for k, v in asdict(cfg).items()},
            'iterations': [],
            'rf_auc_history': [],
            'llm_auc_history': [],
            'best_rf_auc': 0,
            'best_llm_auc': 0,
            'best_rf_iteration': 0,
            'best_llm_iteration': 0,
            'converged': False,
            'convergence_reason': None
        }
        start_iteration = 1
    
    # For first iteration, we need an initial RF
    if start_iteration == 1:
        print("\n" + "=" * 80)
        print("📦 ITERATION 0: Creating Initial RF (from pretrained LLM)")
        print("=" * 80)
        
        # Create fresh LLM
        model = create_llm_model(cfg)
        tokenizer = model.tokenizer
        
        # Extract embeddings from pretrained (not fine-tuned) LLM
        all_embeddings = extract_embeddings(model, tokenizer, data['texts'], cfg)
        
        embeddings_train = all_embeddings[data['full_train_idx']]
        embeddings_test = all_embeddings[data['test_idx']]
        
        X_train_tab = data['X_tabular_full'][data['full_train_idx']]
        X_test_tab = data['X_tabular_full'][data['test_idx']]
        y_train = data['y_all'][data['full_train_idx']]
        y_test = data['y_all'][data['test_idx']]
        
        # Train initial RF
        rf_result = train_rf(
            embeddings_train, y_train,
            embeddings_test, y_test,
            X_train_tab, X_test_tab,
            cfg, iteration=0
        )
        
        full_history['iterations'].append({
            'iteration': 0,
            'type': 'initial_rf',
            'rf_test_auc': rf_result['test_auc'],
            'rf_test_recall': rf_result['test_recall']
        })
        full_history['rf_auc_history'].append(rf_result['test_auc'])
        full_history['best_rf_auc'] = rf_result['test_auc']
        full_history['best_rf_iteration'] = 0
        
        current_rf_path = rf_result['rf_path']
        current_pca_path = rf_result['pca_path']
    else:
        # Load from previous iteration
        prev_iter = start_iteration - 1
        current_rf_path = str(cfg.base_dir / f"models/rf_v2_iter{prev_iter:02d}_pca{cfg.pca_dim}.pkl")
        current_pca_path = str(cfg.base_dir / f"models/pca_v2_iter{prev_iter:02d}_pca{cfg.pca_dim}.pkl")
    
    # Main iterative loop
    iteration = start_iteration
    no_improve_count = 0  # Counter for BOTH not improving
    
    while True:
        print("\n" + "=" * 80)
        print(f"🔄 ITERATION {iteration}")
        print("=" * 80)
        
        # Load current RF
        rf_data = joblib.load(current_rf_path)
        rf_calibrator = rf_data['calibrator']
        pca = joblib.load(current_pca_path)
        
        print(f"📂 Using RF: {Path(current_rf_path).name}")
        
        # Create fresh LLM for this iteration
        model = create_llm_model(cfg)
        tokenizer = model.tokenizer
        
        # If not first iteration, load previous best LLM
        if iteration > 1:
            prev_checkpoint = cfg.base_dir / f"checkpoints/iterative_v2_ppo_iter{iteration-1:02d}/best.pt"
            if prev_checkpoint.exists():
                model, _ = load_llm_checkpoint(model, prev_checkpoint)
                print(f"📂 Loaded LLM from iteration {iteration-1}")
        
        # Train PPO
        ppo_result = train_ppo(model, tokenizer, data, rf_calibrator, pca, cfg, iteration)
        
        # Load best LLM checkpoint
        best_checkpoint = cfg.base_dir / f"checkpoints/iterative_v2_ppo_iter{iteration:02d}/best.pt"
        model, _ = load_llm_checkpoint(model, best_checkpoint)
        
        # Extract new embeddings
        print("\n📦 Extracting embeddings for RF update...")
        all_embeddings = extract_embeddings(model, tokenizer, data['texts'], cfg)
        
        embeddings_train = all_embeddings[data['full_train_idx']]
        embeddings_test = all_embeddings[data['test_idx']]
        
        X_train_tab = data['X_tabular_full'][data['full_train_idx']]
        X_test_tab = data['X_tabular_full'][data['test_idx']]
        y_train = data['y_all'][data['full_train_idx']]
        y_test = data['y_all'][data['test_idx']]
        
        # Train new RF
        rf_result = train_rf(
            embeddings_train, y_train,
            embeddings_test, y_test,
            X_train_tab, X_test_tab,
            cfg, iteration=iteration
        )
        
        # Update history
        iter_record = {
            'iteration': iteration,
            'ppo': {
                'best_epoch': ppo_result['best_epoch'],
                'best_val_auc': ppo_result['best_val_auc'],
                'test_auc': ppo_result['test_auc'],
                'test_recall': ppo_result['test_recall'],
                'test_f1': ppo_result['test_f1'],
                'checkpoint': ppo_result['checkpoint_path']
            },
            'rf': {
                'test_auc': rf_result['test_auc'],
                'test_pr_auc': rf_result['test_pr_auc'],
                'test_recall': rf_result['test_recall'],
                'test_f1': rf_result['test_f1'],
                'model_path': rf_result['rf_path']
            }
        }
        full_history['iterations'].append(iter_record)
        full_history['rf_auc_history'].append(rf_result['test_auc'])
        full_history['llm_auc_history'].append(ppo_result['test_auc'])
        
        # Check for improvement (NEW: both must improve to reset counter)
        rf_improved = rf_result['test_auc'] > full_history['best_rf_auc']
        llm_improved = ppo_result['test_auc'] > full_history['best_llm_auc']
        
        if rf_improved:
            full_history['best_rf_auc'] = rf_result['test_auc']
            full_history['best_rf_iteration'] = iteration
            print(f"\n🌟 NEW BEST RF AUC: {rf_result['test_auc']:.4f}")
        
        if llm_improved:
            full_history['best_llm_auc'] = ppo_result['test_auc']
            full_history['best_llm_iteration'] = iteration
            print(f"🌟 NEW BEST LLM AUC: {ppo_result['test_auc']:.4f}")
        
        # NEW EARLY STOPPING: only increment if BOTH don't improve
        if rf_improved or llm_improved:
            no_improve_count = 0
            print(f"\n✅ At least one improved, resetting counter")
        else:
            no_improve_count += 1
            print(f"\n⚠️ Neither RF nor LLM improved ({no_improve_count}/{cfg.convergence_patience})")
        
        # Save history
        with open(history_path, 'w') as f:
            json.dump(full_history, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "-" * 60)
        print(f"📊 Iteration {iteration} Summary:")
        print(f"  LLM Test AUC: {ppo_result['test_auc']:.4f} (best: {full_history['best_llm_auc']:.4f} @ iter {full_history['best_llm_iteration']})")
        print(f"  RF Test AUC:  {rf_result['test_auc']:.4f} (best: {full_history['best_rf_auc']:.4f} @ iter {full_history['best_rf_iteration']})")
        print(f"  No-improve count: {no_improve_count}/{cfg.convergence_patience}")
        print("-" * 60)
        
        # Check early stopping
        if no_improve_count >= cfg.convergence_patience:
            print("\n" + "=" * 80)
            print("🛑 CONVERGENCE REACHED!")
            print(f"   Neither RF nor LLM improved for {cfg.convergence_patience} consecutive iterations")
            print("=" * 80)
            
            full_history['converged'] = True
            full_history['convergence_reason'] = f"Neither RF nor LLM improved for {cfg.convergence_patience} iterations"
            full_history['total_iterations'] = iteration
            
            with open(history_path, 'w') as f:
                json.dump(full_history, f, indent=2, default=str)
            
            break
        
        # Update paths for next iteration
        current_rf_path = rf_result['rf_path']
        current_pca_path = rf_result['pca_path']
        iteration += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    print(f"Total iterations: {iteration}")
    print(f"Best RF AUC: {full_history['best_rf_auc']:.4f} (iteration {full_history['best_rf_iteration']})")
    print(f"Best LLM AUC: {full_history['best_llm_auc']:.4f} (iteration {full_history['best_llm_iteration']})")
    print(f"\nRF AUC progression: {' → '.join([f'{x:.4f}' for x in full_history['rf_auc_history']])}")
    print(f"LLM AUC progression: {' → '.join([f'{x:.4f}' for x in full_history['llm_auc_history']])}")
    print(f"\n💾 History saved: {history_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
