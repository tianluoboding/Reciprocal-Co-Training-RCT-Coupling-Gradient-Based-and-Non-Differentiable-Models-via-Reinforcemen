#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Cross-Entropy Baseline for Diabetes (BRFSS 2015) Dataset

Trains BioClinicalBERT + LoRA with standard cross-entropy loss.
No RF involvement, no reward shaping, no value head.
Uses the same data split and model config as the iterative framework.
"""

import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score, accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

import os
os.environ['PYTHONUNBUFFERED'] = '1'

BASE_DIR = Path(__file__).parent.parent  # Diabetes_exp/
sys.path.insert(0, str(BASE_DIR / "src"))

from models.llm_wrapper import BioClinicalBERTWithLoRA

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

# --- Config (matched to iterative framework) ---
NUM_EPOCHS = 50
BATCH_SIZE = 32
LR = 3e-5
PATIENCE = 8
VAL_RATIO = 0.15
MAX_LENGTH = 512
SAMPLES_PER_EPOCH = 2000
Y1_WEIGHT = 1.5


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, tokenizer, texts, labels, split_idx):
    model.eval()
    all_probs, all_labels = [], []

    for i in range(0, len(split_idx), BATCH_SIZE):
        batch_idx = split_idx[i:i + BATCH_SIZE]
        batch_texts = [texts[j] for j in batch_idx]
        batch_y = labels[batch_idx]

        inputs = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors='pt'
        ).to(DEVICE)

        logits, _ = model(
            inputs['input_ids'], inputs['attention_mask'],
            return_embeddings=True
        )
        probs = F.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch_y)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    y_pred = (all_probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_labels, y_pred).ravel()
    return {
        'auc': auc,
        'pr_auc': average_precision_score(all_labels, all_probs),
        'accuracy': accuracy_score(all_labels, y_pred),
        'recall': recall_score(all_labels, y_pred, zero_division=0),
        'precision': precision_score(all_labels, y_pred, zero_division=0),
        'f1': f1_score(all_labels, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    }


def main():
    set_seed(SEED)

    print("=" * 80)
    print("LLM Cross-Entropy Baseline: Diabetes (BRFSS 2015) Dataset")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print(f"Device: {DEVICE}")

    # ---- Load data ----
    data_dir = BASE_DIR / "data"

    with open(data_dir / 'raw/descriptive_text.json') as f:
        text_data = json.load(f)
    texts = [item['prompt'] for item in text_data]

    y_all = np.load(data_dir / 'raw/y_all.npy')

    with open(data_dir / 'raw/train_idxs.json') as f:
        train_idx = json.load(f)
    with open(data_dir / 'raw/test_idxs.json') as f:
        test_idx = json.load(f)

    train_sub_idx, val_idx = train_test_split(
        train_idx, test_size=VAL_RATIO,
        random_state=SEED, stratify=y_all[train_idx]
    )

    print(f"\n  Train: {len(train_sub_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    print(f"  Positive rate (train): {y_all[train_sub_idx].mean():.2%}")

    # ---- Create model ----
    model = BioClinicalBERTWithLoRA(
        model_name_or_path='emilyalsentzer/Bio_ClinicalBERT',
        lora_config={'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': 0.05,
                     'target_modules': ['query', 'value']},
        num_labels=2, device=DEVICE
    )
    tokenizer = model.tokenizer

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Balanced dataset so equal weights
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    checkpoint_dir = BASE_DIR / "checkpoints" / "llm_ce_baseline"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"\n{'='*80}")
    print("Training LLM with Cross-Entropy Loss")
    print(f"{'='*80}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()

        train_labels = y_all[train_sub_idx]
        weights = np.where(train_labels == 1, Y1_WEIGHT, 1.0)
        weights = weights / weights.sum()
        n_samples = min(SAMPLES_PER_EPOCH, len(train_sub_idx))
        sample_idx = np.random.choice(train_sub_idx, size=n_samples, replace=True, p=weights)

        epoch_losses = []
        for i in range(0, n_samples, BATCH_SIZE):
            batch_idx = sample_idx[i:i + BATCH_SIZE]
            batch_texts = [texts[j] for j in batch_idx]
            batch_y = torch.tensor(y_all[batch_idx], dtype=torch.long, device=DEVICE)

            inputs = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors='pt'
            ).to(DEVICE)

            logits, _ = model(
                inputs['input_ids'], inputs['attention_mask'],
                return_embeddings=True
            )
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_losses.append(loss.item())

        val_metrics = evaluate(model, tokenizer, texts, y_all, val_idx)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{NUM_EPOCHS} ({elapsed:.1f}s):  "
              f"Loss: {np.mean(epoch_losses):.4f}  "
              f"Val AUC: {val_metrics['auc']:.4f}  "
              f"Val Acc: {val_metrics['accuracy']:.4f}")

        history.append({'epoch': epoch, 'loss': float(np.mean(epoch_losses)),
                        **{f'val_{k}': v for k, v in val_metrics.items()}})

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_auc': val_metrics['auc']},
                       checkpoint_dir / 'best.pt')
            print(f"  -> NEW BEST (saved)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ---- Final test ----
    ckpt = torch.load(checkpoint_dir / 'best.pt', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate(model, tokenizer, texts, y_all, test_idx)

    print(f"\nBest Model (epoch {best_epoch}):")
    print(f"  Val AUC:         {best_val_auc:.4f}")
    print(f"  Test AUC:        {test_metrics['auc']:.4f}")
    print(f"  Test PR AUC:     {test_metrics['pr_auc']:.4f}")
    print(f"  Test Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"  Test Recall:     {test_metrics['recall']:.4f}")
    print(f"  Test Precision:  {test_metrics['precision']:.4f}")
    print(f"  Test Specificity:{test_metrics['specificity']:.4f}")
    print(f"  Test F1:         {test_metrics['f1']:.4f}")

    # ---- Save results ----
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': 'Diabetes (BRFSS 2015)',
        'baseline': 'LLM + Cross-Entropy',
        'timestamp': str(datetime.now()),
        'best_epoch': best_epoch,
        'best_val_auc': float(best_val_auc),
        'metrics': {k: float(v) for k, v in test_metrics.items()},
        'history': history,
        'config': {
            'num_epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE,
            'lr': LR, 'patience': PATIENCE, 'max_length': MAX_LENGTH,
            'samples_per_epoch': SAMPLES_PER_EPOCH,
            'lora_r': 8, 'lora_alpha': 16, 'seed': SEED,
        }
    }
    results_path = results_dir / 'llm_ce_baseline.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
