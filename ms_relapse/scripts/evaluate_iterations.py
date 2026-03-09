#!/usr/bin/env python3
"""
Post-hoc Evaluation of V2 Iterative Training (No Validation Set)

Loads all saved RF models and LLM checkpoints from V2 training,
runs inference on the test set, and outputs:
  1. Full metrics table at threshold=0.5 for every iteration
  2. Threshold-matched evaluation at 80% recall for every iteration
  3. Summary table selecting the best iteration

No retraining — only model.eval() forward passes and calibrator.predict_proba().

USAGE:
  python 85_evaluate_v2_threshold.py
  python 85_evaluate_v2_threshold.py --target-recall 0.80
"""

import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    recall_score, precision_score, f1_score,
    confusion_matrix, roc_curve
)
import joblib

import os
os.environ['PYTHONUNBUFFERED'] = '1'

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from data.loader import DataLoader
from models.llm_wrapper import BioClinicalBERTWithLoRA

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
PCA_DIM = 5
MAX_LENGTH = 512
BATCH_SIZE = 32


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, proba, threshold=0.5):
    auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'accuracy': float(acc),
        'recall': float(rec),
        'specificity': float(spec),
        'precision': float(prec),
        'f1': float(f1),
        'auc': float(auc),
        'pr_auc': float(pr_auc),
    }


def find_threshold_at_recall(y_true, proba, target_recall=0.80):
    """Find the highest threshold achieving at least target_recall."""
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    valid = tpr >= target_recall
    if not valid.any():
        return 0.0
    idx = np.where(valid)[0][0]
    if idx == 0:
        return float(thresholds[0]) if len(thresholds) > 0 else 0.0
    return float(thresholds[idx])


def metrics_at_threshold(y_true, proba, threshold):
    auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'threshold': float(threshold),
        'accuracy': float(acc),
        'recall': float(rec),
        'specificity': float(spec),
        'precision': float(prec),
        'f1': float(f1),
        'auc': float(auc),
        'pr_auc': float(pr_auc),
    }


def load_data():
    """Load data with original train/test split (no validation sub-split)."""
    loader = DataLoader(BASE_DIR / "data")
    all_data = loader.load_all()

    texts = all_data['texts']
    y_all = all_data['y']
    train_idx = all_data['train_idx']
    test_idx = all_data['test_idx']

    X_train_tab = np.load(BASE_DIR / 'data/processed/X_train_selected.npy')
    X_test_tab = np.load(BASE_DIR / 'data/processed/X_test_selected.npy')

    X_tabular_full = np.zeros((len(y_all), X_train_tab.shape[1]), dtype=np.float32)
    X_tabular_full[train_idx] = X_train_tab
    X_tabular_full[test_idx] = X_test_tab

    return {
        'texts': texts,
        'y_all': y_all,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'X_tabular_full': X_tabular_full,
    }


def create_model():
    lora_config = {
        'lora_r': 8, 'lora_alpha': 16,
        'lora_dropout': 0.05, 'target_modules': ['query', 'value'],
    }
    return BioClinicalBERTWithLoRA(
        model_name_or_path='emilyalsentzer/Bio_ClinicalBERT',
        lora_config=lora_config, num_labels=2, device=DEVICE
    )


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE)
    return model


@torch.no_grad()
def extract_embeddings(model, tokenizer, texts):
    model.eval()
    all_emb = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors='pt'
        ).to(DEVICE)
        _, emb = model(inputs['input_ids'], inputs['attention_mask'], return_embeddings=True)
        all_emb.append(emb.cpu().numpy())
    return np.vstack(all_emb)


@torch.no_grad()
def llm_predict_proba(model, tokenizer, texts):
    model.eval()
    all_probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors='pt'
        ).to(DEVICE)
        logits, _ = model(inputs['input_ids'], inputs['attention_mask'], return_embeddings=True)
        probs = F.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


def print_table(title, rows, columns, col_labels=None):
    if col_labels is None:
        col_labels = columns
    header = f"{'Iter':>4}  " + "  ".join(f"{c:>11}" for c in col_labels)
    sep = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        it = row['iter']
        vals = "  ".join(f"{row[c]:>11.4f}" for c in columns)
        print(f"{it:>4}  {vals}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Threshold-matched evaluation for V2 iterative training")
    parser.add_argument('--target-recall', type=float, default=0.80)
    args = parser.parse_args()

    target = args.target_recall
    tag = "v2"

    set_seed(SEED)
    print("=" * 80)
    print(f"Post-hoc Evaluation — V2 Iterative Training (no validation set)")
    print(f"Timestamp: {datetime.now()}")
    print(f"Device: {DEVICE}")
    print(f"Target recall for threshold matching: {target}")
    print("=" * 80)

    data = load_data()
    texts = data['texts']
    y_all = data['y_all']
    test_idx = data['test_idx']
    y_test = y_all[test_idx]
    X_tab_test = data['X_tabular_full'][test_idx]
    test_texts = [texts[j] for j in test_idx]

    print(f"\n  Train: {len(data['train_idx'])}, Test: {len(test_idx)}")
    print(f"  Test set: {len(y_test)} samples, pos rate: {y_test.mean():.2%}")

    rf_iters = sorted([
        int(p.stem.split('_iter')[1].split('_')[0])
        for p in (BASE_DIR / "models").glob(f"rf_v2_iter*_pca{PCA_DIM}.pkl")
    ])
    llm_iters = sorted([
        int(p.parent.name.split('_iter')[1])
        for p in (BASE_DIR / "checkpoints").glob("iterative_v2_ppo_iter*/best.pt")
    ])

    print(f"  RF iterations found:  {rf_iters}")
    print(f"  LLM iterations found: {llm_iters}")

    if not rf_iters:
        print("ERROR: No RF models found. Has training completed?")
        return

    # --- LLM evaluation ---
    test_emb_cache = {}

    print("\n--- Evaluating LLM checkpoints ---")

    print("  Iter 0: pretrained Bio-ClinicalBERT (no checkpoint)")
    model_fresh = create_model()
    tokenizer = model_fresh.tokenizer
    test_emb_cache[0] = extract_embeddings(model_fresh, tokenizer, test_texts)
    del model_fresh
    torch.cuda.empty_cache()

    llm_results_05 = []
    llm_results_thresh = []

    for it in llm_iters:
        ckpt_path = BASE_DIR / f"checkpoints/iterative_v2_ppo_iter{it:02d}/best.pt"
        if not ckpt_path.exists():
            print(f"  Iter {it}: [SKIP] {ckpt_path} not found")
            continue

        print(f"  Iter {it}: loading {ckpt_path.name}")
        model = create_model()
        model = load_checkpoint(model, ckpt_path)

        proba_llm = llm_predict_proba(model, tokenizer, test_texts)

        m_05 = compute_metrics(y_test, proba_llm)
        m_05['iter'] = it
        llm_results_05.append(m_05)

        thresh = find_threshold_at_recall(y_test, proba_llm, target)
        m_t = metrics_at_threshold(y_test, proba_llm, thresh)
        m_t['iter'] = it
        llm_results_thresh.append(m_t)

        print(f"    @0.5:  AUC={m_05['auc']:.4f}  Recall={m_05['recall']:.4f}  "
              f"Spec={m_05['specificity']:.4f}  F1={m_05['f1']:.4f}")
        print(f"    @{target}: t={thresh:.4f}  AUC={m_t['auc']:.4f}  Recall={m_t['recall']:.4f}  "
              f"Spec={m_t['specificity']:.4f}  F1={m_t['f1']:.4f}")

        test_emb_cache[it] = extract_embeddings(model, tokenizer, test_texts)
        del model
        torch.cuda.empty_cache()

    # --- RF evaluation ---
    print("\n--- Evaluating RF models ---")
    rf_results_05 = []
    rf_results_thresh = []

    for it in rf_iters:
        rf_path = BASE_DIR / f"models/rf_v2_iter{it:02d}_pca{PCA_DIM}.pkl"
        pca_path = BASE_DIR / f"models/pca_v2_iter{it:02d}_pca{PCA_DIM}.pkl"

        if not rf_path.exists() or not pca_path.exists():
            print(f"  Iter {it}: [SKIP] model files not found")
            continue

        rf_data = joblib.load(rf_path)
        calibrator = rf_data['calibrator']
        pca = joblib.load(pca_path)

        emb_key = it if it in test_emb_cache else 0
        emb_test = test_emb_cache[emb_key]
        phi_test = pca.transform(emb_test)
        X_test_combined = np.concatenate([X_tab_test, phi_test], axis=1)

        proba_rf = calibrator.predict_proba(X_test_combined)[:, 1]

        m_05 = compute_metrics(y_test, proba_rf)
        m_05['iter'] = it
        rf_results_05.append(m_05)

        thresh = find_threshold_at_recall(y_test, proba_rf, target)
        m_t = metrics_at_threshold(y_test, proba_rf, thresh)
        m_t['iter'] = it
        rf_results_thresh.append(m_t)

        print(f"  Iter {it}: @0.5 AUC={m_05['auc']:.4f} Recall={m_05['recall']:.4f} "
              f"Spec={m_05['specificity']:.4f} F1={m_05['f1']:.4f}")
        print(f"           @{target} t={thresh:.4f} AUC={m_t['auc']:.4f} Recall={m_t['recall']:.4f} "
              f"Spec={m_t['specificity']:.4f} F1={m_t['f1']:.4f}")

    # --- Print tables ---
    cols = ['accuracy', 'recall', 'specificity', 'precision', 'f1', 'auc', 'pr_auc']
    thresh_cols = ['threshold', 'accuracy', 'recall', 'specificity', 'precision', 'f1', 'auc', 'pr_auc']

    print_table(f"RF Metrics @ threshold=0.5 — {tag}", rf_results_05, cols)
    print_table(f"LLM Metrics @ threshold=0.5 — {tag}", llm_results_05, cols)
    print_table(f"RF Metrics @ {int(target*100)}% Recall — {tag}", rf_results_thresh, thresh_cols)
    print_table(f"LLM Metrics @ {int(target*100)}% Recall — {tag}", llm_results_thresh, thresh_cols)

    # --- Best iteration summary ---
    if llm_results_05:
        best_llm_05 = max(llm_results_05, key=lambda x: x['auc'])
        best_llm_t = max(llm_results_thresh, key=lambda x: x['auc'])
    else:
        best_llm_05 = best_llm_t = None

    best_rf_05 = max(rf_results_05, key=lambda x: x['auc'])
    best_rf_t = max(rf_results_thresh, key=lambda x: x['auc'])

    print(f"\n{'='*80}")
    print(f"SUMMARY — V2 Iterative Training (no validation set)")
    print(f"{'='*80}")
    print(f"\nBest RF  @0.5: iter {best_rf_05['iter']}  AUC={best_rf_05['auc']:.4f}  "
          f"Recall={best_rf_05['recall']:.4f}  F1={best_rf_05['f1']:.4f}")
    print(f"Best RF  @{int(target*100)}%:  iter {best_rf_t['iter']}  AUC={best_rf_t['auc']:.4f}  "
          f"Recall={best_rf_t['recall']:.4f}  Spec={best_rf_t['specificity']:.4f}  "
          f"F1={best_rf_t['f1']:.4f}  thresh={best_rf_t['threshold']:.4f}")
    if best_llm_05:
        print(f"Best LLM @0.5: iter {best_llm_05['iter']}  AUC={best_llm_05['auc']:.4f}  "
              f"Recall={best_llm_05['recall']:.4f}  F1={best_llm_05['f1']:.4f}")
        print(f"Best LLM @{int(target*100)}%:  iter {best_llm_t['iter']}  AUC={best_llm_t['auc']:.4f}  "
              f"Recall={best_llm_t['recall']:.4f}  Spec={best_llm_t['specificity']:.4f}  "
              f"F1={best_llm_t['f1']:.4f}  thresh={best_llm_t['threshold']:.4f}")

    # --- Save JSON ---
    output = {
        'timestamp': str(datetime.now()),
        'device': DEVICE,
        'experiment_tag': tag,
        'description': 'V2 iterative training (no validation set, fn_heavy reward)',
        'target_recall': target,
        'test_size': len(y_test),
        'test_pos_rate': float(y_test.mean()),
        'rf_metrics_05': rf_results_05,
        'llm_metrics_05': llm_results_05,
        'rf_metrics_threshold': rf_results_thresh,
        'llm_metrics_threshold': llm_results_thresh,
        'best_rf_05': best_rf_05,
        'best_rf_threshold': best_rf_t,
        'best_llm_05': best_llm_05,
        'best_llm_threshold': best_llm_t,
    }
    out_path = BASE_DIR / "results/iterative_v2_full_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
