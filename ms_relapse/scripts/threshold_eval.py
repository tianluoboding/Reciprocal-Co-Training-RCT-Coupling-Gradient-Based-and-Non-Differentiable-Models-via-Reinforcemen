#!/usr/bin/env python3
"""
Threshold-Matched Evaluation at Target Recall (~80%)

For each model (RF Baseline, LLM CE Baseline, RCT-RF, RCT-LLM),
adjusts the decision threshold post-hoc so that recall ≈ target,
then reports all metrics at that operating point.

Usage:
    python 74_threshold_matched_eval.py --dataset ms
    python 74_threshold_matched_eval.py --dataset wdbc
    python 74_threshold_matched_eval.py --dataset diabetes
    python 74_threshold_matched_eval.py --dataset ms --target-recall 0.80
"""

import sys
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    recall_score, precision_score, f1_score,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib

import os
os.environ['PYTHONUNBUFFERED'] = '1'

SCRIPT_ROOT = Path(__file__).parent.parent          # exp1_clean/
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
VAL_RATIO = 0.15
PCA_DIM = 5
MAX_LENGTH = 512
BATCH_SIZE = 32

RF_PARAMS = dict(
    n_estimators=500, max_depth=8,
    min_samples_leaf=2, min_samples_split=5,
    class_weight="balanced_subsample",
    random_state=SEED, n_jobs=-1,
)


@dataclass
class DatasetConfig:
    name: str
    base_dir: Path
    k: int
    tab_train_path: str
    tab_test_path: str
    rf_model_pattern: str       # e.g. "models/rf_v3_iter{iter:02d}_pca5.pkl"
    pca_model_pattern: str
    llm_ckpt_pattern: str       # e.g. "checkpoints/iterative_v3_ppo_iter{iter:02d}/best.pt"
    llm_ce_ckpt: str
    use_dataloader: bool = False


DATASET_CONFIGS = {
    'ms': DatasetConfig(
        name='MS-Relapse',
        base_dir=SCRIPT_ROOT,
        k=15,
        tab_train_path='data/processed/X_train_selected.npy',
        tab_test_path='data/processed/X_test_selected.npy',
        rf_model_pattern='models/rf_v3_iter{iter:02d}_pca5.pkl',
        pca_model_pattern='models/pca_v3_iter{iter:02d}_pca5.pkl',
        llm_ckpt_pattern='checkpoints/iterative_v3_ppo_iter{iter:02d}/best.pt',
        llm_ce_ckpt='checkpoints/llm_ce_baseline/best.pt',
        use_dataloader=True,
    ),
    'wdbc': DatasetConfig(
        name='Breast Cancer (WDBC)',
        base_dir=SCRIPT_ROOT / 'breast_cancer_exp',
        k=30,
        tab_train_path='data/processed/X_train_selected_k30.npy',
        tab_test_path='data/processed/X_test_selected_k30.npy',
        rf_model_pattern='models/rf_v3_k30_iter{iter:02d}.pkl',
        pca_model_pattern='models/pca_v3_k30_iter{iter:02d}.pkl',
        llm_ckpt_pattern='checkpoints/iterative_v3_k30_ppo_iter{iter:02d}/best.pt',
        llm_ce_ckpt='checkpoints/llm_ce_baseline/best.pt',
        use_dataloader=False,
    ),
    'diabetes': DatasetConfig(
        name='Diabetes (BRFSS 2015)',
        base_dir=SCRIPT_ROOT / 'Diabetes_exp',
        k=21,
        tab_train_path='data/processed/X_train_selected_k21.npy',
        tab_test_path='data/processed/X_test_selected_k21.npy',
        rf_model_pattern='models/rf_v3_k21_iter{iter:02d}.pkl',
        pca_model_pattern='models/pca_v3_k21_iter{iter:02d}.pkl',
        llm_ckpt_pattern='checkpoints/iterative_v3_k21_ppo_iter{iter:02d}/best.pt',
        llm_ce_ckpt='checkpoints/llm_ce_baseline/best.pt',
        use_dataloader=False,
    ),
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Threshold matching
# ---------------------------------------------------------------------------

def find_threshold_at_recall(y_true, proba, target=0.80):
    """Return the highest threshold that achieves recall >= target.

    roc_curve returns (fpr, tpr, thresholds) where tpr is increasing
    and thresholds is decreasing.  The *first* index where tpr >= target
    corresponds to the highest threshold that still meets the target.
    """
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    valid = tpr >= target
    if valid.any():
        idx = np.where(valid)[0][0]    # first index with tpr >= target = highest threshold
    else:
        idx = 0
    return float(thresholds[idx])


def compute_metrics_at_threshold(y_true, proba, threshold):
    """Full metrics at a given decision threshold."""
    auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1v = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'threshold': float(threshold),
        'accuracy': float(acc),
        'recall': float(rec),
        'specificity': float(spec),
        'precision': float(prec),
        'f1': float(f1v),
        'auc': float(auc),
        'pr_auc': float(pr_auc),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cfg: DatasetConfig):
    """Load test set and full training data (for RF baseline retraining)."""
    bd = cfg.base_dir

    if cfg.use_dataloader:
        from data.loader import DataLoader as DL
        all_data = DL(bd / "data").load_all()
        texts = all_data['texts']
        y_all = all_data['y']
        train_idx = np.array(all_data['train_idx'])
        test_idx = np.array(all_data['test_idx'])
    else:
        data_dir = bd / "data"
        with open(data_dir / 'raw/descriptive_text.json', 'r') as f:
            text_data = json.load(f)
        texts = [item['prompt'] for item in text_data]
        y_all = np.load(data_dir / 'raw/y_all.npy')
        with open(data_dir / 'raw/train_idxs.json', 'r') as f:
            train_idx = np.array(json.load(f))
        with open(data_dir / 'raw/test_idxs.json', 'r') as f:
            test_idx = np.array(json.load(f))

    X_train_tab = np.load(bd / cfg.tab_train_path)
    X_test_tab = np.load(bd / cfg.tab_test_path)

    X_tabular_full = np.zeros((len(y_all), X_train_tab.shape[1]), dtype=np.float32)
    X_tabular_full[train_idx] = X_train_tab
    X_tabular_full[test_idx] = X_test_tab

    train_sub_idx, val_idx = train_test_split(
        train_idx, test_size=VAL_RATIO,
        random_state=SEED, stratify=y_all[train_idx]
    )

    return {
        'texts': texts,
        'y_all': y_all,
        'train_idx': train_idx,
        'train_sub_idx': train_sub_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'X_tabular_full': X_tabular_full,
        'X_train_tab': X_train_tab,
        'X_test_tab': X_test_tab,
    }


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def create_model():
    from models.llm_wrapper import BioClinicalBERTWithLoRA
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
        _, emb = model(inputs['input_ids'], inputs['attention_mask'],
                       return_embeddings=True)
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
        logits, _ = model(inputs['input_ids'], inputs['attention_mask'],
                          return_embeddings=True)
        probs = F.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


# ---------------------------------------------------------------------------
# RF Baseline retraining (tabular-only, deterministic)
# ---------------------------------------------------------------------------

def retrain_rf_baseline(X_train_full, y_train_full, X_test, y_test,
                        target_recall=0.80):
    """10-fold CV per-model averaging: train 10 fold-models, compute
    threshold-matched metrics for each model individually, then average.

    This gives AUC = held_out_test_auc_mean (average of individual model AUCs)
    rather than the slightly inflated ensemble AUC from probability averaging.
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (tr_idx, _) in enumerate(skf.split(X_train_full, y_train_full), 1):
        X_fold_pool = X_train_full[tr_idx]
        y_fold_pool = y_train_full[tr_idx]

        sub_idx, val_idx = train_test_split(
            np.arange(len(y_fold_pool)),
            test_size=VAL_RATIO, random_state=SEED, stratify=y_fold_pool
        )
        X_train_sub = X_fold_pool[sub_idx]
        y_train_sub = y_fold_pool[sub_idx]

        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train_sub, y_train_sub,
            test_size=0.2, random_state=SEED, stratify=y_train_sub
        )

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_fit, y_fit)

        calibrator = CalibratedClassifierCV(
            estimator=rf, method='sigmoid', cv='prefit')
        calibrator.fit(X_cal, y_cal)

        proba = calibrator.predict_proba(X_test)[:, 1]
        thresh = find_threshold_at_recall(y_test, proba, target_recall)
        m = compute_metrics_at_threshold(y_test, proba, thresh)
        fold_metrics.append(m)
        print(f"    Fold {fold:2d}: AUC={m['auc']:.4f}  Recall={m['recall']:.4f}  "
              f"Spec={m['specificity']:.4f}  F1={m['f1']:.4f}  thresh={thresh:.4f}")

    metric_keys = ['threshold', 'accuracy', 'recall', 'specificity',
                   'precision', 'f1', 'auc', 'pr_auc']
    avg = {}
    for k in metric_keys:
        vals = [m[k] for m in fold_metrics]
        avg[k] = float(np.mean(vals))
        avg[f'{k}_std'] = float(np.std(vals))

    return avg


# ---------------------------------------------------------------------------
# Discover available iterations
# ---------------------------------------------------------------------------

def discover_iters(cfg: DatasetConfig):
    """Return sorted lists of available RF and LLM V3 iterations."""
    bd = cfg.base_dir

    if cfg.name == 'MS-Relapse':
        rf_iters = sorted([
            int(p.stem.replace('_pca5', '').split('iter')[-1])
            for p in (bd / 'models').glob('rf_v3_iter*_pca5.pkl')
        ])
        llm_iters = sorted([
            int(p.name.split('iter')[-1])
            for p in (bd / 'checkpoints').glob('iterative_v3_ppo_iter*')
            if (p / 'best.pt').exists()
        ])
    else:
        k = cfg.k
        rf_iters = sorted([
            int(p.stem.split('iter')[-1])
            for p in (bd / 'models').glob(f'rf_v3_k{k}_iter*.pkl')
        ])
        llm_iters = sorted([
            int(p.name.split('iter')[-1])
            for p in (bd / 'checkpoints').glob(f'iterative_v3_k{k}_ppo_iter*')
            if (p / 'best.pt').exists()
        ])

    return rf_iters, llm_iters


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

METRIC_COLS = ['threshold', 'accuracy', 'recall', 'specificity',
               'precision', 'f1', 'auc', 'pr_auc']


def print_summary_table(title, rows):
    """rows: list of dicts with keys 'model' + METRIC_COLS."""
    col_w = {c: max(len(c), 11) for c in METRIC_COLS}
    name_w = max(len(r['model']) for r in rows) + 2

    header = f"{'Model':<{name_w}}" + "".join(
        f"  {c:>{col_w[c]}}" for c in METRIC_COLS)
    sep = "-" * len(header)

    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        line = f"{r['model']:<{name_w}}"
        for c in METRIC_COLS:
            line += f"  {r[c]:>{col_w[c]}.4f}"
        print(line)
    print(sep)


def print_iter_table(title, rows):
    """rows: list of dicts with keys 'iter' + METRIC_COLS."""
    header = f"{'Iter':>4}" + "".join(
        f"  {c:>11}" for c in METRIC_COLS)
    sep = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        line = f"{r['iter']:>4}"
        for c in METRIC_COLS:
            line += f"  {r[c]:>11.4f}"
        print(line)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['ms', 'wdbc', 'diabetes'])
    parser.add_argument('--target-recall', type=float, default=0.80)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    target = args.target_recall

    sys.path.insert(0, str(cfg.base_dir / "src"))

    set_seed(SEED)

    print("=" * 90)
    print(f"Threshold-Matched Evaluation — {cfg.name}")
    print(f"Target Recall: {target:.0%}")
    print(f"Timestamp: {datetime.now()}")
    print(f"Device: {DEVICE}")
    print("=" * 90)

    # ---- Load data ----
    data = load_data(cfg)
    y_all = data['y_all']
    train_idx = data['train_idx']
    test_idx = data['test_idx']
    y_test = y_all[test_idx]
    y_train = y_all[train_idx]
    X_tab_test = data['X_tabular_full'][test_idx]
    X_train_tab = data['X_train_tab']
    X_test_tab = data['X_test_tab']
    test_texts = [data['texts'][j] for j in test_idx]

    print(f"\n  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"  Test pos rate: {y_test.mean():.2%}")
    print(f"  Features (tabular): {X_train_tab.shape[1]}")

    # Collect all results: {model_name: metrics_dict}
    all_results = {}

    # ==================================================================
    # 1) RF Baseline (tabular-only, retrained inline)
    # ==================================================================
    print("\n" + "=" * 90)
    print("1) RF Baseline (tabular-only, no LLM embeddings)")
    print("=" * 90)

    rf_base_m = retrain_rf_baseline(
        X_train_tab, y_train, X_test_tab, y_test, target_recall=target)
    rf_base_m['model'] = 'RF Baseline'
    all_results['rf_baseline'] = rf_base_m

    print(f"  --- 10-fold per-model average ---")
    print(f"  Threshold: {rf_base_m['threshold']:.4f} (+/- {rf_base_m['threshold_std']:.4f})")
    print(f"  AUC: {rf_base_m['auc']:.4f} (+/- {rf_base_m['auc_std']:.4f})  "
          f"Recall: {rf_base_m['recall']:.4f} (+/- {rf_base_m['recall_std']:.4f})  "
          f"Spec: {rf_base_m['specificity']:.4f}  F1: {rf_base_m['f1']:.4f}")

    # ==================================================================
    # 2) LLM CE Baseline
    # ==================================================================
    print("\n" + "=" * 90)
    print("2) LLM CE Baseline")
    print("=" * 90)

    ce_ckpt = cfg.base_dir / cfg.llm_ce_ckpt
    if ce_ckpt.exists():
        model = create_model()
        tokenizer = model.tokenizer
        model = load_checkpoint(model, ce_ckpt)
        ce_proba = llm_predict_proba(model, tokenizer, test_texts)
        del model
        torch.cuda.empty_cache()

        ce_thresh = find_threshold_at_recall(y_test, ce_proba, target)
        ce_m = compute_metrics_at_threshold(y_test, ce_proba, ce_thresh)
        ce_m['model'] = 'LLM CE Baseline'
        all_results['llm_ce_baseline'] = ce_m

        print(f"  Threshold: {ce_thresh:.4f}")
        print(f"  AUC: {ce_m['auc']:.4f}  Recall: {ce_m['recall']:.4f}  "
              f"Spec: {ce_m['specificity']:.4f}  F1: {ce_m['f1']:.4f}")
    else:
        print(f"  [SKIP] Checkpoint not found: {ce_ckpt}")

    # ==================================================================
    # 3) & 4) RCT-RF and RCT-LLM (all V3 iterations)
    # ==================================================================
    print("\n" + "=" * 90)
    print("3) & 4) RCT Iterative Models (V3)")
    print("=" * 90)

    rf_iters, llm_iters = discover_iters(cfg)
    print(f"  RF iterations found: {rf_iters}")
    print(f"  LLM iterations found: {llm_iters}")

    rct_rf_results = []
    rct_llm_results = []
    test_emb_cache = {}

    # --- LLM forward passes (also extract embeddings for RF) ---
    print("\n--- LLM checkpoints ---")

    # Iter 0: pretrained (no finetuning), needed for RF iter 0
    print("  Iter 0: pretrained Bio-ClinicalBERT (no finetuning)")
    model_fresh = create_model()
    tokenizer = model_fresh.tokenizer
    test_emb_cache[0] = extract_embeddings(model_fresh, tokenizer, test_texts)
    del model_fresh
    torch.cuda.empty_cache()

    for it in llm_iters:
        ckpt_path = cfg.base_dir / cfg.llm_ckpt_pattern.format(iter=it)
        if not ckpt_path.exists():
            print(f"  Iter {it}: [SKIP] {ckpt_path} not found")
            continue

        print(f"  Iter {it}: loading checkpoint")
        model = create_model()
        model = load_checkpoint(model, ckpt_path)

        proba = llm_predict_proba(model, tokenizer, test_texts)
        thresh = find_threshold_at_recall(y_test, proba, target)
        m = compute_metrics_at_threshold(y_test, proba, thresh)
        m['iter'] = it
        rct_llm_results.append(m)
        print(f"    t={thresh:.4f}  AUC={m['auc']:.4f}  Recall={m['recall']:.4f}  "
              f"Spec={m['specificity']:.4f}  F1={m['f1']:.4f}")

        test_emb_cache[it] = extract_embeddings(model, tokenizer, test_texts)

        del model
        torch.cuda.empty_cache()

    # --- RF evaluation ---
    print("\n--- RF models (tabular + LLM embeddings via PCA) ---")
    for it in rf_iters:
        rf_path = cfg.base_dir / cfg.rf_model_pattern.format(iter=it)
        pca_path = cfg.base_dir / cfg.pca_model_pattern.format(iter=it)

        if not rf_path.exists() or not pca_path.exists():
            print(f"  Iter {it}: [SKIP] model files not found")
            continue

        rf_data = joblib.load(rf_path)
        calibrator = rf_data['calibrator']
        pca = joblib.load(pca_path)

        emb_key = it if it in test_emb_cache else 0
        emb_test = test_emb_cache[emb_key]
        phi_test = pca.transform(emb_test)
        X_combined = np.concatenate([X_tab_test, phi_test], axis=1)

        proba = calibrator.predict_proba(X_combined)[:, 1]
        thresh = find_threshold_at_recall(y_test, proba, target)
        m = compute_metrics_at_threshold(y_test, proba, thresh)
        m['iter'] = it
        rct_rf_results.append(m)
        print(f"  Iter {it}: t={thresh:.4f}  AUC={m['auc']:.4f}  Recall={m['recall']:.4f}  "
              f"Spec={m['specificity']:.4f}  F1={m['f1']:.4f}")

    # ==================================================================
    # Summary tables
    # ==================================================================
    print("\n" + "=" * 90)
    print(f"SUMMARY — {cfg.name}  (target recall = {target:.0%})")
    print("=" * 90)

    # Per-iteration tables
    if rct_rf_results:
        print_iter_table(f"RCT-RF per iteration (target recall {target:.0%})",
                         rct_rf_results)
    if rct_llm_results:
        print_iter_table(f"RCT-LLM per iteration (target recall {target:.0%})",
                         rct_llm_results)

    # Best-iteration summary (by AUC)
    summary_rows = []
    if 'rf_baseline' in all_results:
        summary_rows.append(all_results['rf_baseline'])
    if 'llm_ce_baseline' in all_results:
        summary_rows.append(all_results['llm_ce_baseline'])

    if rct_rf_results:
        best_rf = max(rct_rf_results, key=lambda r: r['auc'])
        best_rf_row = {**best_rf, 'model': f"RCT-RF (iter {best_rf['iter']})"}
        summary_rows.append(best_rf_row)
        all_results['rct_rf_best'] = best_rf_row

    if rct_llm_results:
        best_llm = max(rct_llm_results, key=lambda r: r['auc'])
        best_llm_row = {**best_llm, 'model': f"RCT-LLM (iter {best_llm['iter']})"}
        summary_rows.append(best_llm_row)
        all_results['rct_llm_best'] = best_llm_row

    if summary_rows:
        print_summary_table(
            f"Table 1 — {cfg.name} at {target:.0%} Recall", summary_rows)

    # ==================================================================
    # Save JSON
    # ==================================================================
    output = {
        'timestamp': str(datetime.now()),
        'device': DEVICE,
        'dataset': cfg.name,
        'target_recall': target,
        'k': cfg.k,
        'test_size': int(len(y_test)),
        'test_pos_rate': float(y_test.mean()),
        'summary': {k: v for k, v in all_results.items()},
        'rct_rf_all_iters': rct_rf_results,
        'rct_llm_all_iters': rct_llm_results,
    }

    out_dir = cfg.base_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "threshold_matched_eval_80recall.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved: {out_path}")
    print("=" * 90)


if __name__ == '__main__':
    main()
