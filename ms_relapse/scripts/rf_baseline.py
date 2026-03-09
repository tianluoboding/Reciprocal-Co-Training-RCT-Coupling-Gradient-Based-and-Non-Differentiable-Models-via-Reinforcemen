#!/usr/bin/env python3
"""
RF-Only Baseline: 10-Fold CV with Internal Validation Split

Matches the V3 iterative training protocol:
  - For each fold, split 9 folds into train_sub (85%) + val (15%)
  - RF trains on train_sub (with 80/20 fit/cal inside)
  - RF evaluates on val and on the held-out fold (test)
  - Reports mean +/- std for val AUC and test AUC across 10 folds
  - Also evaluates all 10 fold-models on the original held-out test (439)
"""

import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score, accuracy_score,
    confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold

import os
os.environ['PYTHONUNBUFFERED'] = '1'

BASE_DIR = Path(__file__).parent.parent
SEED = 42
VAL_RATIO = 0.15

RF_PARAMS = dict(
    n_estimators=500, max_depth=8,
    min_samples_leaf=2, min_samples_split=5,
    class_weight="balanced_subsample",
    random_state=SEED, n_jobs=-1,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def eval_metrics(y_true, proba):
    auc_val = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    y_pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return dict(auc=float(auc_val), pr_auc=float(pr_auc), accuracy=float(acc),
                recall=float(rec), precision=float(prec), f1=float(f1),
                specificity=float(spec))


def train_rf_with_val(X_train_sub, y_train_sub, X_val, y_val, X_test, y_test):
    """Train RF on train_sub (with internal fit/cal split), evaluate on val + test."""
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_sub, y_train_sub,
        test_size=0.2, random_state=SEED, stratify=y_train_sub
    )

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_fit, y_fit)

    calibrator = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv='prefit')
    calibrator.fit(X_cal, y_cal)

    train_proba = calibrator.predict_proba(X_train_sub)[:, 1]
    val_proba = calibrator.predict_proba(X_val)[:, 1]
    test_proba = calibrator.predict_proba(X_test)[:, 1]

    return (
        eval_metrics(y_train_sub, train_proba),
        eval_metrics(y_val, val_proba),
        eval_metrics(y_test, test_proba),
        calibrator,
    )


def main():
    set_seed(SEED)

    print("=" * 80)
    print("RF-Only Baseline: 10-Fold CV with Internal Validation Split")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 80)

    data_dir = BASE_DIR / "data"
    y_all = np.load(data_dir / 'raw/y_all.npy')
    train_idx = json.load(open(data_dir / 'raw/train_idxs.json'))
    test_idx = json.load(open(data_dir / 'raw/test_idxs.json'))

    X_train_full = np.load(data_dir / 'processed/X_train_selected.npy')
    X_test_held = np.load(data_dir / 'processed/X_test_selected.npy')
    y_train_full = y_all[train_idx]
    y_test_held = y_all[test_idx]

    print(f"\n  CV pool: {len(y_train_full)} samples")
    print(f"  Held-out test: {len(y_test_held)} samples")
    print(f"  Features: {X_train_full.shape[1]}")
    print(f"  Val ratio inside each fold: {VAL_RATIO}")

    # ---- Part A: Single split matching V3 protocol ----
    print("\n" + "=" * 80)
    print("PART A: Single Split with Validation (V3 protocol)")
    print("=" * 80)

    train_sub_mask, val_mask = train_test_split(
        np.arange(len(y_train_full)),
        test_size=VAL_RATIO, random_state=SEED, stratify=y_train_full
    )

    X_train_sub = X_train_full[train_sub_mask]
    X_val = X_train_full[val_mask]
    y_train_sub = y_train_full[train_sub_mask]
    y_val = y_train_full[val_mask]

    print(f"  Train_sub: {len(y_train_sub)}, Val: {len(y_val)}, Test: {len(y_test_held)}")

    train_m, val_m, test_m, _ = train_rf_with_val(
        X_train_sub, y_train_sub, X_val, y_val, X_test_held, y_test_held
    )

    print(f"\n  Train AUC:  {train_m['auc']:.4f}")
    print(f"  Val AUC:    {val_m['auc']:.4f}  (PR-AUC: {val_m['pr_auc']:.4f}, Recall: {val_m['recall']:.4f})")
    print(f"  Test AUC:   {test_m['auc']:.4f}  (PR-AUC: {test_m['pr_auc']:.4f}, Recall: {test_m['recall']:.4f})")

    # ---- Part B: 10-Fold CV with internal val ----
    print("\n" + "=" * 80)
    print("PART B: 10-Fold Stratified CV with Internal Validation")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    fold_results = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        X_fold_pool = X_train_full[tr_idx]
        y_fold_pool = y_train_full[tr_idx]
        X_fold_test = X_train_full[te_idx]
        y_fold_test = y_train_full[te_idx]

        # Internal val split from the 9-fold training pool
        sub_idx, val_idx = train_test_split(
            np.arange(len(y_fold_pool)),
            test_size=VAL_RATIO, random_state=SEED, stratify=y_fold_pool
        )
        X_fold_train_sub = X_fold_pool[sub_idx]
        y_fold_train_sub = y_fold_pool[sub_idx]
        X_fold_val = X_fold_pool[val_idx]
        y_fold_val = y_fold_pool[val_idx]

        _, fold_val_m, fold_test_m, _ = train_rf_with_val(
            X_fold_train_sub, y_fold_train_sub,
            X_fold_val, y_fold_val,
            X_fold_test, y_fold_test
        )

        print(f"  Fold {fold:2d}: val AUC={fold_val_m['auc']:.4f}  "
              f"test AUC={fold_test_m['auc']:.4f}  "
              f"test Recall={fold_test_m['recall']:.4f}  "
              f"test F1={fold_test_m['f1']:.4f}  "
              f"(train_sub={len(y_fold_train_sub)}, val={len(y_fold_val)}, test={len(y_fold_test)})")

        fold_results.append({
            'fold': fold,
            'n_train_sub': len(y_fold_train_sub),
            'n_val': len(y_fold_val),
            'n_test': len(y_fold_test),
            'val_auc': fold_val_m['auc'],
            'val_pr_auc': fold_val_m['pr_auc'],
            'val_recall': fold_val_m['recall'],
            'val_f1': fold_val_m['f1'],
            'test_auc': fold_test_m['auc'],
            'test_pr_auc': fold_test_m['pr_auc'],
            'test_recall': fold_test_m['recall'],
            'test_f1': fold_test_m['f1'],
            'test_specificity': fold_test_m['specificity'],
        })

    # ---- Summary ----
    val_aucs = [f['val_auc'] for f in fold_results]
    test_aucs = [f['test_auc'] for f in fold_results]
    test_pr_aucs = [f['test_pr_auc'] for f in fold_results]
    test_recalls = [f['test_recall'] for f in fold_results]
    test_f1s = [f['test_f1'] for f in fold_results]
    test_specs = [f['test_specificity'] for f in fold_results]

    print(f"\n  {'='*60}")
    print(f"  10-Fold CV Summary (with internal val)")
    print(f"  {'='*60}")
    print(f"  Val AUC:       {np.mean(val_aucs):.4f} +/- {np.std(val_aucs):.4f}  "
          f"(range: {np.min(val_aucs):.4f} - {np.max(val_aucs):.4f})")
    print(f"  Test AUC:      {np.mean(test_aucs):.4f} +/- {np.std(test_aucs):.4f}  "
          f"(range: {np.min(test_aucs):.4f} - {np.max(test_aucs):.4f})")
    print(f"  Test PR-AUC:   {np.mean(test_pr_aucs):.4f} +/- {np.std(test_pr_aucs):.4f}")
    print(f"  Test Recall:   {np.mean(test_recalls):.4f} +/- {np.std(test_recalls):.4f}")
    print(f"  Test F1:       {np.mean(test_f1s):.4f} +/- {np.std(test_f1s):.4f}")
    print(f"  Test Spec:     {np.mean(test_specs):.4f} +/- {np.std(test_specs):.4f}")

    # ---- Held-out test: evaluate each fold's model ----
    print(f"\n  Evaluating all 10 fold-models on held-out test ({len(y_test_held)} samples)...")
    held_test_aucs = []
    held_test_pr_aucs = []
    held_test_accs = []
    held_test_recalls = []
    held_test_specs = []
    held_test_precs = []
    held_test_f1s = []
    skf2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for fold, (tr_idx, _) in enumerate(skf2.split(X_train_full, y_train_full), 1):
        X_fold_pool = X_train_full[tr_idx]
        y_fold_pool = y_train_full[tr_idx]

        sub_idx, val_idx_inner = train_test_split(
            np.arange(len(y_fold_pool)),
            test_size=VAL_RATIO, random_state=SEED, stratify=y_fold_pool
        )

        _, _, held_m, _ = train_rf_with_val(
            X_fold_pool[sub_idx], y_fold_pool[sub_idx],
            X_fold_pool[val_idx_inner], y_fold_pool[val_idx_inner],
            X_test_held, y_test_held
        )
        held_test_aucs.append(held_m['auc'])
        held_test_pr_aucs.append(held_m['pr_auc'])
        held_test_accs.append(held_m['accuracy'])
        held_test_recalls.append(held_m['recall'])
        held_test_specs.append(held_m['specificity'])
        held_test_precs.append(held_m['precision'])
        held_test_f1s.append(held_m['f1'])

    print(f"  Held-out test AUC:       {np.mean(held_test_aucs):.4f} +/- {np.std(held_test_aucs):.4f}")
    print(f"  Held-out test PR-AUC:    {np.mean(held_test_pr_aucs):.4f} +/- {np.std(held_test_pr_aucs):.4f}")
    print(f"  Held-out test Accuracy:  {np.mean(held_test_accs):.4f} +/- {np.std(held_test_accs):.4f}")
    print(f"  Held-out test Recall:    {np.mean(held_test_recalls):.4f} +/- {np.std(held_test_recalls):.4f}")
    print(f"  Held-out test Spec:      {np.mean(held_test_specs):.4f} +/- {np.std(held_test_specs):.4f}")
    print(f"  Held-out test Precision: {np.mean(held_test_precs):.4f} +/- {np.std(held_test_precs):.4f}")
    print(f"  Held-out test F1:        {np.mean(held_test_f1s):.4f} +/- {np.std(held_test_f1s):.4f}")

    # ---- Comparison ----
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"\n  Single split (V3 protocol):")
    print(f"    Train AUC:  {train_m['auc']:.4f}")
    print(f"    Val AUC:    {val_m['auc']:.4f}")
    print(f"    Test AUC:   {test_m['auc']:.4f}")
    print(f"\n  10-Fold CV (fold-test, with internal val):")
    print(f"    Val AUC:    {np.mean(val_aucs):.4f} +/- {np.std(val_aucs):.4f}")
    print(f"    Test AUC:   {np.mean(test_aucs):.4f} +/- {np.std(test_aucs):.4f}")
    print(f"\n  10-Fold models -> held-out test (439):")
    print(f"    Test AUC:   {np.mean(held_test_aucs):.4f} +/- {np.std(held_test_aucs):.4f}")

    # ---- Save ----
    results = {
        'dataset': 'MS-Relapse',
        'baseline': 'RF-only 10-Fold CV with internal val (V3 protocol)',
        'n_features': int(X_train_full.shape[1]),
        'val_ratio': VAL_RATIO,
        'n_folds': 10,
        'timestamp': str(datetime.now()),
        'single_split': {
            'n_train_sub': len(y_train_sub),
            'n_val': len(y_val),
            'n_test': len(y_test_held),
            'train_auc': train_m['auc'],
            'val_auc': val_m['auc'],
            'test_auc': test_m['auc'],
            'test_metrics': test_m,
        },
        'cv_summary': {
            'val_auc_mean': float(np.mean(val_aucs)),
            'val_auc_std': float(np.std(val_aucs)),
            'test_auc_mean': float(np.mean(test_aucs)),
            'test_auc_std': float(np.std(test_aucs)),
            'test_auc_min': float(np.min(test_aucs)),
            'test_auc_max': float(np.max(test_aucs)),
            'test_pr_auc_mean': float(np.mean(test_pr_aucs)),
            'test_pr_auc_std': float(np.std(test_pr_aucs)),
            'test_recall_mean': float(np.mean(test_recalls)),
            'test_recall_std': float(np.std(test_recalls)),
            'test_f1_mean': float(np.mean(test_f1s)),
            'test_f1_std': float(np.std(test_f1s)),
            'test_specificity_mean': float(np.mean(test_specs)),
            'test_specificity_std': float(np.std(test_specs)),
            'held_out_test_auc_mean': float(np.mean(held_test_aucs)),
            'held_out_test_auc_std': float(np.std(held_test_aucs)),
            'held_out_test_pr_auc_mean': float(np.mean(held_test_pr_aucs)),
            'held_out_test_pr_auc_std': float(np.std(held_test_pr_aucs)),
            'held_out_test_accuracy_mean': float(np.mean(held_test_accs)),
            'held_out_test_accuracy_std': float(np.std(held_test_accs)),
            'held_out_test_recall_mean': float(np.mean(held_test_recalls)),
            'held_out_test_recall_std': float(np.std(held_test_recalls)),
            'held_out_test_specificity_mean': float(np.mean(held_test_specs)),
            'held_out_test_specificity_std': float(np.std(held_test_specs)),
            'held_out_test_precision_mean': float(np.mean(held_test_precs)),
            'held_out_test_precision_std': float(np.std(held_test_precs)),
            'held_out_test_f1_mean': float(np.mean(held_test_f1s)),
            'held_out_test_f1_std': float(np.std(held_test_f1s)),
        },
        'fold_details': fold_results,
        'config': {k: v for k, v in RF_PARAMS.items() if k != 'n_jobs'},
    }

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / 'rf_only_baseline_val_10fold_cv.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\n  Results saved: {out}")
    print("=" * 80)


if __name__ == '__main__':
    main()
