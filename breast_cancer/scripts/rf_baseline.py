#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF-Only Baseline for WDBC Breast Cancer Dataset

Trains a Random Forest on tabular features ONLY (no LLM embeddings, no PCA).
Uses the same hyperparameters and data split as the iterative framework.
"""

import json
import random
import argparse
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
from sklearn.model_selection import train_test_split
import joblib

import os
os.environ['PYTHONUNBUFFERED'] = '1'

BASE_DIR = Path(__file__).parent.parent  # breast_cancer_exp/
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="RF-only baseline for WDBC")
    parser.add_argument('--k', type=int, default=30, choices=[5, 10, 15, 20, 30],
                        help='Number of k-best features')
    args = parser.parse_args()
    k = args.k

    set_seed(SEED)

    print("=" * 80)
    print(f"RF-Only Baseline: WDBC Breast Cancer Dataset  [k={k}]")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")

    # ---- Load data ----
    data_dir = BASE_DIR / "data"

    y_all = np.load(data_dir / 'raw/y_all.npy')

    with open(data_dir / 'raw/train_idxs.json') as f:
        train_idx = json.load(f)
    with open(data_dir / 'raw/test_idxs.json') as f:
        test_idx = json.load(f)

    X_train_tab = np.load(data_dir / f'processed/X_train_selected_k{k}.npy')
    X_test_tab = np.load(data_dir / f'processed/X_test_selected_k{k}.npy')

    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    n_features = X_train_tab.shape[1]
    print(f"\n  Train: {len(train_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    print(f"  Features: {n_features} (tabular only, no LLM)")
    print(f"  Positive rate (train): {y_train.mean():.2%}")

    # ---- Train RF ----
    print("\n" + "=" * 80)
    print("Training Random Forest (tabular only)")
    print("=" * 80)

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train_tab, y_train,
        test_size=0.2, random_state=SEED, stratify=y_train
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        min_samples_split=5,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)

    calibrator = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv='prefit')
    calibrator.fit(X_cal, y_cal)

    # ---- Evaluate ----
    proba_train = calibrator.predict_proba(X_train_tab)[:, 1]
    proba_test = calibrator.predict_proba(X_test_tab)[:, 1]

    train_auc = roc_auc_score(y_train, proba_train)
    test_auc = roc_auc_score(y_test, proba_test)
    test_pr_auc = average_precision_score(y_test, proba_test)

    y_pred = (proba_test >= 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n  Train AUC:        {train_auc:.4f}")
    print(f"  Test AUC:         {test_auc:.4f}")
    print(f"  Test PR AUC:      {test_pr_auc:.4f}")
    print(f"  Test Accuracy:    {test_accuracy:.4f}")
    print(f"  Test Recall:      {test_recall:.4f}")
    print(f"  Test Precision:   {test_precision:.4f}")
    print(f"  Test Specificity: {test_specificity:.4f}")
    print(f"  Test F1:          {test_f1:.4f}")

    # ---- Save results ----
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': 'WDBC Breast Cancer',
        'baseline': 'RF-only (tabular)',
        'k': k,
        'n_features': n_features,
        'timestamp': str(datetime.now()),
        'metrics': {
            'train_auc': float(train_auc),
            'test_auc': float(test_auc),
            'test_pr_auc': float(test_pr_auc),
            'test_accuracy': float(test_accuracy),
            'test_recall': float(test_recall),
            'test_precision': float(test_precision),
            'test_specificity': float(test_specificity),
            'test_f1': float(test_f1),
        },
        'config': {
            'n_estimators': 500,
            'max_depth': 8,
            'min_samples_leaf': 2,
            'min_samples_split': 5,
            'class_weight': 'balanced_subsample',
            'calibration_split': 0.2,
            'seed': SEED,
        }
    }

    results_path = results_dir / f'rf_only_baseline_k{k}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f'rf_only_baseline_k{k}.pkl'
    joblib.dump({'rf': rf, 'calibrator': calibrator}, model_path)
    print(f"  Model saved:   {model_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
