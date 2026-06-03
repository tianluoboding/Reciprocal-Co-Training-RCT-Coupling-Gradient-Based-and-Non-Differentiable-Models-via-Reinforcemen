"""Validation-only threshold selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import recall_score

from .metrics import metric_block


def select_threshold_for_recall(y_val: np.ndarray, p_val: np.ndarray, target_recall: float) -> dict[str, Any]:
    y = np.asarray(y_val).astype(int).reshape(-1)
    p = np.asarray(p_val, dtype="float64").reshape(-1)
    candidates = np.unique(p)
    rows = []
    for threshold in candidates:
        recall = recall_score(y, (p >= threshold).astype(int), zero_division=0)
        rows.append((float(threshold), float(recall)))
    valid = [row for row in rows if row[1] >= target_recall]
    if valid:
        threshold, recall = max(valid, key=lambda row: row[0])
        fallback = False
    else:
        threshold, recall = max(rows, key=lambda row: (row[1], row[0]))
        fallback = True
    return {
        "target_recall": float(target_recall),
        "threshold": float(threshold),
        "validation_recall": float(recall),
        "threshold_source": "validation_only",
        "fallback_used": bool(fallback),
    }


def threshold_matched_metrics(y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray, targets: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target in targets:
        selected = select_threshold_for_recall(y_val, p_val, float(target))
        threshold = float(selected["threshold"])
        out[str(target)] = {"selection": selected, "val": metric_block(y_val, p_val, threshold), "test": metric_block(y_test, p_test, threshold)}
    return out
