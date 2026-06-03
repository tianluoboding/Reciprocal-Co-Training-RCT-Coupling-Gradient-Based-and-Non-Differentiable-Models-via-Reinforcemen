"""Metric and probability-file utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    return 1.0 / (1.0 + np.exp(-x))


def stat_block(x: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(x, dtype="float64").reshape(-1)
    return {
        "n": int(arr.size),
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size else float("nan"),
    }


def metric_block(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(probs, dtype="float64").reshape(-1)
    if y.size != p.size:
        raise ValueError("Metric inputs have different lengths")
    if y.size == 0 or len(np.unique(y)) < 2:
        raise ValueError("Metrics require both binary classes")
    if not np.isfinite(p).all() or p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("Probabilities must be finite and inside [0, 1]")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(y.size),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "positive_prediction_rate": float(pred.mean()),
        "probability": stat_block(p),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def split_metrics(data: dict[str, Any], probabilities: dict[str, np.ndarray], threshold: float = 0.5) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train_sub", "val", "test"):
        idx = data[f"{split}_idx"]
        out[split] = metric_block(data["y"][idx], probabilities[split], threshold)
    return out


def save_probability_csv(path: Path, split: str, idx: np.ndarray, y: np.ndarray, probs: np.ndarray, *, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, index in enumerate(idx):
        prob = float(probs[i])
        rows.append({"split": split, "index": int(index), "y_true": int(y[i]), "prob": prob, "pred_0p5": int(prob >= 0.5), "source": source})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_probability_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        label_col = next((c for c in ("y_true", "label", "target", "y") if c in reader.fieldnames), None)
        prob_col = next((c for c in ("prob", "probability", "score", "p_rf", "p_llm", "qwen2_proba") if c in reader.fieldnames), None)
        if label_col is None or prob_col is None:
            raise ValueError(f"Missing label/probability columns in {path}: {reader.fieldnames}")
        y, p = [], []
        for row in reader:
            y.append(int(float(row[label_col])))
            p.append(float(row[prob_col]))
    return np.asarray(y, dtype=int), np.asarray(p, dtype="float64")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
