"""Fixed-grid ROC/PR curve CSV generation for probability files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .data_loading import load_config, resolve_path
from .metrics import read_probability_csv, write_json

GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def write_csv(path: Path, rows: list[dict[str, float]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def roc_rows(y: np.ndarray, p: np.ndarray) -> tuple[list[dict[str, float]], float]:
    fpr, tpr, _thresholds = roc_curve(y, p)
    auc = float(roc_auc_score(y, p))
    interp = np.interp(GRID, fpr, tpr)
    interp[0] = 0.0
    interp[-1] = 1.0
    return ([{"fpr": float(x), "tpr": float(v), "roc_auc": auc} for x, v in zip(GRID, interp)], auc)


def pr_rows(y: np.ndarray, p: np.ndarray) -> tuple[list[dict[str, float]], float]:
    precision, recall, _thresholds = precision_recall_curve(y, p)
    ap = float(average_precision_score(y, p))
    order = np.argsort(recall, kind="mergesort")
    recall_inc = recall[order]
    precision_inc = precision[order]
    unique_recall = np.unique(recall_inc)
    unique_precision = np.asarray([precision_inc[recall_inc == r].max() for r in unique_recall], dtype="float64")
    interp = np.interp(GRID, unique_recall, unique_precision)
    return ([{"recall": float(x), "precision": float(v), "pr_auc": ap} for x, v in zip(GRID, interp)], ap)


def run_curve_generation(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    out_root = resolve_path(config, config.get("outputs", {}).get("root", "outputs/qwen2_7b")) / "curves"
    sources = config.get("curve_csvs", {}).get("probability_csvs", [])
    generated = []
    for source in sources:
        name = str(source["name"])
        y, p = read_probability_csv(resolve_path(config, source["path"]))
        roc, auc = roc_rows(y, p)
        pr, ap = pr_rows(y, p)
        roc_path = out_root / f"{name}_roc_curve.csv"
        pr_path = out_root / f"{name}_pr_curve.csv"
        write_csv(roc_path, roc, ["fpr", "tpr", "roc_auc"])
        write_csv(pr_path, pr, ["recall", "precision", "pr_auc"])
        generated.append({"name": name, "roc_path": str(roc_path), "pr_path": str(pr_path), "roc_auc": auc, "pr_auc": ap, "rows": 101})
    result = {"method": "qwen2_curve_csv_generation", "grid": "0.00_to_1.00_step_0.01", "generated": generated}
    write_json(out_root / "curve_generation_summary.json", result)
    return result
