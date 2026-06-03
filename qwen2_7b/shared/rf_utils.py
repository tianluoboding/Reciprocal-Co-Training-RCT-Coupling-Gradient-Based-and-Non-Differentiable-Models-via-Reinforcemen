"""Random Forest and PCA helpers for RCT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def rf_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "n_estimators": 500,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "min_samples_split": 5,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": int(config.get("seed", 42)),
    }
    defaults.update(config.get("rf", {}))
    return defaults


def train_pca_rf(data: dict[str, Any], embeddings: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[Any, Any, dict[str, np.ndarray], dict[str, np.ndarray]]:
    pca_dim = int(config.get("pca", {}).get("n_components", 5))
    seed = int(config.get("seed", 42))
    pca = PCA(n_components=pca_dim, random_state=seed)
    phi = {"train_sub": pca.fit_transform(embeddings["train_sub"]).astype("float32")}
    for split in ("val", "test"):
        phi[split] = pca.transform(embeddings[split]).astype("float32")
    x_rf = {}
    for split in ("train_sub", "val", "test"):
        idx = data[f"{split}_idx"]
        x_rf[split] = np.concatenate([data["x_all"][idx].astype("float32"), phi[split]], axis=1)
    rf = RandomForestClassifier(**rf_config(config))
    rf.fit(x_rf["train_sub"], data["y"][data["train_sub_idx"]])
    probs = {split: rf.predict_proba(x_rf[split])[:, 1].astype("float32") for split in ("train_sub", "val", "test")}
    return rf, pca, x_rf, probs


def save_rf_artifacts(out_dir: str | Path, rf: Any, pca: Any, prefix: str) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rf_path = out / f"{prefix}_rf.pkl"
    pca_path = out / f"{prefix}_pca.pkl"
    joblib.dump(rf, rf_path)
    joblib.dump(pca, pca_path)
    return {"rf": str(rf_path), "pca": str(pca_path)}
