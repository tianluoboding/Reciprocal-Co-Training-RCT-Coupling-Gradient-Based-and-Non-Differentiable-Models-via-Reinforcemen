"""Data loading helpers for Qwen2-7B RCT experiments.

The public repository intentionally does not include raw clinical data. These
helpers expect users to provide preprocessed prompt, label, split, and tabular
feature files through an example config.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config file."""
    cfg_path = Path(path)
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install pyyaml or provide a JSON config file") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    data["config_path"] = str(cfg_path)
    data["config_dir"] = str(cfg_path.parent)
    return data


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve relative paths against project_root, then config_dir."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed in public configs: {path}")
    base = Path(config.get("project_root", "."))
    candidate = base / path
    if candidate.exists() or "project_root" in config:
        return candidate
    return Path(config.get("config_dir", ".")) / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_prompt_records(path: Path, prompt_key: str = "prompt") -> list[str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list of prompt records: {path}")
    prompts: list[str] = []
    for i, item in enumerate(records):
        if not isinstance(item, dict) or prompt_key not in item:
            raise ValueError(f"Missing prompt key {prompt_key!r} at row {i}")
        prompts.append(str(item[prompt_key]))
    return prompts


def read_indices(path: Path) -> np.ndarray:
    values = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(values, dtype=int).reshape(-1)


def train_val_split(train_idx: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    train_sub, val = train_test_split(
        train_idx,
        test_size=float(val_ratio),
        random_state=int(seed),
        stratify=y[train_idx],
    )
    return np.asarray(train_sub, dtype=int), np.asarray(val, dtype=int)


def load_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Load prompt text, labels, split indices, and selected tabular features."""
    data_cfg = config["data"]
    prompt_key = data_cfg.get("prompt_key", "prompt")
    prompts = read_prompt_records(resolve_path(config, data_cfg["prompts_json"]), prompt_key)
    y = np.load(resolve_path(config, data_cfg["labels_npy"])).astype(int).reshape(-1)
    train_idx = read_indices(resolve_path(config, data_cfg["train_indices_json"]))
    test_idx = read_indices(resolve_path(config, data_cfg["test_indices_json"]))

    if len(prompts) != len(y):
        raise ValueError(f"Prompt/label length mismatch: {len(prompts)} vs {len(y)}")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Only binary labels 0/1 are supported")

    split_cfg = config.get("split", {})
    val_ratio = float(split_cfg.get("val_ratio", 0.15))
    seed = int(config.get("seed", 42))
    if "val_indices_json" in data_cfg:
        train_sub_idx = train_idx
        val_idx = read_indices(resolve_path(config, data_cfg["val_indices_json"]))
    else:
        train_sub_idx, val_idx = train_val_split(train_idx, y, val_ratio, seed)

    x_train = np.load(resolve_path(config, data_cfg["x_train_npy"])).astype("float32")
    x_test = np.load(resolve_path(config, data_cfg["x_test_npy"])).astype("float32")
    if x_train.shape[0] != len(train_idx) or x_test.shape[0] != len(test_idx):
        raise ValueError("Tabular feature rows do not align with train/test index files")
    x_all = np.zeros((len(y), x_train.shape[1]), dtype="float32")
    x_all[train_idx] = x_train
    x_all[test_idx] = x_test

    overlap = {
        "train_sub_val": len(set(train_sub_idx.tolist()) & set(val_idx.tolist())),
        "train_sub_test": len(set(train_sub_idx.tolist()) & set(test_idx.tolist())),
        "val_test": len(set(val_idx.tolist()) & set(test_idx.tolist())),
    }
    if any(overlap.values()):
        raise ValueError({"split_overlap": overlap})

    return {
        "prompts": prompts,
        "y": y,
        "x_all": x_all,
        "train_idx": train_idx,
        "train_sub_idx": train_sub_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "split_audit": {
            "val_ratio": val_ratio,
            "seed": seed,
            "sizes": {"train_sub": int(len(train_sub_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "test_used_for_split": False,
            "overlap": {k: int(v) for k, v in overlap.items()},
        },
    }
