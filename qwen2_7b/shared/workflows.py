"""Reusable public workflows for Qwen2-7B RCT scripts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .data_loading import load_config, load_dataset, resolve_path, set_seed
from .embedding_utils import classifier_embeddings, embed_texts_batched
from .metrics import metric_block, read_probability_csv, save_probability_csv, split_metrics, write_json
from .model_wrappers import build_classifier, create_lora_model, load_lora_adapter, load_qwen2_base, load_tokenizer, save_classifier_checkpoint
from .ppo_utils import combined_reward
from .rf_utils import save_rf_artifacts, train_pca_rf
from .thresholding import threshold_matched_metrics


def output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    out = config.get("outputs", {})
    dirs = {
        "outputs": resolve_path(config, out.get("root", "outputs/qwen2_7b")),
        "checkpoints": resolve_path(config, out.get("checkpoints", "checkpoints/qwen2_7b")),
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def split_texts(data: dict[str, Any], split: str) -> list[str]:
    return [data["prompts"][int(i)] for i in data[f"{split}_idx"]]


def run_frozen_embedding_rf(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    dirs = output_dirs(config)
    data = load_dataset(config)
    model_name = config.get("model", {}).get("name", "Qwen/Qwen2-7B-Instruct")
    tokenizer = load_tokenizer(model_name)
    base = load_qwen2_base(model_name, quantized=bool(config.get("model", {}).get("quantized", True)), trainable=False)
    emb_cfg = config.get("embedding", {})
    embeddings = {}
    for split in ("train_sub", "val", "test"):
        embeddings[split] = embed_texts_batched(
            base,
            tokenizer,
            split_texts(data, split),
            max_seq_length=int(emb_cfg.get("max_seq_length", 320)),
            batch_size=int(emb_cfg.get("batch_size", 4)),
        )
    rf, pca, _x_rf, probs = train_pca_rf(data, embeddings, config)
    prefix = f"{config.get('dataset', 'dataset')}_qwen2_stage1"
    artifacts = save_rf_artifacts(dirs["outputs"], rf, pca, prefix)
    for split in ("train_sub", "val", "test"):
        idx = data[f"{split}_idx"]
        save_probability_csv(dirs["outputs"] / f"{prefix}_{split}_probs.csv", split, idx, data["y"][idx], probs[split], source="qwen2_frozen_embedding_rf")
    result = {
        "method": "qwen2_frozen_embedding_rf",
        "dataset": config.get("dataset"),
        "split_audit": data["split_audit"],
        "metrics": split_metrics(data, probs),
        "artifacts": artifacts,
    }
    write_json(dirs["outputs"] / f"{prefix}.json", result)
    return result


def train_supervised_lora(config_path: str | Path, *, stage_name: str = "qwen2_lora_supervised") -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    class PromptDataset(Dataset):
        def __init__(self, texts: list[str], labels: np.ndarray) -> None:
            self.texts = texts
            self.labels = labels.astype("float32")
        def __len__(self) -> int:
            return len(self.texts)
        def __getitem__(self, i: int) -> dict[str, Any]:
            return {"text": self.texts[i], "label": float(self.labels[i])}

    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    dirs = output_dirs(config)
    data = load_dataset(config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("supervised", {})
    model_name = model_cfg.get("name", "Qwen/Qwen2-7B-Instruct")
    tokenizer = load_tokenizer(model_name)
    base = load_qwen2_base(model_name, quantized=bool(model_cfg.get("quantized", True)), trainable=True)
    adapter = create_lora_model(base, config.get("lora", {}))
    model = build_classifier(adapter, with_value_head=False)
    dev = next(model.base_model.parameters()).device

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        enc = tokenizer([row["text"] for row in batch], return_tensors="pt", padding=True, truncation=True, max_length=int(model_cfg.get("max_seq_length", 320)))
        enc["labels"] = torch.tensor([row["label"] for row in batch], dtype=torch.float32)
        return enc

    train_idx = data["train_sub_idx"]
    val_idx = data["val_idx"]
    train_ds = PromptDataset(split_texts(data, "train_sub"), data["y"][train_idx])
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg.get("batch_size", 1)), shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(train_cfg.get("learning_rate", 2e-4)))
    best = {"val_roc_auc": -1.0, "epoch": -1, "checkpoint": None}
    patience = int(train_cfg.get("early_stopping_patience", 2))
    stale = 0
    for epoch in range(1, int(train_cfg.get("max_epochs", 3)) + 1):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(dev)
            batch = {key: value.to(dev) for key, value in batch.items()}
            logits = model(**batch)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        probs = predict_lora_splits(model, tokenizer, data, config)["val"]
        val_auc = metric_block(data["y"][val_idx], probs)["roc_auc"]
        if val_auc > best["val_roc_auc"]:
            best.update({"val_roc_auc": float(val_auc), "epoch": epoch})
            ckpt = dirs["checkpoints"] / f"{config.get('dataset', 'dataset')}_{stage_name}_best"
            save_classifier_checkpoint(model, ckpt, {"stage": stage_name, "epoch": epoch, "selection_metric": "validation_roc_auc"})
            best["checkpoint"] = str(ckpt)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    probs_all = predict_lora_splits(model, tokenizer, data, config)
    prefix = f"{config.get('dataset', 'dataset')}_{stage_name}"
    for split in ("train_sub", "val", "test"):
        idx = data[f"{split}_idx"]
        save_probability_csv(dirs["outputs"] / f"{prefix}_{split}_probs.csv", split, idx, data["y"][idx], probs_all[split], source=stage_name)
    result = {"method": stage_name, "dataset": config.get("dataset"), "best": best, "metrics": split_metrics(data, probs_all), "split_audit": data["split_audit"]}
    write_json(dirs["outputs"] / f"{prefix}.json", result)
    return result


def predict_lora_splits(model: Any, tokenizer: Any, data: dict[str, Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    import torch

    max_seq_length = int(config.get("model", {}).get("max_seq_length", 320))
    batch_size = int(config.get("evaluation", {}).get("batch_size", 4))
    dev = next(model.base_model.parameters()).device
    probs: dict[str, np.ndarray] = {}
    model.eval()
    for split in ("train_sub", "val", "test"):
        texts = split_texts(data, split)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                enc = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
                enc = {key: value.to(dev) for key, value in enc.items()}
                logits = model(**enc)
                chunks.append(torch.sigmoid(logits).float().cpu().numpy())
        probs[split] = np.concatenate(chunks).astype("float32")
    return probs


def run_ppo_rf_refresh(config_path: str | Path) -> dict[str, Any]:
    """Run the public RCT outer loop: PPO updates, then RF refresh on new embeddings."""
    import torch
    import torch.nn.functional as F

    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    dirs = output_dirs(config)
    data = load_dataset(config)
    model_cfg = config.get("model", {})
    ppo_cfg = config.get("ppo", {})
    model_name = model_cfg.get("name", "Qwen/Qwen2-7B-Instruct")
    tokenizer = load_tokenizer(model_name)
    base = load_qwen2_base(model_name, quantized=bool(model_cfg.get("quantized", True)), trainable=True)
    if config.get("start_checkpoint"):
        adapter = load_lora_adapter(base, resolve_path(config, config["start_checkpoint"]) / "lora_adapter", trainable=True)
    else:
        adapter = create_lora_model(base, config.get("lora", {}))
    model = build_classifier(adapter, with_value_head=True)
    dev = next(model.base_model.parameters()).device
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(ppo_cfg.get("learning_rate", 5e-6)))
    history = []
    best_val_auc = -1.0
    stale = 0
    max_outer = int(ppo_cfg.get("max_outer_iterations", 10))
    for outer in range(1, max_outer + 1):
        t0 = time.time()
        teacher_embeddings = {}
        for split in ("train_sub", "val", "test"):
            teacher_embeddings[split] = classifier_embeddings(
                model,
                tokenizer,
                split_texts(data, split),
                max_seq_length=int(model_cfg.get("max_seq_length", 320)),
                batch_size=int(config.get("evaluation", {}).get("batch_size", 4)),
            )
        teacher_rf, teacher_pca, _teacher_x_rf, teacher_probs = train_pca_rf(data, teacher_embeddings, config)

        train_idx = data["train_sub_idx"]
        rng = np.random.default_rng(int(config.get("seed", 42)) + outer)
        sample_n = min(int(ppo_cfg.get("samples_per_outer", 512)), len(train_idx))
        sampled = rng.choice(train_idx, size=sample_n, replace=True)
        labels_np = data["y"][sampled]
        sample_to_teacher = {int(idx): float(prob) for idx, prob in zip(train_idx, teacher_probs["train_sub"])}
        tab_dim = data["x_all"].shape[1]
        texts = [data["prompts"][int(i)] for i in sampled]
        for start in range(0, len(texts), int(ppo_cfg.get("batch_size", 1))):
            batch_texts = texts[start:start + int(ppo_cfg.get("batch_size", 1))]
            batch_idx = sampled[start:start + len(batch_texts)]
            batch_labels_np = labels_np[start:start + len(batch_texts)]
            batch_labels = torch.tensor(batch_labels_np, dtype=torch.float32, device=dev)
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=int(model_cfg.get("max_seq_length", 320)))
            enc = {key: value.to(dev) for key, value in enc.items()}
            out = model.pooled_outputs(enc["input_ids"], enc["attention_mask"])
            logits = out["logits"]
            supervised_loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            values = out["values"]
            policy_probs = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
            actions = torch.bernoulli(policy_probs).detach()
            action_features = out["embedding"].detach().float().cpu().numpy()
            if action_features.shape[1] > tab_dim:
                action_features[:, :tab_dim] += actions.detach().cpu().numpy().reshape(-1, 1)
            action_phi = teacher_pca.transform(action_features).astype("float32")
            action_x = np.concatenate([data["x_all"][batch_idx].astype("float32"), action_phi], axis=1)
            p_rf_action = teacher_rf.predict_proba(action_x)[:, 1].astype("float32")
            p_rf_reference = np.asarray([sample_to_teacher[int(idx)] for idx in batch_idx], dtype="float32")
            reward_np = combined_reward(actions.detach().cpu().numpy(), batch_labels_np, p_rf_action, p_rf_reference, config).astype("float32")
            rewards = torch.tensor(reward_np, dtype=torch.float32, device=dev)
            advantage = rewards - values.detach()
            policy_loss = -(torch.log(torch.where(actions > 0.5, policy_probs, 1 - policy_probs)) * advantage).mean()
            value_loss = F.mse_loss(values, rewards)
            loss = policy_loss + float(ppo_cfg.get("value_coef", 0.5)) * value_loss + float(ppo_cfg.get("supervised_coef", 0.5)) * supervised_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(ppo_cfg.get("max_grad_norm", 0.5)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        emb = {}
        for split in ("train_sub", "val", "test"):
            emb[split] = classifier_embeddings(model, tokenizer, split_texts(data, split), max_seq_length=int(model_cfg.get("max_seq_length", 320)), batch_size=int(config.get("evaluation", {}).get("batch_size", 4)))
        rf, pca, _x_rf, probs = train_pca_rf(data, emb, config)
        metrics = split_metrics(data, probs)
        val_auc = metrics["val"]["roc_auc"]
        prefix = f"{config.get('dataset', 'dataset')}_qwen2_rct_outer{outer:02d}"
        save_rf_artifacts(dirs["outputs"], rf, pca, prefix)
        for split in ("train_sub", "val", "test"):
            idx = data[f"{split}_idx"]
            save_probability_csv(dirs["outputs"] / f"{prefix}_{split}_rf_probs.csv", split, idx, data["y"][idx], probs[split], source="rct_rf")
        history.append({"outer": outer, "seconds": time.time() - t0, "metrics": metrics})
        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            stale = 0
            save_classifier_checkpoint(model, dirs["checkpoints"] / f"{prefix}_qwen2_ppo_best", {"stage": "qwen2_rct", "outer": outer, "selection_metric": "rf_validation_roc_auc"})
        else:
            stale += 1
            if stale >= int(ppo_cfg.get("convergence_patience", 3)):
                break
    result = {"method": "qwen2_ppo_rf_refresh_rct", "dataset": config.get("dataset"), "history": history, "best_val_auc": best_val_auc, "split_audit": data["split_audit"]}
    write_json(dirs["outputs"] / f"{config.get('dataset', 'dataset')}_qwen2_rct_history.json", result)
    return result


def run_threshold_eval(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    dirs = output_dirs(config)
    eval_cfg = config.get("threshold_eval", {})
    y_val, p_val = read_probability_csv(resolve_path(config, eval_cfg["val_prob_csv"]))
    y_test, p_test = read_probability_csv(resolve_path(config, eval_cfg["test_prob_csv"]))
    targets = [float(x) for x in eval_cfg.get("target_recalls", [0.8])]
    result = {"method": "threshold_matched_eval", "threshold_source": "validation_only", "metrics": threshold_matched_metrics(y_val, p_val, y_test, p_test, targets)}
    write_json(dirs["outputs"] / f"{config.get('dataset', 'dataset')}_qwen2_threshold_eval.json", result)
    return result


def run_table_metrics(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    dirs = output_dirs(config)
    rows = []
    for item in config.get("table_metrics", {}).get("probability_csvs", []):
        y, p = read_probability_csv(resolve_path(config, item["path"]))
        row = {"name": item.get("name", item["path"]), **metric_block(y, p)}
        rows.append(row)
    result = {"method": "qwen2_table_metrics", "rows": rows}
    write_json(dirs["outputs"] / f"{config.get('dataset', 'dataset')}_qwen2_table_metrics.json", result)
    return result
