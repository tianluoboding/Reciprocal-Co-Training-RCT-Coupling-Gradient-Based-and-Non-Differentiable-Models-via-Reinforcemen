"""Embedding extraction utilities for Qwen2-7B."""

from __future__ import annotations

from typing import Any

import numpy as np


def mean_pool_last_hidden(hidden: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=hidden.dtype)
    return summed / denom


def embed_texts_batched(model: Any, tokenizer: Any, texts: list[str], *, max_seq_length: int, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    dev = next(model.parameters()).device
    pooled_batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
            enc = {key: value.to(dev) for key, value in enc.items()}
            out = model(**enc, output_hidden_states=True, return_dict=True)
            pooled = mean_pool_last_hidden(out.hidden_states[-1], enc["attention_mask"])
            pooled_batches.append(pooled.float().cpu())
    return torch.cat(pooled_batches, dim=0).numpy().astype("float32")


def classifier_embeddings(model: Any, tokenizer: Any, texts: list[str], *, max_seq_length: int, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    dev = next(model.base_model.parameters()).device
    pooled_batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
            enc = {key: value.to(dev) for key, value in enc.items()}
            pooled_batches.append(model.pooled_outputs(enc["input_ids"], enc["attention_mask"])["embedding"].float().cpu())
    return torch.cat(pooled_batches, dim=0).numpy().astype("float32")
