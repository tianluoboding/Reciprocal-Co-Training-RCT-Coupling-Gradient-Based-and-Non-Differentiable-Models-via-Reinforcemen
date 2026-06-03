"""PPO utilities for reciprocal co-training with RF rewards."""

from __future__ import annotations

from typing import Any

import numpy as np


def rf_advantage_reward(p_rf_action: np.ndarray, p_rf_reference: np.ndarray, *, scale: float, clip: tuple[float, float]) -> np.ndarray:
    advantage = np.asarray(p_rf_action, dtype="float64") - np.asarray(p_rf_reference, dtype="float64")
    return scale * np.clip(advantage, clip[0], clip[1])


def accuracy_reward(actions: np.ndarray, labels: np.ndarray, mode: str = "fn_heavy") -> np.ndarray:
    a = np.asarray(actions).astype(int)
    y = np.asarray(labels).astype(int)
    if mode != "fn_heavy":
        return (a == y).astype("float64")
    rewards = np.zeros_like(y, dtype="float64")
    rewards[(a == 1) & (y == 1)] = 1.0
    rewards[(a == 0) & (y == 1)] = -1.5
    rewards[(a == 0) & (y == 0)] = 0.2
    rewards[(a == 1) & (y == 0)] = -0.2
    return rewards


def combined_reward(actions: np.ndarray, labels: np.ndarray, p_rf_action: np.ndarray, p_rf_reference: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    reward_cfg = config.get("reward", {})
    r_rf = rf_advantage_reward(
        p_rf_action,
        p_rf_reference,
        scale=float(reward_cfg.get("scale", 2.0)),
        clip=tuple(reward_cfg.get("clip", [-0.45, 0.45])),
    )
    r_acc = accuracy_reward(actions, labels, reward_cfg.get("accuracy_mode", "fn_heavy"))
    return r_rf + float(reward_cfg.get("lambda_acc", 0.5)) * r_acc


def ppo_clipped_policy_loss(log_prob: Any, old_log_prob: Any, advantage: Any, clip_epsilon: float) -> Any:
    import torch

    ratio = torch.exp(log_prob - old_log_prob)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    return -torch.mean(torch.minimum(unclipped, clipped))


def normalize_advantages(values: Any) -> Any:
    import torch

    if values.numel() <= 1:
        return values
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-8)
