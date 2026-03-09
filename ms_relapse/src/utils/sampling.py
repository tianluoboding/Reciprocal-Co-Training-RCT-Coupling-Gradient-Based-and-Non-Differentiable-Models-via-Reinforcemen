"""
加权采样工具 - 用于PPO训练中平衡类别分布

这个模块提供加权采样功能，模拟ADASYN的平衡效果，但保留真实样本索引。
目标是让LLM在平衡的数据分布上训练，与RF训练时的ADASYN保持一致。
"""

import numpy as np
from typing import Tuple, Dict


def weighted_sample_indices(train_idx: np.ndarray,
                            y_labels: np.ndarray,
                            size: int,
                            y1_weight: float = 1.8,
                            seed: int = None,
                            replace: bool = True) -> np.ndarray:
    """
    对训练索引进行加权采样，给y=1更高的权重。
    
    这种方法模拟ADASYN的效果，但保留真实的样本索引（不生成synthetic样本）。
    通过提高y=1样本的采样概率，使采样后的分布接近1:1平衡。
    
    Args:
        train_idx: 训练集的索引数组
        y_labels: 所有样本的标签（需要索引到train_idx对应的标签）
        size: 需要采样的样本数量
        y1_weight: y=1样本的权重（相对于y=0的权重1.0）
        seed: 随机种子（用于可重复性）
        replace: 是否允许重复采样（True模拟oversampling）
    
    Returns:
        采样后的索引数组 (shape: [size])
    
    Example:
        >>> train_idx = np.array([0, 1, 2, 3, 4])
        >>> y_labels = np.array([0, 0, 1, 0, 1])
        >>> sampled = weighted_sample_indices(train_idx, y_labels, size=10, y1_weight=2.0)
        >>> # y=1的样本会被更频繁地采样
        
        原始分布: y=0: 64%, y=1: 36%
        y1_weight=1.8时期望: y=0: 56%, y=1: 44% (接近1:1平衡)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 获取train_idx对应的标签
    y_train = y_labels[train_idx]
    
    # 计算采样权重
    sample_weights = np.ones(len(train_idx))
    sample_weights[y_train == 1] = y1_weight  # y=1的样本权重更高
    
    # 归一化为概率
    sample_probs = sample_weights / sample_weights.sum()
    
    # 加权采样
    sampled_indices = np.random.choice(
        train_idx,
        size=size,
        replace=replace,
        p=sample_probs
    )
    
    return sampled_indices


def compute_expected_distribution(y_train: np.ndarray,
                                  y1_weight: float,
                                  size: int) -> Dict[str, float]:
    """
    计算加权采样后的期望分布。
    
    用于验证采样策略是否达到期望的平衡效果。
    
    Args:
        y_train: 训练集标签
        y1_weight: y=1的采样权重
        size: 采样数量
    
    Returns:
        包含期望分布信息的字典：
        - expected_y0: 期望的y=0样本数
        - expected_y1: 期望的y=1样本数
        - expected_y1_ratio: 期望的y=1比例
        - original_y1_ratio: 原始的y=1比例
    
    Example:
        >>> y_train = np.array([0, 0, 0, 1, 1])  # 40% y=1
        >>> dist = compute_expected_distribution(y_train, y1_weight=1.5, size=100)
        >>> print(f"Expected y=1 ratio: {dist['expected_y1_ratio']:.2f}")
        Expected y=1 ratio: 0.50  # 接近平衡
    """
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    
    # 计算总权重
    total_weight = n_neg * 1.0 + n_pos * y1_weight
    
    # 计算期望数量
    expected_pos = (n_pos * y1_weight / total_weight) * size
    expected_neg = (n_neg * 1.0 / total_weight) * size
    
    return {
        'expected_y0': int(expected_neg),
        'expected_y1': int(expected_pos),
        'expected_y1_ratio': expected_pos / size,
        'original_y1_ratio': n_pos / len(y_train)
    }


def analyze_sampling_distribution(sampled_indices: np.ndarray,
                                  y_labels: np.ndarray) -> Dict[str, any]:
    """
    分析采样结果的实际分布。
    
    Args:
        sampled_indices: 采样后的索引
        y_labels: 所有样本的标签
    
    Returns:
        包含实际分布信息的字典
    """
    sampled_y = y_labels[sampled_indices]
    
    n_y0 = np.sum(sampled_y == 0)
    n_y1 = np.sum(sampled_y == 1)
    total = len(sampled_y)
    
    # 统计重复采样情况
    unique_indices = np.unique(sampled_indices)
    n_unique = len(unique_indices)
    
    return {
        'n_y0': int(n_y0),
        'n_y1': int(n_y1),
        'y1_ratio': n_y1 / total,
        'n_unique_samples': n_unique,
        'repetition_rate': 1.0 - (n_unique / total)
    }


__all__ = [
    'weighted_sample_indices',
    'compute_expected_distribution',
    'analyze_sampling_distribution'
]

