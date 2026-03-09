"""
Random seed utilities for reproducibility.

This module provides functions to set random seeds for all libraries
to ensure reproducible experiments.
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_random_state():
    """
    Get current random state for all libraries.
    
    Returns:
        Dict with random states
    """
    pass


def set_random_state(state_dict):
    """
    Restore random state from saved state.
    
    Args:
        state_dict: Dict with random states
    """
    pass

