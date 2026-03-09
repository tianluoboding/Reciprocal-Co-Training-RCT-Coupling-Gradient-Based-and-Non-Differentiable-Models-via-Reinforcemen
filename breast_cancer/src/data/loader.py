"""
Data loading module for Breast Cancer WDBC dataset.

This module handles loading of all data files required for the LLM-RF
iterative training framework.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any


def load_text_data(data_dir: Path, return_format='prompt') -> List[str]:
    """
    Load text data from JSON file (descriptive_text.json).
    
    Args:
        data_dir: Path to data directory (should contain raw/ subdirectory)
        return_format: What to return - 'prompt', 'response', 'both', or 'dict'
        
    Returns:
        List of text strings or list of dicts if return_format='dict'
    """
    text_file = data_dir / 'raw' / 'descriptive_text.json'
    
    if not text_file.exists():
        raise FileNotFoundError(f"Text data file not found: {text_file}")
    
    with open(text_file, 'r', encoding='utf-8') as f:
        text_data = json.load(f)
    
    # Validation
    if not isinstance(text_data, list):
        raise ValueError(f"Expected list, got {type(text_data)}")
    
    if len(text_data) == 0:
        raise ValueError("Text data is empty")
    
    # Check if items are dictionaries with expected keys
    if not isinstance(text_data[0], dict):
        return text_data
    
    # Return based on format
    if return_format == 'dict':
        return text_data
    elif return_format == 'prompt':
        texts = [item['prompt'] for item in text_data]
    elif return_format == 'response':
        texts = [item.get('response', '') for item in text_data]
    elif return_format == 'both':
        texts = [f"{item['prompt']} {item.get('response', '')}" for item in text_data]
    else:
        raise ValueError(f"Invalid return_format: {return_format}")
    
    return texts


def load_tabular_data(data_dir: Path) -> pd.DataFrame:
    """
    Load tabular features from CSV file (X_tabular.csv).
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        DataFrame with tabular features
    """
    tabular_file = data_dir / 'raw' / 'X_tabular.csv'
    
    if not tabular_file.exists():
        raise FileNotFoundError(f"Tabular data file not found: {tabular_file}")
    
    df = pd.read_csv(tabular_file)
    print(f"  ✓ Loaded tabular data: {df.shape}")
    
    return df


def load_labels(data_dir: Path) -> np.ndarray:
    """Load labels from numpy file."""
    labels_file = data_dir / 'raw' / 'y_all.npy'
    
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    y = np.load(labels_file)
    print(f"  ✓ Loaded labels: {y.shape}")
    
    return y


def load_train_test_split(data_dir: Path) -> Tuple[List[int], List[int]]:
    """Load train/test split indices."""
    train_file = data_dir / 'raw' / 'train_idxs.json'
    test_file = data_dir / 'raw' / 'test_idxs.json'
    
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError("Train/test split files not found")
    
    with open(train_file, 'r') as f:
        train_idx = json.load(f)
    with open(test_file, 'r') as f:
        test_idx = json.load(f)
    
    print(f"  ✓ Loaded train/test split: {len(train_idx)}/{len(test_idx)}")
    
    return train_idx, test_idx


def load_selected_features(data_dir: Path, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-selected features for a specific k value.
    
    Args:
        data_dir: Path to data directory
        k: Number of features (k-best)
        
    Returns:
        Tuple of (X_train_selected, X_test_selected)
    """
    train_file = data_dir / 'processed' / f'X_train_selected_k{k}.npy'
    test_file = data_dir / 'processed' / f'X_test_selected_k{k}.npy'
    
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Selected features for k={k} not found. Run preprocessing first.")
    
    X_train = np.load(train_file)
    X_test = np.load(test_file)
    
    print(f"  ✓ Loaded selected features (k={k}): train={X_train.shape}, test={X_test.shape}")
    
    return X_train, X_test


def load_feature_selection_metadata(data_dir: Path, k: int) -> Dict:
    """Load feature selection metadata for a specific k."""
    meta_file = data_dir / 'processed' / f'feature_selection_k{k}.json'
    
    if not meta_file.exists():
        raise FileNotFoundError(f"Feature selection metadata for k={k} not found")
    
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata


class DataLoader:
    """
    Main data loader class for WDBC breast cancer dataset.
    
    Usage:
        loader = DataLoader(data_dir, k=30)
        data = loader.load_all()
    """
    
    def __init__(self, data_dir: Path, k: int = 30):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to data directory
            k: Number of features to use (k-best selection)
        """
        self.data_dir = Path(data_dir)
        self.k = k
        
    def load_all(self) -> Dict[str, Any]:
        """
        Load all data files.
        
        Returns:
            Dictionary with:
                - 'texts': List of text strings
                - 'X_tabular': Full tabular DataFrame
                - 'y': Labels array
                - 'train_idx': Training indices
                - 'test_idx': Test indices
                - 'X_train_selected': Selected training features
                - 'X_test_selected': Selected test features
                - 'feature_names': Names of selected features
        """
        print("Loading all data files...")
        
        # Load text data
        texts = load_text_data(self.data_dir)
        print(f"  ✓ Loaded text data: {len(texts)} samples")
        
        # Load tabular data
        X_tabular = load_tabular_data(self.data_dir)
        
        # Load labels
        y = load_labels(self.data_dir)
        
        # Load train/test split
        train_idx, test_idx = load_train_test_split(self.data_dir)
        
        # Load selected features
        X_train_selected, X_test_selected = load_selected_features(self.data_dir, self.k)
        
        # Load feature metadata
        metadata = load_feature_selection_metadata(self.data_dir, self.k)
        
        data = {
            'texts': texts,
            'X_tabular': X_tabular,
            'y': y,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'X_train_selected': X_train_selected,
            'X_test_selected': X_test_selected,
            'feature_names': metadata['selected_names'],
            'k': self.k
        }
        
        # Validate
        self._validate_data(data)
        
        return data
    
    def _validate_data(self, data: Dict) -> None:
        """Validate data consistency."""
        print("\nValidating data consistency...")
        
        n_texts = len(data['texts'])
        n_tabular = len(data['X_tabular'])
        n_labels = len(data['y'])
        n_total = len(data['train_idx']) + len(data['test_idx'])
        
        if not (n_texts == n_tabular == n_labels == n_total):
            raise ValueError(
                f"Sample count mismatch: texts={n_texts}, tabular={n_tabular}, "
                f"labels={n_labels}, train+test={n_total}"
            )
        
        print(f"  ✓ Sample counts consistent: {n_texts} samples")
        print(f"  ✓ Train/test split valid: {len(data['train_idx'])}/{len(data['test_idx'])}")
        print(f"  ✓ Positive rate: {data['y'].mean()*100:.2f}%")
        print(f"  ✓ Selected features (k={self.k}): {len(data['feature_names'])}")
        print("  ✓ All validation checks passed!")
