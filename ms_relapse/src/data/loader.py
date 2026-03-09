"""
Data loading module.

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
    Load text data from JSON file (descriptive_text_2.json).
    
    This loads the AB test winner text format (Structured).
    The JSON contains a list of dictionaries with 'prompt' and 'response' keys.
    
    Args:
        data_dir: Path to data directory (should contain raw/ subdirectory)
        return_format: What to return - 'prompt', 'response', 'both', or 'dict'
                      - 'prompt': Return list of prompt strings (default, for LLM input)
                      - 'response': Return list of response strings
                      - 'both': Return list of combined "prompt + response" strings
                      - 'dict': Return original list of dicts
        
    Returns:
        List of text strings (2192 samples expected) or list of dicts if return_format='dict'
        
    Raises:
        FileNotFoundError: If text data file not found
        ValueError: If data format is invalid
    """
    text_file = data_dir / 'raw' / 'descriptive_text_2.json'
    
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
        # Fallback: assume it's already a list of strings
        return text_data
    
    # Verify all items have 'prompt' key
    missing_prompt = sum(1 for item in text_data if 'prompt' not in item)
    if missing_prompt > 0:
        raise ValueError(f"Found {missing_prompt} items without 'prompt' key")
    
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
    
    # Check for None or empty strings
    none_count = sum(1 for t in texts if t is None or t == '')
    if none_count > 0:
        print(f"Warning: Found {none_count} None or empty text entries")
    
    return texts


def load_tabular_data(data_dir: Path) -> pd.DataFrame:
    """
    Load tabular features from CSV file (X_tabular.csv).
    
    NOTE: This loads RAW data (29 columns) that will need preprocessing
    in Phase 2.2 (feature selection, encoding, normalization).
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        DataFrame with tabular features (2192 rows × 29 columns expected)
        
    Raises:
        FileNotFoundError: If tabular data file not found
        ValueError: If data shape is invalid
    """
    tabular_file = data_dir / 'raw' / 'X_tabular.csv'
    
    if not tabular_file.exists():
        raise FileNotFoundError(f"Tabular data file not found: {tabular_file}")
    
    # Load CSV
    df = pd.read_csv(tabular_file)
    
    # Basic validation
    if df.empty:
        raise ValueError("Tabular data is empty")
    
    if df.shape[1] != 29:
        print(f"Warning: Expected 29 columns, got {df.shape[1]}")
    
    return df


def load_labels(data_dir: Path) -> np.ndarray:
    """
    Load labels from numpy file (y_all.npy).
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Array of binary labels (2192 samples expected)
        
    Raises:
        FileNotFoundError: If labels file not found
        ValueError: If labels are not binary
    """
    labels_file = data_dir / 'raw' / 'y_all.npy'
    
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    # Load numpy array
    y = np.load(labels_file)
    
    # Validation
    if y.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {y.shape}")
    
    if len(y) == 0:
        raise ValueError("Labels array is empty")
    
    # Check if binary
    unique_values = np.unique(y)
    if not np.array_equal(unique_values, np.array([0, 1])) and \
       not np.array_equal(unique_values, np.array([0.])) and \
       not np.array_equal(unique_values, np.array([1.])) and \
       not (len(unique_values) == 2 and set(unique_values).issubset({0, 1, 0., 1.})):
        print(f"Warning: Expected binary labels (0, 1), found unique values: {unique_values}")
    
    return y


def load_train_test_split(data_dir: Path) -> Tuple[List[int], List[int]]:
    """
    Load train and test indices from JSON files.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Tuple of (train_indices, test_indices)
        Expected: train=1753, test=439
        
    Raises:
        FileNotFoundError: If index files not found
        ValueError: If indices are invalid
    """
    train_file = data_dir / 'raw' / 'train_idxs.json'
    test_file = data_dir / 'raw' / 'test_idxs.json'
    
    if not train_file.exists():
        raise FileNotFoundError(f"Train indices file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test indices file not found: {test_file}")
    
    # Load indices
    with open(train_file, 'r') as f:
        train_idx = json.load(f)
    with open(test_file, 'r') as f:
        test_idx = json.load(f)
    
    # Validation
    if not isinstance(train_idx, list) or not isinstance(test_idx, list):
        raise ValueError("Indices must be lists")
    
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Index lists cannot be empty")
    
    # Check for overlap
    train_set = set(train_idx)
    test_set = set(test_idx)
    overlap = train_set & test_set
    if overlap:
        raise ValueError(f"Train/test overlap detected: {len(overlap)} samples")
    
    return train_idx, test_idx


class DataLoader:
    """
    Comprehensive data loader for the LLM-RF framework.
    
    Loads all required data files and performs validation checks.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to data directory (should contain raw/ subdirectory)
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_all(self, validate: bool = True) -> Dict[str, Any]:
        """
        Load all data files at once.
        
        Args:
            validate: Whether to run validation checks
            
        Returns:
            Dictionary containing:
                - 'texts': List of text strings
                - 'X_tabular': DataFrame of tabular features
                - 'y': Array of labels
                - 'train_idx': List of train indices
                - 'test_idx': List of test indices
                
        Raises:
            Various exceptions if data loading or validation fails
        """
        print("Loading all data files...")
        
        # Load each component
        texts = load_text_data(self.data_dir)
        print(f"  ✓ Loaded text data: {len(texts)} samples")
        
        X_tabular = load_tabular_data(self.data_dir)
        print(f"  ✓ Loaded tabular data: {X_tabular.shape}")
        
        y = load_labels(self.data_dir)
        print(f"  ✓ Loaded labels: {y.shape}")
        
        train_idx, test_idx = load_train_test_split(self.data_dir)
        print(f"  ✓ Loaded train/test split: {len(train_idx)}/{len(test_idx)}")
        
        data = {
            'texts': texts,
            'X_tabular': X_tabular,
            'y': y,
            'train_idx': train_idx,
            'test_idx': test_idx
        }
        
        if validate:
            self._validate_data(data)
        
        return data
    
    def _validate_data(self, data: Dict[str, Any]) -> None:
        """
        Validate loaded data for consistency.
        
        Args:
            data: Dictionary of loaded data
            
        Raises:
            ValueError: If validation fails
        """
        print("\nValidating data consistency...")
        
        n_texts = len(data['texts'])
        n_tabular = data['X_tabular'].shape[0]
        n_labels = len(data['y'])
        n_train = len(data['train_idx'])
        n_test = len(data['test_idx'])
        
        # Check sample counts match
        if not (n_texts == n_tabular == n_labels):
            raise ValueError(
                f"Sample count mismatch: texts={n_texts}, "
                f"tabular={n_tabular}, labels={n_labels}"
            )
        
        n_total = n_texts
        
        # Check train/test split covers all samples
        if n_train + n_test != n_total:
            raise ValueError(
                f"Train/test split doesn't cover all samples: "
                f"{n_train} + {n_test} != {n_total}"
            )
        
        # Check indices are within range
        max_train_idx = max(data['train_idx'])
        max_test_idx = max(data['test_idx'])
        if max_train_idx >= n_total or max_test_idx >= n_total:
            raise ValueError(
                f"Index out of range: max_train={max_train_idx}, "
                f"max_test={max_test_idx}, n_samples={n_total}"
            )
        
        # Check label distribution
        positive_rate = data['y'].mean()
        print(f"  ✓ Sample counts consistent: {n_total} samples")
        print(f"  ✓ Train/test split valid: {n_train}/{n_test}")
        print(f"  ✓ Positive rate: {positive_rate:.2%}")
        
        # Check for missing values in tabular data
        missing_counts = data['X_tabular'].isnull().sum()
        total_missing = missing_counts.sum()
        if total_missing > 0:
            print(f"  ⚠ Warning: {total_missing} missing values in tabular data")
            cols_with_missing = missing_counts[missing_counts > 0]
            print(f"    Columns with missing values: {len(cols_with_missing)}")
        else:
            print(f"  ✓ No missing values in tabular data")
        
        print("  ✓ All validation checks passed!")
    
    def get_column_info(self) -> pd.DataFrame:
        """
        Get information about tabular columns.
        
        Returns:
            DataFrame with column information (name, dtype, missing count, etc.)
        """
        X = load_tabular_data(self.data_dir)
        
        info = pd.DataFrame({
            'column': X.columns,
            'dtype': X.dtypes.values,
            'missing': X.isnull().sum().values,
            'missing_pct': (X.isnull().sum() / len(X) * 100).values,
            'unique_values': [X[col].nunique() for col in X.columns]
        })
        
        return info

