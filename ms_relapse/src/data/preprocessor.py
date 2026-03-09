"""
Tabular data preprocessing module.

This module handles all preprocessing steps for the 29 raw tabular features,
including column filtering, encoding, normalization, etc.

Following Ben's preprocessing pipeline from single_run_voting_ensemble.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any
import json


# Configuration constants based on Ben's code
COLUMNS_TO_DELETE = [
    "BMI",  # May have missing values
    "ETHNICITY_DESC",  # Redundant with RACE_DESC
    "NfLValue",  # Neurofilament value, may have missing values
    "HighNfL Binary",  # Derived from NfLValue
]

COLUMNS_TARGET_RELATED = [
    "RelapseInYearBeforeFVBinary",  # Too directly related to target
    "RelapseInThe3YearsBeforeFVBinary",  # Too directly related to target
]

COLUMNS_WITH_X = [
    "PYRAMIDAL_FUNCTION", "CEREBELLAR_FUNCTION", "BRAINSTEM_FUNCTION",
    "SENSORY_FUNCTION", "BOWEL_BLADDER_FUNCTION", "VISUAL_FUNCTION", "MENTAL_FUNCTION"
]

# Columns already binary (0/1) in our data
BINARY_COLUMNS = ["SEX", "SMOKING_EVER", "TreatmentBeforeFV", "Treatment with Injectable Med"]

# Columns for one-hot encoding
CATEGORICAL_COLUMNS = [
    "RACE_DESC", "FAMILY_MS", "DISEASE_CATEGORY_DESC_FV",
    "NewT2lesionYearBeforeFV", "NewGadLesionYearBeforeFV"
]

# Columns to normalize (continuous and ordinal)
CONTINUOUS_COLUMNS = [
    "AgeatFV", "DiseasedurationatFV", "EDSS_FV",
    "PYRAMIDAL_FUNCTION", "CEREBELLAR_FUNCTION", "BRAINSTEM_FUNCTION",
    "SENSORY_FUNCTION", "BOWEL_BLADDER_FUNCTION", "VISUAL_FUNCTION", "MENTAL_FUNCTION",
    "TotalnumberofrelapsesbeforeFV", "Numberofrelapsesinthe3yearsbeforeFV",
    "Numberofrelapsesinthe1yearbeforeFV", "timeSinceLastAttack"
]


def filter_columns(df: pd.DataFrame, 
                   exclude_columns: List[str] = None,
                   exclude_target_related: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filter out unnecessary or problematic columns.
    
    Args:
        df: Input DataFrame
        exclude_columns: List of column names to exclude (if None, use default)
        exclude_target_related: Whether to exclude target-related columns
        
    Returns:
        Tuple of (filtered DataFrame, metadata dict)
    """
    if exclude_columns is None:
        exclude_columns = COLUMNS_TO_DELETE.copy()
    
    if exclude_target_related:
        exclude_columns.extend(COLUMNS_TARGET_RELATED)
    
    # Only drop columns that exist in the DataFrame
    existing_exclude = [col for col in exclude_columns if col in df.columns]
    dropped_cols = existing_exclude
    kept_cols = [col for col in df.columns if col not in existing_exclude]
    
    df_filtered = df.drop(columns=existing_exclude, errors='ignore')
    
    metadata = {
        'dropped_columns': dropped_cols,
        'kept_columns': kept_cols,
        'original_shape': df.shape,
        'filtered_shape': df_filtered.shape
    }
    
    print(f"  Filtered columns: {df.shape[1]} → {df_filtered.shape[1]}")
    print(f"  Dropped: {dropped_cols}")
    
    return df_filtered, metadata


def handle_special_values(df: pd.DataFrame, 
                          strategy: str = 'remove') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle special values like 'X' in function score columns.
    
    Args:
        df: Input DataFrame
        strategy: 'remove' (drop rows) or 'impute' (convert to specific value)
        
    Returns:
        Tuple of (processed DataFrame, metadata dict)
    """
    original_len = len(df)
    rows_with_x = 0
    
    # Check for 'X' values in function columns
    for col in COLUMNS_WITH_X:
        if col in df.columns:
            # Check if column contains 'X' (as string)
            if df[col].dtype == 'object':
                mask = df[col].astype(str).str.contains('X', na=False)
                rows_with_x += mask.sum()
    
    if rows_with_x > 0:
        print(f"  Found {rows_with_x} rows with 'X' values")
        
        if strategy == 'remove':
            # Remove rows with 'X'
            df_processed = df[~df[COLUMNS_WITH_X].isin(['X']).any(axis=1)].copy()
            print(f"  Removed rows: {original_len} → {len(df_processed)}")
        else:
            # Impute 'X' with a specific value (e.g., median or -1)
            df_processed = df.copy()
            for col in COLUMNS_WITH_X:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].replace('X', -1)
            print(f"  Imputed 'X' values with -1")
    else:
        print(f"  No 'X' values found (already clean)")
        df_processed = df.copy()
    
    metadata = {
        'rows_with_x': rows_with_x,
        'strategy': strategy,
        'original_length': original_len,
        'final_length': len(df_processed),
        'rows_removed': original_len - len(df_processed)
    }
    
    return df_processed, metadata


def encode_binary_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode binary features (if not already encoded).
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (encoded DataFrame, metadata dict)
    """
    df_encoded = df.copy()
    encoding_info = {}
    
    for col in BINARY_COLUMNS:
        if col in df.columns:
            unique_vals = df[col].unique()
            
            # Check if already binary (0/1)
            if set(unique_vals).issubset({0, 1, 0., 1.}):
                encoding_info[col] = 'already_binary'
            else:
                # Need encoding (e.g., M/F, Y/N)
                if col == "SEX":
                    df_encoded[col] = df_encoded[col].apply(lambda x: 1 if x == 'M' else 0)
                    encoding_info[col] = 'M→1, F→0'
                else:
                    df_encoded[col] = df_encoded[col].apply(lambda x: 1 if x == 'Y' else 0)
                    encoding_info[col] = 'Y→1, N→0'
    
    print(f"  Binary encoding: {encoding_info}")
    
    metadata = {
        'encoding_info': encoding_info,
        'binary_columns': BINARY_COLUMNS
    }
    
    return df_encoded, metadata


def encode_categorical_features(df: pd.DataFrame,
                                categorical_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply one-hot encoding to categorical features.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical columns (if None, use default)
        
    Returns:
        Tuple of (encoded DataFrame, metadata dict)
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLUMNS.copy()
    
    df_encoded = df.copy()
    encoding_mapping = {}
    created_columns = []
    
    for col in categorical_cols:
        if col in df.columns:
            # Get unique values before encoding
            unique_vals = df[col].unique()
            
            # One-hot encode
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
            
            # Track the mapping
            dummy_cols = list(dummies.columns)
            encoding_mapping[col] = {
                'unique_values': list(unique_vals),
                'encoded_columns': dummy_cols
            }
            created_columns.extend(dummy_cols)
            
            print(f"  {col}: {len(unique_vals)} values → {len(dummy_cols)} columns")
    
    metadata = {
        'encoding_mapping': encoding_mapping,
        'created_columns': created_columns,
        'original_categorical_columns': categorical_cols,
        'shape_before': df.shape,
        'shape_after': df_encoded.shape
    }
    
    return df_encoded, metadata


def normalize_continuous_features(df_train: pd.DataFrame,
                                  df_test: pd.DataFrame = None,
                                  continuous_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Dict[str, Any]]:
    """
    Normalize continuous features using StandardScaler.
    
    IMPORTANT: Fits on train data only, then applies to both train and test.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame (optional)
        continuous_cols: List of continuous columns (if None, use default)
        
    Returns:
        Tuple of (normalized train DataFrame, normalized test DataFrame, fitted scaler, metadata)
    """
    if continuous_cols is None:
        continuous_cols = CONTINUOUS_COLUMNS.copy()
    
    # Only normalize columns that exist in the DataFrame
    existing_cols = [col for col in continuous_cols if col in df_train.columns]
    
    if not existing_cols:
        print(f"  Warning: No continuous columns found to normalize")
        return df_train, df_test, None, {'normalized_columns': []}
    
    # Initialize and fit scaler on TRAIN data only
    scaler = StandardScaler()
    df_train_normalized = df_train.copy()
    df_train_normalized[existing_cols] = scaler.fit_transform(df_train[existing_cols])
    
    print(f"  Normalized {len(existing_cols)} continuous columns")
    print(f"  Columns: {existing_cols}")
    
    # Transform test data if provided
    df_test_normalized = None
    if df_test is not None:
        df_test_normalized = df_test.copy()
        df_test_normalized[existing_cols] = scaler.transform(df_test[existing_cols])
        print(f"  Applied same normalization to test data")
    
    metadata = {
        'normalized_columns': existing_cols,
        'scaler_mean': scaler.mean_.tolist() if scaler.mean_ is not None else None,
        'scaler_std': scaler.scale_.tolist() if scaler.scale_ is not None else None
    }
    
    return df_train_normalized, df_test_normalized, scaler, metadata


def preprocess_tabular_data(X_raw: pd.DataFrame,
                            train_idx: List[int],
                            test_idx: List[int],
                            config: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Main preprocessing pipeline for tabular data.
    
    This orchestrates all preprocessing steps following Ben's approach:
    1. Filter columns (remove BMI, NfLValue, etc.)
    2. Handle special values ('X' in function scores)
    3. Encode binary features (if needed)
    4. One-hot encode categorical features
    5. Normalize continuous features
    
    CRITICAL: Ensures no data leakage by:
    - Splitting train/test BEFORE normalization
    - Fitting scaler on train data only
    - Applying trained scaler to test data
    
    Args:
        X_raw: Raw tabular DataFrame (all samples)
        train_idx: List of training indices
        test_idx: List of test indices
        config: Configuration dictionary (optional)
        
    Returns:
        Tuple of (X_train_processed, X_test_processed, preprocessing_objects)
    """
    print("\n" + "="*80)
    print("Tabular Data Preprocessing Pipeline")
    print("="*80)
    print(f"Input shape: {X_raw.shape}")
    print(f"Train samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")
    
    # Use default config if not provided
    if config is None:
        config = {}
    
    preprocessing_objects = {
        'config': config,
        'original_columns': list(X_raw.columns),
        'steps': []
    }
    
    # Step 1: Filter columns
    print("\n1. Filtering columns...")
    X_filtered, filter_meta = filter_columns(
        X_raw,
        exclude_target_related=config.get('exclude_target_related', True)
    )
    preprocessing_objects['filter_metadata'] = filter_meta
    preprocessing_objects['steps'].append('filter_columns')
    
    # Step 2: Handle special values
    print("\n2. Handling special values...")
    X_cleaned, special_meta = handle_special_values(
        X_filtered,
        strategy=config.get('special_value_strategy', 'remove')
    )
    preprocessing_objects['special_value_metadata'] = special_meta
    preprocessing_objects['steps'].append('handle_special_values')
    
    # IMPORTANT: If rows were removed, update indices
    if special_meta['rows_removed'] > 0:
        print(f"  WARNING: {special_meta['rows_removed']} rows were removed!")
        print(f"  This will require updating train/test indices!")
        # For now, we'll raise an error to handle this case explicitly
        raise ValueError(
            f"Special value handling removed {special_meta['rows_removed']} rows. "
            "This requires updating train/test indices. Please handle this case."
        )
    
    # Step 3: Encode binary features
    print("\n3. Encoding binary features...")
    X_binary, binary_meta = encode_binary_features(X_cleaned)
    preprocessing_objects['binary_encoding_metadata'] = binary_meta
    preprocessing_objects['steps'].append('encode_binary_features')
    
    # Step 4: One-hot encode categorical features
    print("\n4. One-hot encoding categorical features...")
    X_categorical, categorical_meta = encode_categorical_features(X_binary)
    preprocessing_objects['categorical_encoding_metadata'] = categorical_meta
    preprocessing_objects['steps'].append('encode_categorical_features')
    
    # Step 5: Split into train/test BEFORE normalization (critical for no data leakage)
    print("\n5. Splitting train/test...")
    X_train = X_categorical.iloc[train_idx].copy()
    X_test = X_categorical.iloc[test_idx].copy()
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    
    # Step 6: Normalize continuous features
    print("\n6. Normalizing continuous features...")
    X_train_norm, X_test_norm, scaler, norm_meta = normalize_continuous_features(
        X_train, X_test
    )
    preprocessing_objects['scaler'] = scaler
    preprocessing_objects['normalization_metadata'] = norm_meta
    preprocessing_objects['steps'].append('normalize_continuous_features')
    
    # Final column names
    preprocessing_objects['final_column_names'] = list(X_train_norm.columns)
    preprocessing_objects['final_shape'] = {
        'train': X_train_norm.shape,
        'test': X_test_norm.shape
    }
    
    # Convert to numpy arrays
    X_train_final = X_train_norm.values.astype(np.float64)
    X_test_final = X_test_norm.values.astype(np.float64)
    
    print("\n" + "="*80)
    print("Preprocessing Complete!")
    print("="*80)
    print(f"Final train shape: {X_train_final.shape}")
    print(f"Final test shape: {X_test_final.shape}")
    print(f"Total features: {X_train_final.shape[1]}")
    print("="*80)
    
    return X_train_final, X_test_final, preprocessing_objects
