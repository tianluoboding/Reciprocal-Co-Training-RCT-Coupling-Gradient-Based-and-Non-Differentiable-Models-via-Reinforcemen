#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WDBC Data Preprocessing Script

This script preprocesses the Wisconsin Diagnostic Breast Cancer dataset:
1. Load raw data from wdbc.data
2. Generate structured text descriptions (similar to MS dataset format)
3. Split into train/test sets
4. Perform feature selection with different k values
5. Save processed data

Usage:
    python 01_preprocess_wdbc.py
    python 01_preprocess_wdbc.py --k 15  # Specific k value
    python 01_preprocess_wdbc.py --k_values 5 10 15 20 30  # Multiple k values

Output:
    - data/raw/descriptive_text.json  (generated text)
    - data/raw/X_tabular.csv
    - data/raw/y_all.npy
    - data/raw/train_idxs.json
    - data/raw/test_idxs.json
    - data/processed/X_train_selected_k{K}.npy
    - data/processed/X_test_selected_k{K}.npy
    - data/processed/feature_selection_k{K}.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Base directory
BASE_DIR = Path(__file__).parent.parent  # breast_cancer_exp/
SOURCE_DATA = BASE_DIR.parent / "data/breast_dataset/breast+cancer+wisconsin+diagnostic"


# Feature names for WDBC dataset
FEATURE_NAMES_BASE = [
    'radius', 'texture', 'perimeter', 'area', 'smoothness',
    'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
]

FEATURE_NAMES = []
for stat in ['mean', 'se', 'worst']:
    for feat in FEATURE_NAMES_BASE:
        FEATURE_NAMES.append(f"{feat}_{stat}")


def load_raw_data():
    """Load raw WDBC data"""
    print("\n" + "=" * 60)
    print("Loading Raw WDBC Data")
    print("=" * 60)
    
    data_file = SOURCE_DATA / "wdbc.data"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load data (no header)
    # Format: ID, Diagnosis (M/B), 30 features
    df = pd.read_csv(data_file, header=None)
    
    print(f"  ✓ Loaded {len(df)} samples")
    print(f"  ✓ Columns: {df.shape[1]}")
    
    # Extract components
    ids = df.iloc[:, 0].values
    diagnosis = df.iloc[:, 1].values  # M = malignant, B = benign
    features = df.iloc[:, 2:].values
    
    # Convert diagnosis to binary (1 = Malignant, 0 = Benign)
    y = (diagnosis == 'M').astype(int)
    
    print(f"  ✓ Malignant: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  ✓ Benign: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
    
    return ids, features, y, diagnosis


def generate_text_descriptions(ids, features, diagnosis):
    """
    Generate structured text descriptions similar to MS dataset format.
    """
    print("\n" + "=" * 60)
    print("Generating Text Descriptions")
    print("=" * 60)
    
    texts = []
    
    for i in range(len(ids)):
        feat = features[i]
        
        # Build structured text
        text = f"""Breast Tumor Fine Needle Aspirate Analysis

Patient ID: {ids[i]}

Cell Nuclei Measurements (Mean Values):
- Radius: {feat[0]:.4f}
- Texture: {feat[1]:.4f}
- Perimeter: {feat[2]:.4f}
- Area: {feat[3]:.4f}
- Smoothness: {feat[4]:.4f}
- Compactness: {feat[5]:.4f}
- Concavity: {feat[6]:.4f}
- Concave Points: {feat[7]:.4f}
- Symmetry: {feat[8]:.4f}
- Fractal Dimension: {feat[9]:.4f}

Cell Nuclei Measurements (Standard Error):
- Radius SE: {feat[10]:.4f}
- Texture SE: {feat[11]:.4f}
- Perimeter SE: {feat[12]:.4f}
- Area SE: {feat[13]:.4f}
- Smoothness SE: {feat[14]:.4f}
- Compactness SE: {feat[15]:.4f}
- Concavity SE: {feat[16]:.4f}
- Concave Points SE: {feat[17]:.4f}
- Symmetry SE: {feat[18]:.4f}
- Fractal Dimension SE: {feat[19]:.4f}

Cell Nuclei Measurements (Worst/Largest Values):
- Worst Radius: {feat[20]:.4f}
- Worst Texture: {feat[21]:.4f}
- Worst Perimeter: {feat[22]:.4f}
- Worst Area: {feat[23]:.4f}
- Worst Smoothness: {feat[24]:.4f}
- Worst Compactness: {feat[25]:.4f}
- Worst Concavity: {feat[26]:.4f}
- Worst Concave Points: {feat[27]:.4f}
- Worst Symmetry: {feat[28]:.4f}
- Worst Fractal Dimension: {feat[29]:.4f}"""
        
        # Create prompt-response format
        response = "Malignant" if diagnosis[i] == 'M' else "Benign"
        texts.append({
            "prompt": text,
            "response": f"Diagnosis: {response}"
        })
    
    print(f"  ✓ Generated {len(texts)} text descriptions")
    
    return texts


def create_train_test_split(n_samples, y, test_size=0.2, random_state=42):
    """Create stratified train/test split"""
    print("\n" + "=" * 60)
    print("Creating Train/Test Split")
    print("=" * 60)
    
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, 
        random_state=random_state, stratify=y
    )
    
    print(f"  ✓ Train: {len(train_idx)} samples ({y[train_idx].mean()*100:.1f}% malignant)")
    print(f"  ✓ Test: {len(test_idx)} samples ({y[test_idx].mean()*100:.1f}% malignant)")
    
    return train_idx.tolist(), test_idx.tolist()


def perform_feature_selection(X_train, y_train, X_test, k_values):
    """
    Perform feature selection with different k values.
    Returns dict with results for each k.
    """
    print("\n" + "=" * 60)
    print("Performing Feature Selection")
    print("=" * 60)
    
    # First, standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for k in k_values:
        if k > X_train.shape[1]:
            print(f"k={k} exceeds number of features ({X_train.shape[1]}), using all features")
            k = X_train.shape[1]
        
        print(f"\n  Processing k={k}...")
        
        if k == X_train.shape[1]:
            # Use all features
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
            selected_indices = list(range(k))
            selected_names = FEATURE_NAMES[:k]
            scores = f_classif(X_train_scaled, y_train)[0]
        else:
            # Select k best features
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            selected_indices = selector.get_support(indices=True).tolist()
            selected_names = [FEATURE_NAMES[i] for i in selected_indices]
            scores = selector.scores_
        
        print(f"    Selected features: {selected_names[:5]}..." if len(selected_names) > 5 else f"    Selected features: {selected_names}")
        
        results[k] = {
            'X_train': X_train_selected,
            'X_test': X_test_selected,
            'selected_indices': selected_indices,
            'selected_names': selected_names,
            'scores': scores.tolist(),
            'scaler': scaler
        }
    
    return results


def save_data(ids, features, y, texts, train_idx, test_idx, feature_results):
    """Save all processed data"""
    print("\n" + "=" * 60)
    print("Saving Processed Data")
    print("=" * 60)
    
    raw_dir = BASE_DIR / "data/raw"
    processed_dir = BASE_DIR / "data/processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw data
    # Text descriptions
    with open(raw_dir / "descriptive_text.json", 'w') as f:
        json.dump(texts, f, indent=2)
    print(f"  ✓ Saved: descriptive_text.json")
    
    # Tabular features
    df_features = pd.DataFrame(features, columns=FEATURE_NAMES)
    df_features.to_csv(raw_dir / "X_tabular.csv", index=False)
    print(f"  ✓ Saved: X_tabular.csv")
    
    # Labels
    np.save(raw_dir / "y_all.npy", y)
    print(f"  ✓ Saved: y_all.npy")
    
    # Train/test indices
    with open(raw_dir / "train_idxs.json", 'w') as f:
        json.dump(train_idx, f)
    with open(raw_dir / "test_idxs.json", 'w') as f:
        json.dump(test_idx, f)
    print(f"  ✓ Saved: train_idxs.json, test_idxs.json")
    
    # Save processed data for each k
    for k, result in feature_results.items():
        # Save selected features
        np.save(processed_dir / f"X_train_selected_k{k}.npy", result['X_train'])
        np.save(processed_dir / f"X_test_selected_k{k}.npy", result['X_test'])
        
        # Save metadata
        metadata = {
            'k': k,
            'n_features_in': features.shape[1],
            'n_features_out': k,
            'selected_indices': result['selected_indices'],
            'selected_names': result['selected_names'],
            'feature_scores': result['scores'],
            'score_func': 'f_classif'
        }
        with open(processed_dir / f"feature_selection_k{k}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved: k={k} features ({result['X_train'].shape[1]} dims)")
    
    # Save dataset summary
    summary = {
        'dataset': 'Wisconsin Diagnostic Breast Cancer (WDBC)',
        'n_samples': len(y),
        'n_features_raw': features.shape[1],
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'positive_rate': float(y.mean()),
        'k_values_tested': list(feature_results.keys()),
        'feature_names': FEATURE_NAMES,
        'timestamp': str(datetime.now())
    }
    with open(processed_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: dataset_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Preprocess WDBC dataset")
    parser.add_argument('--k', type=int, default=None, help='Single k value for feature selection')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 15, 20, 30], 
                        help='Multiple k values to test (default: 5 10 15 20 30)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("WDBC Data Preprocessing")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print(f"Output directory: {BASE_DIR}")
    
    # Determine k values
    if args.k is not None:
        k_values = [args.k]
    else:
        k_values = args.k_values
    
    print(f"K values to test: {k_values}")
    
    # Load raw data
    ids, features, y, diagnosis = load_raw_data()
    
    # Generate text descriptions
    texts = generate_text_descriptions(ids, features, diagnosis)
    
    # Create train/test split
    train_idx, test_idx = create_train_test_split(
        len(y), y, test_size=args.test_size, random_state=args.seed
    )
    
    # Perform feature selection
    X_train = features[train_idx]
    X_test = features[test_idx]
    y_train = y[train_idx]
    
    feature_results = perform_feature_selection(X_train, y_train, X_test, k_values)
    
    # Save all data
    save_data(ids, features, y, texts, train_idx, test_idx, feature_results)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Run training with: python 02_train_baseline_rf.py --k 30")
    print(f"  2. Or run iterative training: python 10_iterative_training.py --k 30")


if __name__ == "__main__":
    main()
