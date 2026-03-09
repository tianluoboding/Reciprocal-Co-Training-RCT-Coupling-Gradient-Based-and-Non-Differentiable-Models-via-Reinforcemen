#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diabetes Dataset Preprocessing Script

This script preprocesses the Diabetes (BRFSS 2015) dataset:
1. Load raw data from diabetes_binary_5050split CSV
2. Generate structured text descriptions (similar to MS dataset format)
3. Split into train/test sets (80/20)
4. Perform feature selection with different k values
5. Save processed data

Dataset: CDC BRFSS 2015 - Diabetes Health Indicators
Samples: 70,692 (balanced 50-50)
Features: 21

Usage:
    python 01_preprocess_diabetes.py
    python 01_preprocess_diabetes.py --k_values 5 10 21

Output:
    - data/raw/descriptive_text.json
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
BASE_DIR = Path(__file__).parent.parent  # Diabetes_exp/
SOURCE_DATA = BASE_DIR / "archive (2)"

# Feature names and their descriptions
FEATURE_INFO = {
    'HighBP': {'desc': 'High Blood Pressure', 'type': 'binary'},
    'HighChol': {'desc': 'High Cholesterol', 'type': 'binary'},
    'CholCheck': {'desc': 'Cholesterol Check in past 5 years', 'type': 'binary'},
    'BMI': {'desc': 'Body Mass Index', 'type': 'numeric'},
    'Smoker': {'desc': 'Smoked at least 100 cigarettes', 'type': 'binary'},
    'Stroke': {'desc': 'History of Stroke', 'type': 'binary'},
    'HeartDiseaseorAttack': {'desc': 'Coronary Heart Disease or Myocardial Infarction', 'type': 'binary'},
    'PhysActivity': {'desc': 'Physical Activity in past 30 days', 'type': 'binary'},
    'Fruits': {'desc': 'Consumes Fruit 1+ times per day', 'type': 'binary'},
    'Veggies': {'desc': 'Consumes Vegetables 1+ times per day', 'type': 'binary'},
    'HvyAlcoholConsump': {'desc': 'Heavy Alcohol Consumption', 'type': 'binary'},
    'AnyHealthcare': {'desc': 'Has Healthcare Coverage', 'type': 'binary'},
    'NoDocbcCost': {'desc': 'Could not see doctor due to cost', 'type': 'binary'},
    'GenHlth': {'desc': 'General Health', 'type': 'ordinal', 'scale': {1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'}},
    'MentHlth': {'desc': 'Days of poor mental health (past 30 days)', 'type': 'numeric'},
    'PhysHlth': {'desc': 'Days of poor physical health (past 30 days)', 'type': 'numeric'},
    'DiffWalk': {'desc': 'Difficulty Walking or Climbing Stairs', 'type': 'binary'},
    'Sex': {'desc': 'Sex', 'type': 'binary', 'scale': {0: 'Female', 1: 'Male'}},
    'Age': {'desc': 'Age Category', 'type': 'ordinal', 'scale': {
        1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
        6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
        11: '70-74', 12: '75-79', 13: '80+'
    }},
    'Education': {'desc': 'Education Level', 'type': 'ordinal', 'scale': {
        1: 'Never attended school', 2: 'Elementary', 3: 'Some high school',
        4: 'High school graduate', 5: 'Some college', 6: 'College graduate'
    }},
    'Income': {'desc': 'Income Level', 'type': 'ordinal', 'scale': {
        1: '<$10k', 2: '$10k-$15k', 3: '$15k-$20k', 4: '$20k-$25k',
        5: '$25k-$35k', 6: '$35k-$50k', 7: '$50k-$75k', 8: '$75k+'
    }}
}

FEATURE_NAMES = list(FEATURE_INFO.keys())


def load_raw_data():
    """Load raw Diabetes data"""
    print("\n" + "=" * 60)
    print("Loading Raw Diabetes Data")
    print("=" * 60)
    
    data_file = SOURCE_DATA / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    
    print(f"  ✓ Loaded {len(df)} samples")
    print(f"  ✓ Columns: {df.shape[1]}")
    
    # Extract target and features
    y = df['Diabetes_binary'].values.astype(int)
    X = df.drop('Diabetes_binary', axis=1).values
    feature_names = [col for col in df.columns if col != 'Diabetes_binary']
    
    print(f"  ✓ Diabetes (positive): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  ✓ No Diabetes: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
    print(f"  ✓ Features: {len(feature_names)}")
    
    return X, y, df, feature_names


def get_value_label(feature, value):
    """Get human-readable label for a feature value"""
    info = FEATURE_INFO.get(feature, {})
    
    if info.get('type') == 'binary':
        if feature == 'Sex':
            return 'Male' if value == 1 else 'Female'
        return 'Yes' if value == 1 else 'No'
    
    if 'scale' in info:
        return info['scale'].get(int(value), str(int(value)))
    
    return str(value)


def generate_text_descriptions(df, y):
    """
    Generate structured text descriptions similar to MS dataset format.
    """
    print("\n" + "=" * 60)
    print("Generating Text Descriptions")
    print("=" * 60)
    
    texts = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        # Get values
        age_cat = int(row['Age'])
        sex = 'Male' if row['Sex'] == 1 else 'Female'
        age_label = FEATURE_INFO['Age']['scale'].get(age_cat, f'{age_cat}')
        
        # Build structured text (similar to MS dataset format)
        text = f"""Age: {age_label} years
Sex: {sex}
Education: {get_value_label('Education', row['Education'])}
Income: {get_value_label('Income', row['Income'])}

Health Conditions:
- High Blood Pressure: {get_value_label('HighBP', row['HighBP'])}
- High Cholesterol: {get_value_label('HighChol', row['HighChol'])}
- Cholesterol Check (past 5 years): {get_value_label('CholCheck', row['CholCheck'])}
- History of Stroke: {get_value_label('Stroke', row['Stroke'])}
- Heart Disease or Heart Attack: {get_value_label('HeartDiseaseorAttack', row['HeartDiseaseorAttack'])}
- Difficulty Walking: {get_value_label('DiffWalk', row['DiffWalk'])}

Physical Measurements:
- BMI: {row['BMI']:.1f}

Lifestyle Factors:
- Smoker (100+ cigarettes lifetime): {get_value_label('Smoker', row['Smoker'])}
- Physical Activity (past 30 days): {get_value_label('PhysActivity', row['PhysActivity'])}
- Fruit Consumption (1+ per day): {get_value_label('Fruits', row['Fruits'])}
- Vegetable Consumption (1+ per day): {get_value_label('Veggies', row['Veggies'])}
- Heavy Alcohol Consumption: {get_value_label('HvyAlcoholConsump', row['HvyAlcoholConsump'])}

Healthcare Access:
- Has Healthcare Coverage: {get_value_label('AnyHealthcare', row['AnyHealthcare'])}
- Could Not See Doctor Due to Cost: {get_value_label('NoDocbcCost', row['NoDocbcCost'])}

Self-Reported Health Status:
- General Health: {get_value_label('GenHlth', row['GenHlth'])}
- Mental Health (poor days in past 30): {int(row['MentHlth'])}
- Physical Health (poor days in past 30): {int(row['PhysHlth'])}"""
        
        # Create prompt-response format
        response = "Diabetes or Prediabetes" if y[idx] == 1 else "No Diabetes"
        texts.append({
            "prompt": text,
            "response": f"Diabetes Status: {response}"
        })
        
        if (idx + 1) % 10000 == 0:
            print(f"  Processing: {idx + 1}/{len(df)}")
    
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
    
    print(f"  ✓ Train: {len(train_idx)} samples ({y[train_idx].mean()*100:.1f}% diabetes)")
    print(f"  ✓ Test: {len(test_idx)} samples ({y[test_idx].mean()*100:.1f}% diabetes)")
    
    return train_idx.tolist(), test_idx.tolist()


def perform_feature_selection(X_train, y_train, X_test, k_values, feature_names):
    """
    Perform feature selection with different k values.
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
            print(f"  k={k} exceeds number of features ({X_train.shape[1]}), using all features")
            k = X_train.shape[1]
        
        print(f"\n  Processing k={k}...")
        
        if k == X_train.shape[1]:
            # Use all features
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
            selected_indices = list(range(k))
            selected_names = feature_names[:k]
            scores = f_classif(X_train_scaled, y_train)[0]
        else:
            # Select k best features
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            selected_indices = selector.get_support(indices=True).tolist()
            selected_names = [feature_names[i] for i in selected_indices]
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


def save_data(X, y, texts, train_idx, test_idx, feature_results, feature_names):
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
        json.dump(texts, f)
    print(f"  ✓ Saved: descriptive_text.json")
    
    # Tabular features
    df_features = pd.DataFrame(X, columns=feature_names)
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
            'n_features_in': X.shape[1],
            'n_features_out': k if k <= X.shape[1] else X.shape[1],
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
        'dataset': 'Diabetes Health Indicators (BRFSS 2015)',
        'source_file': 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv',
        'n_samples': len(y),
        'n_features_raw': X.shape[1],
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'positive_rate': float(y.mean()),
        'k_values_tested': list(feature_results.keys()),
        'feature_names': feature_names,
        'timestamp': str(datetime.now())
    }
    with open(processed_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: dataset_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Diabetes dataset")
    parser.add_argument('--k', type=int, default=None, help='Single k value for feature selection')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 21], 
                        help='Multiple k values to test (default: 5 10 21)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Diabetes Dataset Preprocessing")
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
    X, y, df, feature_names = load_raw_data()
    
    # Generate text descriptions
    texts = generate_text_descriptions(df, y)
    
    # Create train/test split
    train_idx, test_idx = create_train_test_split(
        len(y), y, test_size=args.test_size, random_state=args.seed
    )
    
    # Perform feature selection
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    
    feature_results = perform_feature_selection(X_train, y_train, X_test, k_values, feature_names)
    
    # Save all data
    save_data(X, y, texts, train_idx, test_idx, feature_results, feature_names)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nDataset Summary:")
    print(f"  - Total samples: {len(y)}")
    print(f"  - Train samples: {len(train_idx)}")
    print(f"  - Test samples: {len(test_idx)}")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - K values prepared: {k_values}")
    print(f"\nNext steps:")
    print(f"  1. Run training with: python 10_iterative_training.py --k 21")


if __name__ == "__main__":
    main()
