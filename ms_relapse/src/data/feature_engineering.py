"""
Feature engineering module.

This module handles feature selection, transformation, and combination of
tabular and LLM features.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from typing import Dict, List, Tuple, Any, Optional


def select_tabular_features(X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_test: np.ndarray = None,
                            n_features: int = 15,
                            score_func=f_classif,
                            feature_names: List[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray], SelectKBest, Dict[str, Any]]:
    """
    Select top k tabular features using SelectKBest.
    
    Uses univariate feature selection (ANOVA F-value) to identify the most
    discriminative features between classes.
    
    IMPORTANT: 
    - n_features is configurable, not fixed at 15
    - Fits on train data only to prevent data leakage
    
    Args:
        X_train: Training feature matrix (n_samples, n_features)
        y_train: Training target labels (n_samples,)
        X_test: Test feature matrix (optional)
        n_features: Number of features to select (default 15, but configurable)
        score_func: Scoring function (default: f_classif)
        feature_names: List of feature names (optional)
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, selector, metadata)
    """
    print(f"\n{'='*80}")
    print(f"Feature Selection: Selecting top {n_features} features")
    print(f"{'='*80}")
    print(f"Input shape: {X_train.shape}")
    
    # Initialize selector and fit on TRAIN data only
    selector = SelectKBest(score_func=score_func, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # Get feature scores and selected indicesselect_tabular_features
    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    
    print(f"Selected {n_features} features from {X_train.shape[1]}")
    print(f"Selected indices: {selected_indices.tolist()}")
    
    # Transform test data if provided
    X_test_selected = None
    if X_test is not None:
        X_test_selected = selector.transform(X_test)
        print(f"Applied to test data: {X_test.shape} → {X_test_selected.shape}")
    
    # Build metadata
    metadata = {
        'n_features': n_features,
        'score_func': score_func.__name__,
        'selected_indices': selected_indices.tolist(),
        'feature_scores': scores.tolist(),
        'n_features_in': X_train.shape[1],
        'n_features_out': X_train_selected.shape[1]
    }
    
    # If feature names provided, include selected feature names
    if feature_names is not None:
        selected_names = [feature_names[i] for i in selected_indices]
        metadata['selected_feature_names'] = selected_names
        metadata['feature_importance_ranking'] = {
            name: {'index': int(idx), 'score': float(scores[idx])}
            for name, idx in zip(selected_names, selected_indices)
        }
        print(f"\nSelected features:")
        # Sort by score descending
        sorted_features = sorted(zip(selected_names, selected_indices, scores[selected_indices]), 
                                key=lambda x: x[2], reverse=True)
        for i, (name, idx, score) in enumerate(sorted_features, 1):
            print(f"  {i:2d}. {name:<50s} (idx={idx:2d}, score={score:8.2f})")
    
    print(f"{'='*80}\n")
    
    return X_train_selected, X_test_selected, selector, metadata


def evaluate_feature_selection(X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               k_values: List[int],
                               feature_names: List[str] = None) -> Dict[int, Dict[str, Any]]:
    """
    Evaluate different values of k for feature selection.
    
    For each k, select features and train a simple RF classifier to evaluate
    performance. This helps choose the optimal number of features.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        k_values: List of k values to try (e.g., [10, 15, 20, 25])
        feature_names: List of feature names (optional)
        
    Returns:
        Dictionary mapping k → evaluation metrics
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    print(f"\n{'='*80}")
    print("Evaluating Feature Selection with Different k Values")
    print(f"{'='*80}")
    
    results = {}
    
    for k in k_values:
        print(f"\n--- Testing k={k} ---")
        
        # Select features
        X_train_k, X_test_k, selector, meta = select_tabular_features(
            X_train, y_train, X_test, n_features=k, feature_names=feature_names
        )
        
        # Train a simple RF
        print(f"Training RF classifier...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Calibrate RF
        rf_calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=3)
        rf_calibrated.fit(X_train_k, y_train)
        
        # Evaluate
        y_train_pred = rf_calibrated.predict(X_train_k)
        y_test_pred = rf_calibrated.predict(X_test_k)
        
        y_train_proba = rf_calibrated.predict_proba(X_train_k)[:, 1]
        y_test_proba = rf_calibrated.predict_proba(X_test_k)[:, 1]
        
        train_metrics = {
            'auc': roc_auc_score(y_train, y_train_proba),
            'recall': recall_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred)
        }
        
        test_metrics = {
            'auc': roc_auc_score(y_test, y_test_proba),
            'recall': recall_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred)
        }
        
        gap = train_metrics['auc'] - test_metrics['auc']
        
        print(f"Train AUC:      {train_metrics['auc']:.4f}")
        print(f"Test AUC:       {test_metrics['auc']:.4f}")
        print(f"Gap:            {gap:.4f}")
        print(f"Test Recall:    {test_metrics['recall']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test F1:        {test_metrics['f1']:.4f}")
        
        results[k] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'gap': gap,
            'selected_indices': meta['selected_indices'],
            'selected_feature_names': meta.get('selected_feature_names', [])
        }
    
    print(f"\n{'='*80}")
    print("Feature Selection Evaluation Complete")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"{'k':<5} {'Train AUC':<12} {'Test AUC':<12} {'Gap':<10} {'Test Recall':<12} {'Test F1':<12}")
    print("-" * 75)
    for k in k_values:
        r = results[k]
        print(f"{k:<5} {r['train_metrics']['auc']:<12.4f} {r['test_metrics']['auc']:<12.4f} "
              f"{r['gap']:<10.4f} {r['test_metrics']['recall']:<12.4f} {r['test_metrics']['f1']:<12.4f}")
    
    # Find best k based on different criteria
    best_k_gap = min(k_values, key=lambda k: results[k]['gap'])
    best_k_recall = max(k_values, key=lambda k: results[k]['test_metrics']['recall'])
    best_k_f1 = max(k_values, key=lambda k: results[k]['test_metrics']['f1'])
    
    print(f"\nRecommendations:")
    print(f"  Best k for minimum gap:     {best_k_gap}")
    print(f"  Best k for maximum recall:  {best_k_recall}")
    print(f"  Best k for maximum F1:      {best_k_f1}")
    print(f"{'='*80}\n")
    
    return results


def apply_pca_to_embeddings(phi_llm: np.ndarray,
                            n_components: int = 10,
                            pca: PCA = None,
                            fit: bool = True) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA to reduce LLM embeddings dimensionality.
    
    Reduces high-dimensional LLM embeddings (e.g., 768-dim) to lower dimensions
    (e.g., 10-dim) to prevent overfitting in RF.
    
    Args:
        phi_llm: LLM embeddings (n_samples, embedding_dim), e.g., (2192, 768)
        n_components: Number of PCA components (default 10)
        pca: Optional pre-fitted PCA transformer (if provided, just transform)
        fit: Whether to fit PCA (True) or just transform (False)
        
    Returns:
        Tuple of (reduced_embeddings, pca_transformer)
    """
    if pca is None:
        pca = PCA(n_components=n_components, random_state=42)
    
    if fit:
        phi_llm_pca = pca.fit_transform(phi_llm)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA: {phi_llm.shape[1]} → {n_components} dims (explained variance: {explained_var:.2%})")
    else:
        phi_llm_pca = pca.transform(phi_llm)
        print(f"PCA transform: {phi_llm.shape[1]} → {n_components} dims")
    
    return phi_llm_pca, pca


def construct_rf_input_features(X_tabular: np.ndarray,
                                phi_llm_pca: np.ndarray = None,
                                p_llm: np.ndarray = None) -> np.ndarray:
    """
    Construct input features for RF model.
    
    Combines: [x_tab (k dims), phi_llm_pca (10 dims), p_llm (1 dim)]
    Total: (k + 10 + 1) dimensions
    
    NOTE: k is not fixed at 15, it depends on feature selection.
    
    Args:
        X_tabular: Selected tabular features (n_samples, k)
        phi_llm_pca: PCA-reduced LLM embeddings (n_samples, 10), optional
        p_llm: LLM prediction probabilities (n_samples, 1 or n_samples,), optional
        
    Returns:
        Combined feature matrix for RF (n_samples, k+10+1 or k+10 or k)
    """
    features_list = [X_tabular]
    feature_dims = [f"x_tab({X_tabular.shape[1]})"]
    
    if phi_llm_pca is not None:
        if phi_llm_pca.ndim == 1:
            phi_llm_pca = phi_llm_pca.reshape(-1, 1)
        features_list.append(phi_llm_pca)
        feature_dims.append(f"phi_llm({phi_llm_pca.shape[1]})")
    
    if p_llm is not None:
        if p_llm.ndim == 1:
            p_llm = p_llm.reshape(-1, 1)
        features_list.append(p_llm)
        feature_dims.append(f"p_llm({p_llm.shape[1]})")
    
    combined = np.hstack(features_list)
    
    feature_str = " + ".join(feature_dims)
    print(f"Combined features: {feature_str} = {combined.shape}")
    
    return combined
