"""
Reward functions for PPO training.

This module implements different reward computation strategies that use
the RF model output and true labels to generate training signals for the LLM.

Key insight: y_true is used to correct RF bias, not as direct reward.
This avoids data leakage while improving the quality of the reward signal.
"""

import numpy as np
import torch


def compute_reward_binary(p_rf, y_true, threshold=0.5):
    """
    Binary reward: 1 if RF prediction direction matches true label, 0 otherwise.
    
    Logic:
        if (p_rf > threshold and y_true == 1) or (p_rf <= threshold and y_true == 0):
            reward = 1  # Correct direction
        else:
            reward = 0  # Incorrect direction
    
    Args:
        p_rf: RF predicted probability for y=1 (shape: [batch_size])
        y_true: True labels (shape: [batch_size])
        threshold: Decision threshold (default: 0.5)
        
    Returns:
        Rewards of shape [batch_size], values in {0, 1}
    """
    # Convert to numpy if needed
    if torch.is_tensor(p_rf):
        p_rf = p_rf.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    
    # Ensure numpy arrays
    p_rf = np.asarray(p_rf)
    y_true = np.asarray(y_true)
    
    # Compute predictions from RF probabilities
    predictions = (p_rf > threshold).astype(int)
    
    # Reward = 1 if prediction matches true label, 0 otherwise
    rewards = (predictions == y_true).astype(float)
    
    return rewards


def compute_reward_continuous(p_rf, y_true):
    """
    Continuous reward: RF probability distance to the correct answer.
    
    Logic:
        if y_true == 1:
            reward = p_rf  # Higher p_rf is better
        else:
            reward = 1 - p_rf  # Lower p_rf is better
    
    This preserves confidence information and provides smooth gradients.
    
    Args:
        p_rf: RF predicted probability for y=1 (shape: [batch_size])
        y_true: True labels (shape: [batch_size])
        
    Returns:
        Rewards of shape [batch_size], values in [0, 1]
    """
    # Convert to numpy if needed
    if torch.is_tensor(p_rf):
        p_rf = p_rf.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    
    # Ensure numpy arrays
    p_rf = np.asarray(p_rf)
    y_true = np.asarray(y_true)
    
    # Compute continuous reward based on true label
    # For y=1: reward = p_rf (越接近1越好)
    # For y=0: reward = 1 - p_rf (越接近0, 即1-p_rf越接近1越好)
    rewards = np.where(y_true == 1, p_rf, 1 - p_rf)
    
    return rewards


def compute_reward_with_bias_correction(p_rf, y_true, correction_strength=0.5):
    """
    Bias-corrected reward: Adjust RF output based on systematic bias.
    
    This is a more sophisticated approach that detects and corrects
    RF's tendency to over- or under-predict certain classes.
    
    Args:
        p_rf: RF predicted probability for y=1 (shape: [batch_size])
        y_true: True labels (shape: [batch_size])
        correction_strength: How strongly to correct bias [0, 1]
        
    Returns:
        Corrected rewards of shape [batch_size], values in [0, 1]
    """
    pass


def compute_rl_rewards(indices, actions, llm_model, rf_model, pca, feature_selector,
                       X_tabular, y_labels, texts, tokenizer, reward_type='continuous',
                       device='cpu'):
    """
    Compute RL rewards for a batch of actions.
    
    This is the main function called during PPO training.
    
    Steps:
    1. Extract phi_llm and p_llm from LLM
    2. Apply PCA to phi_llm
    3. Construct RF input [x_tab, phi_llm_pca, p_llm]
    4. Get RF prediction p_rf
    5. Compare with y_true to compute reward
    
    Args:
        indices: Batch indices (np.ndarray or list)
        actions: Sampled actions from policy (not used for reward, only for logging)
        llm_model: Current LLM model (BioClinicalBERTWithLoRA)
        rf_model: Current RF model (CalibratedClassifierCV)
        pca: Current PCA transformer
        feature_selector: Feature selector for tabular features
        X_tabular: Full tabular data (pd.DataFrame, preprocessed)
        y_labels: Full labels (np.ndarray)
        texts: Full text data (list of strings)
        tokenizer: Tokenizer for LLM
        reward_type: Type of reward ('binary' or 'continuous')
        device: Device to run computations ('cpu' or 'cuda')
        
    Returns:
        Rewards tensor of shape [batch_size]
    """
    # Ensure indices are numpy array
    if torch.is_tensor(indices):
        indices = indices.cpu().numpy()
    indices = np.asarray(indices)
    
    batch_size = len(indices)
    
    # Step 1: Extract batch data
    batch_texts = [texts[idx] for idx in indices]
    batch_y_true = y_labels[indices]
    
    # Get selected tabular features for this batch
    # X_tabular should already be preprocessed and selected (15 features)
    if isinstance(X_tabular, np.ndarray):
        X_tab_batch = X_tabular[indices]
    else:
        X_tab_batch = X_tabular.iloc[indices].values
    
    # Step 2: Extract LLM features (phi_llm and p_llm)
    llm_model.eval()
    with torch.no_grad():
        # Extract features using LLM model's method
        phi_llm_batch, p_llm_batch = llm_model.extract_features(
            batch_texts,
            batch_size=batch_size,
            show_progress=False
        )
    
    # phi_llm_batch: (batch_size, 768)
    # p_llm_batch: (batch_size,)
    
    # Step 3: Apply PCA to phi_llm (768D → 10D)
    phi_llm_pca_batch = pca.transform(phi_llm_batch)
    
    # Step 4: Construct RF input features [x_tab(15), phi_llm_pca(10), p_llm(1)] = 26D
    # Combine: tabular + PCA embeddings + p_llm
    p_llm_reshaped = p_llm_batch.reshape(-1, 1)  # (batch_size, 1)
    rf_input = np.concatenate([X_tab_batch, phi_llm_pca_batch, p_llm_reshaped], axis=1)
    
    # Verify shape
    assert rf_input.shape == (batch_size, 26), f"RF input shape mismatch: {rf_input.shape}"
    
    # Step 5: Get RF prediction (p_rf)
    p_rf_batch = rf_model.predict_proba(rf_input)[:, 1]  # Probability for class 1
    
    # Step 6: Compute reward based on reward_type
    if reward_type == 'binary':
        rewards = compute_reward_binary(p_rf_batch, batch_y_true)
    elif reward_type == 'continuous':
        rewards = compute_reward_continuous(p_rf_batch, batch_y_true)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}. Must be 'binary' or 'continuous'.")
    
    # Convert to torch tensor
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    # Check for NaN or Inf
    if torch.isnan(rewards_tensor).any() or torch.isinf(rewards_tensor).any():
        print(f"Warning: NaN or Inf detected in rewards!")
        print(f"  p_rf range: [{p_rf_batch.min():.4f}, {p_rf_batch.max():.4f}]")
        print(f"  rewards range: [{rewards.min():.4f}, {rewards.max():.4f}]")
        # Replace NaN/Inf with 0
        rewards_tensor = torch.nan_to_num(rewards_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    
    return rewards_tensor

