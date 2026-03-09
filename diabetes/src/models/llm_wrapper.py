"""
LLM Wrapper for Bio-ClinicalBERT with LoRA

This module provides a wrapper class for Bio-ClinicalBERT that:
1. Loads the pretrained model from HuggingFace
2. Integrates LoRA adapters for efficient fine-tuning
3. Extracts embeddings (phi_llm) from the [CLS] token
4. Computes prediction probabilities (p_llm)
5. Supports batch processing for efficiency

Architecture:
    Input text → Tokenizer → BERT (+ LoRA) → [CLS] embedding (768-dim)
                                            ↓
                                      Classifier head
                                            ↓
                              [phi_llm (768), p_llm (prob)]

Usage:
    # Initialize model
    model = BioClinicalBERTWithLoRA(
        model_name_or_path="emilyalsentzer/Bio_ClinicalBERT",
        lora_config={'r': 8, 'lora_alpha': 16},
        device='cuda'
    )
    
    # Extract features
    phi_llm, p_llm = model.extract_features(texts, batch_size=32)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
from typing import Dict, Tuple, List, Optional
import numpy as np
from tqdm import tqdm


class BioClinicalBERTWithLoRA(nn.Module):
    """
    Bio-ClinicalBERT wrapper with optional LoRA adapters.
    
    This class wraps the Bio-ClinicalBERT model and adds:
    - Optional LoRA adapters for parameter-efficient fine-tuning
    - A classification head for binary prediction
    - Methods to extract embeddings and probabilities
    - Batch processing capabilities
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        lora_config: Dictionary with LoRA configuration (optional)
        num_labels: Number of output classes (default: 2)
        device: Device to run model on ('cuda' or 'cpu')
    """
    
    def __init__(self, 
                 model_name_or_path: str,
                 lora_config: Optional[Dict] = None,
                 num_labels: int = 2,
                 device: str = 'cuda'):
        super().__init__()
        
        # Auto-detect device if CUDA not available
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = device
        self.num_labels = num_labels
        
        print(f"Loading Bio-ClinicalBERT from '{model_name_or_path}'...")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            print(f"✓ Tokenizer loaded")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise
        
        # Load base BERT model
        try:
            config = AutoConfig.from_pretrained(model_name_or_path)
            config.num_labels = num_labels
            # Try loading model (auto-detect format: safetensors or pytorch_model.bin)
            # Don't force safetensors if local cache has pytorch_model.bin
            self.bert = AutoModel.from_pretrained(
                model_name_or_path, 
                config=config,
                local_files_only=True  # Use local cache only, no network access
            )
            self.hidden_size = config.hidden_size  # Should be 768 for BERT
            print(f"✓ Base model loaded (hidden_size={self.hidden_size})")
        except Exception as e:
            print(f"Error loading base model: {e}")
            raise
        
        # Add LoRA if configured
        self.lora_enabled = False
        if lora_config:
            try:
                self._add_lora_adapters(lora_config)
                self.lora_enabled = True
            except Exception as e:
                print(f"Error adding LoRA adapters: {e}")
                raise
        else:
            print("LoRA not enabled (lora_config is None)")
        
        # Classification head
        # Note: Using config's hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        print(f"✓ Classification head added ({self.hidden_size} → {num_labels})")
        
        # ⭐ Value head (critic) for PPO
        self.value_head = nn.Linear(self.hidden_size, 1)
        print(f"✓ Value head added ({self.hidden_size} → 1)")
        
        # Move model to device
        self.to(device)
        print(f"✓ Model moved to {device}")
        
        # Print summary
        self._print_model_summary()
    
    def _add_lora_adapters(self, lora_config: Dict):
        """
        Add LoRA adapters to the BERT model.
        
        Args:
            lora_config: Dictionary containing:
                - r: LoRA rank (default: 8)
                - lora_alpha: LoRA scaling factor (default: 16)
                - lora_dropout: Dropout for LoRA layers (default: 0.05)
                - target_modules: List of module names to apply LoRA to
        """
        # Default values
        r = lora_config.get('r', 8)
        lora_alpha = lora_config.get('lora_alpha', 16)
        lora_dropout = lora_config.get('lora_dropout', 0.05)
        target_modules = lora_config.get('target_modules', ["query", "value"])
        
        # Create PEFT config
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION"  # We're doing feature extraction + classification
        )
        
        # Apply LoRA to the model
        self.bert = get_peft_model(self.bert, peft_config)
        
        print(f"✓ LoRA adapters added:")
        print(f"  - r (rank): {r}")
        print(f"  - lora_alpha: {lora_alpha}")
        print(f"  - lora_dropout: {lora_dropout}")
        print(f"  - target_modules: {target_modules}")
        
        # Print trainable parameters
        self._print_trainable_parameters()
    
    def _print_trainable_parameters(self):
        """Print the number of trainable vs total parameters."""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percentage = 100 * trainable_params / all_param
        
        print(f"  - Trainable params: {trainable_params:,}")
        print(f"  - All params: {all_param:,}")
        print(f"  - Trainable%: {trainable_percentage:.4f}%")
    
    def _print_model_summary(self):
        """Print a summary of the model."""
        print("\n" + "="*80)
        print("Model Summary")
        print("="*80)
        print(f"Model: Bio-ClinicalBERT")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Number of labels: {self.num_labels}")
        print(f"LoRA enabled: {self.lora_enabled}")
        print(f"Device: {self.device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("="*80 + "\n")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False,
        return_values: bool = False,
    ):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            return_embeddings: If True, return embeddings (phi_llm)
            return_values: If True, return value estimates V(s) for PPO
            
        Returns:
            Depends on flags:
            - (logits,)                           if both False
            - (logits, embeddings)                if return_embeddings=True
            - (logits, values)                    if return_values=True
            - (logits, embeddings, values)        if both True
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract [CLS] token embedding (first token)
        # This is our phi_llm representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        pooled_output_dropout = self.dropout(pooled_output)
        
        # Classification (actor)
        logits = self.classifier(pooled_output_dropout)  # (batch_size, num_labels)
        
        # ⭐ Value head (critic) for PPO
        if return_values:
            values = self.value_head(pooled_output_dropout).squeeze(-1)  # (batch_size,)
            
            if return_embeddings:
                return logits, pooled_output, values
            else:
                return logits, values
        
        if return_embeddings:
            # Return original pooled_output (without dropout) as embedding
            return logits, pooled_output
        
        return logits
    
    def extract_features(self, 
                        texts: List[str], 
                        batch_size: int = 32,
                        show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract phi_llm embeddings and p_llm probabilities for a list of texts.
        
        This is the main feature extraction method used in the RF pipeline.
        
        Args:
            texts: List of text strings (patient descriptions)
            batch_size: Batch size for processing (adjust based on memory)
            show_progress: Whether to show progress bar
            
        Returns:
            (phi_llm, p_llm):
                phi_llm: (n_samples, hidden_size) embeddings from [CLS] token
                p_llm: (n_samples,) probabilities for positive class (relapse)
        """
        self.eval()  # Set to evaluation mode
        
        all_embeddings = []
        all_probs = []
        
        # Create progress bar
        num_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Extracting features")
        
        with torch.no_grad():  # No gradient computation
            for i in iterator:
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                try:
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(self.device)
                except Exception as e:
                    print(f"\nError tokenizing batch {i//batch_size}: {e}")
                    raise
                
                # Forward pass
                try:
                    logits, embeddings = self.forward(
                        inputs['input_ids'],
                        inputs['attention_mask'],
                        return_embeddings=True
                    )
                except Exception as e:
                    print(f"\nError in forward pass for batch {i//batch_size}: {e}")
                    raise
                
                # Get probabilities using softmax
                # probs[:, 1] is P(y=1|x) - probability of positive class (relapse)
                probs = torch.softmax(logits, dim=1)[:, 1]
                
                # Move to CPU and convert to numpy
                all_embeddings.append(embeddings.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        # Concatenate all batches
        phi_llm = np.vstack(all_embeddings)  # (n_samples, hidden_size)
        p_llm = np.concatenate(all_probs)    # (n_samples,)
        
        return phi_llm, p_llm
    
    def freeze_base_model(self):
        """
        Freeze all parameters except LoRA adapters and classifier.
        
        This is useful for fine-tuning only the adapters and classification head
        while keeping the base BERT weights fixed.
        """
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.named_parameters():
            # Keep LoRA and classifier parameters trainable
            if 'lora' in name.lower() or 'classifier' in name.lower():
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        
        print(f"Frozen {frozen_count:,} parameters")
        print(f"Trainable {trainable_count:,} parameters")
        self._print_trainable_parameters()
    
    def save_pretrained(self, save_directory: str):
        """
        Save the model and tokenizer.
        
        Args:
            save_directory: Directory to save to
        """
        print(f"Saving model to {save_directory}...")
        
        # Save BERT (with LoRA if enabled)
        if self.lora_enabled:
            self.bert.save_pretrained(save_directory)
        else:
            self.bert.save_pretrained(save_directory)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save classifier head
        torch.save({
            'classifier': self.classifier.state_dict(),
            'hidden_size': self.hidden_size,
            'num_labels': self.num_labels
        }, f"{save_directory}/classifier.pt")
        
        print(f"✓ Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, save_directory: str, device: str = 'cuda'):
        """
        Load a saved model.
        
        Args:
            save_directory: Directory to load from
            device: Device to load model on
            
        Returns:
            Loaded model instance
        """
        print(f"Loading model from {save_directory}...")
        
        # Load classifier config
        classifier_path = f"{save_directory}/classifier.pt"
        checkpoint = torch.load(classifier_path, map_location=device)
        
        # Initialize model (BERT and tokenizer will be loaded automatically)
        model = cls(
            model_name_or_path=save_directory,
            lora_config=None,  # LoRA config is saved with the model
            num_labels=checkpoint['num_labels'],
            device=device
        )
        
        # Load classifier weights
        model.classifier.load_state_dict(checkpoint['classifier'])
        
        print(f"✓ Model loaded from {save_directory}")
        
        return model
