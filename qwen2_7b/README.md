# Qwen2-7B RCT Implementation

This folder contains a public, anonymized Qwen2-7B implementation of Reciprocal Co-Training (RCT). It is parallel to the ClinicalBERT code in the dataset folders at the repository root.

## Method Summary

The Qwen2-7B version follows the same RCT loop:

1. Qwen2-7B-Instruct encodes patient descriptions with LoRA adapters.
2. LLM embeddings are reduced with PCA.
3. A Random Forest is trained on tabular features plus PCA-reduced LLM embeddings.
4. RF feedback and supervised label feedback are used during PPO-style LLM updates.
5. The updated LLM refreshes embeddings, PCA, and RF in an alternating outer loop.

## Layout

```
qwen2_7b/
├── configs/              # Example config files with relative placeholder paths
├── shared/               # Reusable data, model, embedding, RF, PPO, and metric utilities
├── ms_relapse/scripts/   # MS-Relapse entry points
├── breast_cancer/scripts/# WDBC entry points
├── diabetes/scripts/     # Diabetes entry points
└── examples/             # Shell examples for end-to-end runs
```

## Data Policy

MS-Relapse data is proprietary and is not included. Do not commit raw clinical data, derived private embeddings, trained model weights, or RF/PCA artifacts trained on private data. The example configs describe the expected file names and shapes only.

Public datasets such as WDBC and Diabetes can be prepared locally and placed under `data/` using the paths in `configs/*.example.yaml`.

## Example Usage

From the repository root:

```bash
python qwen2_7b/ms_relapse/scripts/qwen2_llm_baseline_train.py --config qwen2_7b/configs/ms_relapse_qwen2_config.example.yaml
python qwen2_7b/ms_relapse/scripts/qwen2_stage1_frozen_embedding_rf.py --config qwen2_7b/configs/ms_relapse_qwen2_config.example.yaml
python qwen2_7b/ms_relapse/scripts/qwen2_stage5_ppo_rf_refresh.py --config qwen2_7b/configs/ms_relapse_qwen2_config.example.yaml
python qwen2_7b/ms_relapse/scripts/qwen2_threshold_matched_eval.py --config qwen2_7b/configs/ms_relapse_qwen2_config.example.yaml
```

A CUDA GPU is strongly recommended for Qwen2-7B. The scripts support 4-bit loading through `bitsandbytes` when available.
