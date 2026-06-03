# Reciprocal Co-Training (RCT): Coupling Gradient-Based and Non-Differentiable Models via Reinforcement Learning

This repository contains anonymized source code for Reciprocal Co-Training (RCT), an iterative framework that couples a gradient-based language model with a non-differentiable Random Forest classifier for binary clinical classification.

## Method Overview

RCT is implemented for both ClinicalBERT and Qwen2-7B variants:

1. A language model processes natural-language patient descriptions and produces embeddings.
2. PCA reduces the embedding dimension.
3. A Random Forest is trained on tabular features plus PCA-reduced LLM embeddings.
4. RF reward and supervised label reward update the LLM through PPO-style training.
5. The loop alternates between LLM updates and RF refreshes until convergence.

The root dataset folders contain the ClinicalBERT-era implementation. The `qwen2_7b/` folder contains a clean parallel Qwen2-7B implementation with configurable paths and example configs.

## Repository Structure

```
RCT-Clinical-Classification/
├── README.md
├── requirements.txt
├── .gitignore
├── ms_relapse/                  # ClinicalBERT MS-Relapse experiment
│   ├── src/
│   └── scripts/
├── breast_cancer/               # ClinicalBERT WDBC experiment
│   ├── src/
│   └── scripts/
├── diabetes/                    # ClinicalBERT Diabetes experiment
│   ├── src/
│   └── scripts/
└── qwen2_7b/                    # Qwen2-7B RCT implementation
    ├── README.md
    ├── configs/
    ├── shared/
    ├── ms_relapse/
    ├── breast_cancer/
    ├── diabetes/
    └── examples/
```

## Requirements

Python 3.9 or newer is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

Qwen2-7B experiments additionally require a CUDA-capable GPU for practical runtimes. The public scripts use `transformers`, `peft`, `bitsandbytes`, `accelerate`, `torch`, `scikit-learn`, `pandas`, and `numpy`.

## Datasets

### MS-Relapse

The MS-Relapse dataset is proprietary clinical data and is not included. To reproduce this experiment, place preprocessed local files under a private `data/ms_relapse/` directory matching the example config paths. Do not commit raw MS patient data, derived private embeddings, trained weights, RF/PCA models, logs, or result dumps.

Expected private layout:

```
data/ms_relapse/
├── raw/
│   ├── descriptive_text.json
│   ├── y_all.npy
│   ├── train_idxs.json
│   └── test_idxs.json
└── processed/
    ├── X_train_selected_k15.npy
    └── X_test_selected_k15.npy
```

### Breast Cancer Wisconsin Diagnostic

WDBC is publicly available from the UCI Machine Learning Repository. Prepare the local raw file and derived prompt, split, label, and feature arrays according to the dataset scripts.

### Diabetes BRFSS

The Diabetes BRFSS 2015 dataset is publicly available from Kaggle. Prepare the local raw file and derived prompt, split, label, and feature arrays according to the dataset scripts.

## Running ClinicalBERT Experiments

Each root dataset folder follows the same pattern:

```bash
cd ms_relapse/scripts
python rf_baseline.py
python llm_ce_baseline.py
python iterative_training.py
python threshold_eval.py --dataset ms --target-recall 0.80
```

## Running Qwen2-7B Experiments

From the repository root:

```bash
python qwen2_7b/ms_relapse/scripts/qwen2_llm_baseline_train.py --config qwen2_7b/configs/ms_relapse_qwen2_config.example.yaml
python qwen2_7b/ms_relapse/scripts/qwen2_stage1_frozen_embedding_rf.py --config qwen2_7b/configs/ms_relapse_qwen2_config.example.yaml
python qwen2_7b/ms_relapse/scripts/qwen2_stage5_ppo_rf_refresh.py --config qwen2_7b/configs/ms_relapse_qwen2_config.example.yaml
```

Breast Cancer and Diabetes use their own Qwen2 entry points under `qwen2_7b/breast_cancer/scripts/` and `qwen2_7b/diabetes/scripts/` for LLM baseline training, PPO+RF refresh, threshold evaluation, and curve CSV generation. The MS folder additionally includes explicit Stage 1 frozen-embedding RF and Stage 4B LoRA warm-start entry points.

## Anonymization Note

This repository has been anonymized for review. User-specific paths, institution identifiers, emails, server names, raw clinical data, checkpoints, and trained model artifacts have been removed.

## License

This code is provided for academic research purposes.
