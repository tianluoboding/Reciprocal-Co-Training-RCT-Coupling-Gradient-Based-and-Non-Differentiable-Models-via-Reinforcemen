# Reciprocal Co-Training (RCT): Coupling Gradient-Based and Non-Differentiable Models via Reinforcement Learning

This repository contains the source code for the **Reinforcement Collaborative Training (RCT)** framework, which iteratively co-trains a Large Language Model (LLM) and a Random Forest (RF) classifier for binary clinical classification tasks.

## Method Overview

RCT is an iterative student-teacher loop:

1. The LLM (Bio-ClinicalBERT + LoRA) processes natural-language patient descriptions and produces CLS-token embeddings (768-d).
2. Embeddings are reduced to 5 dimensions via PCA and concatenated with tabular features.
3. A calibrated Random Forest is trained on this combined representation.
4. The RF's predicted probability advantage is used as a reward signal to fine-tune the LLM via Proximal Policy Optimization (PPO).
5. Steps 1-4 repeat until convergence (early stopping on test AUC).

## Repository Structure

```
RCT-Clinical-Classification/
├── README.md
├── requirements.txt
├── .gitignore
├── ms_relapse/              # MS-Relapse dataset experiment
│   ├── src/                 # Shared source library
│   │   ├── models/          #   Bio-ClinicalBERT + LoRA wrapper
│   │   ├── data/            #   Data loading & preprocessing
│   │   ├── training/        #   PPO trainer, reward functions, early stopping
│   │   ├── evaluation/      #   Metrics & threshold search
│   │   └── utils/           #   Sampling, logging, seeding, I/O
│   └── scripts/
│       ├── iterative_training.py      # Main RCT iterative training loop
│       ├── rf_baseline.py             # RF-only baseline (10-fold CV)
│       ├── llm_ce_baseline.py         # LLM cross-entropy baseline
│       ├── threshold_eval.py          # Threshold-matched evaluation (>=80% recall)
│       └── evaluate_iterations.py     # Per-iteration metric extraction
├── breast_cancer/           # WDBC dataset experiment
│   ├── src/                 # Dataset-specific loader + shared LLM wrapper
│   └── scripts/
│       ├── preprocess.py              # Raw data -> text descriptions + splits
│       ├── iterative_training.py      # RCT iterative training loop
│       ├── rf_baseline.py             # RF-only baseline
│       └── llm_ce_baseline.py         # LLM cross-entropy baseline
└── diabetes/                # Diabetes (BRFSS 2015) experiment
    ├── src/                 # Dataset-specific loader + shared LLM wrapper
    └── scripts/
        ├── preprocess.py              # Raw data -> text descriptions + splits
        ├── iterative_training.py      # RCT iterative training loop
        ├── rf_baseline.py             # RF-only baseline
        └── llm_ce_baseline.py         # LLM cross-entropy baseline
```

## Requirements

- Python >= 3.9
- CUDA-capable GPU (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

### MS-Relapse (Proprietary)

The MS-Relapse dataset is proprietary clinical data and **cannot be shared**. To reproduce this experiment, place the preprocessed data files in `ms_relapse/data/` following the expected directory structure:

```
ms_relapse/data/
├── raw/
│   ├── descriptive_text.json    # Natural-language patient descriptions
│   ├── train_idxs.json          # Train split indices
│   └── test_idxs.json           # Test split indices
└── processed/
    ├── X_train_selected.npy     # Selected tabular features (train)
    ├── X_test_selected.npy      # Selected tabular features (test)
    ├── y_train.npy              # Labels (train)
    └── y_test.npy               # Labels (test)
```

### Breast Cancer Wisconsin (Diagnostic) -- WDBC

Publicly available from the UCI Machine Learning Repository.

1. Download from: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
2. Place the `wdbc.data` file in `breast_cancer/data/raw/`
3. Run preprocessing:

```bash
cd breast_cancer/scripts
python preprocess.py --k_values 5 10 15 20 30
```

### Diabetes (CDC BRFSS 2015)

Publicly available on Kaggle.

1. Download from: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
2. Place `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` in `diabetes/data/raw/`
3. Run preprocessing:

```bash
cd diabetes/scripts
python preprocess.py --k_values 5 10 21
```

## Running Experiments

Each experiment follows the same workflow. Navigate to the dataset directory and run scripts from the `scripts/` folder.

### 1. Baselines

**RF-only baseline** (tabular features only, no LLM):

```bash
cd {dataset}/scripts
python rf_baseline.py
```

**LLM cross-entropy baseline** (text-only, no RF):

```bash
cd {dataset}/scripts
python llm_ce_baseline.py
```

### 2. RCT Iterative Training

```bash
cd {dataset}/scripts
python iterative_training.py
```

The training loop will:
- Initialize the LLM and extract embeddings
- Train RF on tabular + PCA-reduced LLM embeddings
- Fine-tune LLM via PPO using RF reward signal
- Repeat until early stopping triggers (5 consecutive iterations without RF test AUC improvement)

### 3. Evaluation

For MS-Relapse, threshold-matched evaluation at a target recall:

```bash
cd ms_relapse/scripts
python threshold_eval.py --dataset ms --target-recall 0.80
```

## Model Configuration

| Component | Configuration |
|-----------|--------------|
| **LLM** | Bio-ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) |
| **Fine-tuning** | LoRA (rank=8, alpha=16, dropout=0.05, target: query + value) |
| **Trainable params** | 297,219 / 108,607,491 (0.27%) |
| **Embedding** | CLS token, 768-d -> PCA 5-d |
| **Random Forest** | 500 trees, max_depth=8, min_samples_leaf=2, min_samples_split=5, class_weight=balanced_subsample |
| **Calibration** | Sigmoid (Platt scaling), 20% calibration split |
| **PPO** | lr=2e-5, batch_size=32, 4 epochs/batch, entropy_coef=0.05 |
| **Reward** | R = R_RF + lambda * R_acc (fn_heavy: TP=+1.0, FN=-1.5, TN=+0.2, FP=-0.2) |
| **Feature selection** | SelectKBest (f_classif): k=15 (MS), k=30 (WDBC), k=21 (Diabetes) |
| **Early stopping** | RF test AUC, patience=5 iterations |

## License

This code is provided for academic research purposes.
