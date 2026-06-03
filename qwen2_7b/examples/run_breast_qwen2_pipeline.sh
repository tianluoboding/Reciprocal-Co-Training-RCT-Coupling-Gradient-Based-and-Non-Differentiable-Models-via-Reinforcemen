#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-qwen2_7b/configs/breast_cancer_qwen2_config.example.yaml}"

python qwen2_7b/breast_cancer/scripts/qwen2_llm_baseline_train.py --config "$CONFIG"
python qwen2_7b/breast_cancer/scripts/qwen2_stage5_ppo_rf_refresh.py --config "$CONFIG"
python qwen2_7b/breast_cancer/scripts/qwen2_threshold_matched_eval.py --config "$CONFIG"
