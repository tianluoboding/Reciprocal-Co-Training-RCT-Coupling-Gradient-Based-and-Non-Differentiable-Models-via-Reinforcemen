#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-qwen2_7b/configs/diabetes_qwen2_config.example.yaml}"

python qwen2_7b/diabetes/scripts/qwen2_llm_baseline_train.py --config "$CONFIG"
python qwen2_7b/diabetes/scripts/qwen2_stage5_ppo_rf_refresh.py --config "$CONFIG"
python qwen2_7b/diabetes/scripts/qwen2_threshold_matched_eval.py --config "$CONFIG"
