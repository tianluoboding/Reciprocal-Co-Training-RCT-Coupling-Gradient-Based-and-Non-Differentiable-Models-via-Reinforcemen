#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-qwen2_7b/configs/ms_relapse_qwen2_config.example.yaml}"

python qwen2_7b/ms_relapse/scripts/qwen2_llm_baseline_train.py --config "$CONFIG"
python qwen2_7b/ms_relapse/scripts/qwen2_stage1_frozen_embedding_rf.py --config "$CONFIG"
python qwen2_7b/ms_relapse/scripts/qwen2_stage4b_lora_warmstart.py --config "$CONFIG"
python qwen2_7b/ms_relapse/scripts/qwen2_stage5_ppo_rf_refresh.py --config "$CONFIG"
python qwen2_7b/ms_relapse/scripts/qwen2_threshold_matched_eval.py --config "$CONFIG"
