# Breast Cancer WDBC Qwen2-7B RCT

Prepare the public WDBC files locally before running.

Run scripts from the repository root with a config file from `qwen2_7b/configs/`. The scripts are thin entry points over `qwen2_7b/shared/` so that path settings, output locations, and model parameters remain configurable.

Typical workflow:

```bash
python qwen2_7b/breast_cancer/scripts/qwen2_llm_baseline_train.py --config qwen2_7b/configs/breast_cancer_qwen2_config.example.yaml
python qwen2_7b/breast_cancer/scripts/qwen2_stage5_ppo_rf_refresh.py --config qwen2_7b/configs/breast_cancer_qwen2_config.example.yaml
python qwen2_7b/breast_cancer/scripts/qwen2_threshold_matched_eval.py --config qwen2_7b/configs/breast_cancer_qwen2_config.example.yaml
```
