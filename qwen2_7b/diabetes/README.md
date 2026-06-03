# Diabetes BRFSS Qwen2-7B RCT

Prepare the public Diabetes BRFSS files locally before running.

Run scripts from the repository root with a config file from `qwen2_7b/configs/`. The scripts are thin entry points over `qwen2_7b/shared/` so that path settings, output locations, and model parameters remain configurable.

Typical workflow:

```bash
python qwen2_7b/diabetes/scripts/qwen2_llm_baseline_train.py --config qwen2_7b/configs/diabetes_qwen2_config.example.yaml
python qwen2_7b/diabetes/scripts/qwen2_stage5_ppo_rf_refresh.py --config qwen2_7b/configs/diabetes_qwen2_config.example.yaml
python qwen2_7b/diabetes/scripts/qwen2_threshold_matched_eval.py --config qwen2_7b/configs/diabetes_qwen2_config.example.yaml
```
