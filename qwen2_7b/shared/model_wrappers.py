"""Qwen2-7B LoRA model wrappers used by the public RCT scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_pad_token(tokenizer: Any) -> dict[str, Any]:
    info = {"pad_token_id_before": getattr(tokenizer, "pad_token_id", None)}
    if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        info["action"] = "set_pad_token_to_eos"
    else:
        info["action"] = "pad_token_present"
    info["pad_token_id_after"] = tokenizer.pad_token_id
    return info


def load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    ensure_pad_token(tokenizer)
    return tokenizer


def qwen_dtype() -> Any:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_qwen2_base(model_name: str, *, quantized: bool = True, trainable: bool = False) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs.update({"device_map": "auto", "torch_dtype": qwen_dtype()})
        if quantized:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=qwen_dtype(),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
    else:
        kwargs.update({"torch_dtype": torch.float32, "low_cpu_mem_usage": True})
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if hasattr(model, "config") and model.config is not None:
        model.config.use_cache = False
    if not trainable:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
    return model


def create_lora_model(base_model: Any, config: dict[str, Any]) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if config.get("prepare_kbit", True):
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)))
    lora_cfg = LoraConfig(
        r=int(config.get("r", 8)),
        lora_alpha=int(config.get("alpha", 16)),
        lora_dropout=float(config.get("dropout", 0.05)),
        bias="none",
        target_modules=list(config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(base_model, lora_cfg)


def load_lora_adapter(base_model: Any, adapter_dir: str | Path, *, trainable: bool) -> Any:
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, Path(adapter_dir), is_trainable=trainable)


class Qwen2LoRAClassifierMixin:
    """Methods shared by the classifier and value-head wrapper."""

    def pooled_outputs(self, input_ids: Any, attention_mask: Any) -> dict[str, Any]:
        import torch

        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1]
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_idx, lengths].to(dtype=self.classifier.weight.dtype)
        logits = self.classifier(pooled).squeeze(-1)
        values = self.value_head(pooled).squeeze(-1) if hasattr(self, "value_head") else None
        return {"embedding": pooled, "logits": logits, "values": values}

    def forward(self, input_ids: Any, attention_mask: Any) -> Any:
        return self.pooled_outputs(input_ids, attention_mask)["logits"]


def build_classifier(base_or_adapter: Any, *, with_value_head: bool = False) -> Any:
    import torch.nn as nn

    hidden_size = int(getattr(base_or_adapter.config, "hidden_size", 3584))

    class Qwen2LoRAClassifier(Qwen2LoRAClassifierMixin, nn.Module):
        def __init__(self, base_model: Any, hidden: int) -> None:
            super().__init__()
            self.base_model = base_model
            self.classifier = nn.Linear(hidden, 1)
            if with_value_head:
                self.value_head = nn.Linear(hidden, 1)

    model = Qwen2LoRAClassifier(base_or_adapter, hidden_size)
    dev = next(base_or_adapter.parameters()).device
    model.classifier.to(dev)
    if with_value_head:
        model.value_head.to(dev)
    return model


def save_classifier_checkpoint(model: Any, out_dir: str | Path, metadata: dict[str, Any]) -> None:
    import json
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(out / "lora_adapter")
    torch.save(model.classifier.state_dict(), out / "classifier_head.pt")
    if hasattr(model, "value_head"):
        torch.save(model.value_head.state_dict(), out / "value_head.pt")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
