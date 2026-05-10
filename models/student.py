"""Student model loader using Unsloth FastVisionModel.

Loads Qwen3.5-VL with QLoRA adapters via Unsloth for memory-efficient
fine-tuning on a single 24GB GPU.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class StudentConfig:
    """Configuration for the student model."""
    model_name: str = "unsloth/Qwen3.5-4B"
    max_seq_length: int = 4096
    load_in_4bit: bool = False
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # Target modules for LoRA (None means use Unsloth auto-detection)
    target_modules: Optional[List[str]] = None
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False
    loftq_config: Optional[Any] = None
    # Vision-specific fine-tuning flags (for Qwen3.5-VL style)
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


def load_student(
    config: Optional[StudentConfig] = None,
) -> Tuple[Any, Any]:
    """Load student model with Unsloth QLoRA.

    Args:
        config: Student model configuration.

    Returns:
        Tuple of (model, tokenizer/processor).
    """
    from unsloth import FastVisionModel

    cfg = config or StudentConfig()

    logger.info("Loading student model: %s", cfg.model_name)
    logger.info("  4-bit quantization: %s", cfg.load_in_4bit)
    logger.info("  Max sequence length: %d", cfg.max_seq_length)

    model, tokenizer = FastVisionModel.from_pretrained(
        cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
    )

    logger.info("Applying LoRA adapters (r=%d, alpha=%d)", cfg.lora_r, cfg.lora_alpha)

    peft_kwargs = dict(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        use_rslora=cfg.use_rslora,
        loftq_config=cfg.loftq_config,
    )

    # For explicit target_modules list (Qwen2.5-VL style)
    if cfg.target_modules:
        peft_kwargs["target_modules"] = cfg.target_modules
    else:
        # Use vision-layer fine-tuning flags (Qwen3.5-VL / newer unsloth style)
        peft_kwargs["finetune_vision_layers"] = cfg.finetune_vision_layers
        peft_kwargs["finetune_language_layers"] = cfg.finetune_language_layers
        peft_kwargs["finetune_attention_modules"] = cfg.finetune_attention_modules
        peft_kwargs["finetune_mlp_modules"] = cfg.finetune_mlp_modules

    model = FastVisionModel.get_peft_model(model, **peft_kwargs)

    # Log parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Student model loaded: %d trainable / %d total parameters (%.2f%%)",
        trainable, total, 100 * trainable / total if total > 0 else 0,
    )

    return model, tokenizer


def get_student_vision_encoder(model: Any) -> torch.nn.Module:
    """Extract the vision encoder (ViT) from the student model.

    Used for feature-level distillation — hooks are attached to specific
    layers of this encoder to extract intermediate representations.
    """
    # Qwen2.5-VL / Qwen3-VL architecture: model.visual or model.model.visual
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    # For PEFT-wrapped models
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model"):
            inner = base.model
            if hasattr(inner, "visual"):
                return inner.visual
            if hasattr(inner, "model") and hasattr(inner.model, "visual"):
                return inner.model.visual
    raise AttributeError(
        "Cannot locate vision encoder in student model. "
        "Expected model.visual or model.model.visual"
    )
