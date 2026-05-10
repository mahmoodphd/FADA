"""Model loader with dual Unsloth/transformers+PEFT fallback.

On ZeroGPU, the GPU is not available at import time. Models must be
loaded inside @spaces.GPU-decorated functions. This module provides
a lazy-loading singleton that caches the model after first load.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level cache
_model: Optional[Any] = None
_processor: Optional[Any] = None
_backend: Optional[str] = None
_loading_lock = threading.Lock()

# HF model repo
MODEL_REPO = "mshz88/FADA-SKD"
BASE_MODEL = "unsloth/Qwen3.5-4B"
MAX_SEQ_LENGTH = 4096


def get_model_and_processor() -> Tuple[Any, Any]:
    """Get the cached model and processor, loading on first call.

    Must be called from within a @spaces.GPU-decorated function.
    """
    global _model, _processor, _backend

    if _model is not None and _processor is not None:
        return _model, _processor

    with _loading_lock:
        # Double-check after acquiring lock
        if _model is not None and _processor is not None:
            return _model, _processor

        _model, _processor, _backend = _load_model()
        logger.info("Model loaded via %s backend", _backend)
        return _model, _processor


def _load_model() -> Tuple[Any, Any, str]:
    """Load model with Unsloth primary, transformers+PEFT fallback."""
    import torch

    # Ensure HF_TOKEN is available for private repo access
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logger.info("HF_TOKEN found, will use for private repo access")

    # Strategy A: Unsloth (primary)
    try:
        from unsloth import FastVisionModel
        logger.info("Attempting Unsloth loading from %s", MODEL_REPO)
        model, processor = FastVisionModel.from_pretrained(
            MODEL_REPO,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=False,
            token=hf_token,
        )
        FastVisionModel.for_inference(model)
        return model, processor, "unsloth"
    except Exception as e:
        logger.warning("Unsloth loading failed: %s. Trying transformers+PEFT.", e)

    # Strategy B: transformers + PEFT fallback
    try:
        from transformers import AutoProcessor
        from peft import PeftModel

        # transformers 5.x renamed AutoModelForVision2Seq
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM

        logger.info("Loading base model: %s", BASE_MODEL)
        base_model = AutoVLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )

        logger.info("Applying PEFT adapter from %s", MODEL_REPO)
        model = PeftModel.from_pretrained(
            base_model, MODEL_REPO, token=hf_token)
        model.eval()

        processor = AutoProcessor.from_pretrained(
            MODEL_REPO, trust_remote_code=True, token=hf_token)

        return model, processor, "transformers+peft"
    except Exception as e:
        logger.error("Transformers+PEFT loading also failed: %s", e)
        raise RuntimeError(
            f"Could not load model via Unsloth or transformers+PEFT. "
            f"Ensure MODEL_REPO='{MODEL_REPO}' is correct and accessible."
        ) from e


def get_backend() -> Optional[str]:
    """Return which backend loaded the model ('unsloth' or 'transformers+peft')."""
    return _backend
