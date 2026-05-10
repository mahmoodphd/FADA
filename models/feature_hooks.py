"""Forward hook manager for extracting intermediate features.

Attaches PyTorch forward hooks to specific layers of student and teacher
vision encoders to capture intermediate representations for feature-level
knowledge distillation.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeatureHookManager:
    """Manages forward hooks on model layers to capture intermediate features.

    Usage:
        manager = FeatureHookManager()
        manager.attach(model.layer4, "student_layer4")
        output = model(input)
        features = manager.get("student_layer4")
        manager.clear()  # Clear stored features between batches
    """

    def __init__(self):
        self._hooks: Dict[str, torch.utils.hooks.RemovableHook] = {}
        self._features: Dict[str, torch.Tensor] = {}

    def attach(
        self,
        module: nn.Module,
        name: str,
        transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """Attach a forward hook to a module.

        Args:
            module: The nn.Module to hook into.
            name: Unique identifier for the hooked feature.
            transform_fn: Optional function to apply to the captured output.
        """
        if name in self._hooks:
            logger.warning("Hook '%s' already exists, removing old hook", name)
            self._hooks[name].remove()

        def hook_fn(mod: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                feat = output
            elif isinstance(output, tuple):
                feat = output[0]
            elif isinstance(output, dict):
                feat = output.get("last_hidden_state", next(iter(output.values())))
            else:
                logger.warning("Unexpected output type from hook '%s': %s", name, type(output))
                return

            if transform_fn is not None:
                feat = transform_fn(feat)

            self._features[name] = feat

        handle = module.register_forward_hook(hook_fn)
        self._hooks[name] = handle
        logger.debug("Attached hook '%s' to %s", name, type(module).__name__)

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Get captured feature by name."""
        return self._features.get(name)

    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all captured features."""
        return dict(self._features)

    def clear(self) -> None:
        """Clear all captured features (call between batches)."""
        self._features.clear()

    def remove(self, name: str) -> None:
        """Remove a specific hook."""
        if name in self._hooks:
            self._hooks[name].remove()
            del self._hooks[name]
        self._features.pop(name, None)

    def remove_all(self) -> None:
        """Remove all hooks and clear features."""
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()
        self._features.clear()

    @property
    def names(self) -> List[str]:
        """Names of all registered hooks."""
        return list(self._hooks.keys())


def auto_attach_hooks(
    student_encoder: nn.Module,
    teacher_model: nn.Module,
    student_layer_indices: List[int],
    teacher_layer_indices: List[int],
    hook_manager: FeatureHookManager,
    student_prefix: str = "student",
    teacher_prefix: str = "teacher",
) -> List[Tuple[str, str]]:
    """Automatically attach hooks to corresponding layers of student and teacher.

    Attempts to find the main transformer block list (e.g., model.blocks,
    model.layers, model.encoder.layers) and hooks into the specified indices.

    Args:
        student_encoder: Student's vision encoder.
        teacher_model: Teacher model or its encoder.
        student_layer_indices: Which student layers to hook.
        teacher_layer_indices: Which teacher layers to hook (must match length).
        hook_manager: The hook manager to use.
        student_prefix: Prefix for student hook names.
        teacher_prefix: Prefix for teacher hook names.

    Returns:
        List of (student_hook_name, teacher_hook_name) pairs for loss computation.
    """
    assert len(student_layer_indices) == len(teacher_layer_indices), (
        "Must provide equal number of student and teacher layer indices"
    )

    student_blocks = _find_blocks(student_encoder)
    teacher_blocks = _find_blocks(teacher_model)

    if student_blocks is None:
        raise ValueError("Cannot find transformer blocks in student encoder")
    if teacher_blocks is None:
        raise ValueError("Cannot find transformer blocks in teacher model")

    pairs = []
    for s_idx, t_idx in zip(student_layer_indices, teacher_layer_indices):
        s_name = f"{student_prefix}_layer{s_idx}"
        t_name = f"{teacher_prefix}_layer{t_idx}"

        if s_idx < len(student_blocks):
            hook_manager.attach(student_blocks[s_idx], s_name)
        else:
            logger.warning(
                "Student layer index %d out of range (max %d)",
                s_idx, len(student_blocks) - 1,
            )
            continue

        if t_idx < len(teacher_blocks):
            hook_manager.attach(teacher_blocks[t_idx], t_name)
        else:
            logger.warning(
                "Teacher layer index %d out of range (max %d)",
                t_idx, len(teacher_blocks) - 1,
            )
            continue

        pairs.append((s_name, t_name))

    logger.info(
        "Attached %d hook pairs: %s",
        len(pairs),
        [(s, t) for s, t in pairs],
    )
    return pairs


def _find_blocks(model: nn.Module) -> Optional[nn.ModuleList]:
    """Find the main transformer block list in a model.

    Searches common attribute patterns across architectures:
    - timm ViT / MAE: model.blocks
    - mmpretrain / SAM: model.layers
    - open_clip CLIP: model.visual.transformer.resblocks
    - HuggingFace: model.encoder.layers
    """
    block_attrs = ["blocks", "layers", "encoder_layers", "resblocks"]

    # Direct attributes: model.blocks, model.layers, etc.
    for attr in block_attrs:
        blocks = getattr(model, attr, None)
        if isinstance(blocks, (nn.ModuleList, nn.Sequential)):
            return blocks

    # One level deep: model.X.blocks, model.X.layers, etc.
    container_attrs = ["encoder", "transformer", "backbone", "visual", "trunk"]
    for c_attr in container_attrs:
        container = getattr(model, c_attr, None)
        if container is not None:
            for attr in block_attrs:
                blocks = getattr(container, attr, None)
                if isinstance(blocks, (nn.ModuleList, nn.Sequential)):
                    return blocks

    # Two levels deep: model.visual.transformer.resblocks (open_clip CLIP)
    for c_attr in container_attrs:
        container = getattr(model, c_attr, None)
        if container is not None:
            for inner_attr in ["transformer", "encoder", "trunk"]:
                inner = getattr(container, inner_attr, None)
                if inner is not None:
                    for attr in block_attrs:
                        blocks = getattr(inner, attr, None)
                        if isinstance(blocks, (nn.ModuleList, nn.Sequential)):
                            return blocks

    return None
