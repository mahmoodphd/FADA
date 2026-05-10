"""Attention transfer distillation loss (L_attn).

Transfers attention patterns from teacher ViT attention heads to student,
encouraging the student to attend to similar spatial regions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionTransferLoss(nn.Module):
    """Attention map transfer loss between teacher and student ViTs.

    Uses the attention rollout or per-layer attention maps to compute
    an MSE loss between normalized student and teacher attention patterns.

    Attention maps are extracted via forward hooks on attention layers.
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: If True, L2-normalize attention maps before comparison.
        """
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        student_attn_maps: Dict[str, torch.Tensor],
        teacher_attn_maps: Dict[str, torch.Tensor],
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> torch.Tensor:
        """Compute attention transfer loss.

        Args:
            student_attn_maps: {name: [B, H, N, N]} attention matrices.
            teacher_attn_maps: {name: [B, H, N, N]} attention matrices.
            pairs: List of (student_name, teacher_name) pairs.

        Returns:
            Scalar loss tensor.
        """
        if pairs is None or not pairs:
            device = "cpu"
            if student_attn_maps:
                device = next(iter(student_attn_maps.values())).device
            return torch.tensor(0.0, device=device)

        total_loss = torch.tensor(0.0)
        count = 0

        for s_name, t_name in pairs:
            s_attn = student_attn_maps.get(s_name)
            t_attn = teacher_attn_maps.get(t_name)

            if s_attn is None or t_attn is None:
                continue

            loss = self._compute_pair_loss(s_attn, t_attn)
            total_loss = total_loss.to(loss.device) + loss
            count += 1

        if count == 0:
            device = "cpu"
            if student_attn_maps:
                device = next(iter(student_attn_maps.values())).device
            return torch.tensor(0.0, device=device)

        return total_loss / count

    def _compute_pair_loss(
        self,
        student_attn: torch.Tensor,
        teacher_attn: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for a single attention map pair.

        Handles:
        - Different number of heads (averages across heads)
        - Different spatial dimensions (interpolates)
        """
        # Average across heads: [B, H, N, N] -> [B, N, N]
        s_map = student_attn.float().mean(dim=1)
        t_map = teacher_attn.float().mean(dim=1)

        # Align spatial dimensions if needed
        if s_map.shape != t_map.shape:
            target_n = t_map.shape[-1]
            s_map = F.interpolate(
                s_map.unsqueeze(1),  # [B, 1, N_s, N_s]
                size=(target_n, target_n),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # [B, N_t, N_t]

        if self.normalize:
            # Row-wise L2 normalization
            s_map = F.normalize(s_map, p=2, dim=-1)
            t_map = F.normalize(t_map, p=2, dim=-1)

        return F.mse_loss(s_map, t_map)


def extract_attention_hook(
    module: nn.Module,
    input: any,
    output: any,
) -> None:
    """Hook function that extracts attention weights from a multi-head attention layer.

    Compatible with PyTorch's nn.MultiheadAttention (when need_weights=True)
    and HuggingFace-style attention modules that return (output, attn_weights).
    """
    if isinstance(output, tuple) and len(output) >= 2:
        attn_weights = output[1]
        if attn_weights is not None:
            module._captured_attention = attn_weights
