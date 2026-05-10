"""Cosine similarity loss for knowledge distillation.

Measures angular alignment between student and teacher feature vectors.
Unlike L2-norm + MSE (which approximates 2*(1 - cos_sim)), this is a
direct formulation that is numerically cleaner and focuses purely on
directional alignment, ignoring magnitude entirely.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_alignment_loss(
    s_feat: torch.Tensor,
    t_feat: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Cosine similarity distillation loss.

    Args:
        s_feat: Student features [B, D] (or [B, N, D]).
        t_feat: Teacher features [B, D] (or [B, N, D]).
        dim: Dimension along which to compute cosine similarity.

    Returns:
        Scalar loss = mean(1 - cos_sim). Range [0, 2].
    """
    cos_sim = F.cosine_similarity(
        s_feat.float(), t_feat.float(), dim=dim,
    )
    return (1.0 - cos_sim).mean()
