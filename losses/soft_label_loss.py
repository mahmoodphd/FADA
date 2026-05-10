"""Soft-label distillation loss (L_soft).

KL divergence between student and teacher prediction distributions,
operating on the logits of the language model head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLabelLoss(nn.Module):
    """KL-divergence based soft-label distillation loss.

    Computes:
        L_soft = KL(softmax(teacher_logits / T), log_softmax(student_logits / T)) * T^2

    where T is the temperature parameter that controls softness of distributions.
    """

    def __init__(self, temperature: float = 4.0):
        """
        Args:
            temperature: Softmax temperature. Higher = softer distributions.
                Typical range: 2.0 - 8.0 for VLM distillation.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute soft-label distillation loss.

        Args:
            student_logits: Student LM head output [B, T_seq, V].
            teacher_logits: Teacher LM head output [B, T_seq, V].
            mask: Optional boolean mask [B, T_seq] — True for tokens to include.
                Typically masks out padding and image tokens.

        Returns:
            Scalar loss tensor.
        """
        if student_logits.shape != teacher_logits.shape:
            # Align sequence lengths (take the shorter)
            min_len = min(student_logits.shape[1], teacher_logits.shape[1])
            student_logits = student_logits[:, :min_len, :]
            teacher_logits = teacher_logits[:, :min_len, :]
            if mask is not None:
                mask = mask[:, :min_len]

        # Scale by temperature
        s_scaled = student_logits.float() / self.temperature
        t_scaled = teacher_logits.float() / self.temperature

        # KL divergence: KL(p_teacher || q_student)
        # = sum(p_teacher * (log(p_teacher) - log(q_student)))
        s_log_probs = F.log_softmax(s_scaled, dim=-1)
        t_probs = F.softmax(t_scaled, dim=-1)

        # Per-token KL divergence
        kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)  # [B, T]

        if mask is not None:
            kl = kl * mask.float()
            loss = kl.sum() / mask.float().sum().clamp(min=1.0)
        else:
            loss = kl.mean()

        # Scale by T^2 (standard distillation convention)
        return loss * (self.temperature ** 2)
