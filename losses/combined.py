"""Combined distillation loss.

Weighted combination of all loss components:
- L_task: Standard SFT cross-entropy loss (from SFTTrainer)
- L_feat: Feature alignment loss
- L_soft: Soft-label distillation loss
- L_attn: Attention transfer loss
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .attention_loss import AttentionTransferLoss
from .feature_loss import FeatureAlignmentLoss
from .soft_label_loss import SoftLabelLoss

logger = logging.getLogger(__name__)


@dataclass
class DistillationLossConfig:
    """Weights and settings for the combined distillation loss."""
    # Loss weights
    w_task: float = 1.0       # Weight for SFT task loss
    w_feat: float = 0.5       # Weight for feature alignment
    w_soft: float = 0.3       # Weight for soft-label distillation
    w_attn: float = 0.2       # Weight for attention transfer
    # Soft-label settings
    temperature: float = 4.0
    # Feature alignment settings
    normalize_features: bool = True
    # Attention transfer settings
    normalize_attention: bool = True


class CombinedDistillationLoss(nn.Module):
    """Weighted sum of all distillation loss components.

    Total loss = w_task * L_task + w_feat * L_feat + w_soft * L_soft + w_attn * L_attn

    The task loss (L_task) is computed by SFTTrainer directly; this module
    computes the additional distillation losses to be added.
    """

    def __init__(self, config: Optional[DistillationLossConfig] = None):
        super().__init__()
        self.config = config or DistillationLossConfig()

        self.feature_loss = FeatureAlignmentLoss(
            normalize=self.config.normalize_features,
        )
        self.soft_label_loss = SoftLabelLoss(
            temperature=self.config.temperature,
        )
        self.attention_loss = AttentionTransferLoss(
            normalize=self.config.normalize_attention,
        )

        logger.info(
            "CombinedDistillationLoss: w_task=%.2f, w_feat=%.2f, w_soft=%.2f, w_attn=%.2f",
            self.config.w_task,
            self.config.w_feat,
            self.config.w_soft,
            self.config.w_attn,
        )

    def forward(
        self,
        task_loss: torch.Tensor,
        student_features: Optional[Dict[str, torch.Tensor]] = None,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
        feature_pairs: Optional[List[Tuple[str, str]]] = None,
        projector_bank: Optional[nn.Module] = None,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        logit_mask: Optional[torch.Tensor] = None,
        student_attn_maps: Optional[Dict[str, torch.Tensor]] = None,
        teacher_attn_maps: Optional[Dict[str, torch.Tensor]] = None,
        attn_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined distillation loss.

        Args:
            task_loss: SFT cross-entropy loss from the trainer.
            student_features: Captured student intermediate features.
            teacher_features: Captured teacher intermediate features.
            feature_pairs: (student_name, teacher_name) pairs for L_feat.
            projector_bank: Projectors for dimension alignment.
            student_logits: Student LM head logits.
            teacher_logits: Teacher LM head logits.
            logit_mask: Mask for which tokens to include in L_soft.
            student_attn_maps: Student attention maps.
            teacher_attn_maps: Teacher attention maps.
            attn_pairs: (student_name, teacher_name) pairs for L_attn.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual loss values for logging.
        """
        device = task_loss.device
        loss_dict = {"task": task_loss.item()}
        total = self.config.w_task * task_loss

        # Feature alignment loss
        l_feat = torch.tensor(0.0, device=device)
        if (
            self.config.w_feat > 0
            and student_features
            and teacher_features
            and feature_pairs
        ):
            l_feat = self.feature_loss(
                student_features, teacher_features,
                projector_bank=projector_bank,
                pairs=feature_pairs,
            )
            total = total + self.config.w_feat * l_feat
        loss_dict["feat"] = l_feat.item()

        # Soft-label loss
        l_soft = torch.tensor(0.0, device=device)
        if (
            self.config.w_soft > 0
            and student_logits is not None
            and teacher_logits is not None
        ):
            l_soft = self.soft_label_loss(
                student_logits, teacher_logits, mask=logit_mask,
            )
            total = total + self.config.w_soft * l_soft
        loss_dict["soft"] = l_soft.item()

        # Attention transfer loss
        l_attn = torch.tensor(0.0, device=device)
        if (
            self.config.w_attn > 0
            and student_attn_maps
            and teacher_attn_maps
            and attn_pairs
        ):
            l_attn = self.attention_loss(
                student_attn_maps, teacher_attn_maps, pairs=attn_pairs,
            )
            total = total + self.config.w_attn * l_attn
        loss_dict["attn"] = l_attn.item()

        loss_dict["total"] = total.item()

        return total, loss_dict
