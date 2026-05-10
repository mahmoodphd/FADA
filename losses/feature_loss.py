"""Feature alignment distillation loss (L_feat).

Computes MSE loss between projected student features and teacher features
at matched intermediate layers.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlignmentLoss(nn.Module):
    """MSE-based feature alignment loss across matched layer pairs.

    For each (student_feat, teacher_feat) pair:
        L = MSE(projector(student_feat), teacher_feat)

    Spatial dimensions are pooled if they don't match.
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: If True, L2-normalize features before computing loss.
                This stabilizes training when teacher/student scales differ.
        """
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        projector_bank: Optional[nn.Module] = None,
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> torch.Tensor:
        """Compute feature alignment loss.

        Args:
            student_features: {hook_name: tensor} from student.
            teacher_features: {hook_name: tensor} from teacher.
            projector_bank: ProjectorBank for dimension alignment.
            pairs: List of (student_name, teacher_name) pairs.

        Returns:
            Scalar loss tensor.
        """
        if pairs is None:
            return torch.tensor(0.0, device=next(iter(student_features.values())).device)

        total_loss = torch.tensor(0.0)
        count = 0

        for s_name, t_name in pairs:
            s_feat = student_features.get(s_name)
            t_feat = teacher_features.get(t_name)

            if s_feat is None or t_feat is None:
                continue

            # Project student features if projector is available
            proj_name = f"{s_name}_to_{t_name}"
            if projector_bank is not None and hasattr(projector_bank, "project"):
                try:
                    s_feat = projector_bank.project(proj_name, s_feat)
                except KeyError:
                    pass  # Dims already match, no projector needed

            # Handle spatial dimension mismatch via adaptive pooling
            s_feat, t_feat = self._align_spatial(s_feat, t_feat)

            if self.normalize:
                s_feat = F.normalize(s_feat, dim=-1)
                t_feat = F.normalize(t_feat, dim=-1)

            loss = F.mse_loss(s_feat.float(), t_feat.float())
            total_loss = total_loss.to(loss.device) + loss
            count += 1

        if count == 0:
            device = next(iter(student_features.values())).device
            return torch.tensor(0.0, device=device)

        return total_loss / count

    @staticmethod
    def _align_spatial(
        s: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align spatial dimensions between student and teacher features.

        Handles cases where feature tensors have different sequence lengths
        (e.g., different patch grid sizes).
        """
        if s.shape == t.shape:
            return s, t

        # Both are 2D [B, D] (already mean-pooled) -> no alignment needed
        if s.dim() == 2 and t.dim() == 2:
            return s, t

        # Both are [B, N, D] with different N -> pool to match
        if s.dim() == 3 and t.dim() == 3 and s.shape[0] == t.shape[0]:
            if s.shape[1] != t.shape[1]:
                # Adaptive average pool along sequence dimension
                # Interpolate student to match teacher's seq length
                s = s.permute(0, 2, 1)  # [B, D, N_s]
                s = F.adaptive_avg_pool1d(s, t.shape[1])  # [B, D, N_t]
                s = s.permute(0, 2, 1)  # [B, N_t, D]

        return s, t
