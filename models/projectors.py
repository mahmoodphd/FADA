"""Lightweight MLP projectors for feature alignment.

Used to project student features to match teacher feature dimensions
for feature-level distillation loss computation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeatureProjector(nn.Module):
    """MLP projector to align feature dimensions between student and teacher.

    Architecture: Linear -> LayerNorm -> GELU -> Linear
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (student_dim + teacher_dim) // 2

        self.projector = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_dim),
        )
        self._student_dim = student_dim
        self._teacher_dim = teacher_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project student features to teacher dimension.

        Args:
            x: Student features [B, N, D_s] or [B, D_s].

        Returns:
            Projected features [B, N, D_t] or [B, D_t].
        """
        return self.projector(x)

    @property
    def student_dim(self) -> int:
        return self._student_dim

    @property
    def teacher_dim(self) -> int:
        return self._teacher_dim


class ProjectorBank(nn.Module):
    """Collection of projectors for multi-teacher, multi-layer distillation.

    Manages one projector per (teacher, layer) pair, allowing efficient
    feature alignment across multiple distillation pathways.
    """

    def __init__(self):
        super().__init__()
        self._projectors = nn.ModuleDict()

    def add_projector(
        self,
        name: str,
        student_dim: int,
        teacher_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> FeatureProjector:
        """Add a new projector.

        Args:
            name: Unique name for this projector (e.g., "fetal_clip_layer8").
            student_dim: Student feature dimension.
            teacher_dim: Teacher feature dimension.
            hidden_dim: Optional hidden layer dimension.

        Returns:
            The created projector.
        """
        proj = FeatureProjector(student_dim, teacher_dim, hidden_dim)
        self._projectors[name] = proj
        logger.info(
            "Added projector '%s': %d -> %d (hidden=%d)",
            name, student_dim, teacher_dim,
            hidden_dim or (student_dim + teacher_dim) // 2,
        )
        return proj

    def project(self, name: str, features: torch.Tensor) -> torch.Tensor:
        """Project features through a named projector.

        Args:
            name: Name of the projector to use.
            features: Input features to project.

        Returns:
            Projected features.
        """
        if name not in self._projectors:
            raise KeyError(f"Projector '{name}' not found. Available: {list(self._projectors.keys())}")
        return self._projectors[name](features)

    def setup_for_distillation(
        self,
        hook_pairs: List[Tuple[str, str]],
        student_dims: Dict[str, int],
        teacher_dims: Dict[str, int],
    ) -> None:
        """Automatically create projectors for all hook pairs.

        Args:
            hook_pairs: List of (student_hook_name, teacher_hook_name).
            student_dims: {hook_name: feature_dim} for student.
            teacher_dims: {hook_name: feature_dim} for teacher.
        """
        for s_name, t_name in hook_pairs:
            s_dim = student_dims.get(s_name)
            t_dim = teacher_dims.get(t_name)
            if s_dim is None or t_dim is None:
                logger.warning(
                    "Missing dims for pair (%s, %s): s=%s, t=%s",
                    s_name, t_name, s_dim, t_dim,
                )
                continue

            proj_name = f"{s_name}_to_{t_name}"
            if s_dim != t_dim:
                self.add_projector(proj_name, s_dim, t_dim)
            else:
                # Identity projector when dims match
                logger.info("Dims match for '%s', no projector needed", proj_name)

    @property
    def projector_names(self) -> List[str]:
        return list(self._projectors.keys())
