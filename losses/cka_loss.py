"""CKA (Centered Kernel Alignment) loss for knowledge distillation.

CKA measures structural similarity between two feature representations.
Unlike MSE which compares element-wise, CKA compares the relational
structure (which samples are similar to each other in each space).

NOTE: Linear CKA degenerates when B << D (CKA -> 1.0 for any features).
To address this, we apply random projection to reduce D before computing
CKA when the feature dimension is large relative to batch size.

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _centering_matrix_hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) with centering.

    HSIC(K, L) = trace(K @ H @ L @ H) / (n-1)^2
    where H = I - (1/n) * ones is the centering matrix.

    Instead of explicitly constructing H, we center K and L first.
    """
    n = K.shape[0]
    row_mean = K.mean(dim=1, keepdim=True)
    col_mean = K.mean(dim=0, keepdim=True)
    grand_mean = K.mean()
    K_c = K - row_mean - col_mean + grand_mean

    row_mean = L.mean(dim=1, keepdim=True)
    col_mean = L.mean(dim=0, keepdim=True)
    grand_mean = L.mean()
    L_c = L - row_mean - col_mean + grand_mean

    return (K_c * L_c).sum() / max((n - 1) ** 2, 1)


def _random_project(
    X: torch.Tensor,
    target_dim: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply random projection to reduce feature dimension (Johnson-Lindenstrauss).

    Args:
        X: [B, D] input features.
        target_dim: Target dimension (should be << D).
        generator: Random generator for reproducibility within a step.

    Returns:
        [B, target_dim] projected features.
    """
    D = X.shape[-1]
    if D <= target_dim:
        return X

    proj = torch.randn(D, target_dim, device=X.device, dtype=X.dtype, generator=generator)
    proj = proj / math.sqrt(target_dim)
    return X @ proj


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute linear CKA between two feature matrices.

    Args:
        X: [B, D1] feature matrix from model 1.
        Y: [B, D2] feature matrix from model 2.

    Returns:
        Scalar CKA value in [0, 1]. Higher means more similar structure.
    """
    K = X @ X.T  # [B, B]
    L = Y @ Y.T  # [B, B]

    hsic_kl = _centering_matrix_hsic(K, L)
    hsic_kk = _centering_matrix_hsic(K, K)
    hsic_ll = _centering_matrix_hsic(L, L)

    denom = torch.sqrt(hsic_kk * hsic_ll).clamp(min=1e-8)
    return hsic_kl / denom


def linear_cka_loss(
    X: torch.Tensor,
    Y: torch.Tensor,
    proj_dim: int = 128,
) -> torch.Tensor:
    """CKA-based distillation loss with random projection for high-D features.

    When feature dim >> batch size, linear CKA degenerates (approaches 1.0
    for any features). Random projection to proj_dim dimensions preserves
    pairwise distances (Johnson-Lindenstrauss) while making CKA discriminative.

    Args:
        X: [B, D1] student features (B >= 2 required).
        Y: [B, D2] teacher features.
        proj_dim: Dimension to project to before CKA. Set to 0 to disable.

    Returns:
        Scalar loss in [0, 1].
    """
    if X.shape[0] < 2:
        return torch.tensor(0.0, device=X.device, requires_grad=True)

    X_f = X.float()
    Y_f = Y.float()

    # Apply random projection if features are high-dimensional
    if proj_dim > 0 and (X_f.shape[-1] > proj_dim or Y_f.shape[-1] > proj_dim):
        gen = torch.Generator(device=X.device)
        gen.manual_seed(42)
        X_f = _random_project(X_f, proj_dim, gen)
        gen.manual_seed(42)
        Y_f = _random_project(Y_f, proj_dim, gen)

    cka_val = linear_cka(X_f, Y_f)
    return 1.0 - cka_val
