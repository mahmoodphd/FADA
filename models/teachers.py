"""Teacher model registry and loaders.

Manages loading of 4 foundation teacher models:
- FetalCLIP: CLIP-based model pre-trained on fetal ultrasound (open_clip)
- UltraFedFM: Federated foundation model for ultrasound (MAE ViT-Base)
- UltraSAM: SAM variant fine-tuned for ultrasound segmentation (ViTSAM)
- USF-MAE: Masked autoencoder for ultrasound features (MAE ViT-Base)

All teachers are loaded in eval mode with frozen weights.
On a 24GB GPU, all 4 teachers can coexist (~12-14GB total with student).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class TeacherSpec:
    """Specification for a single teacher model."""
    name: str
    model_path: str
    model_type: str  # "clip", "sam", "mae", "vit"
    feature_dim: int  # Output feature dimension from the encoder
    load_fn: Optional[str] = None  # Custom loader function name
    device: str = "cuda"
    dtype: str = "float16"  # "float16" or "bfloat16"


# Default teacher specifications
DEFAULT_TEACHERS: Dict[str, TeacherSpec] = {
    "fetal_clip": TeacherSpec(
        name="fetal_clip",
        model_path="",  # Set via config
        model_type="clip",
        feature_dim=768,
    ),
    "ultra_fed_fm": TeacherSpec(
        name="ultra_fed_fm",
        model_path="",
        model_type="mae",
        feature_dim=768,
    ),
    "ultra_sam": TeacherSpec(
        name="ultra_sam",
        model_path="",
        model_type="sam",
        feature_dim=256,
    ),
    "usf_mae": TeacherSpec(
        name="usf_mae",
        model_path="",
        model_type="mae",
        feature_dim=768,
    ),
}


# ---------------------------------------------------------------------------
# Lightweight SAM ViT backbone (avoids mmcv/mmpretrain dependency)
# ---------------------------------------------------------------------------

class _LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)


def _window_unpartition(
    windows: torch.Tensor, window_size: int, H: int, W: int,
) -> torch.Tensor:
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


def _get_rel_pos(
    q_size: int, k_size: int, rel_pos: torch.Tensor,
) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None]
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :]
    relative_coords = (q_coords - k_coords) + (k_size - 1)
    return rel_pos_resized[relative_coords.long()]


def _add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = _get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = _get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    )
    return attn.view(B, q_h * q_w, k_h * k_w)


class _SAMAttention(nn.Module):
    """Multi-head attention with decomposed relative positional encoding."""

    def __init__(self, dim: int, num_heads: int = 12, rel_pos_size: int = 27):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_pos_size, self.head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_pos_size, self.head_dim))

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = _add_decomposed_rel_pos(
            attn.reshape(B * self.num_heads, N, N),
            q.reshape(B * self.num_heads, N, self.head_dim),
            self.rel_pos_h,
            self.rel_pos_w,
            hw,
            hw,
        ).reshape(B, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _SAMFFN(nn.Module):
    """FFN matching mmpretrain key format: layers.0.0 (Linear+GELU) + layers.1 (Linear)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU()),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _SAMBlock(nn.Module):
    """Transformer block with optional windowed attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        window_size: int = 0,
        rel_pos_size: int = 27,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _SAMAttention(dim, num_heads, rel_pos_size)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = _SAMFFN(dim, int(dim * mlp_ratio))
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        shortcut = x
        x = self.ln1(x)

        if self.window_size > 0:
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            Hp, Wp = x.shape[1], x.shape[2]
            x = _window_partition(x, self.window_size)
            hw = (self.window_size, self.window_size)
        else:
            Hp, Wp = H, W
            hw = (H, W)

        x = x.view(x.shape[0], -1, C)
        x = self.attn(x, hw)

        if self.window_size > 0:
            x = x.view(-1, self.window_size, self.window_size, C)
            x = _window_unpartition(x, self.window_size, Hp, Wp)
            if Hp > H or Wp > W:
                x = x[:, :H, :W, :]
        else:
            x = x.view(B, H, W, C)

        x = shortcut + x
        x = x + self.ffn(self.ln2(x))
        return x


class _SAMPatchEmbed(nn.Module):
    """Patch embedding matching mmpretrain key format (uses .projection)."""

    def __init__(self, in_channels: int, embed_dim: int, kernel_size: int):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class SAMViTBackbone(nn.Module):
    """Lightweight SAM ViT backbone matching mmpretrain ViTSAM state_dict.

    Implements the backbone / image-encoder portion of UltraSAM without
    requiring the full mmcv/mmpretrain stack.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_channels: int = 256,
        window_size: int = 14,
        global_attn_indices: Tuple[int, ...] = (2, 5, 8, 11),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        feat_size = img_size // patch_size

        self.patch_embed = _SAMPatchEmbed(3, embed_dim, patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, feat_size, feat_size, embed_dim),
        )

        self.layers = nn.ModuleList()
        for i in range(depth):
            is_global = i in global_attn_indices
            ws = 0 if is_global else window_size
            rps = 2 * feat_size - 1 if is_global else 2 * window_size - 1
            self.layers.append(
                _SAMBlock(embed_dim, num_heads, mlp_ratio, ws, rps),
            )

        self.channel_reduction = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, 1, bias=False),
            _LayerNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _LayerNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)              # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)            # (B, H, W, C)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = x.permute(0, 3, 1, 2)            # (B, C, H, W)
        x = self.channel_reduction(x)         # (B, out_ch, H, W)
        return x


# ---------------------------------------------------------------------------
# MAE Encoder (uses timm components directly, no external repo import)
# ---------------------------------------------------------------------------

class MAEEncoder(nn.Module):
    """MAE ViT-Base encoder built from timm components.

    Matches the state_dict key format of MaskedAutoencoderViT from
    UltraFedFM/USF-MAE checkpoints (encoder portion only).
    Decoder keys are ignored when loading with strict=False.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        from timm.models.vision_transformer import PatchEmbed, Block

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


# ---------------------------------------------------------------------------
# TeacherModel wrapper
# ---------------------------------------------------------------------------

class TeacherModel:
    """Wrapper around a loaded teacher model.

    Provides a uniform interface for feature extraction regardless
    of the underlying model architecture.
    """

    def __init__(self, spec: TeacherSpec, model: nn.Module, transform: Optional[Callable] = None):
        self.spec = spec
        self.model = model
        self.transform = transform
        self._feature_dim = spec.feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def name(self) -> str:
        return self.spec.name

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of images.

        Args:
            images: Batch of images [B, C, H, W] in the teacher's expected format.

        Returns:
            Feature tensor from the teacher's encoder.
        """
        if self.transform is not None:
            images = self.transform(images)

        if self.spec.model_type == "clip":
            return self._extract_clip_features(images)
        elif self.spec.model_type == "sam":
            return self._extract_sam_features(images)
        elif self.spec.model_type == "mae":
            return self._extract_mae_features(images)
        else:
            return self._extract_vit_features(images)

    def _extract_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual features from a CLIP model."""
        if hasattr(self.model, "encode_image"):
            return self.model.encode_image(images)
        if hasattr(self.model, "visual"):
            return self.model.visual(images)
        # Model is the visual encoder itself
        return self.model(images)

    def _extract_sam_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image encoder features from a SAM model."""
        if hasattr(self.model, "image_encoder"):
            return self.model.image_encoder(images)
        # Model is the image encoder / backbone itself
        return self.model(images)

    def _extract_mae_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from an MAE encoder (without masking)."""
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(images)
        return self.model(images)

    def _extract_vit_features(self, images: torch.Tensor) -> torch.Tensor:
        """Generic ViT feature extraction."""
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(images)
        return self.model(images)


# ---------------------------------------------------------------------------
# TeacherRegistry
# ---------------------------------------------------------------------------

class TeacherRegistry:
    """Registry for managing multiple teacher models.

    Handles loading, caching, and providing a unified interface
    for extracting features from multiple teachers.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._teachers: Dict[str, TeacherModel] = {}
        self._loaders: Dict[str, Callable] = {}
        self._register_default_loaders()

    def _register_default_loaders(self) -> None:
        self._loaders["clip"] = self._load_clip_model
        self._loaders["sam"] = self._load_sam_model
        self._loaders["mae"] = self._load_mae_model
        self._loaders["vit"] = self._load_vit_model

    def register_loader(self, model_type: str, loader_fn: Callable) -> None:
        self._loaders[model_type] = loader_fn

    def load_teacher(self, spec: TeacherSpec) -> TeacherModel:
        """Load a teacher model from its specification."""
        if spec.name in self._teachers:
            logger.info("Teacher '%s' already loaded, using cached", spec.name)
            return self._teachers[spec.name]

        logger.info("Loading teacher: %s (type=%s)", spec.name, spec.model_type)

        loader = self._loaders.get(spec.model_type)
        if loader is None:
            raise ValueError(
                f"No loader registered for model type '{spec.model_type}'. "
                f"Available: {list(self._loaders.keys())}"
            )

        model, transform = loader(spec)

        # Freeze and move to device
        dtype = torch.float16 if spec.dtype == "float16" else torch.bfloat16
        model = model.to(device=self.device, dtype=dtype)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        teacher = TeacherModel(spec, model, transform)
        self._teachers[spec.name] = teacher

        param_count = sum(p.numel() for p in model.parameters())
        mem_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)
        logger.info(
            "  %s loaded: %d params, ~%.0f MB VRAM",
            spec.name, param_count, mem_mb,
        )

        return teacher

    def load_all(self, specs: List[TeacherSpec]) -> Dict[str, TeacherModel]:
        result = {}
        for spec in specs:
            try:
                result[spec.name] = self.load_teacher(spec)
            except Exception:
                logger.exception("Failed to load teacher: %s", spec.name)
        return result

    def get_teacher(self, name: str) -> Optional[TeacherModel]:
        return self._teachers.get(name)

    @property
    def teachers(self) -> Dict[str, TeacherModel]:
        return dict(self._teachers)

    def total_vram_mb(self) -> float:
        total = 0.0
        for t in self._teachers.values():
            total += sum(
                p.numel() * p.element_size() for p in t.model.parameters()
            ) / (1024 * 1024)
        return total

    # --- Loaders ---

    def _load_clip_model(
        self, spec: TeacherSpec,
    ) -> Tuple[nn.Module, Optional[Callable]]:
        """Load FetalCLIP via open_clip with custom config registration."""
        import open_clip

        # Look for FetalCLIP config JSON alongside the weights file
        config_path = spec.model_path.replace("_weights.pt", "_config.json")
        if os.path.exists(config_path):
            logger.info("  Registering FetalCLIP config from %s", config_path)
            with open(config_path, "r") as f:
                config = json.load(f)
            open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config
            model, _, preprocess = open_clip.create_model_and_transforms(
                "FetalCLIP", pretrained=spec.model_path,
            )
            return model, preprocess

        # Fallback: generic open_clip model
        logger.info("  No FetalCLIP config found, trying generic ViT-B-16")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=spec.model_path,
        )
        return model.visual, preprocess

    def _load_sam_model(
        self, spec: TeacherSpec,
    ) -> Tuple[nn.Module, Optional[Callable]]:
        """Load UltraSAM backbone from mmpretrain-format checkpoint."""
        # Try mmpretrain first (if mmcv is available)
        try:
            from mmpretrain.models.backbones import ViTSAM
            logger.info("  Using mmpretrain ViTSAM backbone")
            backbone = ViTSAM(
                arch="base",
                img_size=1024,
                patch_size=16,
                out_channels=256,
                use_abs_pos=True,
                use_rel_pos=True,
                window_size=14,
                out_indices=[11],
            )
            checkpoint = torch.load(
                spec.model_path, map_location="cpu", weights_only=False,
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            backbone_dict = {
                k.replace("backbone.", ""): v
                for k, v in state_dict.items()
                if k.startswith("backbone.")
            }
            backbone.load_state_dict(backbone_dict, strict=False)
            return backbone, None
        except ImportError:
            pass

        # Fallback: lightweight standalone SAM ViT backbone
        logger.info("  mmpretrain not available, using standalone SAMViTBackbone")
        backbone = SAMViTBackbone(
            img_size=1024,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            out_channels=256,
            window_size=14,
            global_attn_indices=(2, 5, 8, 11),
        )
        checkpoint = torch.load(
            spec.model_path, map_location="cpu", weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        backbone_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        missing, unexpected = backbone.load_state_dict(backbone_dict, strict=False)
        if missing:
            logger.warning("  SAM backbone missing keys: %s", missing)
        if unexpected:
            logger.warning("  SAM backbone unexpected keys: %s", unexpected)
        return backbone, None

    def _load_mae_model(
        self, spec: TeacherSpec,
    ) -> Tuple[nn.Module, Optional[Callable]]:
        """Load MAE ViT-Base encoder from UltraFedFM or USF-MAE checkpoint."""
        encoder = MAEEncoder(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
        )

        checkpoint = torch.load(
            spec.model_path, map_location="cpu", weights_only=False,
        )

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Raw state_dict (USF-MAE format)
            state_dict = checkpoint

        # Remove 'module.' prefix if present (DataParallel artifacts)
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }

        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        # Decoder keys are expected to be "unexpected" since we only load encoder
        decoder_unexpected = [k for k in unexpected if "decoder" in k or "mask_token" in k]
        real_unexpected = [k for k in unexpected if k not in decoder_unexpected]
        if missing:
            logger.warning("  MAE encoder missing keys: %s", missing)
        if real_unexpected:
            logger.warning("  MAE encoder unexpected keys: %s", real_unexpected)
        if decoder_unexpected:
            logger.debug(
                "  Skipped %d decoder keys (expected)", len(decoder_unexpected),
            )

        return encoder, None

    def _load_vit_model(
        self, spec: TeacherSpec,
    ) -> Tuple[nn.Module, Optional[Callable]]:
        """Load a generic ViT model via timm (fallback)."""
        try:
            import timm
            model = timm.create_model(
                "vit_base_patch16_224", pretrained=False, num_classes=0,
            )
            if spec.model_path:
                state_dict = torch.load(
                    spec.model_path, map_location="cpu", weights_only=True,
                )
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
                model.load_state_dict(state_dict, strict=False)
            return model, None
        except ImportError:
            pass

        return self._load_checkpoint(spec)

    def _load_checkpoint(
        self, spec: TeacherSpec,
    ) -> Tuple[nn.Module, Optional[Callable]]:
        """Generic checkpoint loading fallback."""
        if not spec.model_path:
            raise ValueError(f"No model path specified for teacher '{spec.name}'")

        checkpoint = torch.load(
            spec.model_path, map_location="cpu", weights_only=False,
        )

        if isinstance(checkpoint, nn.Module):
            return checkpoint, None

        if isinstance(checkpoint, dict):
            for key in ["model", "state_dict", "encoder", "visual"]:
                if key in checkpoint and isinstance(checkpoint[key], nn.Module):
                    return checkpoint[key], None

        raise ValueError(
            f"Cannot load teacher '{spec.name}' from {spec.model_path}. "
            "Provide a custom loader via register_loader()."
        )
