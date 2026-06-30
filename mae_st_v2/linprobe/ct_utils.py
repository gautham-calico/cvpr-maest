"""
ct_utils.py — Shared CT loading, preprocessing, and frozen ViT feature extraction.

Reuses preprocessing logic from dataset_ct_maest.py and wraps the MAE-ST
VisionTransformer encoder for feature extraction without modifying models_vit.py.
"""

import sys
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# CT preprocessing constants (must match dataset_ct_maest.py exactly)
# ---------------------------------------------------------------------------
HU_MIN = -1024.0
HU_MAX = 3071.0
TARGET_HW = 224
NUM_FRAMES = 16  # sliding window depth


# ---------------------------------------------------------------------------
# Volume I/O
# ---------------------------------------------------------------------------

def load_and_preprocess_volume(path):
    """Load NIfTI, transpose (X,Y,Z)->(Z,Y,X), HU clip, normalize to [0,1]."""
    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=np.float32).transpose(2, 0, 1)  # (D, H, W)
    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    return vol


def load_mask(path):
    """Load NIfTI mask, transpose (X,Y,Z)->(Z,Y,X), return binary uint8."""
    img = nib.load(str(path))
    mask = np.asarray(img.dataobj, dtype=np.float32).transpose(2, 0, 1)
    return (mask > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# FeatureExtractor — wraps VisionTransformer for frozen feature extraction
# ---------------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    """
    Wraps a VisionTransformer and provides forward_features() that returns
    (B, embed_dim) or (B, 4*embed_dim) by running the encoder forward pass
    without dropout/head.

    Args:
        vit: nn.Module with patch_embed, blocks, norm, etc.
        multi_layer: If True, concatenate mean-pooled features from layers
                     [7, 15, 23] + final norm. Returns (B, 4*embed_dim).
    """

    def __init__(self, vit, multi_layer=False):
        super().__init__()
        self.vit = vit
        self.multi_layer = multi_layer
        # Layers to collect features from (0-indexed block indices)
        self.collect_layers = [7, 15, 23]

    @torch.no_grad()
    def forward_features(self, x):
        """
        x: (B, 1, T, H, W) — single-channel CT window
        Returns: (B, embed_dim) or (B, 4*embed_dim) if multi_layer=True
        """
        vit = self.vit

        # Patch embed
        x = vit.patch_embed(x)
        N, T, L, C = x.shape
        x = x.view(N, T * L, C)

        # Prepend CLS token
        if vit.cls_embed:
            cls_tokens = vit.cls_token.expand(N, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        if vit.sep_pos_embed:
            pos_embed = vit.pos_embed_spatial.repeat(
                1, vit.input_size[0], 1
            ) + torch.repeat_interleave(
                vit.pos_embed_temporal,
                vit.input_size[1] * vit.input_size[2],
                dim=1,
            )
            if vit.cls_embed:
                pos_embed = torch.cat(
                    [
                        vit.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            pos_embed = vit.pos_embed[:, :, :]
        x = x + pos_embed

        # Reshape if attention requires T-shape
        requires_t_shape = (
            len(vit.blocks) > 0
            and hasattr(vit.blocks[0].attn, "requires_t_shape")
            and vit.blocks[0].attn.requires_t_shape
        )
        if requires_t_shape:
            x = x.view(N, T, L, C)

        # Transformer blocks (with optional multi-layer collection)
        intermediate_feats = []
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            if self.multi_layer and i in self.collect_layers:
                if requires_t_shape:
                    feat = x.view(N, T * L + (1 if vit.cls_embed else 0), C)
                else:
                    feat = x
                intermediate_feats.append(feat[:, 1:, :].mean(dim=1))  # (B, C)

        if requires_t_shape:
            x = x.view(N, T * L + (1 if vit.cls_embed else 0), C)

        # Final LayerNorm + pool
        final = vit.norm(x[:, 1:, :].mean(dim=1))  # (B, C)

        if self.multi_layer:
            intermediate_feats.append(final)
            return torch.cat(intermediate_feats, dim=-1)  # (B, 4*C)
        else:
            return final


def build_feature_extractor(checkpoint_path, device="cuda", multi_layer=False):
    """
    Create a ViT-Large with CT config, load MAE encoder weights, freeze all params.

    Args:
        checkpoint_path: Path to MAE pretrain checkpoint (.pth)
        device: Device to place model on
        multi_layer: If True, concatenate features from layers [7,15,23]+final
                     yielding 4096-dim features instead of 1024.

    Returns:
        FeatureExtractor on device, in eval mode with all params frozen.
    """
    # Import locally to avoid issues with mae_st.util.logging print override
    # when running as standalone script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from mae_st_v2.util.video_vit import Attention, Block, PatchEmbed

    vit = _build_vit_large(PatchEmbed, Block, Attention)
    _load_mae_encoder_weights(vit, checkpoint_path)

    # Freeze all parameters
    for p in vit.parameters():
        p.requires_grad = False

    model = FeatureExtractor(vit, multi_layer=multi_layer)
    model.to(device)
    model.eval()
    return model


def _build_vit_large(PatchEmbed, Block, Attention):
    """Construct ViT-Large matching our MAE-ST pretraining config."""
    from functools import partial

    embed_dim = 1024
    depth = 24
    num_heads = 16
    mlp_ratio = 4.0
    norm_layer = partial(nn.LayerNorm, eps=1e-6)
    num_frames = NUM_FRAMES
    t_patch_size = 4
    img_size = TARGET_HW
    patch_size = 16
    in_chans = 1

    patch_embed = PatchEmbed(
        img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size
    )
    num_patches = patch_embed.num_patches
    input_size = patch_embed.input_size

    # Build the ViT manually (same as VisionTransformer.__init__ but without
    # dropout/head which we don't need for feature extraction)
    vit = nn.Module()
    vit.patch_embed = patch_embed
    vit.input_size = input_size
    vit.cls_embed = True
    vit.sep_pos_embed = True

    vit.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    vit.pos_embed_spatial = nn.Parameter(
        torch.zeros(1, input_size[1] * input_size[2], embed_dim)
    )
    vit.pos_embed_temporal = nn.Parameter(
        torch.zeros(1, input_size[0], embed_dim)
    )
    vit.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))

    vit.blocks = nn.ModuleList(
        [
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer,
                drop_path=0.0,
                attn_func=partial(Attention, input_size=input_size),
            )
            for _ in range(depth)
        ]
    )
    vit.norm = norm_layer(embed_dim)

    return vit


def _load_mae_encoder_weights(vit, checkpoint_path):
    """Load encoder-only weights from MAE checkpoint, filtering decoder/mask_token keys."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Handle both 'model' and 'model_state' keys
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    # Filter out decoder and mask_token keys
    encoder_state = {}
    skip_prefixes = ("decoder_", "mask_token")
    skip_keys = {"decoder_embed.weight", "decoder_embed.bias",
                 "decoder_norm.weight", "decoder_norm.bias",
                 "decoder_pred.weight", "decoder_pred.bias",
                 "decoder_cls_token"}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in skip_prefixes):
            continue
        if k in skip_keys:
            continue
        # Also skip head weights (from finetuning checkpoints)
        if k.startswith("head."):
            continue
        if k.startswith("dropout."):
            continue
        encoder_state[k] = v

    msg = vit.load_state_dict(encoder_state, strict=False)
    print(f"Loaded MAE encoder weights: {msg}")


# ---------------------------------------------------------------------------
# Sliding window feature extraction
# ---------------------------------------------------------------------------

def _prepare_windows(vol, num_frames=NUM_FRAMES, target_hw=TARGET_HW, stride=None):
    """
    Split volume into sliding windows of num_frames slices with configurable stride.

    Args:
        vol: (D, H, W) numpy float32 array in [0, 1]
        num_frames: Window size (temporal dimension)
        target_hw: Spatial resize target
        stride: Step between windows. Default: num_frames // 2 (50% overlap).
                Set to num_frames for non-overlapping (legacy behavior).

    Returns:
        Tensor of shape (num_windows, 1, num_frames, target_hw, target_hw)
    """
    if stride is None:
        stride = num_frames // 2  # 50% overlap by default

    D, H, W = vol.shape

    # Pad depth so we have at least num_frames slices
    if D < num_frames:
        pad_d = num_frames - D
        vol = np.pad(vol, ((0, pad_d), (0, 0), (0, 0)), mode="edge")
        D = vol.shape[0]

    windows = []
    start = 0
    while start + num_frames <= D:
        window = vol[start : start + num_frames]  # (T, H, W)

        # Convert to tensor and resize spatially
        t = torch.from_numpy(window).unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
        if H != target_hw or W != target_hw:
            t = F.interpolate(
                t,
                size=(num_frames, target_hw, target_hw),
                mode="trilinear",
                align_corners=False,
            )
        windows.append(t)
        start += stride

    # If we haven't covered the last slices, add a final window ending at D
    if start - stride + num_frames < D and D >= num_frames:
        window = vol[D - num_frames : D]
        t = torch.from_numpy(window).unsqueeze(0).unsqueeze(0)
        if H != target_hw or W != target_hw:
            t = F.interpolate(
                t,
                size=(num_frames, target_hw, target_hw),
                mode="trilinear",
                align_corners=False,
            )
        windows.append(t)

    return torch.cat(windows, dim=0)  # (num_windows, 1, T, H, W)


@torch.no_grad()
def extract_volume_features(model, vol, batch_size=8, device="cuda"):
    """
    Non-overlapping sliding window over full depth, mean-pool across windows.

    Args:
        model: FeatureExtractor instance
        vol: (D, H, W) numpy float32 in [0, 1]
        batch_size: Number of windows to process at once
        device: CUDA device

    Returns:
        (embed_dim,) tensor — mean-pooled feature vector
    """
    windows = _prepare_windows(vol)  # (num_windows, 1, T, H, W)
    all_features = []

    for i in range(0, len(windows), batch_size):
        batch = windows[i : i + batch_size].to(device)
        feats = model.forward_features(batch)  # (B, embed_dim)
        all_features.append(feats.cpu())

    all_features = torch.cat(all_features, dim=0)  # (num_windows, embed_dim)
    return all_features.mean(dim=0)  # (embed_dim,)


@torch.no_grad()
def extract_volume_features_with_mask(model, vol, mask_vol, pad_slices=8,
                                       batch_size=8, device="cuda"):
    """
    Find depth range where mask is active, pad by ±pad_slices, crop volume,
    then apply sliding window extraction on the cropped region.

    Args:
        model: FeatureExtractor instance
        vol: (D, H, W) numpy float32 in [0, 1]
        mask_vol: (D, H, W) binary numpy uint8
        pad_slices: Number of context slices to add around mask range
        batch_size: Batch size for forward pass
        device: CUDA device

    Returns:
        (embed_dim,) tensor — mean-pooled feature vector from mask region
    """
    D = vol.shape[0]

    # Find active depth range
    active_slices = np.any(mask_vol, axis=(1, 2))  # (D,) bool
    if not active_slices.any():
        # Fallback to global features if mask is empty
        return extract_volume_features(model, vol, batch_size, device)

    active_indices = np.where(active_slices)[0]
    z_min = max(0, active_indices[0] - pad_slices)
    z_max = min(D, active_indices[-1] + 1 + pad_slices)

    # Ensure at least num_frames slices
    if z_max - z_min < NUM_FRAMES:
        center = (z_min + z_max) // 2
        z_min = max(0, center - NUM_FRAMES // 2)
        z_max = min(D, z_min + NUM_FRAMES)
        z_min = max(0, z_max - NUM_FRAMES)

    cropped = vol[z_min:z_max]
    return extract_volume_features(model, cropped, batch_size, device)
