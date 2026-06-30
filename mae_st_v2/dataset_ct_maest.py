"""
dataset_ct_maest.py
CT NIfTI → MAE-ST video-format adapter.

Returns tensors of shape (1, T, H, W) suitable for MAE-ST pretraining,
treating the depth (Z) axis as the temporal axis.
"""

import random
from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# HU clipping range for CT
HU_MIN = -1024.0
HU_MAX = 3071.0


class CTVideoDataset(Dataset):
    """
    Loads a .nii.gz CT volume and returns a random contiguous window
    of `num_frames` axial slices, resized to `target_hw` × `target_hw`,
    as a float32 tensor of shape (1, T, H, W) in [0, 1].

    Args:
        file_list: Path to a plain-text file with one absolute NIfTI path
                   per line, OR a list of path strings.
        num_frames: Number of consecutive slices to sample (temporal dim T).
        target_hw: Spatial size each slice is resized to (square).
        augment: If True, apply random horizontal flip (consistent across all slices).
        hu_min / hu_max: HU clipping window.
    """

    def __init__(
        self,
        file_list: Union[str, Path, list],
        num_frames: int = 16,
        target_hw: int = 224,
        augment: bool = True,
        hu_min: float = HU_MIN,
        hu_max: float = HU_MAX,
    ):
        if isinstance(file_list, (str, Path)):
            with open(file_list, "r") as f:
                self.paths = [line.strip() for line in f if line.strip()]
        else:
            self.paths = [str(p) for p in file_list]

        self.num_frames = num_frames
        self.target_hw = target_hw
        self.augment = augment
        self.hu_min = hu_min
        self.hu_max = hu_max

    def __len__(self) -> int:
        return len(self.paths)

    def _load_volume(self, path: str) -> np.ndarray:
        """Load NIfTI, return float32 array of shape (D, H, W)."""
        img = nib.load(path)
        # Standard FLARE storage: (X, Y, Z) → transpose to (Z, Y, X) = (D, H, W)
        vol = np.asarray(img.dataobj, dtype=np.float32).transpose(2, 0, 1)
        return vol

    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        """Clip HU and scale to [0, 1]."""
        vol = np.clip(vol, self.hu_min, self.hu_max)
        vol = (vol - self.hu_min) / (self.hu_max - self.hu_min)
        return vol

    def _sample_frames(self, vol: np.ndarray) -> np.ndarray:
        """
        Random contiguous crop of `num_frames` slices from depth axis.
        If volume is shorter than num_frames, tile/repeat it first.
        Returns array of shape (T, H, W).
        """
        D = vol.shape[0]
        T = self.num_frames

        # Handle volumes shallower than T by repeating
        if D < T:
            repeats = (T // D) + 1
            vol = np.tile(vol, (repeats, 1, 1))[:T]
            D = T

        # Random contiguous window
        start = random.randint(0, D - T)
        frames = vol[start : start + T]  # (T, H, W)
        return frames

    def _resize_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Resize each slice (H, W) → (target_hw, target_hw) using bilinear interp.
        Input: (T, H, W) numpy float32
        Output: (1, T, target_hw, target_hw) torch float32
        """
        # (T, H, W) → (1, T, H, W) for F.interpolate
        t = torch.from_numpy(frames).unsqueeze(0)
        t = F.interpolate(
            t.unsqueeze(0),  # (1, 1, T, H, W)
            size=(self.num_frames, self.target_hw, self.target_hw),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)  # (1, T, H, W)
        return t

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.paths[idx]

        try:
            vol = self._load_volume(path)
            vol = self._normalize(vol)
            frames = self._sample_frames(vol)  # (T, H, W) float32

            # Resize if needed
            H, W = frames.shape[1], frames.shape[2]
            if H != self.target_hw or W != self.target_hw:
                # (T, H, W) → (1, T, H, W) for interpolate, then back
                t = torch.from_numpy(frames).unsqueeze(0).unsqueeze(0)  # (1,1,T,H,W)
                t = F.interpolate(
                    t,
                    size=(self.num_frames, self.target_hw, self.target_hw),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)  # (1, T, H, W)
            else:
                t = torch.from_numpy(frames).unsqueeze(0)  # (1, T, H, W)

            # --- Augmentations (consistent across all slices) ---
            if self.augment:
                # Random horizontal flip (50%)
                if random.random() < 0.5:
                    t = torch.flip(t, dims=[-1])  # flip W axis

                # Random vertical flip (50%)
                if random.random() < 0.5:
                    t = torch.flip(t, dims=[-2])  # flip H axis

                # Random 90-degree axial rotation (25% chance each: 0/90/180/270)
                k = random.randint(0, 3)
                if k > 0:
                    t = torch.rot90(t, k, dims=[-2, -1])

                # Random intensity shift/scale
                shift = random.uniform(-0.05, 0.05)
                scale = random.uniform(0.95, 1.05)
                t = t * scale + shift
                t = t.clamp(0.0, 1.0)

                # Gaussian noise (30% chance, std=0.02)
                if random.random() < 0.3:
                    noise = torch.randn_like(t) * 0.02
                    t = (t + noise).clamp(0.0, 1.0)

        except Exception as e:
            # Fallback: return a zero tensor so training doesn't crash on a bad file
            print(f"[CTVideoDataset] Error loading {path}: {e}")
            t = torch.zeros(1, self.num_frames, self.target_hw, self.target_hw)

        # Label is ignored during MAE pretraining; return 0 as placeholder
        return t, 0


if __name__ == "__main__":
    import sys

    file_list = sys.argv[1] if len(sys.argv) > 1 else "train_files.txt"
    ds = CTVideoDataset(file_list, num_frames=16, target_hw=224, augment=True)
    print(f"Dataset size: {len(ds)}")
    x, label = ds[0]
    print(f"Sample shape: {x.shape}, min={x.min():.3f}, max={x.max():.3f}, dtype={x.dtype}")
    assert x.shape == (1, 16, 224, 224), f"Unexpected shape: {x.shape}"
    print("OK")
