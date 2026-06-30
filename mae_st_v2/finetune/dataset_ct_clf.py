"""
dataset_ct_clf.py — Multi-disease CT classification dataset.

Reads 15 label CSVs and builds a unified (N, 15) label matrix.
Returns (volume_tensor, label_vector, label_mask) per sample.
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Same HU range as pretraining
HU_MIN = -1024.0
HU_MAX = 3071.0

ALL_DISEASES = [
    "adrenal_hyperplasia",
    "ascites",
    "atherosclerosis",
    "cholecystitis",
    "colorectal_cancer",
    "fatty_liver",
    "gallstone",
    "hydronephrosis",
    "kidney_stone",
    "liver_calcifications",
    "liver_cyst",
    "liver_lesion",
    "lymphadenopathy",
    "renal_cyst",
    "splenomegaly",
]


def load_all_labels(label_dir: str, split: str = "train") -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load all 15 label CSVs and build unified label matrix.

    Returns:
        case_ids: sorted list of unique case IDs in the requested split
        labels: (N, 15) float32 array, NaN for unlabeled entries
        mask: (N, 15) float32 array, 1.0 where labeled, 0.0 where NaN
    """
    label_dir = Path(label_dir)

    # Collect all case_id -> {disease_idx: label} mappings
    case_labels: Dict[str, Dict[int, int]] = {}

    for d_idx, disease in enumerate(ALL_DISEASES):
        csv_path = label_dir / f"{disease}.csv"
        if not csv_path.exists():
            print(f"Warning: label file not found: {csv_path}")
            continue

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"].strip() != split:
                    continue
                case_id = row["case_id"].replace(".nii.gz", "")
                label = int(row[disease])
                if case_id not in case_labels:
                    case_labels[case_id] = {}
                case_labels[case_id][d_idx] = label

    case_ids = sorted(case_labels.keys())
    N = len(case_ids)
    labels = np.full((N, 15), np.nan, dtype=np.float32)
    mask = np.zeros((N, 15), dtype=np.float32)

    for i, cid in enumerate(case_ids):
        for d_idx, label in case_labels[cid].items():
            labels[i, d_idx] = label
            mask[i, d_idx] = 1.0

    return case_ids, labels, mask


def compute_pos_weights(labels: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """Compute per-disease pos_weight for BCE loss from training labels."""
    pos_weights = []
    for d in range(15):
        valid = mask[:, d] > 0
        if valid.sum() == 0:
            pos_weights.append(1.0)
            continue
        y = labels[valid, d]
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0:
            pos_weights.append(1.0)
        else:
            pos_weights.append(n_neg / n_pos)
    return torch.tensor(pos_weights, dtype=torch.float32)


class CTClassificationDataset(Dataset):
    """
    Multi-disease CT classification dataset.

    Returns:
        tensor: (1, T, H, W) float32 volume in [0, 1]
        labels: (15,) float32 label vector (NaN where unlabeled)
        mask: (15,) float32 mask (1.0 where labeled)
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        split: str = "train",
        num_frames: int = 16,
        target_hw: int = 224,
        augment: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.num_frames = num_frames
        self.target_hw = target_hw
        self.augment = augment and (split == "train")

        # Load all labels
        self.case_ids, self.labels, self.label_mask = load_all_labels(label_dir, split)

        # Verify images exist
        valid_indices = []
        for i, cid in enumerate(self.case_ids):
            img_path = self.image_dir / f"{cid}.nii.gz"
            if img_path.exists():
                valid_indices.append(i)

        if len(valid_indices) < len(self.case_ids):
            print(f"[CTClassificationDataset] {len(self.case_ids) - len(valid_indices)} "
                  f"cases missing images, keeping {len(valid_indices)}")
            self.case_ids = [self.case_ids[i] for i in valid_indices]
            self.labels = self.labels[valid_indices]
            self.label_mask = self.label_mask[valid_indices]

        print(f"[CTClassificationDataset] split={split}, {len(self.case_ids)} samples, "
              f"labeled entries: {int(self.label_mask.sum())}/{ len(self.case_ids) * 15}")

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        case_id = self.case_ids[idx]
        img_path = self.image_dir / f"{case_id}.nii.gz"

        try:
            # Load and preprocess
            img = nib.load(str(img_path))
            vol = np.asarray(img.dataobj, dtype=np.float32).transpose(2, 0, 1)  # (D, H, W)
            vol = np.clip(vol, HU_MIN, HU_MAX)
            vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)

            # Sample frames
            D = vol.shape[0]
            T = self.num_frames
            if D < T:
                repeats = (T // D) + 1
                vol = np.tile(vol, (repeats, 1, 1))
                D = vol.shape[0]
            start = random.randint(0, D - T)
            frames = vol[start: start + T]

            # Resize
            H, W = frames.shape[1], frames.shape[2]
            if H != self.target_hw or W != self.target_hw:
                t = torch.from_numpy(frames).unsqueeze(0).unsqueeze(0)
                t = F.interpolate(
                    t,
                    size=(T, self.target_hw, self.target_hw),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
            else:
                t = torch.from_numpy(frames).unsqueeze(0)

            # Augmentations (same as pretraining)
            if self.augment:
                if random.random() < 0.5:
                    t = torch.flip(t, dims=[-1])
                if random.random() < 0.5:
                    t = torch.flip(t, dims=[-2])
                k = random.randint(0, 3)
                if k > 0:
                    t = torch.rot90(t, k, dims=[-2, -1])
                shift = random.uniform(-0.05, 0.05)
                scale = random.uniform(0.95, 1.05)
                t = (t * scale + shift).clamp(0.0, 1.0)
                if random.random() < 0.3:
                    t = (t + torch.randn_like(t) * 0.02).clamp(0.0, 1.0)

        except Exception as e:
            print(f"[CTClassificationDataset] Error loading {img_path}: {e}")
            t = torch.zeros(1, self.num_frames, self.target_hw, self.target_hw)

        label_vec = torch.from_numpy(self.labels[idx])
        mask_vec = torch.from_numpy(self.label_mask[idx])

        return t, label_vec, mask_vec
