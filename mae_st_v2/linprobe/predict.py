#!/usr/bin/env python3
"""
predict.py — Docker feature extraction for CVPR 2026 CT competition Task 1 (LP).

The Docker container extracts feature embeddings from frozen ViT encoder
and saves them as .h5 files. The competition organizers then run their own
linear probing pipeline on these features.

Output format per volume: {case_id}.h5 containing dataset 'y_hat' with shape (embed_dim,)

Reads MASKS_DIR env var to determine mode:
  - If set (ROI mode): extract mask-guided features for volumes in /workspace/inputs/
  - If unset (non-ROI mode): extract global features for volumes in /workspace/inputs/

Input:  /workspace/inputs/*.nii.gz
Output: /workspace/outputs/{case_id}.h5
"""

import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mae_st_v2.linprobe.ct_utils import (
    build_feature_extractor,
    extract_volume_features,
    extract_volume_features_with_mask,
    load_and_preprocess_volume,
    load_mask,
)

# Paths inside Docker container
INPUT_DIR = Path("/workspace/inputs")
OUTPUT_DIR = Path("/workspace/outputs")
CHECKPOINT_PATH = SCRIPT_DIR / "encoder_checkpoint.pth"


def get_case_id(filename):
    return Path(filename).name.replace(".nii.gz", "")


def save_features_h5(case_id, features, output_dir):
    """Save feature vector as HDF5 file with 'y_hat' dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}.h5"
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("y_hat", data=features.numpy())
    return output_path


def extract_roi_mode(model, masks_dir, device):
    """ROI mode: extract mask-guided features for each volume."""
    masks_dir = Path(masks_dir)

    mask_files = sorted(masks_dir.glob("*.nii.gz"))
    if not mask_files:
        mask_files = sorted(masks_dir.rglob("*.nii.gz"))

    print(f"ROI mode: {len(mask_files)} masks in {masks_dir}")

    for mask_path in mask_files:
        case_id = get_case_id(mask_path.name)
        img_path = INPUT_DIR / mask_path.name

        if not img_path.exists():
            print(f"  WARNING: image not found for {case_id}, skipping")
            continue

        t0 = time.time()
        vol = load_and_preprocess_volume(img_path)
        mask_vol = load_mask(mask_path)
        feat = extract_volume_features_with_mask(model, vol, mask_vol, device=device)

        out_path = save_features_h5(case_id, feat, OUTPUT_DIR)
        elapsed = time.time() - t0
        print(f"  {case_id} — {elapsed:.1f}s — saved to {out_path}")


def extract_non_roi_mode(model, device):
    """Non-ROI mode: extract global features for all volumes."""
    nifti_files = sorted(INPUT_DIR.glob("*.nii.gz"))
    print(f"Non-ROI mode: {len(nifti_files)} volumes")

    for path in nifti_files:
        case_id = get_case_id(path.name)
        t0 = time.time()
        vol = load_and_preprocess_volume(path)
        feat = extract_volume_features(model, vol, device=device)

        out_path = save_features_h5(case_id, feat, OUTPUT_DIR)
        elapsed = time.time() - t0
        print(f"  {case_id} — {elapsed:.1f}s — saved to {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("Loading feature extractor...")
    model = build_feature_extractor(str(CHECKPOINT_PATH), device=device)

    # Determine mode
    masks_dir = os.environ.get("MASKS_DIR", "").strip()

    if masks_dir:
        extract_roi_mode(model, masks_dir, device)
    else:
        extract_non_roi_mode(model, device)

    print("Done!")


if __name__ == "__main__":
    main()
