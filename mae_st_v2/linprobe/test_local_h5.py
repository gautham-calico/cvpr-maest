#!/usr/bin/env python3
"""Quick local test of H5 feature extraction."""
import sys, h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from mae_st_v2.linprobe.ct_utils import (
    build_feature_extractor, extract_volume_features,
    extract_volume_features_with_mask, load_and_preprocess_volume, load_mask,
)

CHECKPOINT = "/scratch5/gautham/cvpr/checkpoints/encoder_checkpoint.pth"
TEST_IMG = "/scratch5/gautham/cvpr/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/test_demo/amos_0004.nii.gz"
TEST_MASK = "/scratch5/gautham/cvpr/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/test_demo/fg_masks/adrenal_hyperplasia/amos_0004.nii.gz"
OUT_DIR = "/scratch5/gautham/cvpr/docker_test/h5_outputs"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = build_feature_extractor(CHECKPOINT, device=device)

# Test global features
vol = load_and_preprocess_volume(TEST_IMG)
feat = extract_volume_features(model, vol, device=device)
print(f"Global feature shape: {feat.shape}")

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
with h5py.File(f"{OUT_DIR}/amos_0004_global.h5", "w") as hf:
    hf.create_dataset("y_hat", data=feat.numpy())

# Test ROI features
mask_vol = load_mask(TEST_MASK)
feat_roi = extract_volume_features_with_mask(model, vol, mask_vol, device=device)
print(f"ROI feature shape: {feat_roi.shape}")

with h5py.File(f"{OUT_DIR}/amos_0004_roi.h5", "w") as hf:
    hf.create_dataset("y_hat", data=feat_roi.numpy())

# Verify readback
with h5py.File(f"{OUT_DIR}/amos_0004_global.h5", "r") as hf:
    loaded = hf["y_hat"][:]
    print(f"Loaded shape: {loaded.shape}, dtype: {loaded.dtype}")
    assert loaded.shape == (1024,), f"Expected (1024,), got {loaded.shape}"

print("All H5 tests passed!")
