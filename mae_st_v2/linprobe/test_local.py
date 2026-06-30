#!/usr/bin/env python3
"""
Local test of predict.py logic — simulates Docker environment.
Tests both ROI and non-ROI modes with test_demo data.
"""

import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mae_st_v2.linprobe.ct_utils import (
    build_feature_extractor,
    extract_volume_features,
    extract_volume_features_with_mask,
    load_and_preprocess_volume,
    load_mask,
)

CHECKPOINT = "/scratch5/gautham/cvpr/checkpoints/encoder_checkpoint.pth"
CLASSIFIERS = "/scratch5/gautham/cvpr/linprobe_output/classifiers.pkl"
TEST_INPUT = "/scratch5/gautham/cvpr/docker_test/inputs"
TEST_OUTPUT = "/scratch5/gautham/cvpr/docker_test/outputs"
TEST_MASKS = "/scratch5/gautham/cvpr/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/test_demo/fg_masks/adrenal_hyperplasia"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model
print("Loading feature extractor...")
model = build_feature_extractor(CHECKPOINT, device=device)

# Load classifiers
print("Loading classifiers...")
with open(CLASSIFIERS, "rb") as f:
    classifiers = pickle.load(f)
print(f"Loaded {len(classifiers)} classifiers")

os.makedirs(TEST_OUTPUT, exist_ok=True)

# --- Test 1: ROI mode (adrenal_hyperplasia) ---
print("\n=== Test 1: ROI mode (adrenal_hyperplasia) ===")
disease = "adrenal_hyperplasia"
clf = classifiers[disease]
mask_path = Path(TEST_MASKS) / "amos_0004.nii.gz"
img_path = Path(TEST_INPUT) / "amos_0004.nii.gz"

vol = load_and_preprocess_volume(img_path)
mask_vol = load_mask(mask_path)
print(f"Volume shape: {vol.shape}, Mask shape: {mask_vol.shape}")

feat = extract_volume_features_with_mask(model, vol, mask_vol, device=device)
feat_np = normalize(feat.unsqueeze(0).numpy(), norm="l2")
prob = clf.predict_proba(feat_np)[0, 1]
print(f"amos_0004 — {disease} probability: {prob:.6f}")

# Write output
with open(f"{TEST_OUTPUT}/{disease}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case_id", "probability"])
    w.writerow(["amos_0004", f"{prob:.6f}"])
print(f"Wrote {TEST_OUTPUT}/{disease}.csv")

# --- Test 2: Non-ROI mode (all 4 diseases) ---
print("\n=== Test 2: Non-ROI mode ===")
NON_ROI = ["ascites", "atherosclerosis", "colorectal_cancer", "lymphadenopathy"]

feat_global = extract_volume_features(model, vol, device=device)
feat_global_np = normalize(feat_global.unsqueeze(0).numpy(), norm="l2")

for d in NON_ROI:
    clf = classifiers[d]
    prob = clf.predict_proba(feat_global_np)[0, 1]
    print(f"amos_0004 — {d} probability: {prob:.6f}")

    with open(f"{TEST_OUTPUT}/{d}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "probability"])
        w.writerow(["amos_0004", f"{prob:.6f}"])

print(f"\nAll outputs:")
for f in sorted(Path(TEST_OUTPUT).glob("*.csv")):
    print(f"  {f.name}")

print("\nDone! Local test passed.")
