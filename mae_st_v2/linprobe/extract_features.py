#!/usr/bin/env python3
"""
extract_features.py — Pre-extract and cache ViT features for all AMOS CT volumes.

Outputs:
  {output_dir}/global_features.pt   — dict {case_id: tensor(1024,)} (or 4096 with --multi_layer)
  {output_dir}/roi_features/{disease}.pt — dict {case_id: tensor(1024,)} (or 4096)

Usage:
  python -m mae_st.linprobe.extract_features \
      --checkpoint /path/to/checkpoint.pth \
      --image_dir /path/to/imagesTr \
      --mask_dir /path/to/masks \
      --output_dir /path/to/features
"""

import argparse
import os
import re
import time
from pathlib import Path

import torch

from mae_st_v2.linprobe.ct_utils import (
    build_feature_extractor,
    extract_volume_features,
    extract_volume_features_with_mask,
    load_and_preprocess_volume,
    load_mask,
)

# 11 diseases with ROI masks
ROI_DISEASES = [
    "adrenal_hyperplasia",
    "cholecystitis",
    "fatty_liver",
    "gallstone",
    "hydronephrosis",
    "kidney_stone",
    "liver_calcifications",
    "liver_cyst",
    "liver_lesion",
    "renal_cyst",
    "splenomegaly",
]


def get_case_id(filename):
    """Extract case ID from filename like 'amos_0001.nii.gz' -> 'amos_0001'."""
    return Path(filename).name.replace(".nii.gz", "")


def extract_global_features(model, image_dir, output_dir, device, batch_size=8):
    """Extract global (full-volume) features for all volumes."""
    image_dir = Path(image_dir)
    output_path = Path(output_dir) / "global_features.pt"

    # Load existing features if resuming
    if output_path.exists():
        features = torch.load(output_path)
        print(f"Resuming: {len(features)} global features already cached")
    else:
        features = {}

    nifti_files = sorted(image_dir.glob("*.nii.gz"))
    print(f"Found {len(nifti_files)} volumes in {image_dir}")

    for i, path in enumerate(nifti_files):
        case_id = get_case_id(path.name)
        if case_id in features:
            continue

        t0 = time.time()
        try:
            vol = load_and_preprocess_volume(path)
            feat = extract_volume_features(model, vol, batch_size=batch_size, device=device)
            features[case_id] = feat
        except Exception as e:
            print(f"  ERROR on {case_id}: {e}")
            continue

        elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(nifti_files)}] {case_id} — {elapsed:.1f}s — feat shape {feat.shape}")

        # Save checkpoint every 50 volumes
        if (i + 1) % 50 == 0:
            torch.save(features, output_path)

    torch.save(features, output_path)
    print(f"Saved {len(features)} global features to {output_path}")
    return features


def extract_roi_features(model, image_dir, mask_dir, output_dir, device, batch_size=8):
    """Extract ROI-guided features for each of the 11 ROI diseases."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    roi_dir = Path(output_dir) / "roi_features"
    roi_dir.mkdir(parents=True, exist_ok=True)

    for disease in ROI_DISEASES:
        disease_mask_dir = mask_dir / disease
        if not disease_mask_dir.exists():
            print(f"Skipping {disease}: mask dir {disease_mask_dir} not found")
            continue

        output_path = roi_dir / f"{disease}.pt"

        # Load existing features if resuming
        if output_path.exists():
            features = torch.load(output_path)
            print(f"Resuming {disease}: {len(features)} features already cached")
        else:
            features = {}

        mask_files = sorted(disease_mask_dir.glob("*.nii.gz"))
        print(f"\n{disease}: {len(mask_files)} masks found")

        for i, mask_path in enumerate(mask_files):
            case_id = get_case_id(mask_path.name)
            if case_id in features:
                continue

            # Find corresponding image
            img_path = image_dir / mask_path.name
            if not img_path.exists():
                print(f"  WARNING: image not found for {case_id}")
                continue

            t0 = time.time()
            try:
                vol = load_and_preprocess_volume(img_path)
                mask_vol = load_mask(mask_path)
                feat = extract_volume_features_with_mask(
                    model, vol, mask_vol, batch_size=batch_size, device=device
                )
                features[case_id] = feat
            except Exception as e:
                print(f"  ERROR on {case_id}: {e}")
                continue

            elapsed = time.time() - t0
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(mask_files)}] {case_id} — {elapsed:.1f}s")

            # Save checkpoint every 50 volumes
            if (i + 1) % 50 == 0:
                torch.save(features, output_path)

        torch.save(features, output_path)
        print(f"Saved {len(features)} ROI features for {disease}")


def main():
    parser = argparse.ArgumentParser(description="Pre-extract ViT features for linear probing")
    parser.add_argument("--checkpoint", required=True, help="Path to MAE pretrain checkpoint")
    parser.add_argument("--image_dir", required=True, help="Path to imagesTr directory")
    parser.add_argument("--mask_dir", required=True, help="Path to masks directory")
    parser.add_argument("--output_dir", required=True, help="Path to save extracted features")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for ViT forward")
    parser.add_argument("--global_only", action="store_true", help="Only extract global features")
    parser.add_argument("--roi_only", action="store_true", help="Only extract ROI features")
    parser.add_argument("--multi_layer", action="store_true",
                        help="Concatenate features from layers [7,15,23]+final (4096-dim)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building feature extractor...")
    model = build_feature_extractor(args.checkpoint, device=args.device,
                                     multi_layer=args.multi_layer)
    if args.multi_layer:
        print("Multi-layer feature concatenation enabled (4096-dim)")
    print("Feature extractor ready")

    if not args.roi_only:
        print("\n=== Extracting global features ===")
        extract_global_features(model, args.image_dir, args.output_dir,
                                args.device, args.batch_size)

    if not args.global_only:
        print("\n=== Extracting ROI features ===")
        extract_roi_features(model, args.image_dir, args.mask_dir, args.output_dir,
                             args.device, args.batch_size)

    print("\nDone!")


if __name__ == "__main__":
    main()
