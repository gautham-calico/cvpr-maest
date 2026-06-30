#!/usr/bin/env python3
"""
extract_finetuned.py — Extract features from a fine-tuned encoder.

Reuses the sliding-window pipeline from linprobe/ct_utils.py but loads
the encoder weights from a fine-tuned checkpoint (best_model.pth) instead
of a raw MAE pretrain checkpoint.

Usage:
  python -m mae_st_v2.finetune.extract_finetuned \
      --checkpoint /path/to/best_model.pth \
      --image_dir /path/to/images \
      --mask_dir /path/to/fg_masks \
      --output_dir /path/to/features_finetuned \
      --t_patch_size 4
"""

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from mae_st_v2.linprobe.ct_utils import (
    FeatureExtractor,
    extract_volume_features,
    extract_volume_features_with_mask,
    load_and_preprocess_volume,
    load_mask,
    NUM_FRAMES,
    TARGET_HW,
)

ROI_DISEASES = [
    "adrenal_hyperplasia", "cholecystitis", "fatty_liver", "gallstone",
    "hydronephrosis", "kidney_stone", "liver_calcifications",
    "liver_cyst", "liver_lesion", "renal_cyst", "splenomegaly",
]


def build_encoder_from_finetune(checkpoint_path, t_patch_size=4, device="cuda"):
    """
    Build ViT encoder and load weights from fine-tuned checkpoint.
    The fine-tuned checkpoint contains 'encoder.*' and 'head.*' keys.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from mae_st_v2.util.video_vit import Attention, Block, PatchEmbed

    embed_dim = 1024
    depth = 24
    num_heads = 16
    mlp_ratio = 4.0
    norm_layer = partial(nn.LayerNorm, eps=1e-6)

    vit = nn.Module()
    vit.patch_embed = PatchEmbed(
        TARGET_HW, 16, 1, embed_dim, NUM_FRAMES, t_patch_size
    )
    input_size = vit.patch_embed.input_size
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

    vit.blocks = nn.ModuleList([
        Block(
            embed_dim, num_heads, mlp_ratio,
            qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
            attn_func=partial(Attention, input_size=input_size),
        )
        for _ in range(depth)
    ])
    vit.norm = norm_layer(embed_dim)

    # Load weights from fine-tuned checkpoint
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]

    # Extract encoder.* keys, strip the "encoder." prefix
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v

    msg = vit.load_state_dict(encoder_state, strict=False)
    print(f"Loaded fine-tuned encoder: {msg}")

    for p in vit.parameters():
        p.requires_grad = False

    model = FeatureExtractor(vit)
    model.to(device)
    model.eval()
    return model


def get_case_id(filename):
    return Path(filename).name.replace(".nii.gz", "")


def main():
    parser = argparse.ArgumentParser(description="Extract features from fine-tuned encoder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--t_patch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--global_only", action="store_true")
    parser.add_argument("--roi_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building feature extractor from fine-tuned checkpoint...")
    model = build_encoder_from_finetune(
        args.checkpoint, t_patch_size=args.t_patch_size, device=args.device
    )

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)

    # Global features
    if not args.roi_only:
        print("\n=== Extracting global features ===")
        output_path = Path(args.output_dir) / "global_features.pt"
        features = torch.load(output_path) if output_path.exists() else {}

        nifti_files = sorted(image_dir.glob("*.nii.gz"))
        print(f"Found {len(nifti_files)} volumes")

        for i, path in enumerate(nifti_files):
            case_id = get_case_id(path.name)
            if case_id in features:
                continue
            try:
                vol = load_and_preprocess_volume(path)
                feat = extract_volume_features(model, vol, args.batch_size, args.device)
                features[case_id] = feat
            except Exception as e:
                print(f"  ERROR {case_id}: {e}")
                continue
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(nifti_files)}] {case_id}")
            if (i + 1) % 50 == 0:
                torch.save(features, output_path)

        torch.save(features, output_path)
        print(f"Saved {len(features)} global features")

    # ROI features
    if not args.global_only:
        print("\n=== Extracting ROI features ===")
        roi_dir = Path(args.output_dir) / "roi_features"
        roi_dir.mkdir(parents=True, exist_ok=True)

        for disease in ROI_DISEASES:
            disease_mask_dir = mask_dir / disease
            if not disease_mask_dir.exists():
                print(f"Skipping {disease}: no mask dir")
                continue

            output_path = roi_dir / f"{disease}.pt"
            features = torch.load(output_path) if output_path.exists() else {}
            mask_files = sorted(disease_mask_dir.glob("*.nii.gz"))
            print(f"\n{disease}: {len(mask_files)} masks")

            for i, mask_path in enumerate(mask_files):
                case_id = get_case_id(mask_path.name)
                if case_id in features:
                    continue
                img_path = image_dir / mask_path.name
                if not img_path.exists():
                    continue
                try:
                    vol = load_and_preprocess_volume(img_path)
                    mask_vol = load_mask(mask_path)
                    feat = extract_volume_features_with_mask(
                        model, vol, mask_vol, batch_size=args.batch_size, device=args.device
                    )
                    features[case_id] = feat
                except Exception as e:
                    print(f"  ERROR {case_id}: {e}")
                    continue
                if (i + 1) % 50 == 0:
                    torch.save(features, output_path)

            torch.save(features, output_path)
            print(f"Saved {len(features)} ROI features for {disease}")

    print("\nDone!")


if __name__ == "__main__":
    main()
