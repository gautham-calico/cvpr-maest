#!/usr/bin/env python3
"""
strip_checkpoint.py — Strip decoder/optimizer from MAE checkpoint.

Reduces ~3.8GB full checkpoint to ~1.2GB encoder-only checkpoint for Docker.

Usage:
  python -m mae_st.linprobe.strip_checkpoint \
      --input /path/to/checkpoint-00099.pth \
      --output /path/to/encoder_checkpoint.pth
"""

import argparse

import torch


def strip_checkpoint(input_path, output_path):
    print(f"Loading checkpoint from {input_path}...")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    # Get model state dict
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    # Filter to encoder-only keys
    encoder_state = {}
    skipped = []
    skip_prefixes = ("decoder_", "mask_token")

    for k, v in state_dict.items():
        if any(k.startswith(p) for p in skip_prefixes):
            skipped.append(k)
            continue
        if k.startswith("head.") or k.startswith("dropout."):
            skipped.append(k)
            continue
        encoder_state[k] = v

    print(f"Kept {len(encoder_state)} keys, skipped {len(skipped)} keys")
    if skipped:
        print(f"Skipped key prefixes: {set(k.split('.')[0] for k in skipped)}")

    # Save as a simple dict (no optimizer, no epoch, no scaler)
    torch.save({"model": encoder_state}, output_path)

    # Report sizes
    import os
    in_size = os.path.getsize(input_path) / (1024**3)
    out_size = os.path.getsize(output_path) / (1024**3)
    print(f"Input:  {in_size:.2f} GB")
    print(f"Output: {out_size:.2f} GB")
    print(f"Reduction: {(1 - out_size/in_size)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Strip decoder from MAE checkpoint")
    parser.add_argument("--input", required=True, help="Input MAE checkpoint path")
    parser.add_argument("--output", required=True, help="Output encoder-only checkpoint path")
    args = parser.parse_args()
    strip_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
