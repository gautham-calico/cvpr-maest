#!/usr/bin/env python3
"""
finetune_ct.py — Two-stage fine-tuning of MAE-ST ViT-Large encoder
for 15 binary CT disease classification tasks.

Stage 1: Freeze encoder, train linear head (5 epochs, LR=1e-3)
Stage 2: Unfreeze all, layer-wise LR decay, cosine schedule (25 epochs)

Usage:
  python -m mae_st_v2.finetune.finetune_ct \
      --checkpoint /path/to/pretrain_checkpoint.pth \
      --image_dir /path/to/images \
      --label_dir /path/to/labels \
      --output_dir /path/to/output
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from mae_st_v2.finetune.dataset_ct_clf import (
    ALL_DISEASES,
    CTClassificationDataset,
    compute_pos_weights,
)


def get_args_parser():
    parser = argparse.ArgumentParser("CT fine-tuning", add_help=False)

    # Data
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--label_dir", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to MAE pretrain checkpoint (.pth)")

    # Model
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--t_patch_size", default=4, type=int)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--drop_path", default=0.1, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)

    # Training — Stage 1 (head only)
    parser.add_argument("--stage1_epochs", default=5, type=int)
    parser.add_argument("--stage1_lr", default=1e-3, type=float)

    # Training — Stage 2 (full fine-tune)
    parser.add_argument("--stage2_epochs", default=25, type=int)
    parser.add_argument("--stage2_lr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--warmup_epochs", default=3, type=int)
    parser.add_argument("--layer_decay", default=0.75, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)

    # General
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--output_dir", default="./output_finetune", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--patience", default=7, type=int,
                        help="Early stopping patience (epochs without improvement)")

    return parser


class CTClassifier(nn.Module):
    """ViT encoder + linear classification head for 15 diseases."""

    def __init__(self, encoder, embed_dim=1024, num_classes=15, dropout=0.5):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        x: (B, 1, T, H, W)
        Returns: (B, 15) logits
        """
        enc = self.encoder
        x = enc.patch_embed(x)
        N, T, L, C = x.shape
        x = x.view(N, T * L, C)

        if enc.cls_embed:
            cls_tokens = enc.cls_token.expand(N, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if enc.sep_pos_embed:
            pos_embed = enc.pos_embed_spatial.repeat(
                1, enc.input_size[0], 1
            ) + torch.repeat_interleave(
                enc.pos_embed_temporal,
                enc.input_size[1] * enc.input_size[2],
                dim=1,
            )
            if enc.cls_embed:
                pos_embed = torch.cat(
                    [enc.pos_embed_class.expand(pos_embed.shape[0], -1, -1), pos_embed], 1
                )
        else:
            pos_embed = enc.pos_embed[:, :, :]
        x = x + pos_embed

        for blk in enc.blocks:
            x = blk(x)

        # Global average pool (exclude CLS token)
        x = x[:, 1:, :].mean(dim=1)
        x = enc.norm(x)
        x = self.dropout(x)
        x = self.head(x)
        return x


def build_encoder(args):
    """Build ViT-Large encoder from video_vit components."""
    from functools import partial

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from mae_st_v2.util.video_vit import Attention, Block, PatchEmbed

    embed_dim = 1024
    depth = 24
    num_heads = 16
    mlp_ratio = 4.0
    norm_layer = partial(nn.LayerNorm, eps=1e-6)

    # Stochastic depth
    dpr = [x.item() for x in torch.linspace(0, args.drop_path, depth)]

    encoder = nn.Module()
    encoder.patch_embed = PatchEmbed(
        args.input_size, 16, 1, embed_dim, args.num_frames, args.t_patch_size
    )
    input_size = encoder.patch_embed.input_size
    encoder.input_size = input_size
    encoder.cls_embed = True
    encoder.sep_pos_embed = True

    encoder.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    encoder.pos_embed_spatial = nn.Parameter(
        torch.zeros(1, input_size[1] * input_size[2], embed_dim)
    )
    encoder.pos_embed_temporal = nn.Parameter(
        torch.zeros(1, input_size[0], embed_dim)
    )
    encoder.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))

    encoder.blocks = nn.ModuleList([
        Block(
            embed_dim, num_heads, mlp_ratio,
            qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
            drop_path=dpr[i],
            attn_func=partial(Attention, input_size=input_size),
        )
        for i in range(depth)
    ])
    encoder.norm = norm_layer(embed_dim)

    return encoder


def load_pretrained_weights(encoder, checkpoint_path):
    """Load encoder weights from MAE pretrain checkpoint."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("model_state", ckpt))

    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("decoder_") or k.startswith("mask_token") or k.startswith("head."):
            continue
        encoder_state[k] = v

    msg = encoder.load_state_dict(encoder_state, strict=False)
    print(f"Loaded pretrained encoder: {msg}")


def param_groups_lrd(model, weight_decay, layer_decay, lr):
    """Layer-wise learning rate decay for encoder + head."""
    param_groups = {}
    num_layers = len(model.encoder.blocks) + 1

    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # No weight decay for 1D params (biases, norms)
        if p.ndim == 1:
            this_decay = 0.0
        else:
            this_decay = weight_decay

        # Determine layer
        if name.startswith("head."):
            layer_id = num_layers
        elif name.startswith("encoder."):
            inner = name[len("encoder."):]
            if inner.startswith("blocks."):
                block_id = int(inner.split(".")[1])
                layer_id = block_id + 1
            else:
                layer_id = 0
        else:
            layer_id = num_layers

        group_key = f"layer_{layer_id}_decay_{this_decay}"
        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [],
                "lr": lr * layer_scales[layer_id],
                "lr_scale": layer_scales[layer_id],
                "weight_decay": this_decay,
            }
        param_groups[group_key]["params"].append(p)

    return list(param_groups.values())


def cosine_lr(optimizer, epoch, warmup_epochs, total_epochs, base_lr, min_lr):
    """Cosine decay with warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * epoch / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        if "lr_scale" in pg:
            pg["lr"] = lr * pg["lr_scale"]
        else:
            pg["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model, dataloader, pos_weight, device):
    """Evaluate model on validation set, return mean AUC and per-disease AUCs."""
    model.eval()
    all_logits = []
    all_labels = []
    all_masks = []

    for volumes, labels, masks in dataloader:
        volumes = volumes.to(device, non_blocking=True)
        logits = model(volumes)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
        all_masks.append(masks)

    all_logits = torch.cat(all_logits, dim=0)  # (N, 15)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    probs = torch.sigmoid(all_logits)

    # Per-disease AUC
    aucs = {}
    for d_idx, disease in enumerate(ALL_DISEASES):
        valid = all_masks[:, d_idx] > 0
        if valid.sum() < 2:
            continue
        y_true = all_labels[valid, d_idx].numpy()
        y_prob = probs[valid, d_idx].numpy()
        if len(np.unique(y_true)) < 2:
            aucs[disease] = 0.5
        else:
            aucs[disease] = roc_auc_score(y_true, y_prob)

    mean_auc = np.mean(list(aucs.values())) if aucs else 0.0
    return mean_auc, aucs


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for volumes, labels, masks in dataloader:
        volumes = volumes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(volumes)  # (B, 15)

        # Masked BCE: only compute loss on labeled entries
        loss_per_element = criterion(logits, labels)  # (B, 15)
        loss = (loss_per_element * masks).sum() / masks.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Datasets
    print("Loading datasets...")
    train_ds = CTClassificationDataset(
        args.image_dir, args.label_dir, split="train",
        num_frames=args.num_frames, target_hw=args.input_size, augment=True,
    )
    val_ds = CTClassificationDataset(
        args.image_dir, args.label_dir, split="val",
        num_frames=args.num_frames, target_hw=args.input_size, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Compute pos_weight for BCE loss
    pos_weight = compute_pos_weights(train_ds.labels, train_ds.label_mask).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    print(f"pos_weight: {pos_weight.cpu().numpy()}")

    # Build model
    print("Building model...")
    encoder = build_encoder(args)
    load_pretrained_weights(encoder, args.checkpoint)
    model = CTClassifier(encoder, embed_dim=1024, num_classes=15, dropout=args.dropout)
    model.to(device)

    best_auc = 0.0
    best_epoch = -1
    log_entries = []

    # =====================================================================
    # Stage 1: Freeze encoder, train head only
    # =====================================================================
    print("\n" + "=" * 60)
    print("Stage 1: Training head only (encoder frozen)")
    print("=" * 60)

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.stage1_lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.stage1_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        mean_auc, aucs = evaluate(model, val_loader, pos_weight, device)

        entry = {"stage": 1, "epoch": epoch, "loss": loss, "mean_auc": mean_auc, "aucs": aucs}
        log_entries.append(entry)
        print(f"  [Stage1 Epoch {epoch}] loss={loss:.4f}  val_mean_auc={mean_auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "stage": 1,
                "mean_auc": mean_auc,
            }, os.path.join(args.output_dir, "best_model.pth"))

    # =====================================================================
    # Stage 2: Unfreeze all, layer-wise LR decay
    # =====================================================================
    print("\n" + "=" * 60)
    print("Stage 2: Full fine-tuning with layer-wise LR decay")
    print("=" * 60)

    # Unfreeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = True

    # Layer-wise LR decay param groups
    param_groups = param_groups_lrd(
        model, args.weight_decay, args.layer_decay, args.stage2_lr
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.stage2_lr, betas=(0.9, 0.999))

    patience_counter = 0
    total_stage2_epochs = args.stage2_epochs

    for epoch in range(total_stage2_epochs):
        lr = cosine_lr(
            optimizer, epoch, args.warmup_epochs, total_stage2_epochs,
            args.stage2_lr, args.min_lr
        )

        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        mean_auc, aucs = evaluate(model, val_loader, pos_weight, device)

        entry = {"stage": 2, "epoch": epoch, "loss": loss, "lr": lr,
                 "mean_auc": mean_auc, "aucs": aucs}
        log_entries.append(entry)
        print(f"  [Stage2 Epoch {epoch}] lr={lr:.2e}  loss={loss:.4f}  val_mean_auc={mean_auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "stage": 2,
                "mean_auc": mean_auc,
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"    -> New best! mean_auc={mean_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stopping after {args.patience} epochs without improvement")
                break

    # Save final model
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "stage": 2,
        "mean_auc": mean_auc,
    }, os.path.join(args.output_dir, "final_model.pth"))

    # Save training log
    with open(os.path.join(args.output_dir, "train_log.json"), "w") as f:
        json.dump(log_entries, f, indent=2, default=str)

    # Print final summary
    print("\n" + "=" * 60)
    print(f"Best mean AUC: {best_auc:.4f} at epoch {best_epoch}")
    print("=" * 60)

    # Load best model and print per-disease AUCs
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    mean_auc, aucs = evaluate(model, val_loader, pos_weight, device)

    print(f"\n{'Disease':<25} {'AUC-ROC':<10}")
    print("-" * 35)
    for disease in ALL_DISEASES:
        if disease in aucs:
            print(f"{disease:<25} {aucs[disease]:<10.4f}")
    print("-" * 35)
    print(f"{'MEAN':<25} {mean_auc:<10.4f}")


if __name__ == "__main__":
    main()
