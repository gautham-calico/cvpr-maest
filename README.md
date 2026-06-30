# cvpr-maest — MAE-ST v2 for CT Linear-Probe Diagnosis

Team **maest** submission to the **CVPR 2026 Challenge: Foundation Models for
General CT Image Diagnosis** ([Codabench 12650](https://www.codabench.org/competitions/12650/)),
**Task 1 (Linear Probing) — All-Data track**.

The method treats a 3D CT volume as a short **video** (the axial *Z* axis is the
temporal axis) and pre-trains a spatiotemporal masked autoencoder
(MAE-ST, ViT-Large) from scratch on the challenge's unlabeled CT corpus. At
inference the encoder is **frozen** and emits a single 1024-d descriptor per
volume; the challenge organizers fit a linear probe on those descriptors.

> The accompanying technical report is on the **`paper`** branch.

## Results

| Split | Targets | Balanced Acc | AUROC | Runtime |
|---|---|---|---|---|
| Validation (AMOS, abdomen) | 15 | — | **0.678** | — |
| **Final test (official)** | 17 | **0.600** | **0.624** | **6.27 s/case** |

Final test: **rank 4 / 6** on the All-Data LP leaderboard.

## Method at a glance

- **Backbone:** ViT-Large, 303.6 M params (embed 1024, depth 24, 16 heads).
- **CT-as-video:** input `1×16×224×224`; 3D patch embed `16×16` spatial / `4`
  temporal → `4×14×14 = 784` tokens + CLS.
- **Pre-training:** masked autoencoding, pixel-MSE on masked patches; **dynamic
  mask ratio `U[0.6, 0.9]`** per step; decoder depth 8 / 512-d. AdamW(0.9,0.95),
  blr 1e-3 (eff. lr 9.4e-5), eff. batch 24, 40-ep warmup + cosine, 250 ep
  scheduled, **epoch 130 submitted**.
- **Feature extraction (frozen):** overlapping 16-slice windows (stride 8),
  `LayerNorm(mean patch tokens)` averaged across windows → 1024-d. ROI targets
  are cropped to the provided organ mask ±8 slices.
- **Preprocessing:** HU clip `[-1024, 3071]` → `[0, 1]`.

## Repository layout

```
mae_st_v2/
├── main_pretrain.py / engine_pretrain.py / models_mae.py   # SSL pre-training
├── dataset_ct_maest.py                                     # CT→video dataset
├── scripts/pretrain_ct_maest.sh                            # pre-training launch
├── finetune/                                               # 2-stage FT (ablation only)
├── scripts/finetune_ct_clf.sh
├── linprobe/                                               # SUBMISSION (Task 1 LP)
│   ├── predict.py / ct_utils.py                            #   frozen feature extraction
│   ├── extract_feat_LP.sh                                  #   docker entrypoint
│   ├── Dockerfile
│   ├── strip_checkpoint.py                                 #   encoder-only checkpoint
│   └── train_linprobe.py                                   #   logistic-regression probe
└── util/, models_vit.py, ...                               # backbone + helpers
```

## Reproducing the submission

**1. Pre-train** (3× GPU; see `mae_st_v2/scripts/pretrain_ct_maest.sh`):

```bash
torchrun --nproc_per_node=3 -m mae_st_v2.main_pretrain \
    --ct_data_path <flare_train_files.txt> --in_chans 1 \
    --model mae_vit_large_patch16 --input_size 224 --num_frames 16 \
    --t_patch_size 4 --pred_t_dim 16 --decoder_depth 8 --decoder_embed_dim 512 \
    --mask_ratio_min 0.6 --mask_ratio_max 0.9 \
    --batch_size 4 --accum_iter 2 --epochs 250 --warmup_epochs 40 \
    --blr 1e-3 --weight_decay 0.05 --clip_grad 1.0 \
    --sep_pos_embed --cls_embed --fp32 --distributed \
    --output_dir <ckpt_dir>
```

**2. Strip to an encoder-only checkpoint** (what ships in the container):

```bash
python -m mae_st_v2.linprobe.strip_checkpoint \
    --input <ckpt_dir>/checkpoint-00130.pth --output encoder_checkpoint.pth
```

**3. Build & run the LP Docker** (per the challenge run command):

```bash
docker build -f mae_st_v2/linprobe/Dockerfile -t maest_alldata_lp:latest .
# non-ROI disease
docker run --gpus "device=0" -m 32G --rm \
    -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ \
    maest_alldata_lp:latest /bin/bash -c "sh extract_feat_LP.sh"
# ROI disease (e.g. adrenal_hyperplasia)
docker run --gpus "device=0" -m 32G --rm \
    -e MASKS_DIR=/workspace/inputs/fg_masks/adrenal_hyperplasia \
    -v $PWD/test_demo/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ \
    maest_alldata_lp:latest /bin/bash -c "sh extract_feat_LP.sh"
```

Each `{case_id}.h5` output contains a dataset `y_hat` of shape `(1024,)`.

## Note on scope

This repo contains **only** the maest submission model and code. The
encoder checkpoint (~1.2 GB) and the pre-training data are not committed; see the
challenge page for the data and contact the authors for weights.

## Acknowledgements

Built on Meta's [MAE-ST](https://github.com/facebookresearch/mae_st)
(Feichtenhofer et al., *Masked Autoencoders As Spatiotemporal Learners*, NeurIPS
2022) and [MAE](https://github.com/facebookresearch/mae). Licensed under
CC-BY-NC 4.0 (see `mae_st_v2/LICENSE`).
