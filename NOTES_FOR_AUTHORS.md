# Paper build notes — maest (Task 1 LP, All-Data track)

Source of every fact in `main.tex` is the submitted Docker (`maest_v2_lp.tar.gz`)
and the `mae_st_v2/` source tree. This file lists what is **verified**, what is
**marked `[TBD]`** in red (`\tba{}`), and where each number came from.

## The submission, in one line
MAE-ST v2 = video-style spatiotemporal MAE, ViT-Large (303.6 M params, exact,
counted from `checkpoints/encoder_checkpoint.pth`), pre-trained from scratch on
10,819 FLARE CT volumes, frozen at inference, emits a 1024-d descriptor per
volume. Submitted checkpoint = epoch 130.

## Verified numbers (do not change)
- Architecture: ViT-L, embed 1024, depth 24, 16 heads, mlp 4; patch 16×16 spatial / 4 temporal; input 16×224×224 → 4×14×14=784 tokens + CLS. (`ct_utils.py`, `models_mae.py`)
- Encoder params: **303.6 M** (decoder stripped from shipped checkpoint).
- Pre-training: dynamic mask U[0.6,0.9]; decoder depth 8 / 512-d; pixel-MSE on masked patches (no norm-pix-loss); AdamW(0.9,0.95), wd 0.05, blr 1e-3 → lr 9.4e-5; eff. batch 24 (4/GPU × 2 accum × 3 Titan RTX); 40-ep warmup + cosine→0; grad-clip 1.0; FP32; 250 ep scheduled, ep130 submitted. (`scripts/pretrain_ct_maest.sh`, `main_pretrain.py`)
- Inference: 16-slice windows, stride 8 (50% overlap), trilinear→224²; feature = LayerNorm(mean of patch tokens), averaged over windows; ROI = crop to mask Z-range ±8 slices. (`ct_utils.py`, `predict.py`)
- Preprocessing: HU clip [-1024, 3071] → [0,1]. (`dataset_ct_maest.py`)
- FT ablation: 2-stage (head 5 ep @1e-3 frozen; full 25 ep @1e-4 LLRD 0.75 + cosine/3-warmup, drop-path 0.1, dropout 0.5, wd 0.05, early-stop patience 7); masked weighted BCE pos_weight=n_neg/n_pos. (`finetune/finetune_ct.py`)
- **TEST set = official final ranking (maest = ours, rank 4/6, 17 targets): BalAcc 0.600, AUROC 0.624, runtime 6.273 s/case.** (user-provided; goes in Sec. 4.7 "final testing set")
- **VALIDATION set = AMOS-15 (abdomen, 15 targets)** mean AUROC: VoCo-L frozen 0.549, VoCo-L SSL-FT 0.544, MAE-ST v1 0.627, MAE-ST v2 0.678. (`oa_mae_v2/PROJECT_STATUS.md`; goes in Sec. 4.3 "validation set"). User confirmed AMOS-15 = the validation results.
- v1→v2 per-disease highlights: splenomegaly 0.663→0.841, liver_calc 0.440→0.603, atherosclerosis 0.493→0.612; 12/15 improved.

## `[TBD]` cells you must fill before camera-ready
1. **Authors / affiliations / ORCID / count ≤6** — placeholder is a single Calico author.
2. **Env table**: exact CPU model.
3. **Pre-training wall-clock hours** (Titan RTX ×3) and **FT wall-clock hours** — not in local logs (`/scratch5` unmounted). Check your SLURM logs `maest_v2_pretrain_*.log`.
4. **FLOPs per 16-slice window** — run fvcore on the encoder (224², 784 tokens). Order ~250–300 GFLOPs expected for ViT-L.
5. **Mean BalAcc** for the internal lineage table (Table 3) — only mean AUROC is recorded locally for v1/v2/VoCo.
6. **Chest block + per-target cells** in Table 4 (covid, lung_nodule_malignancy) — request organizer per-target metrics, or re-run local eval after adding the two chest sets (COVID-CT 372, LUNA25 1,200 volumes were downloaded per PROJECT_STATUS).
7. **Per-region runtimes** in Table 4 — derive from your own timing logs (official overall is 6.273 s).
8. **Qualitative figure** (success/failure per region) — Sec. 4.6.
9. Confirm main text ≥ 8 pages; if short, expand related work / qualitative section.

NOTE: The pipeline figure (Fig. 1) is now a **native TikZ figure** drawn inline in
`main.tex` — no external image is needed. `imgs/pipeline.png` is no longer
referenced and can be deleted. The figure uses only standard TikZ libraries
(arrows.meta, positioning, fit, backgrounds, calc), all present on Overleaf.
Could not compile locally (no TeX installed in this environment); compile on
Overleaf or a local TeX Live to render.

## Honesty guardrails already baked in
- Submission is clearly stated as **frozen LP**; the fine-tuning is labeled an **ablation, not the submission** (Task 1 forbids supervised FT).
- Coreset section explicitly marked **not applicable** (All-Data track).
- The 0.678 (abdomen) vs 0.624 (official 17-target) gap is explained by the two unseen chest targets — do not present 0.678 as the challenge result.
- VoCo-L rows are flagged as public-backbone baselines run under our pipeline (public weights are not submitted, per rules).

## To compile
```
cd /home/gautham/Projects/C-MAE/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
