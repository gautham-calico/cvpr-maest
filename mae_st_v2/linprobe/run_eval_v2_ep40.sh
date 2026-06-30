#!/bin/bash
#SBATCH --job-name=eval_v2_40
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch5/gautham/cvpr/logs/eval_v2_ep40_%j.log

export PYTHONUNBUFFERED=1
cd /home/gautham/Projects/C-MAE

PYTHON=/home/gautham/anaconda3/envs/voco/bin/python
CKPT=/scratch5/gautham/cvpr/checkpoints/encoder_checkpoint_v2_ep40.pth
IMG_DIR=/scratch5/gautham/cvpr/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/images
MASK_DIR=/scratch5/gautham/cvpr/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/fg_masks
LABEL_DIR=/scratch5/gautham/cvpr/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/labels
FEAT_DIR=/scratch5/gautham/cvpr/features_v2_ep40
OUT_DIR=/scratch5/gautham/cvpr/linprobe_output_v2_ep40

echo "=== Step 2: Extract features ==="
$PYTHON -m mae_st_v2.linprobe.extract_features \
    --checkpoint $CKPT \
    --image_dir $IMG_DIR \
    --mask_dir $MASK_DIR \
    --output_dir $FEAT_DIR \
    --batch_size 8

echo ""
echo "=== Step 3: Train linear probes ==="
$PYTHON -m mae_st_v2.linprobe.train_linprobe \
    --feature_dir $FEAT_DIR \
    --label_dir $LABEL_DIR \
    --output_dir $OUT_DIR

echo ""
echo "=== Done ==="
