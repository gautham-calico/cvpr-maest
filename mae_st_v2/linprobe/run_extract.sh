#!/bin/bash
#SBATCH --job-name=extract_feats
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch5/gautham/cvpr/logs/extract_feats_%j.log

export PYTHONUNBUFFERED=1
cd /home/gautham/Projects/C-MAE

/home/gautham/anaconda3/envs/mae/bin/python -m mae_st.linprobe.extract_features \
    --checkpoint /scratch5/gautham/cvpr/checkpoints/encoder_checkpoint.pth \
    --image_dir /scratch5/gautham/cvpr/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/images \
    --mask_dir /scratch5/gautham/cvpr/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/fg_masks \
    --output_dir /scratch5/gautham/cvpr/features \
    --batch_size 8 \
    --roi_only
