#!/bin/bash
#SBATCH --job-name=maest_v2_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titan_rtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch5/gautham/cvpr/logs/maest_v2_finetune_%j.log

export PYTHONUNBUFFERED=1
cd /home/gautham/Projects/C-MAE

source /home/gautham/anaconda3/etc/profile.d/conda.sh
conda activate voco

python -m mae_st_v2.finetune.finetune_ct \
    --checkpoint /scratch5/gautham/cvpr/checkpoints/maest_v2/checkpoint-249.pth \
    --image_dir /scratch5/gautham/cvpr/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/images \
    --label_dir /scratch5/gautham/cvpr/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/labels \
    --output_dir /scratch5/gautham/cvpr/checkpoints/maest_v2_finetune \
    --num_frames 16 \
    --t_patch_size 4 \
    --input_size 224 \
    --drop_path 0.1 \
    --dropout 0.5 \
    --stage1_epochs 5 \
    --stage1_lr 1e-3 \
    --stage2_epochs 25 \
    --stage2_lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_epochs 3 \
    --layer_decay 0.75 \
    --weight_decay 0.05 \
    --batch_size 4 \
    --num_workers 8 \
    --patience 7
