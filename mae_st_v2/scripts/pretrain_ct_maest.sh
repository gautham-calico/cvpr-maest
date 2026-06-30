#!/bin/bash
#SBATCH --job-name=maest_v2_pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titan_rtx:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch5/gautham/cvpr/logs/maest_v2_pretrain_%j.log

export PYTHONUNBUFFERED=1
cd /home/gautham/Projects/C-MAE

# Activate environment
source /home/gautham/anaconda3/etc/profile.d/conda.sh
conda activate voco

torchrun --nproc_per_node=3 --master_port=29500 \
    -m mae_st_v2.main_pretrain \
    --ct_data_path /scratch5/gautham/cvpr/flare_train_files.txt \
    --in_chans 1 \
    --model mae_vit_large_patch16 \
    --input_size 224 \
    --num_frames 16 \
    --t_patch_size 4 \
    --pred_t_dim 16 \
    --decoder_depth 8 \
    --decoder_embed_dim 512 \
    --mask_ratio_min 0.6 \
    --mask_ratio_max 0.9 \
    --batch_size 4 \
    --accum_iter 2 \
    --epochs 250 \
    --warmup_epochs 40 \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --clip_grad 1.0 \
    --num_workers 8 \
    --output_dir /scratch5/gautham/cvpr/checkpoints/maest_v2 \
    --log_dir /scratch5/gautham/cvpr/logs/maest_v2_tb \
    --checkpoint_period 10 \
    --sep_pos_embed \
    --cls_embed \
    --fp32 \
    --distributed
