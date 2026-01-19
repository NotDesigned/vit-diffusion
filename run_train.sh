#!/bin/bash

# ViT-Diffusion Training Script
# Usage: ./run_train.sh [optional arguments]

# Setup Environment
export OMP_NUM_THREADS=4

# Config
DATA_PATH="$HOME/.cache/kagglehub/datasets/dimensi0n/afhq-512/versions/1" # Replace with your dataset
# DATA_PATH="./data/afhq_subset_1k"
CHECKPOINT_DIR="checkpoints/run_001"
# RESUME_FROM_CHECKPOINT="$CHECKPOINT_DIR/epoch_190"
MODE="vit" # 'vit' or 'standard'

# Hyperparameters
BATCH_SIZE=64
LR=1e-4
EPOCHS=500
WANDB_PROJECT="vit-diffusion"
WANDB_RUN_NAME="run_v2"

echo "Starting Training in $MODE mode..."
echo "Data: $DATA_PATH"
echo "Checkpoints: $CHECKPOINT_DIR"

# Launch with Accelerate
# Using mixed precision bfloat16 by default
accelerate launch --mixed_precision="bf16" train.py \
    --mode $MODE \
    --data_path "$DATA_PATH" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --sigma_start 2.0 \
    --sigma_end 0.1 \
    --patch_size 2 \
    --hidden_size 384 \
    --depth 12 \
    --num_heads 6 \
    --input_size 32 \
    --vit_num_heads 4 \
    --vit_rank 16 \
    --use_ema \
    --ema_decay 0.9999 \
    --compute_fid \
    --fid_num_samples 1000 \
    --fid_inference_steps 50 \
    --log_every_epoch 25 \
    --ckpt_every_epoch 25 \
    --lambda_gmm 1 \
    --lambda_ortho 0.1 \
    --lambda_div 0 \
    --lambda_repul 0.1 \
    --num_epochs $EPOCHS \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --lr_scheduler cosine \
    --lr_warmup_steps 1000 \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
    --use_latent_cache \
    --latent_cache_dir ./my_cache \
    $@

