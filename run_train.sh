#!/bin/bash

# ViT-Diffusion Training Script
# Usage: ./run_train.sh [optional arguments]

# Setup Environment
export OMP_NUM_THREADS=4

# Config
DATA_PATH="$HOME/.cache/kagglehub/datasets/dimensi0n/afhq-512/versions/1" # Replace with your dataset
CHECKPOINT_DIR="checkpoints/run_001"
MODE="vit" # 'vit' or 'standard'

# Hyperparameters
BATCH_SIZE=32           # Reduced for safety, adjust based on GPU VRAM
LR=1e-4
EPOCHS=100
WANDB_PROJECT="vit-diffusion"
WANDB_RUN_NAME="run_v1"

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
    --only_winner_align \
    --log_every_epoch 2 \
    --num_epochs $EPOCHS \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    $@

