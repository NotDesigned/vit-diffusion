#!/bin/bash

# ViT-Diffusion Synthetic Training Script
# Usage: ./train_synthetic.sh [optional arguments]

# Setup Environment
export OMP_NUM_THREADS=4

# Config
DATASET_TYPE="synthetic"
SYNTHETIC_TYPE=${SYNTHETIC_TYPE:-"swiss_roll"}
# Options: 'swiss_roll', '8gaussians', 'checkerboard', 'olympic'
CHECKPOINT_DIR="checkpoints/synthetic"
MODE="vit" # 'vit' or 'standard'

# Hyperparameters
BATCH_SIZE=2048         # Larger batch size for synthetic 2D data
LR=1e-3                 # Higher LR often works well for low-dim problems
EPOCHS=500
WANDB_PROJECT="vit-diffusion-synthetic"
WANDB_RUN_NAME="synthetic_run_001"

echo "Starting Training in $MODE mode on dataset $SYNTHETIC_TYPE..."
echo "Checkpoints: $CHECKPOINT_DIR"

# Launch with Accelerate
# Using no mixed precision to ensure coordinate precision for simple 2D tasks
accelerate launch --mixed_precision="no" train.py \
    --mode $MODE \
    --dataset_type $DATASET_TYPE \
    --synthetic_type $SYNTHETIC_TYPE \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --sigma_start 2.0 \
    --sigma_end 0.1 \
    --hidden_size 128 \
    --depth 4 \
    --num_heads 4 \
    --vit_num_heads 2 \
    --vit_rank 1 \
    --num_epochs $EPOCHS \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --warmup_steps 1000 \
    --lambda_align 10.0 \
    --lambda_reg 0.1 \
    --lambda_repul 5.0 \
    --temp_anneal_steps 5000 \
    --log_every_epoch 25 \
    $@
