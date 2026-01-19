#!/bin/bash
# =============================================================================
# VIT-Diffusion Docker Training Script
# 一键拉取镜像 + 配置环境 + 多卡训练
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration (Edit these as needed)
# =============================================================================

# Docker image
DOCKER_IMAGE="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel"

# Container name
CONTAINER_NAME="vit-diffusion-train"

# Paths (absolute paths required)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"  # This script's directory
DATA_DIR="${DATA_DIR:-$HOME/.cache/kagglehub/datasets}"  # Default data location
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints}"

# Training config
MODE="${MODE:-vit}"  # 'vit' or 'standard'
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"  # Auto-detect GPUs
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# WandB (optional)
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-vit-diffusion}"
USE_WANDB="${USE_WANDB:-false}"

# =============================================================================
# Colors for output
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Help
# =============================================================================
show_help() {
    cat << EOF
VIT-Diffusion Docker Training Script

Usage: ./docker_train.sh [OPTIONS]

Options:
    --mode MODE             Training mode: 'vit' or 'standard' (default: vit)
    --gpus N                Number of GPUs to use (default: auto-detect)
    --batch-size N          Batch size per GPU (default: 32)
    --epochs N              Number of epochs (default: 500)
    --data-dir PATH         Path to dataset directory
    --output-dir PATH       Path for checkpoints (default: ./checkpoints)
    --wandb                 Enable WandB logging
    --wandb-key KEY         WandB API key
    --mixed-precision TYPE  Mixed precision: bf16, fp16, no (default: bf16)
    --build                 Force rebuild the Docker image
    --shell                 Start interactive shell instead of training
    -h, --help              Show this help message

Environment Variables:
    DATA_DIR                Dataset directory
    OUTPUT_DIR              Checkpoint directory
    NUM_GPUS                Number of GPUs
    WANDB_API_KEY           WandB API key
    MODE                    Training mode

Examples:
    # Train VIT mode with 4 GPUs
    ./docker_train.sh --mode vit --gpus 4

    # Train standard mode with WandB
    ./docker_train.sh --mode standard --wandb --wandb-key your_key

    # Interactive shell for debugging
    ./docker_train.sh --shell
EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================
INTERACTIVE_SHELL=false
FORCE_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --wandb) USE_WANDB=true; shift ;;
        --wandb-key) WANDB_API_KEY="$2"; shift 2 ;;
        --mixed-precision) MIXED_PRECISION="$2"; shift 2 ;;
        --build) FORCE_BUILD=true; shift ;;
        --shell) INTERACTIVE_SHELL=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) log_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# =============================================================================
# Validation
# =============================================================================
log_info "Validating configuration..."

# Check Docker
if ! command -v docker &> /dev/null; then
    log_error "Docker not found. Please install Docker first."
    exit 1
fi

# Check NVIDIA Docker runtime
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    log_warn "NVIDIA Docker runtime not detected. GPU support may not work."
fi

# Check GPUs
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    log_warn "Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available. Using $AVAILABLE_GPUS."
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Print Configuration
# =============================================================================
echo ""
echo "=============================================="
echo "  VIT-Diffusion Docker Training"
echo "=============================================="
echo "  Mode:            $MODE"
echo "  GPUs:            $NUM_GPUS"
echo "  Batch Size:      $BATCH_SIZE (per GPU)"
echo "  Epochs:          $NUM_EPOCHS"
echo "  Mixed Precision: $MIXED_PRECISION"
echo "  Data Dir:        $DATA_DIR"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  WandB:           $USE_WANDB"
echo "  Docker Image:    $DOCKER_IMAGE"
echo "=============================================="
echo ""

# =============================================================================
# Pull Docker Image
# =============================================================================
log_info "Pulling Docker image: $DOCKER_IMAGE"
docker pull "$DOCKER_IMAGE"
log_success "Docker image ready"

# =============================================================================
# Build GPU string for docker
# =============================================================================
if [ "$NUM_GPUS" -eq "$AVAILABLE_GPUS" ]; then
    GPU_FLAG="--gpus all"
else
    # Build comma-separated list: 0,1,2,...
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
    GPU_FLAG="--gpus '\"device=$GPU_LIST\"'"
fi

# =============================================================================
# Prepare Training Command
# =============================================================================

# WandB args
WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="--use_wandb --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_API_KEY" ]; then
        WANDB_ENV="-e WANDB_API_KEY=$WANDB_API_KEY"
    fi
fi

# Mode-specific args
if [ "$MODE" = "vit" ]; then
    MODE_ARGS="--mode vit --vit_num_heads 4 --vit_rank 16 --only_winner_align"
    RUN_NAME="vit_docker_$(date +%Y%m%d_%H%M%S)"
else
    MODE_ARGS="--mode standard"
    RUN_NAME="standard_docker_$(date +%Y%m%d_%H%M%S)"
fi

# Full training command
TRAIN_CMD="cd /workspace && \
pip install -q -r requirements.txt && \
accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    --mixed_precision=$MIXED_PRECISION \
    train.py \
    $MODE_ARGS \
    --data_path /data \
    --checkpoint_dir /output/$RUN_NAME \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate 1e-4 \
    --hidden_size 384 \
    --depth 12 \
    --num_heads 6 \
    --use_ema \
    --ema_decay 0.9999 \
    --lr_scheduler cosine \
    --lr_warmup_steps 1000 \
    --log_every_epoch 25 \
    --ckpt_every_epoch 25 \
    --sigma_start 2.0 \
    --sigma_end 0.1 \
    $WANDB_ARGS \
    --wandb_run_name $RUN_NAME"

# =============================================================================
# Run Container
# =============================================================================

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_info "Stopping existing container..."
    docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true
fi

# Common docker args
DOCKER_ARGS="
    --name $CONTAINER_NAME \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $PROJECT_DIR:/workspace \
    -v $DATA_DIR:/data:ro \
    -v $OUTPUT_DIR:/output \
    -w /workspace \
    $WANDB_ENV"

if [ "$INTERACTIVE_SHELL" = true ]; then
    log_info "Starting interactive shell..."
    log_info "Run 'exit' to leave the container"
    echo ""
    
    docker run -it --rm \
        $DOCKER_ARGS \
        "$DOCKER_IMAGE" \
        /bin/bash
else
    log_info "Starting training..."
    log_info "Logs will be displayed below. Press Ctrl+C to stop."
    echo ""
    
    docker run --rm \
        $DOCKER_ARGS \
        "$DOCKER_IMAGE" \
        /bin/bash -c "$TRAIN_CMD"
    
    log_success "Training completed!"
    log_info "Checkpoints saved to: $OUTPUT_DIR/$RUN_NAME"
fi
