#!/bin/bash
# DDP training script for balanced GPU memory usage
# 
# Usage:
#   方式1 - 使用配置文件 (推荐):
#       bash run_ddp.sh --config src/config/derain_ntu.yaml
#
#   方式2 - 指定GPU数量 + 配置文件:
#       bash run_ddp.sh 8 --config src/config/derain_ntu.yaml
#
#   方式3 - 指定GPU + 覆盖配置:
#       bash run_ddp.sh 8 --config src/config/derain_ntu.yaml --batch_size 32
#
#   方式4 - 使用特定GPU:
#       CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_ddp.sh 4 --config src/config/derain_ntu.yaml

set -e

# Check if first argument is a number (GPU count)
if [[ "$1" =~ ^[0-9]+$ ]]; then
    NUM_GPUS=$1
    shift
else
    # Default: use all available GPUs
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

echo "=========================================="
echo "Starting DDP training with ${NUM_GPUS} GPUs"
echo "All GPUs will have BALANCED memory usage!"
echo "=========================================="
echo ""

# Use torchrun (recommended for PyTorch >= 1.9)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    train_video5.py \
    --use_ddp 1 \
    "$@"
