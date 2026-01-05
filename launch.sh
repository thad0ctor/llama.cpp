#!/bin/bash

# Stop on error
set -e

# Activate conda base environment
source /home/rgilbreth/anaconda3/etc/profile.d/conda.sh
conda activate base

# Setup logging
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/server"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/llama-server_$(date '+%Y-%m-%d_%H-%M-%S').log"

echo "Logging to: $LOG_FILE"

# Start logging (output to both terminal and log file)
exec > >(tee -a "$LOG_FILE") 2>&1

# Set Environment Variables
export CUDA_VISIBLE_DEVICES=0,2
export GGML_CUDA_FORCE_MMQ="1"
export GGML_CUDA_F16="1"
# export GGML_CUDA_GRAPH_FORCE="1"  # DISABLED - graphs crash with SM_120 kernels
export GGML_CUDA_DISABLE_GRAPHS=1    # Disable graphs until kernel is stable
export CUDA_LAUNCH_BLOCKING=1

echo "Environment variables set:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GGML_CUDA_FORCE_MMQ=$GGML_CUDA_FORCE_MMQ"
echo "GGML_CUDA_F16=$GGML_CUDA_F16"
echo "GGML_CUDA_DISABLE_GRAPHS=$GGML_CUDA_DISABLE_GRAPHS"
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "----------------------------------------------------------------"

# Run with compute-sanitizer for memory error detection
# Remove or comment out compute-sanitizer line for normal runs
compute-sanitizer --tool memcheck --show-backtrace yes \
/home/rgilbreth/Desktop/AI-Software/WIP/llama-lite/llama.cpp/build/bin/llama-server \
    -m /media/rgilbreth/AI-M2-2TB/Models/Qwen3/Coder/30B-A3B/Q8_0/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf \
    --threads 24 \
    --threads-batch 48 \
    --ctx-size 131072 \
    --temp 0.6 \
    --min-p 0.01 \
    --tensor-split 1,1 \
    --n-gpu-layers 999 \
    --flash-attn on \
    --port 5001 \
    --no-warmup \
    --jinja
