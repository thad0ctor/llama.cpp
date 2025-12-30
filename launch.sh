#!/bin/bash

# Stop on error
set -e
set -o pipefail

# Activate conda base environment
source /home/rgilbreth/miniconda3/etc/profile.d/conda.sh
conda activate base

# Setup logging
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/server"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/llama-server_$(date '+%Y-%m-%d_%H-%M-%S').log"

echo "Logging to: $LOG_FILE"

# Set Environment Variables
export CUDA_VISIBLE_DEVICES=0,2
export GGML_CUDA_FORCE_MMQ="1"
export GGML_CUDA_F16="1"
export GGML_CUDA_GRAPH_FORCE="1"
# export GGML_CUDA_DISABLE_GRAPHS=1    # Disable graphs until kernel is stable
export CUDA_LAUNCH_BLOCKING=1

# Run everything in a block and pipe to tee for reliable logging
{
    echo "Environment variables set:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "GGML_CUDA_FORCE_MMQ=$GGML_CUDA_FORCE_MMQ"
    echo "GGML_CUDA_F16=$GGML_CUDA_F16"
    echo "GGML_CUDA_DISABLE_GRAPHS=$GGML_CUDA_DISABLE_GRAPHS"
    echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
    echo "----------------------------------------------------------------"

    # Run with compute-sanitizer for memory error detection
    # Remove or comment out compute-sanitizer line for normal runs
    # compute-sanitizer --tool memcheck --show-backtrace yes \
    /home/rgilbreth/Desktop/AI-Software/WIP/llama-lite/llama.cpp/build/bin/llama-server \
        -m /media/rgilbreth/AI-M2-2TB/Models/Qwen3/Coder/30B-A3B/Q8_0/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf \
        --threads 12 \
        --threads-batch 24 \
        --ctx-size 24000 \
        --temp 0.6 \
        --min-p 0.01 \
        --tensor-split 1,1 \
        --n-gpu-layers 999 \
        --fit off \
        --flash-attn on \
        --parallel 1 \
        --port 5001 \
        --no-warmup \
        --jinja
} 2>&1 | stdbuf -oL tee -a "$LOG_FILE"
