#!/bin/bash

# Stop on error
set -e

# Activate the specified virtual environment
VENV_PATH="/home/rgilbreth/Desktop/AI-Software/Venvs/unsloth-xformers/venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "Virtual environment activated."
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
fi

# Set Environment Variables
export CUDA_VISIBLE_DEVICES=0,2
export GGML_CUDA_FORCE_MMQ="1"
export GGML_CUDA_F16="1"
export GGML_CUDA_GRAPH_FORCE="1"
export CUDA_LAUNCH_BLOCKING=1

echo "Environment variables set:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GGML_CUDA_FORCE_MMQ=$GGML_CUDA_FORCE_MMQ"
echo "GGML_CUDA_F16=$GGML_CUDA_F16"
echo "GGML_CUDA_GRAPH_FORCE=$GGML_CUDA_GRAPH_FORCE"
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "----------------------------------------------------------------"

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
