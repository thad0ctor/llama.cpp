#!/bin/bash

# Optimized Blackwell GPU Launch Script for Qwen3-235B
# Addresses memory pool allocation issues while preserving Blackwell optimizations

# Set environment variables for optimal memory management
export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Memory management optimizations for Blackwell
export GGML_CUDA_POOL_SIZE=0.95    # Use 95% of available GPU memory per device
export GGML_CUDA_NO_PEER_COPY=1    # Disable peer-to-peer copying to avoid memory fragmentation
export GGML_CUDA_FORCE_MMQ=0       # Disable MMQ to use Blackwell optimizations

# CUDA memory pool optimizations
export CUDA_MALLOC_HEAP_SIZE=2147483648  # 2GB heap for large allocations
export CUDA_STACK_SIZE=8388608            # 8MB stack for deep kernel calls

echo "=== Blackwell-Optimized Qwen3-235B Launch ==="
echo "GPUs: RTX 5090 x3 with Blackwell optimizations"
echo "Model: 235B parameters, Q3_K_S quantization"
echo "Memory: Optimized pool allocation for large models"
echo

# Activate virtual environment
if [ -d "/home/rgilbreth/pytorch-venv" ]; then
    source /home/rgilbreth/pytorch-venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Launch with conservative but optimized parameters
/media/rgilbreth/AI-M2-2TB/AI_Software/WIP/llama.cpp/build/bin/llama-server \
    -m /media/rgilbreth/AI-M2-2TB/Models/unsloth/Qwen3-235B-A22B-128K-GGUF/Q3_K_S/Qwen3-235B-A22B-Q3_K_S-00001-of-00003.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 16384 \
    --batch-size 256 \
    --ubatch-size 128 \
    --threads 16 \
    --threads-batch 16 \
    --gpu-layers 999 \
    --tensor-split 0.34,0.33,0.33 \
    --main-gpu 0 \
    --memory-f16 \
    --no-mmap \
    --numa isolate \
    --parallel 4 \
    --cont-batching \
    --metrics \
    --log-format json \
    --verbose

echo "Server stopped." 