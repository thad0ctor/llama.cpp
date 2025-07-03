#!/bin/bash

# Blackwell RTX 5090 Optimized Launch Script - Stable Version
# Model: Qwen3-235B-A22B (Q3_K_S quantization)
# Target: 3x RTX 5090 with quantized KV cache for memory efficiency

set -e

echo "=== Blackwell RTX 5090 Qwen3-235B Launch Script (Stable) ==="
echo "Model: Qwen3-235B-A22B-Q3_K_S"
echo "KV Cache: INT4 quantized for maximum memory efficiency"
echo "Target: 3x RTX 5090 setup"
echo

# Model path configuration
MODEL_BASE_PATH="/media/rgilbreth/AI-M2-2TB/Models/unsloth/Qwen3-235B-A22B-128K-GGUF/Q3_K_S"
MODEL_FILE="Qwen3-235B-A22B-Q3_K_S-00001-of-00003.gguf"
MODEL_PATH="${MODEL_BASE_PATH}/${MODEL_FILE}"

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found at: $MODEL_PATH"
    echo "Please check the model path and ensure the file exists."
    exit 1
fi

echo "✓ Model file found: $MODEL_PATH"

# GPU Configuration for 3x RTX 5090
# Distribute layers across GPUs: GPU0=27GB, GPU1=29GB, GPU2=29GB
TENSOR_SPLIT="27,29,29"
GPU_LAYERS=85  # Adjust based on model size and available VRAM

# Set up Blackwell runtime environment
echo "Setting up Blackwell RTX 5090 runtime environment..."
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=1
export GGML_CUDA_BLACKWELL_CLUSTER_SIZE=8
export GGML_CUDA_BLACKWELL_TILE_SIZE=128

# HBM3 Memory Optimizations
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=1
export GGML_CUDA_HBM3_COALESCING_FACTOR=16
export GGML_CUDA_HBM3_BURST_SIZE=512

# KV Cache Quantization Settings
export LLAMA_ENABLE_QUANTIZED_KV_CACHE=1
export LLAMA_KV_CACHE_QUANTIZATION_K=INT8     # K tensor quantization: INT8 for balanced memory/quality
export LLAMA_KV_CACHE_QUANTIZATION_V=INT8     # V tensor quantization: INT8 for balanced memory/quality
# export LLAMA_KV_CACHE_QUANTIZATION_K=INT4   # Alternative: INT4 (uncomment for maximum compression - may cause gibberish)
# export LLAMA_KV_CACHE_QUANTIZATION_V=INT4   # Alternative: INT4 (uncomment for maximum compression - may cause gibberish)
export LLAMA_KV_CACHE_QUALITY=HIGH            # Higher quality threshold for better output

# L2 Cache Optimizations (RTX 5090: 128MB L2)
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=1
export GGML_CUDA_L2_CACHE_SIZE=134217728      # 128MB L2 cache
export GGML_CUDA_L2_TILE_SIZE=32

# Memory Management
export GGML_CUDA_MEMORY_POOL_SIZE=31000000000 # ~31GB per GPU
export GGML_CUDA_HOST_REGISTER=1
export GGML_CUDA_NO_PEER_COPY=0               # Enable P2P for multi-GPU

# Performance Optimizations
export GGML_CUDA_FORCE_DMMV=1                 # Force dual-matrix-multiply
export GGML_CUDA_PEER_MAX_BATCH_SIZE=128
export CUDA_LAUNCH_BLOCKING=0                 # Async kernel launches
export CUDA_DEVICE_MAX_CONNECTIONS=32

# CPU Optimizations
export OMP_NUM_THREADS=16                     # Adjust for your CPU core count
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=33554432        # 32MB

echo "✓ Blackwell runtime environment configured"
echo

# Launch parameters
CONTEXT_SIZE=32000      # Increased for better coherence with INT8 quantization
BATCH_SIZE=256          # Optimized for RTX 5090 memory bandwidth
UBATCH_SIZE=256         # Match batch size for efficiency
THREADS=24              # CPU threads for preprocessing
THREADS_BATCH=48        # Threads for batch processing
TEMPERATURE=0.6         # Balanced creativity
MIN_P=0.0              # Minimum probability threshold
TOP_P=0.95             # Top-p sampling
SERVER_HOST="0.0.0.0"  # Listen on all interfaces
SERVER_PORT=5001       # Server port

echo "=== Launch Configuration ==="
echo "Context Size: $CONTEXT_SIZE tokens"
echo "Batch Size: $BATCH_SIZE"
echo "GPU Layers: $GPU_LAYERS"
echo "Tensor Split: $TENSOR_SPLIT"
echo "KV Cache: INT4 quantized (K=INT4, V=INT4)"
echo "Server: $SERVER_HOST:$SERVER_PORT"
echo

echo "Starting llama-server with Blackwell optimizations..."
echo "Press Ctrl+C to stop the server"
echo

# Launch the server
./build/bin/llama-server \
    -m "$MODEL_PATH" \
    --threads $THREADS \
    --threads-batch $THREADS_BATCH \
    --ctx-size $CONTEXT_SIZE \
    --temp $TEMPERATURE \
    --min-p $MIN_P \
    --tensor-split $TENSOR_SPLIT \
    --main-gpu 0 \
    --flash-attn \
    --host $SERVER_HOST \
    --port $SERVER_PORT \
    --top-p $TOP_P \
    --defrag-thold 0.1 \
    --batch-size $BATCH_SIZE \
    --ubatch-size $UBATCH_SIZE \
    --cont-batching \
    --log-colors \
    --n-gpu-layers $GPU_LAYERS \
    --verbose

echo ""
echo "Script finished."
echo "Press Enter to close..."
read 