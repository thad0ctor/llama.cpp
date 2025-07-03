#!/bin/bash

# Minimal Launch Script - Disable All Optimizations
# Model: Qwen3-235B-A22B (Q3_K_S quantization)
# Target: Basic functionality without crashes

set -e

echo "=== MINIMAL Launch Script (Crash Prevention) ==="
echo "Model: Qwen3-235B-A22B-Q3_K_S"
echo "Mode: Minimal configuration to prevent CUDA crashes"
echo "All optimizations DISABLED for stability"
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

# DISABLE ALL BLACKWELL OPTIMIZATIONS
echo "Disabling all Blackwell optimizations..."
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=0
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=0
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=0
export GGML_CUDA_FORCE_DMMV=0

# DISABLE KV CACHE QUANTIZATION
export LLAMA_ENABLE_QUANTIZED_KV_CACHE=0
export LLAMA_KV_DISABLE_AUTO=1
unset LLAMA_KV_CACHE_QUANTIZATION_K
unset LLAMA_KV_CACHE_QUANTIZATION_V
unset LLAMA_KV_CACHE_QUALITY

# Basic GPU configuration (single GPU first)
TENSOR_SPLIT=""        # Let llama.cpp auto-detect
GPU_LAYERS=40          # Conservative layer count
MAIN_GPU=0

# Conservative memory settings
export GGML_CUDA_MEMORY_POOL_SIZE=20000000000  # 20GB per GPU (conservative)
export GGML_CUDA_HOST_REGISTER=0               # Disable host memory registration
export GGML_CUDA_NO_PEER_COPY=1                # Disable P2P for stability

# Disable advanced CUDA features
export CUDA_LAUNCH_BLOCKING=1                  # Synchronous kernel launches
export CUDA_DEVICE_MAX_CONNECTIONS=1           # Single connection
unset GGML_CUDA_PEER_MAX_BATCH_SIZE

# Basic CPU settings
export OMP_NUM_THREADS=8                       # Conservative thread count

echo "✓ Minimal CUDA configuration applied"
echo

# Conservative launch parameters
CONTEXT_SIZE=4096       # Small context to avoid memory issues
BATCH_SIZE=32           # Very small batch
UBATCH_SIZE=32          
THREADS=8               # Conservative thread count
THREADS_BATCH=8        
TEMPERATURE=0.7         
MIN_P=0.05             
TOP_P=0.95             
SERVER_HOST="0.0.0.0"  
SERVER_PORT=5004       # Different port for minimal config

echo "=== Minimal Configuration ==="
echo "Context Size: $CONTEXT_SIZE tokens"
echo "Batch Size: $BATCH_SIZE"
echo "GPU Layers: $GPU_LAYERS"
echo "Main GPU: $MAIN_GPU"
echo "Tensor Split: AUTO-DETECT"
echo "KV Cache: UNQUANTIZED"
echo "Optimizations: ALL DISABLED"
echo "Server: $SERVER_HOST:$SERVER_PORT"
echo

echo "🎯 Goal: Basic functionality without crashes"
echo "If this works, we can gradually re-enable optimizations"
echo

echo "Starting llama-server with minimal configuration..."
echo "Press Ctrl+C to stop the server"
echo

# Launch with minimal parameters (no flash attention, no advanced features)
./build/bin/llama-server \
    -m "$MODEL_PATH" \
    --threads $THREADS \
    --threads-batch $THREADS_BATCH \
    --ctx-size $CONTEXT_SIZE \
    --temp $TEMPERATURE \
    --min-p $MIN_P \
    --main-gpu $MAIN_GPU \
    --host $SERVER_HOST \
    --port $SERVER_PORT \
    --top-p $TOP_P \
    --batch-size $BATCH_SIZE \
    --ubatch-size $UBATCH_SIZE \
    --log-colors \
    --n-gpu-layers $GPU_LAYERS \
    --verbose

echo ""
echo "Minimal script finished."
echo "Press Enter to close..."
read 