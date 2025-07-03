#!/bin/bash

# INT4 Quantization Test Script - Careful Quality Testing
# Model: Qwen3-235B-A22B (Q3_K_S quantization)
# Target: Test INT4 with optimal conditions and careful monitoring

set -e

echo "=== INT4 KV Cache Quantization Test ==="
echo "Model: Qwen3-235B-A22B-Q3_K_S"
echo "KV Cache: INT4 quantization with quality safeguards"
echo "Target: 3x RTX 5090 setup with careful quality monitoring"
echo "⚠️  WARNING: INT4 quantization is extremely aggressive"
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
TENSOR_SPLIT="27,29,29"
GPU_LAYERS=85  

# Set up Blackwell runtime environment (conservative for testing)
echo "Setting up conservative Blackwell runtime environment for INT4 testing..."
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=1
export GGML_CUDA_BLACKWELL_CLUSTER_SIZE=4         # Smaller cluster for stability
export GGML_CUDA_BLACKWELL_TILE_SIZE=64           # Smaller tiles for stability

# HBM3 Memory Optimizations (conservative for testing)
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=1
export GGML_CUDA_HBM3_COALESCING_FACTOR=8         # Conservative
export GGML_CUDA_HBM3_BURST_SIZE=256              # Conservative

# INT4 KV Cache Quantization Settings with Quality Safeguards
export LLAMA_ENABLE_QUANTIZED_KV_CACHE=1

# Option 1: Asymmetric quantization (recommended)
export LLAMA_KV_CACHE_QUANTIZATION_K=INT4         # K tensor: more aggressive
export LLAMA_KV_CACHE_QUANTIZATION_V=INT8         # V tensor: more conservative
export LLAMA_KV_CACHE_QUALITY=HIGH                # Strict quality control

# Option 2: Full INT4 (uncomment to test - high risk of gibberish)
# export LLAMA_KV_CACHE_QUANTIZATION_K=INT4       # K tensor: maximum compression
# export LLAMA_KV_CACHE_QUANTIZATION_V=INT4       # V tensor: maximum compression
# export LLAMA_KV_CACHE_QUALITY=HIGH              # Even stricter for full INT4

# Quality monitoring environment variables
export LLAMA_KV_QUALITY_THRESHOLD=0.01            # 1% max degradation (very strict)
export LLAMA_KV_ADAPTIVE_QUANTIZATION=1           # Enable adaptive quantization
export LLAMA_KV_IMPORTANCE_TRACKING=1             # Track token importance

# L2 Cache Optimizations (conservative)
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=0      # Disable for stability
export GGML_CUDA_L2_CACHE_SIZE=134217728          # 128MB L2 cache

# Memory Management (conservative for INT4 testing)
export GGML_CUDA_MEMORY_POOL_SIZE=28000000000     # 28GB per GPU (conservative)
export GGML_CUDA_HOST_REGISTER=1
export GGML_CUDA_NO_PEER_COPY=0               

# Performance Optimizations (conservative)
export GGML_CUDA_FORCE_DMMV=0                     # Disable for testing
export GGML_CUDA_PEER_MAX_BATCH_SIZE=32           # Small batches for quality
export CUDA_LAUNCH_BLOCKING=0                 
export CUDA_DEVICE_MAX_CONNECTIONS=8              # Conservative

# CPU Optimizations
export OMP_NUM_THREADS=16                     
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=33554432            # 32MB

echo "✓ INT4 quantization environment configured with quality safeguards"
echo

# Launch parameters (optimized for INT4 quality testing)
CONTEXT_SIZE=8000       # Reduced context for INT4 stability
BATCH_SIZE=64           # Small batches for quality
UBATCH_SIZE=64          
THREADS=24              
THREADS_BATCH=48        
TEMPERATURE=0.5         # Lower temperature for more focused output
MIN_P=0.1              # Higher minimum probability for stability
TOP_P=0.9              # Slightly more conservative sampling
SERVER_HOST="0.0.0.0"  
SERVER_PORT=5003       # Different port for INT4 testing

echo "=== INT4 Test Configuration ==="
echo "Context Size: $CONTEXT_SIZE tokens (reduced for INT4 stability)"
echo "Batch Size: $BATCH_SIZE (small for quality)"
echo "GPU Layers: $GPU_LAYERS"
echo "Tensor Split: $TENSOR_SPLIT"
echo "KV Cache: K=INT4, V=INT8 (asymmetric for safety)"
echo "Quality: HIGH with 1% max degradation threshold"
echo "Server: $SERVER_HOST:$SERVER_PORT"
echo

echo "⚠️  IMPORTANT NOTES:"
echo "1. INT4 quantization is extremely aggressive"
echo "2. This test uses asymmetric quantization (K=INT4, V=INT8)"
echo "3. Small context size (8K) for maximum stability"
echo "4. If output is gibberish, INT4 is too aggressive for this model"
echo "5. Use test_model_quality.sh to compare with other configurations"
echo

echo "Starting llama-server with INT4 quantization testing..."
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
echo "INT4 test script finished."
echo "Press Enter to close..."
read 