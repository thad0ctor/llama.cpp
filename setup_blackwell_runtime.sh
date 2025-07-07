#!/bin/bash
# Blackwell RTX 5090 Runtime Environment Setup
# Source this script before running llama.cpp with Blackwell optimizations

echo "Setting up Blackwell RTX 5090 runtime environment..."

# === CORE BLACKWELL OPTIMIZATIONS ===
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=1
export GGML_CUDA_BLACKWELL_CLUSTER_SIZE=8
export GGML_CUDA_BLACKWELL_TILE_SIZE=128

# === HBM3 MEMORY OPTIMIZATIONS ===
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=1
export GGML_CUDA_HBM3_COALESCING_FACTOR=16
export GGML_CUDA_HBM3_BURST_SIZE=512

# === L2 CACHE OPTIMIZATIONS ===
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=1
export GGML_CUDA_L2_CACHE_SIZE=134217728  # 128MB L2 cache
export GGML_CUDA_L2_TILE_SIZE=32

# === MEMORY MANAGEMENT ===
export GGML_CUDA_MEMORY_POOL_SIZE=31000000000  # ~31GB per GPU
export GGML_CUDA_HOST_REGISTER=1
export GGML_CUDA_NO_PEER_COPY=0                # Enable P2P for multi-GPU

# === PERFORMANCE OPTIMIZATIONS ===
export GGML_CUDA_FORCE_DMMV=1                  # Force dual-matrix-multiply
export GGML_CUDA_PEER_MAX_BATCH_SIZE=128
export CUDA_LAUNCH_BLOCKING=0                  # Async kernel launches
export CUDA_DEVICE_MAX_CONNECTIONS=32

# === CPU OPTIMIZATIONS ===
export OMP_NUM_THREADS=16                      # Adjust for your CPU
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=33554432         # 32MB

# === DEBUG FLAGS (set to 1 for debugging) ===
export GGML_CUDA_BLACKWELL_DEBUG=0
export GGML_CUDA_BLACKWELL_BENCHMARK=0

echo "✅ Blackwell runtime environment configured!"
echo "Usage: ./llama-cli --model <model> --gpu-layers 999 [other options]"
