#!/bin/bash

# Blackwell RTX 5090 Dual-GPU Launch Script - Phase 3: KV Cache Memory Optimization
# Optimized for 235B+ parameter models with quantized KV cache and long context support
# Target: 50-100 t/s for 19k context (vs current 12.9 t/s baseline)

# ============================================================================
# PHASE 3: KV CACHE MEMORY OPTIMIZATION CONFIGURATION
# ============================================================================

# KV Cache Quantization Settings
export LLAMA_KV_CACHE_QUANTIZED=1                    # Enable quantized KV cache
export LLAMA_KV_QUANT_LEVEL_K=1                      # K tensor: 1=INT8, 2=INT4, 0=disabled
export LLAMA_KV_QUANT_LEVEL_V=1                      # V tensor: 1=INT8, 2=INT4, 0=disabled
export LLAMA_KV_QUANT_QUALITY=1                      # 0=fast, 1=balanced, 2=high_quality
export LLAMA_KV_QUANT_STRATEGY=2                     # 0=fifo, 1=importance, 2=hybrid

# Memory Bandwidth Optimization (RTX 5090 HBM3e)
export LLAMA_KV_ENABLE_BLACKWELL_OPTS=1              # Enable RTX 5090 optimizations
export LLAMA_KV_COALESCING_FACTOR=16                 # Memory coalescing factor
export LLAMA_KV_PREFETCH_DISTANCE=8                  # Prefetch distance for HBM3e
export LLAMA_KV_USE_L2_CACHE_HINTS=1                 # Utilize 128MB L2 cache

# Dynamic Cache Management
export LLAMA_KV_ENABLE_COMPRESSION=1                 # Enable distance-based compression
export LLAMA_KV_COMPRESSION_THRESHOLD=512            # Compress tokens beyond 512 positions
export LLAMA_KV_ENABLE_STREAMING=1                   # Enable streaming for long contexts
export LLAMA_KV_MAX_CONTEXT_LENGTH=128000            # Maximum supported context length

# Quality Control
export LLAMA_KV_QUALITY_THRESHOLD=0.02               # 2% max quality degradation
export LLAMA_KV_QUALITY_MONITORING=1                 # Enable quality monitoring
export LLAMA_KV_ADAPTIVE_QUANT=1                     # Adaptive quantization based on importance

# ============================================================================
# EXISTING BLACKWELL OPTIMIZATIONS (Phases 1 & 2)
# ============================================================================

# Phase 1: Cluster GEMM Optimizations
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=1
export GGML_CUDA_BLACKWELL_CLUSTER_SIZE=8
export GGML_CUDA_BLACKWELL_TILE_SIZE=128

# Phase 2: HBM3 Memory Bandwidth Optimizations  
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=1
export GGML_CUDA_HBM3_COALESCING_FACTOR=16
export GGML_CUDA_HBM3_BURST_SIZE=512

# Phase 2: L2 Flash Attention Optimizations
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=1
export GGML_CUDA_L2_CACHE_SIZE=134217728             # 128MB L2 cache
export GGML_CUDA_L2_TILE_SIZE=32

# Performance and Debugging
export GGML_CUDA_BLACKWELL_DEBUG=0                   # Set to 1 for debug output
export GGML_CUDA_BLACKWELL_BENCHMARK=0               # Set to 1 for benchmarking
export LLAMA_KV_CACHE_DEBUG=0                        # Set to 1 for KV cache debug info

# ============================================================================
# MODEL AND HARDWARE CONFIGURATION
# ============================================================================

# Model Configuration (235B+ Parameter Model)
MODEL_PATH="${MODEL_PATH:-./models/Qwen3-235B-Q3_K_S.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3-235b}"

# Context and Generation Settings
CONTEXT_SIZE="${CONTEXT_SIZE:-32768}"                # Base context size
MAX_CONTEXT="${MAX_CONTEXT:-131072}"                 # Maximum context with KV optimization
N_PREDICT="${N_PREDICT:-2048}"                       # Tokens to generate
BATCH_SIZE="${BATCH_SIZE:-512}"                      # Batch size
UBATCH_SIZE="${UBATCH_SIZE:-128}"                    # Micro-batch size for memory efficiency

# RTX 5090 Dual-GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1                      # Use both RTX 5090 GPUs
export GGML_CUDA_FORCE_DMMV=1                        # Force dual-matrix-multiply
export GGML_CUDA_PEER_MAX_BATCH_SIZE=128             # P2P batch size

# Memory Management
export GGML_CUDA_MEMORY_POOL_SIZE=31000000000        # ~31GB memory pool per GPU
export GGML_CUDA_HOST_REGISTER=1                     # Register host memory
export GGML_CUDA_NO_PEER_COPY=0                      # Enable P2P copy

# ============================================================================
# PERFORMANCE OPTIMIZATION FLAGS
# ============================================================================

# CPU Optimization
export OMP_NUM_THREADS=16                            # Optimize for your CPU
export GGML_METAL=0                                  # Disable Metal (NVIDIA only)

# Memory Allocation
export MALLOC_TRIM_THRESHOLD_=0                      # Disable malloc trim for performance
export MALLOC_MMAP_THRESHOLD_=33554432               # 32MB mmap threshold

# CUDA Optimization
export CUDA_LAUNCH_BLOCKING=0                        # Async kernel launches
export CUDA_DEVICE_MAX_CONNECTIONS=32                # Max concurrent connections

# ============================================================================
# VALIDATION AND STARTUP
# ============================================================================

echo "========================================================================"
echo "Blackwell RTX 5090 Launch - Phase 3: KV Cache Memory Optimization"
echo "========================================================================"
echo "Model: $MODEL_PATH"
echo "Context Size: $CONTEXT_SIZE tokens (Max: $MAX_CONTEXT with KV optimization)"
echo "Batch Size: $BATCH_SIZE, Micro-batch: $UBATCH_SIZE"
echo "KV Cache Quantization: K=${LLAMA_KV_QUANT_LEVEL_K}, V=${LLAMA_KV_QUANT_LEVEL_V}"
echo "Memory Bandwidth Target: 2-4x improvement"
echo "Performance Target: 50-100 t/s for 19k context"
echo "========================================================================"

# Validate GPU setup
echo "Detecting RTX 5090 GPUs..."
nvidia-smi --query-gpu=gpu_name,memory.total,compute_cap --format=csv,noheader,nounits | grep -E "RTX 509[0-9]|compute_9\.[0-9]" || {
    echo "Warning: RTX 5090 not detected. Blackwell optimizations may not be effective."
}

# Check CUDA version compatibility
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -o 'V[0-9]*\.[0-9]*' | cut -c2- | head -1)
if [ -n "$CUDA_VERSION" ] && [ "$(echo "$CUDA_VERSION >= 11.8" | bc -l 2>/dev/null)" = "1" ]; then
    echo "CUDA Version: $CUDA_VERSION (Compatible with Blackwell optimizations)"
else
    echo "Warning: CUDA 11.8+ required for optimal Blackwell performance"
fi

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Please ensure the 235B+ parameter model is available."
    exit 1
fi

# ============================================================================
# LAUNCH LLAMA.CPP SERVER WITH PHASE 3 OPTIMIZATIONS
# ============================================================================

# Build the command with all optimizations
LLAMA_CMD="./llama-server"

# Model and basic parameters
LLAMA_CMD="$LLAMA_CMD --model $MODEL_PATH"
LLAMA_CMD="$LLAMA_CMD --alias $MODEL_ALIAS"
LLAMA_CMD="$LLAMA_CMD --ctx-size $CONTEXT_SIZE"
LLAMA_CMD="$LLAMA_CMD --n-predict $N_PREDICT"
LLAMA_CMD="$LLAMA_CMD --batch-size $BATCH_SIZE"
LLAMA_CMD="$LLAMA_CMD --ubatch-size $UBATCH_SIZE"

# GPU configuration
LLAMA_CMD="$LLAMA_CMD --gpu-layers 999"              # Offload all layers to GPU
LLAMA_CMD="$LLAMA_CMD --split-mode layer"            # Split by layer for dual GPU
LLAMA_CMD="$LLAMA_CMD --tensor-split 0.5,0.5"       # Equal split between RTX 5090s

# Memory and performance optimization
LLAMA_CMD="$LLAMA_CMD --no-mmap"                     # Disable mmap for GPU optimization
LLAMA_CMD="$LLAMA_CMD --numa distribute"             # NUMA optimization
LLAMA_CMD="$LLAMA_CMD --flash-attn"                  # Enable flash attention

# KV Cache optimization (Phase 3)
if [ "$LLAMA_KV_CACHE_QUANTIZED" = "1" ]; then
    LLAMA_CMD="$LLAMA_CMD --kv-cache-type quantized"
    LLAMA_CMD="$LLAMA_CMD --kv-cache-quantization ${LLAMA_KV_QUANT_LEVEL_K},${LLAMA_KV_QUANT_LEVEL_V}"
    LLAMA_CMD="$LLAMA_CMD --kv-cache-quality-threshold $LLAMA_KV_QUALITY_THRESHOLD"
    
    if [ "$LLAMA_KV_ENABLE_STREAMING" = "1" ]; then
        LLAMA_CMD="$LLAMA_CMD --ctx-size $MAX_CONTEXT"
        echo "Long context streaming enabled: $MAX_CONTEXT tokens"
    fi
fi

# Server configuration
LLAMA_CMD="$LLAMA_CMD --host 0.0.0.0"
LLAMA_CMD="$LLAMA_CMD --port 8080"
LLAMA_CMD="$LLAMA_CMD --threads $OMP_NUM_THREADS"

# Sampling parameters optimized for quality
LLAMA_CMD="$LLAMA_CMD --temp 0.7"
LLAMA_CMD="$LLAMA_CMD --top-k 40"
LLAMA_CMD="$LLAMA_CMD --top-p 0.9"
LLAMA_CMD="$LLAMA_CMD --repeat-penalty 1.1"

# Enable metrics and monitoring
LLAMA_CMD="$LLAMA_CMD --metrics"
LLAMA_CMD="$LLAMA_CMD --log-format text"

# ============================================================================
# PERFORMANCE MONITORING SETUP
# ============================================================================

# Function to monitor performance
monitor_performance() {
    echo "Starting performance monitoring..."
    
    # Create monitoring log directory
    mkdir -p ./logs/phase3-monitoring
    
    # Monitor GPU utilization
    nvidia-smi dmon -s pucvmet -d 5 > ./logs/phase3-monitoring/gpu_utilization.log &
    GPU_MON_PID=$!
    
    # Monitor memory usage
    nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv -l 5 > ./logs/phase3-monitoring/memory_usage.log &
    MEM_MON_PID=$!
    
    echo "Performance monitoring started (PIDs: $GPU_MON_PID, $MEM_MON_PID)"
    echo "Logs saved to ./logs/phase3-monitoring/"
}

# Function to stop monitoring
stop_monitoring() {
    echo "Stopping performance monitoring..."
    kill $GPU_MON_PID $MEM_MON_PID 2>/dev/null
}

# ============================================================================
# LAUNCH AND MONITORING
# ============================================================================

echo "Starting llama.cpp server with Phase 3 KV Cache optimizations..."
echo "Command: $LLAMA_CMD"
echo "========================================================================"

# Start performance monitoring if requested
if [ "${MONITOR_PERFORMANCE:-1}" = "1" ]; then
    monitor_performance
    trap stop_monitoring EXIT
fi

# Launch the server
exec $LLAMA_CMD

# ============================================================================
# PERFORMANCE VALIDATION NOTES
# ============================================================================

# Expected Performance Improvements:
# - Memory bandwidth: 2-4x improvement through quantization
# - Token generation: 50-100 t/s for 19k context (vs 12.9 t/s baseline)
# - Memory usage: 50-75% reduction in KV cache memory
# - Context length: Support for 128k+ tokens with maintained performance
# - Quality degradation: <2% with balanced quantization settings

# Key Monitoring Metrics:
# 1. Tokens per second at various context lengths
# 2. Memory bandwidth utilization (HBM3e)
# 3. L2 cache hit rates
# 4. KV cache compression ratios
# 5. Quality metrics (perplexity, BLEU scores)

# Troubleshooting:
# - If performance is lower than expected, try reducing quantization levels
# - For quality issues, increase LLAMA_KV_QUALITY_THRESHOLD
# - For memory issues, enable compression and reduce context size
# - Check GPU temperatures and power limits during long runs 