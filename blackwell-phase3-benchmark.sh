#!/bin/bash

# Blackwell Phase 3 KV Cache Memory Optimization Benchmark
# Comprehensive testing for RTX 5090 dual-GPU setup

echo "Blackwell Phase 3 KV Cache Benchmark Starting..."
echo "Testing memory bandwidth optimizations and quantization"

# Simple performance test
MODEL_PATH="${MODEL_PATH:-./models/Qwen3-235B-Q3_K_S.gguf}"
RESULTS_DIR="./benchmark_results/phase3_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $1" | tee -a "$RESULTS_DIR/benchmark.log"
}

# Test baseline vs quantized performance
run_test() {
    local test_name="$1"
    local kv_quant_k="$2"
    local kv_quant_v="$3"
    
    log_message "Running test: $test_name (K=$kv_quant_k, V=$kv_quant_v)"
    
    export LLAMA_KV_CACHE_QUANTIZED=1
    export LLAMA_KV_QUANT_LEVEL_K=$kv_quant_k
    export LLAMA_KV_QUANT_LEVEL_V=$kv_quant_v
    
    ./llama-cli \
        --model "$MODEL_PATH" \
        --ctx-size 19456 \
        --n-predict 256 \
        --batch-size 256 \
        --gpu-layers 999 \
        --split-mode layer \
        --tensor-split 0.5,0.5 \
        --flash-attn \
        --prompt "Explain machine learning principles." \
        > "$RESULTS_DIR/${test_name}_output.txt" 2>&1
    
    log_message "Test $test_name completed"
}

log_message "Starting Phase 3 benchmark tests..."

# Run tests
run_test "baseline" 0 0
run_test "int8_both" 1 1  
run_test "int4_both" 2 2

log_message "Benchmark completed. Results in: $RESULTS_DIR" 