#!/bin/bash

# CUDA Debug Steps Script
# Systematically identify which optimization causes the crash

set -e

echo "=== CUDA Crash Debugging Script ==="
echo "This script will help identify which optimization causes the crash"
echo

# Function to test a specific configuration
test_cuda_config() {
    local config_name="$1"
    local port="$2"
    
    echo "🧪 Testing: $config_name"
    echo "Port: $port"
    echo
    
    # Create a temporary test script
    cat > temp_test_launch.sh << EOF
#!/bin/bash
set -e

# Model configuration
MODEL_BASE_PATH="/media/rgilbreth/AI-M2-2TB/Models/unsloth/Qwen3-235B-A22B-128K-GGUF/Q3_K_S"
MODEL_FILE="Qwen3-235B-A22B-Q3_K_S-00001-of-00003.gguf"
MODEL_PATH="\${MODEL_BASE_PATH}/\${MODEL_FILE}"

# Verify model exists
if [ ! -f "\$MODEL_PATH" ]; then
    echo "ERROR: Model file not found"
    exit 1
fi

# Apply the specific configuration being tested
$3

# Basic settings
export OMP_NUM_THREADS=8
CONTEXT_SIZE=2048
BATCH_SIZE=16
GPU_LAYERS=20

echo "Testing $config_name..."

# Run a quick test (just load the model, don't start server)
timeout 60s ./build/bin/llama-cli \\
    -m "\$MODEL_PATH" \\
    -n 1 \\
    -p "Test prompt" \\
    -ngl \$GPU_LAYERS \\
    -c \$CONTEXT_SIZE \\
    --batch-size \$BATCH_SIZE \\
    --no-display-prompt \\
    2>&1
EOF

    chmod +x temp_test_launch.sh
    
    # Run the test with timeout
    echo "Running test (60 second timeout)..."
    if timeout 120s ./temp_test_launch.sh > test_output.log 2>&1; then
        echo "✅ SUCCESS: $config_name works!"
        echo "No crash detected"
        rm -f temp_test_launch.sh test_output.log
        return 0
    else
        exit_code=$?
        echo "❌ FAILED: $config_name causes issues"
        echo "Exit code: $exit_code"
        
        # Check for specific errors
        if grep -q "ggml_cuda_error" test_output.log; then
            echo "🚨 CUDA runtime error detected"
        fi
        if grep -q "cuBLAS" test_output.log; then
            echo "🚨 cuBLAS error detected"
        fi
        if grep -q "out of memory" test_output.log; then
            echo "🚨 GPU memory error detected"
        fi
        
        echo "Last few lines of output:"
        tail -10 test_output.log
        echo
        
        rm -f temp_test_launch.sh test_output.log
        return 1
    fi
}

echo "Starting systematic CUDA optimization testing..."
echo "This will test each optimization individually to find the problematic one"
echo

# Test 1: Baseline (everything disabled)
echo "========================================"
test_cuda_config "Baseline (All Disabled)" 5001 '
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=0
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=0
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=0
export GGML_CUDA_FORCE_DMMV=0
export LLAMA_ENABLE_QUANTIZED_KV_CACHE=0
export GGML_CUDA_HOST_REGISTER=0
export GGML_CUDA_NO_PEER_COPY=1
export CUDA_LAUNCH_BLOCKING=1
'

# Test 2: Enable cluster GEMM only
echo "========================================"
test_cuda_config "Cluster GEMM Only" 5002 '
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=1
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=0
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=0
export GGML_CUDA_FORCE_DMMV=0
export LLAMA_ENABLE_QUANTIZED_KV_CACHE=0
export GGML_CUDA_HOST_REGISTER=0
export GGML_CUDA_NO_PEER_COPY=1
export CUDA_LAUNCH_BLOCKING=1
'

# Test 3: Enable HBM3 optimizations only
echo "========================================"
test_cuda_config "HBM3 Optimizations Only" 5003 '
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=0
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=1
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=0
export GGML_CUDA_FORCE_DMMV=0
export LLAMA_ENABLE_QUANTIZED_KV_CACHE=0
export GGML_CUDA_HOST_REGISTER=0
export GGML_CUDA_NO_PEER_COPY=1
export CUDA_LAUNCH_BLOCKING=1
'

# Test 4: Enable Flash Attention only
echo "========================================"
test_cuda_config "Flash Attention Only" 5004 '
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=0
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=0
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=1
export GGML_CUDA_FORCE_DMMV=0
export LLAMA_ENABLE_QUANTIZED_KV_CACHE=0
export GGML_CUDA_HOST_REGISTER=0
export GGML_CUDA_NO_PEER_COPY=1
export CUDA_LAUNCH_BLOCKING=1
'

# Test 5: Enable tensor split (3 GPUs)
echo "========================================"
test_cuda_config "Multi-GPU Tensor Split" 5005 '
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=0
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=0
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=0
export GGML_CUDA_FORCE_DMMV=0
export LLAMA_ENABLE_QUANTIZED_KV_CACHE=0
export GGML_CUDA_HOST_REGISTER=0
export GGML_CUDA_NO_PEER_COPY=0
export CUDA_LAUNCH_BLOCKING=1
TENSOR_SPLIT="27,29,29"
'

# Test 6: Enable host memory registration
echo "========================================"
test_cuda_config "Host Memory Registration" 5006 '
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=0
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=0
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=0
export GGML_CUDA_FORCE_DMMV=0
export LLAMA_ENABLE_QUANTIZED_KV_CACHE=0
export GGML_CUDA_HOST_REGISTER=1
export GGML_CUDA_NO_PEER_COPY=1
export CUDA_LAUNCH_BLOCKING=1
'

# Test 7: Enable INT8 KV quantization
echo "========================================"
test_cuda_config "INT8 KV Quantization" 5007 '
export GGML_CUDA_ENABLE_BLACKWELL_CLUSTER_GEMM=0
export GGML_CUDA_ENABLE_HBM3_OPTIMIZATIONS=0
export GGML_CUDA_ENABLE_L2_FLASH_ATTENTION=0
export GGML_CUDA_FORCE_DMMV=0
export LLAMA_ENABLE_QUANTIZED_KV_CACHE=1
export LLAMA_KV_CACHE_QUANTIZATION_K=INT8
export LLAMA_KV_CACHE_QUANTIZATION_V=INT8
export LLAMA_KV_CACHE_QUALITY=HIGH
export GGML_CUDA_HOST_REGISTER=0
export GGML_CUDA_NO_PEER_COPY=1
export CUDA_LAUNCH_BLOCKING=1
'

echo "========================================"
echo "✅ Debugging tests completed!"
echo
echo "📋 Summary:"
echo "- Tests that passed: Safe to use"
echo "- Tests that failed: Causing the CUDA crash"
echo
echo "Next steps:"
echo "1. Use only the optimizations that passed tests"
echo "2. For failed tests, investigate specific CUDA errors"
echo "3. Consider hardware compatibility issues"
echo

# Clean up any remaining temp files
rm -f temp_test_launch.sh test_output.log 