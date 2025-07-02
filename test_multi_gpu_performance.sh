#!/bin/bash

# Multi-GPU Performance Test Script for llama.cpp
# Tests different configurations to identify performance bottlenecks

echo "=== llama.cpp Multi-GPU Performance Test ==="
echo "This script tests various configurations to identify performance issues"
echo

# Function to run a quick test and measure initialization time
run_test() {
    local test_name="$1"
    local env_vars="$2"
    echo "Running: $test_name"
    echo "Environment: $env_vars"
    
    # Use a simple help command to test initialization time
    start_time=$(date +%s.%N)
    env $env_vars ./build/bin/llama-cli --help > /dev/null 2>&1
    end_time=$(date +%s.%N)
    
    duration=$(echo "$end_time - $start_time" | bc -l)
    printf "Initialization time: %.3f seconds\n\n" $duration
}

# Check if llama-cli exists
if [ ! -f "./build/bin/llama-cli" ]; then
    echo "Error: llama-cli not found. Please run: cd build && make -j8 llama-cli"
    exit 1
fi

# Test 1: Default configuration
run_test "Default Configuration" ""

# Test 2: Disable Blackwell detection entirely  
run_test "Blackwell Detection Disabled" "LLAMA_CUDA_NO_BLACKWELL=1"

# Test 3: Enable verbose logging to see what's happening
run_test "Verbose Logging Enabled" "LLAMA_CUDA_VERBOSE=1"

# Test 4: Both disabled
run_test "Minimal Configuration" "LLAMA_CUDA_NO_BLACKWELL=1 LLAMA_CUDA_VERBOSE=1"

echo "=== Performance Test Complete ==="
echo 
echo "If you see significant differences, here's what they mean:"
echo "- Slow 'Default': Our capability detection may be adding overhead"
echo "- Fast 'Blackwell Disabled': Confirms the detection is the bottleneck" 
echo "- 'Verbose' shows detailed device information for debugging"
echo
echo "For production multi-GPU use, consider:"
echo "export LLAMA_CUDA_NO_BLACKWELL=1   # Disable enhanced detection"
echo
echo "This disables Blackwell optimizations but restores original performance."
echo "You can re-enable after we fix the performance regression." 