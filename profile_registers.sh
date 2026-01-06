#!/bin/bash

# Profile Flash Attention v5 kernel register usage
# Two modes:
#   1. Compile-time: nvcc -Xptxas="-v" to see register allocation
#   2. Runtime: ncu to measure actual register usage per kernel

set -e
set -o pipefail

# Activate conda base environment
source /home/rgilbreth/miniconda3/etc/profile.d/conda.sh
conda activate base

# Setup logging
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/profile_registers"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Paths
BUILD_DIR="$SCRIPT_DIR/build"
SERVER_PATH="$BUILD_DIR/bin/llama-server"
MODEL="/media/rgilbreth/AI-M2-2TB/Models/Qwen3/Coder/30B-A3B/Q8_0/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf"

# Set Environment Variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2
export GGML_CUDA_FORCE_MMQ="1"
export GGML_CUDA_F16="1"

# Mode selection
echo ""
echo "Profile Flash Attention v5 Register Usage"
echo "=========================================="
echo ""
echo "Select profiling mode:"
echo "  [1] Compile-time  - Rebuild with nvcc -Xptxas=\"-v\" to see register allocation"
echo "  [2] Runtime NCU   - Use NVIDIA Nsight Compute to measure actual register usage"
echo "  [3] Both          - Compile-time + Runtime profiling"
echo ""
read -p "Enter choice [1]: " choice

PROFILE_MODE="compile"
case "$choice" in
    2|r|R|runtime)
        PROFILE_MODE="runtime"
        ;;
    3|b|B|both)
        PROFILE_MODE="both"
        ;;
    *)
        PROFILE_MODE="compile"
        ;;
esac

# Function: Compile-time register profiling
profile_compile_time() {
    echo ""
    echo "=== Compile-time Register Profiling ==="
    echo ""

    LOG_FILE="$LOG_DIR/compile_registers_$TIMESTAMP.log"
    echo "Logging to: $LOG_FILE"

    # Clean and rebuild with verbose PTX output
    echo "Rebuilding with nvcc -Xptxas=\"-v\"..."
    echo ""

    # Create a temporary CMake cache override
    CACHE_OVERRIDE="$BUILD_DIR/register_profile_flags.cmake"
    cat > "$CACHE_OVERRIDE" << 'EOF'
# Temporary flags for register profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v" CACHE STRING "" FORCE)
EOF

    {
        echo "================================================================"
        echo "Compile-time Register Profile - $(date)"
        echo "================================================================"
        echo ""

        # Rebuild just the CUDA files with verbose output
        cd "$SCRIPT_DIR"

        # Force recompile of attention_v5.cu by touching it
        touch ggml/src/ggml-cuda/attention_v5.cu

        # Build with verbose PTX output
        cmake --build build --config Release -- VERBOSE=1 2>&1 | \
            grep -E "(attention_v5|ptxas|registers|spill|smem)" || true

        echo ""
        echo "================================================================"
        echo "Extracting register info for attention_v5 kernels..."
        echo "================================================================"
        echo ""

        # Parse the build output for register info
        # nvcc -Xptxas="-v" outputs lines like:
        # ptxas info    : Used 224 registers, 98304 bytes smem, 440 bytes cmem[0]

    } 2>&1 | tee "$LOG_FILE"

    # Cleanup
    rm -f "$CACHE_OVERRIDE"

    echo ""
    echo "Register profile saved to: $LOG_FILE"
    echo ""
    echo "Key metrics to look for:"
    echo "  - 'Used N registers' - actual register count per kernel"
    echo "  - 'spill stores/loads' - registers spilled to local memory"
    echo "  - 'bytes smem' - shared memory usage"
}

# Function: Runtime register profiling with NCU
profile_runtime_ncu() {
    echo ""
    echo "=== Runtime Register Profiling (NCU) ==="
    echo ""

    # Check if ncu is available
    if ! command -v ncu &> /dev/null; then
        echo "ERROR: NVIDIA Nsight Compute (ncu) not found!"
        echo "Install with: sudo apt install nsight-compute"
        echo "Or add to PATH: export PATH=/opt/nvidia/nsight-compute/2024.x.x:\$PATH"
        exit 1
    fi

    # Verify server exists
    if [[ ! -x "$SERVER_PATH" ]]; then
        echo "ERROR: Server not found: $SERVER_PATH"
        echo "Run: cmake --build build --config Release"
        exit 1
    fi

    LOG_FILE="$LOG_DIR/ncu_registers_$TIMESTAMP.csv"
    NCU_REPORT="$LOG_DIR/ncu_report_$TIMESTAMP.ncu-rep"

    echo "NCU CSV output: $LOG_FILE"
    echo "NCU Report: $NCU_REPORT"
    echo ""

    # Create a simple test prompt file
    TEST_PROMPT_FILE=$(mktemp)
    cat > "$TEST_PROMPT_FILE" << 'EOF'
{"prompt": "Hello, world!", "n_predict": 16}
EOF

    echo "Starting server with NCU profiling..."
    echo "This will profile the first few kernel launches."
    echo ""

    # Run NCU with targeted metrics for register analysis
    # Filter for attention_v5 kernels specifically
    {
        echo "================================================================"
        echo "NCU Runtime Register Profile - $(date)"
        echo "================================================================"
        echo ""
        echo "Profiling attention_v5 kernels..."
        echo ""

        # Launch with ncu - capture first 10 kernel invocations
        # Focus on register-related metrics
        sudo ncu \
            --target-processes all \
            --kernel-name "regex:attention_v5" \
            --launch-count 10 \
            --metrics \
                launch__registers_per_thread,\
                launch__block_size,\
                launch__grid_size,\
                sm__warps_active.avg.pct_of_peak_sustained_active,\
                l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_localstore.sum,\
                l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_localload.sum,\
                smsp__inst_executed_op_local_ld.sum,\
                smsp__inst_executed_op_local_st.sum \
            --csv \
            --page raw \
            "$SERVER_PATH" \
                -m "$MODEL" \
                --threads 4 \
                --ctx-size 2048 \
                --n-gpu-layers 999 \
                --flash-attn on \
                --parallel 1 \
                --port 15001 \
                -p "Hello world" \
                -n 16 \
                2>&1

    } | tee "$LOG_FILE"

    # Cleanup
    rm -f "$TEST_PROMPT_FILE"

    echo ""
    echo "================================================================"
    echo "NCU Profile Complete"
    echo "================================================================"
    echo ""
    echo "CSV output: $LOG_FILE"
    echo ""
    echo "Key metrics:"
    echo "  - launch__registers_per_thread: Actual registers used"
    echo "  - smsp__inst_executed_op_local_ld/st: Local memory (spill) operations"
    echo "  - sm__warps_active: Occupancy indicator"
    echo ""
    echo "To analyze:"
    echo "  grep 'attention_v5' $LOG_FILE | column -t -s','"
}

# Execute based on mode
case "$PROFILE_MODE" in
    compile)
        profile_compile_time
        ;;
    runtime)
        profile_runtime_ncu
        ;;
    both)
        profile_compile_time
        echo ""
        echo "Press Enter to continue with runtime profiling..."
        read
        profile_runtime_ncu
        ;;
esac

echo ""
echo "Profile complete!"
echo ""
