#!/bin/bash

# Quick NCU profile to check attention_v5 kernel register usage at runtime
# Requires: sudo (for NCU hardware counters)

set -e
set -o pipefail

# Activate conda base environment
source /home/rgilbreth/miniconda3/etc/profile.d/conda.sh
conda activate base

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/ncu"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Paths
SERVER_PATH="$SCRIPT_DIR/build/bin/llama-server"
CLI_PATH="$SCRIPT_DIR/build/bin/llama-cli"
MODEL="/media/rgilbreth/AI-M2-2TB/Models/Qwen3/Coder/30B-A3B/Q8_0/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf"

# Environment
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export GGML_CUDA_FORCE_MMQ="1"
export GGML_CUDA_F16="1"

# Check ncu
if ! command -v ncu &> /dev/null; then
    echo "ERROR: ncu not found. Install NVIDIA Nsight Compute."
    echo ""
    echo "Try:"
    echo "  export PATH=/opt/nvidia/nsight-compute-2024.3.2:\$PATH"
    echo "  # or"
    echo "  sudo apt install nsight-compute"
    exit 1
fi

# Check binary
if [[ ! -x "$CLI_PATH" ]]; then
    echo "ERROR: llama-cli not found: $CLI_PATH"
    echo "Run: cmake --build build --config Release"
    exit 1
fi

echo ""
echo "NCU Register Profile for Flash Attention v5"
echo "============================================"
echo ""
echo "Using: $CLI_PATH"
echo "Model: $MODEL"
echo ""

CSV_FILE="$LOG_DIR/registers_$TIMESTAMP.csv"
REPORT_FILE="$LOG_DIR/report_$TIMESTAMP.ncu-rep"

echo "Output:"
echo "  CSV:    $CSV_FILE"
echo "  Report: $REPORT_FILE"
echo ""

# Run with minimal prompt to trigger flash attention
echo "Profiling (this may take a minute)..."
echo ""

sudo ncu \
    --target-processes all \
    --kernel-name "regex:attention_v5" \
    --launch-count 5 \
    --set full \
    -o "$REPORT_FILE" \
    --csv \
    "$CLI_PATH" \
        -m "$MODEL" \
        --threads 4 \
        --ctx-size 512 \
        --n-gpu-layers 999 \
        --flash-attn \
        -p "What is 2+2?" \
        -n 8 \
        2>&1 | tee "$CSV_FILE"

echo ""
echo "================================================================"
echo "Register Usage Summary"
echo "================================================================"
echo ""

# Extract key metrics
if [[ -f "$CSV_FILE" ]]; then
    echo "Kernel launch metrics:"
    grep -E "attention_v5" "$CSV_FILE" | head -20 || echo "No attention_v5 kernels captured"
    echo ""
fi

echo ""
echo "To view detailed report:"
echo "  ncu-ui $REPORT_FILE"
echo ""
echo "Key metrics to check:"
echo "  - Registers Per Thread (target: <=224)"
echo "  - Local Memory Per Thread (should be 0 for no spill)"
echo "  - Achieved Occupancy"
echo ""
