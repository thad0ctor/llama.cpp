#!/bin/bash

# Stop on error
set -e
set -o pipefail

# Activate conda base environment
source /home/rgilbreth/miniconda3/etc/profile.d/conda.sh
conda activate base

# Setup logging
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/server"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/llama-server_$(date '+%Y-%m-%d_%H-%M-%S').log"

echo "Logging to: $LOG_FILE"

# Paths
LLAMA_LITE="/home/rgilbreth/Desktop/AI-Software/WIP/llama-lite/llama.cpp/build/bin/llama-server"
LLAMA_VANILLA="/home/rgilbreth/Desktop/AI-Software/llama.cpp/build/bin/llama-server"
MODEL="/media/rgilbreth/AI-M2-2TB/Models/Qwen3/Coder/30B-A3B/Q8_0/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf"

# Set Environment Variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2
export GGML_CUDA_FORCE_MMQ="1"
export GGML_CUDA_F16="1"
#export GGML_CUDA_GRAPH_FORCE="1"

# Mode selection
LAUNCH_MODE="normal"
SERVER_PATH="$LLAMA_LITE"
SERVER_NAME="llama-lite"

# Check for command line argument first
if [[ "$1" == "--debug" || "$1" == "-d" ]]; then
    LAUNCH_MODE="debug"
elif [[ "$1" == "--normal" || "$1" == "-n" ]]; then
    LAUNCH_MODE="normal"
elif [[ "$1" == "--vanilla" || "$1" == "-v" ]]; then
    LAUNCH_MODE="vanilla"
elif [[ -z "$1" ]]; then
    # Interactive prompt if no argument provided
    echo ""
    echo "Select launch mode:"
    echo "  [1] Normal      - llama-lite (fast)"
    echo "  [2] Debug       - llama-lite + compute-sanitizer (100x slower)"
    echo "  [3] Vanilla     - upstream llama.cpp (for comparison)"
    echo ""
    read -p "Enter choice [1]: " choice
    case "$choice" in
        2|d|D|debug)
            LAUNCH_MODE="debug"
            ;;
        3|v|V|vanilla)
            LAUNCH_MODE="vanilla"
            ;;
        *)
            LAUNCH_MODE="normal"
            ;;
    esac
fi

# Configure based on mode
LAUNCH_CMD=""
case "$LAUNCH_MODE" in
    debug)
        export CUDA_LAUNCH_BLOCKING=1
        LAUNCH_CMD="compute-sanitizer --tool memcheck --show-backtrace yes"
        SERVER_PATH="$LLAMA_LITE"
        SERVER_NAME="llama-lite (DEBUG)"
        echo ""
        echo "*** DEBUG MODE ENABLED (expect 100x slowdown) ***"
        echo ""
        ;;
    vanilla)
        SERVER_PATH="$LLAMA_VANILLA"
        SERVER_NAME="vanilla llama.cpp"
        ;;
    *)
        SERVER_PATH="$LLAMA_LITE"
        SERVER_NAME="llama-lite"
        ;;
esac

# Verify server exists
if [[ ! -x "$SERVER_PATH" ]]; then
    echo "ERROR: Server not found or not executable: $SERVER_PATH"
    exit 1
fi

# Run everything in a block and pipe to tee for reliable logging
{
    echo "----------------------------------------------------------------"
    echo "Server: $SERVER_NAME"
    echo "Binary: $SERVER_PATH"
    echo "Model:  $MODEL"
    echo ""
    echo "Environment:"
    echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "  GGML_CUDA_FORCE_MMQ=$GGML_CUDA_FORCE_MMQ"
    echo "  GGML_CUDA_F16=$GGML_CUDA_F16"
    if [[ "$LAUNCH_MODE" == "debug" ]]; then
        echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
    fi
    echo "----------------------------------------------------------------"

    $LAUNCH_CMD \
    "$SERVER_PATH" \
        -m "$MODEL" \
        --threads 24 \
        --threads-batch 48 \
        --ctx-size 24000 \
        --temp 0.6 \
        --min-p 0.01 \
        --tensor-split 1,1 \
        --n-gpu-layers 999 \
        --flash-attn on \
        --parallel 1 \
        --port 5001 \
        --no-warmup \
        --jinja \
        --n-predict 500
} 2>&1 | stdbuf -oL tee -a "$LOG_FILE"
