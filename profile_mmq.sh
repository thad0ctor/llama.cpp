#!/bin/bash

# Stop on error
set -e

# Activate the specified virtual environment
VENV_PATH="/home/rgilbreth/Desktop/AI-Software/Venvs/unsloth-xformers/venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "Virtual environment activated."
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
fi

# Set Environment Variables
export CUDA_VISIBLE_DEVICES=0,2
export GGML_CUDA_FORCE_MMQ="1"
export GGML_CUDA_F16="1"
export GGML_CUDA_GRAPH_FORCE="1"

echo "Environment variables set:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GGML_CUDA_FORCE_MMQ=$GGML_CUDA_FORCE_MMQ"
echo "GGML_CUDA_F16=$GGML_CUDA_F16"
echo "GGML_CUDA_GRAPH_FORCE=$GGML_CUDA_GRAPH_FORCE"

# Output filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PROFILE_OUTPUT="mmq_profile_${TIMESTAMP}"
LOG_OUTPUT="server_log_${TIMESTAMP}.txt"

# Add CUDA 13.1 bin to PATH for ncu
export PATH=/usr/local/cuda-13.1/bin:$PATH

echo "Launching NCU Profiler..."
echo "Profile Report: ${PROFILE_OUTPUT}.ncu-rep"
echo "Server Logs:    ${LOG_OUTPUT}"
echo "----------------------------------------------------------------"

# Launch command wrapped in ncu
# --set full: Collects all metrics (high overhead)
# --target-processes all: Profiles child processes if any
# --force-overwrite: Overwrites existing profile file
# --csv: (Optional) If you wanted CSV output, but default report is better for GUI.

# Note: The server will start and wait. NCU will profile everything that happens.
# You will likely need to send a request to the server to trigger the MMQ kernels 
# if they don't run during load. 
# Since the command is interactive, we background it or just run it? 
# The user command had "read -rp ..." suggesting it keeps the terminal open.
# We will just run it directly. Use Ctrl+C to stop profiling when done testing.

ncu --set full \
    --target-processes all \
    -o "${PROFILE_OUTPUT}" \
    --force-overwrite \
    /home/rgilbreth/Desktop/AI-Software/WIP/llama-lite/llama.cpp/build/bin/llama-server \
    -m /media/rgilbreth/AI-M2-2TB/Models/Qwen3/Coder/30B-A3B/Q8_0/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf \
    --threads 24 \
    --threads-batch 48 \
    --ctx-size 131072 \
    --temp 0.6 \
    --min-p 0.01 \
    --tensor-split 1,1 \
    --n-gpu-layers 999 \
    --flash-attn on \
    --port 5001 \
    --no-warmup \
    --jinja 2>&1 | tee "${LOG_OUTPUT}"

echo "Profiling finished."
