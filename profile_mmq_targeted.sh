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
PROFILE_OUTPUT="mmq_profile_targeted_${TIMESTAMP}"
LOG_OUTPUT="server_log_targeted_${TIMESTAMP}.txt"

# Add CUDA 13.1 bin to PATH for ncu
export PATH=/usr/local/cuda-13.1/bin:$PATH

echo "Launching Targeted NCU Profiler..."
echo "Profile CSV:    ${PROFILE_OUTPUT}.csv"
echo "Server Logs:    ${LOG_OUTPUT}"
echo "----------------------------------------------------------------"

# Expanded Metrics List
# Stalls
#   smsp__warps_issue_stalled_long_scoreboard, smsp__warps_issue_stalled_short_scoreboard
#   smsp__warps_issue_stalled_wait, smsp__warp_issue_stalled_mio_throttle, smsp__warp_issue_stalled_barrier
# Throughput
#   sm__throughput.avg.pct_of_peak_sustained_elapsed
#   gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
# Tensor Cores
#   sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
# Cache
#   lts__t_sectors.avg.pct_of_peak_sustained_elapsed
#   l1tex__t_bytes_pipe_lsu_mem_global_op_ldgsts_cache_access.sum
#   l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum
# Occupancy
#   sm__warps_active.avg.pct_of_peak_sustained_active
# Launch config (NEW)
#   launch__registers_per_thread
#   launch__shared_mem_per_block_static, launch__shared_mem_per_block_dynamic
#   launch__grid_size, launch__block_size
# Timing (NEW)
#   gpu__time_duration.sum

METRICS="smsp__warps_issue_stalled_long_scoreboard,smsp__warps_issue_stalled_short_scoreboard,smsp__warps_issue_stalled_wait,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ldgsts_cache_access.sum,smsp__warp_issue_stalled_mio_throttle.avg,smsp__warp_issue_stalled_barrier.avg,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__grid_size,launch__block_size,gpu__time_duration.sum"

# Cleanup function - exports results even on Ctrl+C
cleanup() {
    echo ""
    echo "Exporting results to CSV..."
    sleep 2
    if [ -f "${PROFILE_OUTPUT}.ncu-rep" ]; then
        ncu --import "${PROFILE_OUTPUT}.ncu-rep" --csv > "${PROFILE_OUTPUT}.csv" 2>/dev/null
        echo "Report: ${PROFILE_OUTPUT}.ncu-rep"
        echo "CSV:    ${PROFILE_OUTPUT}.csv"
    fi
}
trap cleanup EXIT INT TERM

# Launch command wrapped in ncu
ncu --metrics "${METRICS}" \
    -k "regex:mul_mat_q|soft_max|rope_|add_f32|mul_f32|quantize_|dequantize_|flash_attn|norm_|scale_|cpy_|concat_|diag_mask|get_rows|argsort|im2col|pool_|clamp_|sum_|repeat_" \
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
