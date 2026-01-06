#!/bin/bash

# Set Environment Variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2
export GGML_CUDA_FORCE_MMQ="1"
export GGML_CUDA_F16="1"
export GGML_CUDA_GRAPH_FORCE="1"

# Create timestamped output directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$SCRIPT_DIR/logs/profile_targeted/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"
PROFILE_OUTPUT="$OUTPUT_DIR/profile"
LOG_OUTPUT="$OUTPUT_DIR/server.log"

# Add CUDA 13.1 bin to PATH for ncu
export PATH=/usr/local/cuda-13.1/bin:$PATH

SERVER_PORT=5001

echo "================================================================"
echo "NCU Profiler - Interactive Mode"
echo "================================================================"
echo "Output Dir:  ${OUTPUT_DIR}"
echo "Profile:     ${PROFILE_OUTPUT}.ncu-rep"
echo "Server Port: ${SERVER_PORT}"
echo ""
echo "Send inference requests to: http://localhost:${SERVER_PORT}/v1/chat/completions"
echo ""
echo "NCU will automatically write the report after profiling 500 kernels."
echo "Press Ctrl+C to stop early (report only written if 500 kernels reached)."
echo "================================================================"
echo ""

METRICS="smsp__warps_issue_stalled_long_scoreboard,smsp__warps_issue_stalled_short_scoreboard,smsp__warps_issue_stalled_wait,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ldgsts_cache_access.sum,smsp__warp_issue_stalled_mio_throttle.avg,smsp__warp_issue_stalled_barrier.avg,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__grid_size,launch__block_size,gpu__time_duration.sum"

# Cleanup function
cleanup() {
    echo ""
    echo "================================================================"
    echo "Waiting for NCU to finish writing report..."
    echo "================================================================"

    # Wait for NCU to exit
    if [ -n "$NCU_PID" ]; then
        wait "$NCU_PID" 2>/dev/null
    fi

    # Kill tail
    if [ -n "$TAIL_PID" ]; then
        kill "$TAIL_PID" 2>/dev/null
    fi

    # Give filesystem time to sync
    sleep 3

    if [ -f "${PROFILE_OUTPUT}.ncu-rep" ]; then
        echo "Exporting to CSV..."
        ncu --import "${PROFILE_OUTPUT}.ncu-rep" --csv > "${PROFILE_OUTPUT}.csv" 2>/dev/null
        echo ""
        echo "SUCCESS!"
        echo "Report: ${PROFILE_OUTPUT}.ncu-rep"
        echo "CSV:    ${PROFILE_OUTPUT}.csv"
        echo ""
        echo "Kernels captured:"
        ncu --import "${PROFILE_OUTPUT}.ncu-rep" --csv 2>/dev/null | cut -d',' -f1 | sort | uniq -c | sort -rn | head -20
    else
        echo "Warning: No .ncu-rep file found"
        echo ""
        echo "This can happen if:"
        echo "  - No kernels matched the filter after --launch-skip 10000"
        echo "  - You pressed Ctrl+C before sending any inference requests"
        echo "  - NCU was interrupted mid-profile (try waiting longer after request completes)"
    fi
    echo "================================================================"
    echo ""
    echo "Done!"
    exit 0
}
trap cleanup EXIT INT TERM

echo "Starting server under NCU profiler..."
echo "(Model loading will be slow due to NCU instrumentation)"
echo ""

# Run NCU in background, redirect to log file
ncu --metrics "${METRICS}" \
    --launch-skip 10000 \
    --launch-count 500 \
    --replay-mode kernel \
    -k "regex:mul_mat|soft_max|softcap|rope_|quantize_|dequantize_|flash_attn|attention_v5|norm|scale_|cpy_|concat_|diag|get_rows|argsort|im2col|pool_|clamp|sum_|repeat_|silu|swiglu|stream_k|acc_f32|convert|topk|unary|argmax|mm_ids|bin_bcast|reduce_rows|pad_f32|rwkv|ssm_conv|gated_linear" \
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
    --port ${SERVER_PORT} \
    --no-warmup \
    --jinja > "${LOG_OUTPUT}" 2>&1 &

NCU_PID=$!
echo "NCU PID: $NCU_PID"
echo "Log: ${LOG_OUTPUT}"
echo ""

# Stream log output
tail -f "${LOG_OUTPUT}" 2>/dev/null &
TAIL_PID=$!

# Wait for NCU to finish (or be interrupted by Ctrl+C)
wait "$NCU_PID" 2>/dev/null

# cleanup will be called via trap
