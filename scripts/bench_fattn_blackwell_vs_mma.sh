#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/bench"
mkdir -p "$LOG_DIR"

BENCH_BIN="$ROOT_DIR/build/bin/llama-bench"
if [[ ! -x "$BENCH_BIN" ]]; then
  echo "ERROR: llama-bench not found or not executable: $BENCH_BIN" >&2
  exit 1
fi

# Defaults (override via env)
MODEL_DEFAULT="/media/rgilbreth/AI-M2-2TB/Models/Qwen3/Coder/30B-A3B/Q8_0/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf"
MODEL="${MODEL:-$MODEL_DEFAULT}"
NGL="${NGL:-999}"
THREADS="${THREADS:-24}"
BATCH="${BATCH:-128}"
UBATCH="${UBATCH:-128}"
REPS="${REPS:-3}"
CUDA_DEVICE_ORDER_DEFAULT="PCI_BUS_ID"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-$CUDA_DEVICE_ORDER_DEFAULT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# Scenarios
PREFILL_PROMPT="${PREFILL_PROMPT:-4096}"
PREFILL_GEN="${PREFILL_GEN:-1}"
DECODE_PROMPT="${DECODE_PROMPT:-1}"
DECODE_GEN="${DECODE_GEN:-512}"

# Kernel selection
ENV_NO_ATTN_V5="${GGML_CUDA_NO_ATTENTION_V5:-1}"
ENV_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-}"

# Profiling modes:
#   0 = bench only (fastest, just tok/s comparison)
#   1 = lightweight NCU (attention kernels only, launch params + time)
#   2 = full NCU (all kernels, all stall metrics — slow)
PROFILE="${PROFILE:-0}"

NCU_BIN="${NCU_BIN:-/opt/nvidia/nsight-compute/2025.4.0/ncu}"

# Lightweight: just launch config + time (enough to verify register/smem changes)
NCU_METRICS_LIGHT="launch__registers_per_thread,launch__shared_mem_per_block_dynamic,launch__grid_size,launch__block_size,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum"
NCU_KERNEL_LIGHT="regex:flash_attn|attention_v5|mul_mat|stream_k|quantize|dequantize|rms_norm|norm|rope_|soft_max|softcap|mm_ids|reduce|silu|swiglu|topk|argmax"

# Full: all kernels, stall breakdown + occupancy + memory hierarchy
NCU_METRICS_FULL="${NCU_METRICS:-launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__grid_size,launch__block_size,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ldgsts_cache_access.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,smsp__warps_issue_stalled_long_scoreboard,smsp__warps_issue_stalled_short_scoreboard,smsp__warps_issue_stalled_wait,smsp__warp_issue_stalled_mio_throttle.avg,smsp__warp_issue_stalled_barrier.avg,smsp__warp_issue_stalled_not_selected.avg,smsp__warp_issue_stalled_drain.avg}"
NCU_KERNEL_FULL="${NCU_KERNEL_REGEX:-regex:flash_attn|attention_v5|mul_mat|stream_k|quantize|dequantize|rms_norm|norm|rope_|soft_max|softcap|mm_ids|reduce|cpy_|concat|diag|get_rows|silu|swiglu|topk|argmax|unary|bin_bcast|pad}"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_CSV="$LOG_DIR/fattn_bench_${RUN_ID}.csv"
LOG_FILE="$LOG_DIR/fattn_bench_${RUN_ID}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Run ID: $RUN_ID"
echo "Model: $MODEL"
echo "Bench: $BENCH_BIN"
echo "Params: NGL=$NGL THREADS=$THREADS BATCH=$BATCH UBATCH=$UBATCH REPS=$REPS"
echo "Prefill: prompt=$PREFILL_PROMPT gen=$PREFILL_GEN"
echo "Decode:  prompt=$DECODE_PROMPT gen=$DECODE_GEN"
echo "Env: CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
echo "Env: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Env: GGML_CUDA_NO_ATTENTION_V5=$ENV_NO_ATTN_V5"
if [[ -n "$ENV_DISABLE_GRAPHS" ]]; then
  echo "Env: GGML_CUDA_DISABLE_GRAPHS=$ENV_DISABLE_GRAPHS"
fi
echo "Env: PROFILE=$PROFILE"

if [[ "$PROFILE" != "0" ]]; then
  export PATH="/usr/local/cuda-13.1/bin:$PATH"
  if [[ ! -x "$NCU_BIN" ]]; then
    if command -v ncu >/dev/null 2>&1; then
      NCU_BIN="ncu"
    else
      echo "WARN: NCU not found. Disabling profiling." >&2
      PROFILE=0
    fi
  fi
  if [[ "$PROFILE" != "0" ]]; then
    echo "Env: NCU_BIN=$NCU_BIN"
    echo "Env: PROFILE mode=$PROFILE (1=light, 2=full)"
  fi
fi

COMPLETED_PROFILES=()

cleanup() {
  echo ""
  echo "--- Interrupted ---"
  if [[ ${#COMPLETED_PROFILES[@]} -gt 0 ]]; then
    echo "Completed profiles:"
    for p in "${COMPLETED_PROFILES[@]}"; do
      echo "  $p"
    done
  fi
  local profile_dir="$LOG_DIR/profile_${RUN_ID}"
  if [[ -d "$profile_dir" ]]; then
    local files
    files=$(find "$profile_dir" \( -name '*.ncu-rep' -o -name '*.csv' \) 2>/dev/null)
    if [[ -n "$files" ]]; then
      echo "All profile files in $profile_dir:"
      echo "$files" | while read -r f; do
        echo "  $f ($(du -h "$f" | cut -f1))"
      done
    else
      echo "No profile files produced yet."
    fi
  fi
  if [[ -f "$OUT_CSV" ]]; then
    echo "Bench CSV: $OUT_CSV"
  fi
  echo "Log: $LOG_FILE"
  exit 130
}
trap cleanup INT TERM

echo ""

auto_append_csv() {
  local tmp_csv="$1"
  local scenario="$2"
  local kernel="$3"
  local no_blackwell="$4"
  local no_attn_v5="$5"
  local disable_graphs="$6"

  if [[ ! -f "$OUT_CSV" ]]; then
    head -n 1 "$tmp_csv" | sed 's/$/,scenario,kernel,ggml_cuda_no_blackwell_f16,ggml_cuda_no_attention_v5,ggml_cuda_disable_graphs/' > "$OUT_CSV"
  fi

  tail -n +2 "$tmp_csv" | sed "s/$/,${scenario},${kernel},${no_blackwell},${no_attn_v5},${disable_graphs}/" >> "$OUT_CSV"
}

run_case() {
  local scenario="$1"
  local n_prompt="$2"
  local n_gen="$3"
  local kernel_label="$4"
  local no_blackwell="$5"

  local tmp_csv="$LOG_DIR/tmp_${RUN_ID}_${scenario}_${kernel_label}.csv"

  local -a env_cmd
  env_cmd=(env)

  if [[ "$no_blackwell" == "1" ]]; then
    env_cmd+=("GGML_CUDA_NO_BLACKWELL_F16=1")
  else
    env_cmd+=("-u" "GGML_CUDA_NO_BLACKWELL_F16")
  fi

  env_cmd+=("CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER")
  env_cmd+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
  env_cmd+=("GGML_CUDA_NO_ATTENTION_V5=$ENV_NO_ATTN_V5")
  if [[ -n "$ENV_DISABLE_GRAPHS" ]]; then
    env_cmd+=("GGML_CUDA_DISABLE_GRAPHS=$ENV_DISABLE_GRAPHS")
  fi

  echo "---"
  echo "Running: scenario=$scenario kernel=$kernel_label no_blackwell=$no_blackwell"

  local -a bench_cmd
  bench_cmd=( "$BENCH_BIN"
    -m "$MODEL"
    -ngl "$NGL"
    -t "$THREADS"
    -fa 1
    --no-warmup
    -r "$REPS"
    -b "$BATCH"
    -ub "$UBATCH"
    -p "$n_prompt"
    -n "$n_gen"
    -o csv
  )

  if [[ "$PROFILE" != "0" ]]; then
    local profile_dir="$LOG_DIR/profile_${RUN_ID}"
    mkdir -p "$profile_dir"
    local profile_out="$profile_dir/${scenario}_${kernel_label}"

    local ncu_metrics ncu_kernel_regex
    if [[ "$PROFILE" == "1" ]]; then
      ncu_metrics="$NCU_METRICS_LIGHT"
      ncu_kernel_regex="$NCU_KERNEL_LIGHT"
    else
      ncu_metrics="$NCU_METRICS_FULL"
      ncu_kernel_regex="$NCU_KERNEL_FULL"
    fi

    # Use 1 rep under NCU (profiling adds replay overhead, multiple reps are wasteful)
    local -a profile_bench_cmd=( "${bench_cmd[@]}" )
    for i in "${!profile_bench_cmd[@]}"; do
      if [[ "${profile_bench_cmd[$i]}" == "-r" ]]; then
        profile_bench_cmd[$((i+1))]="1"
        break
      fi
    done

    local ncu_log="${profile_out}_ncu.log"
    echo "  Profiling with NCU (mode=$PROFILE, replay=kernel)..."
    echo "  Report: ${profile_out}.ncu-rep"
    echo "  Log: $ncu_log"
    # replay-mode=kernel: outputs per-kernel results incrementally as each completes
    # (application mode buffers everything until the end)
    if ! "${env_cmd[@]}" "$NCU_BIN" \
      --metrics "$ncu_metrics" \
      --replay-mode kernel \
      -k "$ncu_kernel_regex" \
      --target-processes all \
      -o "$profile_out" \
      --force-overwrite \
      "${profile_bench_cmd[@]}" \
      2>&1 | tee "$ncu_log"; then
      echo "WARN: NCU profiling failed for ${scenario}/${kernel_label}." >&2
    fi

    # Bench separately for clean CSV (tok/s numbers)
    echo "  Running bench for CSV..."
    if ! "${env_cmd[@]}" "${bench_cmd[@]}" > "$tmp_csv"; then
      echo "WARN: Bench run failed for ${scenario}/${kernel_label}." >&2
    fi

    if [[ -f "${profile_out}.ncu-rep" ]]; then
      echo "  Report: ${profile_out}.ncu-rep ($(du -h "${profile_out}.ncu-rep" | cut -f1))"
      COMPLETED_PROFILES+=("${profile_out}.ncu-rep")
      "$NCU_BIN" --import "${profile_out}.ncu-rep" --csv > "${profile_out}.csv" 2>/dev/null || true
      if [[ -s "${profile_out}.csv" ]]; then
        local csv_lines
        csv_lines=$(wc -l < "${profile_out}.csv")
        echo "  CSV: ${profile_out}.csv ($csv_lines lines)"
      fi
    elif [[ -s "$ncu_log" ]]; then
      echo "  No .ncu-rep but log captured ($ncu_log)"
    else
      echo "WARN: No profile data produced for ${scenario}/${kernel_label}." >&2
    fi
  else
    "${env_cmd[@]}" "${bench_cmd[@]}" > "$tmp_csv"
  fi

  auto_append_csv "$tmp_csv" "$scenario" "$kernel_label" "$no_blackwell" "$ENV_NO_ATTN_V5" "${ENV_DISABLE_GRAPHS:-0}"
}

run_case "prefill" "$PREFILL_PROMPT" "$PREFILL_GEN" "blackwell_f16" "0"
run_case "decode"  "$DECODE_PROMPT" "$DECODE_GEN" "blackwell_f16" "0"

run_case "prefill" "$PREFILL_PROMPT" "$PREFILL_GEN" "mma_f16" "1"
run_case "decode"  "$DECODE_PROMPT" "$DECODE_GEN" "mma_f16" "1"

echo ""
echo "Done. Results: $OUT_CSV"
echo "Log: $LOG_FILE"
