#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/bench"
mkdir -p "$LOG_DIR"

BENCH_BIN_BLACKWELL_DEFAULT="$ROOT_DIR/build/bin/llama-bench"
BENCH_BIN_VANILLA_DEFAULT="/home/rgilbreth/Desktop/AI-Software/llama.cpp/build/bin/llama-bench"
BENCH_BIN_BLACKWELL="${BENCH_BIN_BLACKWELL:-$BENCH_BIN_BLACKWELL_DEFAULT}"
BENCH_BIN_VANILLA="${BENCH_BIN_VANILLA:-$BENCH_BIN_VANILLA_DEFAULT}"
if [[ ! -x "$BENCH_BIN_BLACKWELL" ]]; then
  echo "ERROR: blackwell llama-bench not found or not executable: $BENCH_BIN_BLACKWELL" >&2
  exit 1
fi
if [[ ! -x "$BENCH_BIN_VANILLA" ]]; then
  echo "ERROR: vanilla llama-bench not found or not executable: $BENCH_BIN_VANILLA" >&2
  exit 1
fi

# Defaults (override via env)
MODEL_DEFAULT="/media/rgilbreth/AI-M2-2TB/Models/Qwen3/Coder/30B-A3B/Q8_0/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf"
MODEL="${MODEL:-$MODEL_DEFAULT}"
NGL="${NGL:-999}"
THREADS="${THREADS:-24}"
BATCH="${BATCH:-1024}"
UBATCH="${UBATCH:-1024}"
REPS="${REPS:-3}"
CUDA_DEVICE_ORDER_DEFAULT="PCI_BUS_ID"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-$CUDA_DEVICE_ORDER_DEFAULT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
ENV_FORCE_MMQ="${GGML_CUDA_FORCE_MMQ:-1}"
ENV_CUDA_F16="${GGML_CUDA_F16:-1}"

# Mode: "sweep" (default, threshold tuning) or "scenarios" (original behavior)
MODE="${MODE:-sweep}"

# Sweep mode: batch sizes to test (space-separated powers of 2)
SWEEP_BATCH_SIZES="${SWEEP_BATCH_SIZES:-1 2 4 8 16 32 64 128 256 512}"
# Context lengths to hold fixed while sweeping batch size (decode context)
SWEEP_CONTEXTS="${SWEEP_CONTEXTS:-4096}"

# Scenarios (used in MODE=scenarios)
PREFILL_PROMPT="${PREFILL_PROMPT:-4096}"
PREFILL_GEN="${PREFILL_GEN:-1}"
DECODE_PROMPT="${DECODE_PROMPT:-1}"
DECODE_GEN="${DECODE_GEN:-512}"
LONG_PROMPT="${LONG_PROMPT:-40000}"
LONG_GEN="${LONG_GEN:-500}"
COMBINED_PROMPT="${COMBINED_PROMPT:-$PREFILL_PROMPT}"
COMBINED_GEN="${COMBINED_GEN:-$DECODE_PROMPT}"
COMBINED_LONG_PROMPT="${COMBINED_LONG_PROMPT:-$LONG_PROMPT}"
COMBINED_LONG_GEN="${COMBINED_LONG_GEN:-$LONG_GEN}"

# Kernel selection
ENV_NO_ATTN_V5="${GGML_CUDA_NO_ATTENTION_V5-}"
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
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Bench (blackwell): $BENCH_BIN_BLACKWELL"
echo "Bench (vanilla):  $BENCH_BIN_VANILLA"
echo "Params: NGL=$NGL THREADS=$THREADS BATCH=$BATCH UBATCH=$UBATCH REPS=$REPS"
if [[ "$MODE" == "sweep" ]]; then
  echo "Sweep batch sizes: $SWEEP_BATCH_SIZES"
  echo "Sweep contexts: $SWEEP_CONTEXTS"
else
  echo "Prefill: prompt=$PREFILL_PROMPT gen=$PREFILL_GEN"
  echo "Decode:  prompt=$DECODE_PROMPT gen=$DECODE_GEN"
  echo "Long:    prompt=$LONG_PROMPT gen=$LONG_GEN"
  echo "Combined:       prompt=$COMBINED_PROMPT gen=$COMBINED_GEN"
  echo "Combined Long:  prompt=$COMBINED_LONG_PROMPT gen=$COMBINED_LONG_GEN"
fi
echo "Env: CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
echo "Env: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Env: GGML_CUDA_FORCE_MMQ=$ENV_FORCE_MMQ"
echo "Env: GGML_CUDA_F16=$ENV_CUDA_F16"
if [[ -n "$ENV_NO_ATTN_V5" ]]; then
  echo "Env: GGML_CUDA_NO_ATTENTION_V5=$ENV_NO_ATTN_V5"
fi
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
  local batch_override="${6:-}"
  local ubatch_override="${7:-}"

  local tmp_csv="$LOG_DIR/tmp_${RUN_ID}_${scenario}_${kernel_label}.csv"

  local -a env_cmd env_opts env_vars
  env_cmd=(env)

  if [[ "$no_blackwell" == "1" ]]; then
    env_vars+=("GGML_CUDA_NO_BLACKWELL_F16=1")
  else
    env_opts+=("-u" "GGML_CUDA_NO_BLACKWELL_F16")
  fi

  env_vars+=("CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER")
  env_vars+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
  env_vars+=("GGML_CUDA_FORCE_MMQ=$ENV_FORCE_MMQ")
  env_vars+=("GGML_CUDA_F16=$ENV_CUDA_F16")
  if [[ -n "$ENV_NO_ATTN_V5" ]]; then
    env_vars+=("GGML_CUDA_NO_ATTENTION_V5=$ENV_NO_ATTN_V5")
  else
    env_opts+=("-u" "GGML_CUDA_NO_ATTENTION_V5")
  fi
  if [[ -n "$ENV_DISABLE_GRAPHS" ]]; then
    env_vars+=("GGML_CUDA_DISABLE_GRAPHS=$ENV_DISABLE_GRAPHS")
  fi

  local use_batch="${batch_override:-$BATCH}"
  local use_ubatch="${ubatch_override:-$UBATCH}"

  echo "---"
  echo "Running: scenario=$scenario kernel=$kernel_label no_blackwell=$no_blackwell b=$use_batch ub=$use_ubatch"

  local bench_bin
  if [[ "$no_blackwell" == "1" ]]; then
    bench_bin="$BENCH_BIN_VANILLA"
  else
    bench_bin="$BENCH_BIN_BLACKWELL"
  fi

  local -a bench_cmd
  bench_cmd=( "$bench_bin"
    -m "$MODEL"
    -ngl "$NGL"
    -t "$THREADS"
    -fa 1
    --no-warmup
    -r "$REPS"
    -b "$use_batch"
    -ub "$use_ubatch"
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
    if ! "${env_cmd[@]}" "${env_opts[@]}" "${env_vars[@]}" "$NCU_BIN" \
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
  "${env_cmd[@]}" "${env_opts[@]}" "${env_vars[@]}" "${bench_cmd[@]}" > "$tmp_csv"
  fi

  auto_append_csv "$tmp_csv" "$scenario" "$kernel_label" "$no_blackwell" "$ENV_NO_ATTN_V5" "${ENV_DISABLE_GRAPHS:-0}"
}

# ============================================================
# Sweep mode: find blackwell_f16 vs mma_f16 crossover point
# ============================================================
#
# For each batch size N in SWEEP_BATCH_SIZES, run prefill-only
# (-p N -n 0) with -b N -ub N so the entire prompt is one
# attention call with Q->ne[1] = N.  Compare tok/s between
# blackwell_f16 and mma_f16 to find where blackwell starts winning.
#
# Usage:
#   MODE=sweep scripts/bench_fattn_blackwell_vs_mma.sh
#   MODE=sweep SWEEP_BATCH_SIZES="1 4 16 64 256 1024" scripts/bench_fattn_blackwell_vs_mma.sh
#   MODE=sweep SWEEP_CONTEXTS="2048 8192" scripts/bench_fattn_blackwell_vs_mma.sh
# ============================================================

# Extract avg tok/s from llama-bench CSV output.
# llama-bench CSV header has "avg_ts" for the speed column.
# With -p N -n 0 there is exactly one data row (prompt processing).
extract_pp_toks() {
  local csv_file="$1"
  awk -F',' '
    NR==1 {
      for (i=1; i<=NF; i++) {
        if ($i == "avg_ts" || $i == "\"avg_ts\"") { ts_col=i }
      }
      next
    }
    NR==2 {
      if (ts_col) { gsub(/"/, "", $ts_col); print $ts_col }
    }
  ' "$csv_file" 2>/dev/null
}

if [[ "$MODE" == "sweep" ]]; then
  SWEEP_CSV="$LOG_DIR/fattn_sweep_${RUN_ID}.csv"
  echo "n_queries,context,blackwell_f16_tps,mma_f16_tps,delta_pct,faster" > "$SWEEP_CSV"

  # Collect results for the summary table
  declare -a SWEEP_RESULTS=()

  for ctx in $SWEEP_CONTEXTS; do
    echo ""
    echo "========================================"
    echo "  Sweep: context=$ctx"
    echo "========================================"

    for bs in $SWEEP_BATCH_SIZES; do
      # Prefill-only: -p sets prompt length (= Q->ne[1] per ubatch),
      # -n 0 means no decode.  -b and -ub match so it's a single batch.
      local_b="$bs"
      local_ub="$bs"

      # If bs > ctx, skip (prompt can't exceed context)
      if [[ "$bs" -gt "$ctx" ]]; then
        echo "  Skipping n_queries=$bs (exceeds context=$ctx)"
        continue
      fi

      echo ""
      echo "--- n_queries=$bs context=$ctx ---"

      # Run blackwell_f16 (fork binary, no env disable)
      run_case "sweep_pp${bs}_ctx${ctx}" "$bs" "0" "blackwell_f16" "0" "$local_b" "$local_ub"
      bw_csv="$LOG_DIR/tmp_${RUN_ID}_sweep_pp${bs}_ctx${ctx}_blackwell_f16.csv"
      bw_tps=$(extract_pp_toks "$bw_csv")

      # Run mma_f16 (vanilla binary, blackwell disabled)
      run_case "sweep_pp${bs}_ctx${ctx}" "$bs" "0" "mma_f16" "1" "$local_b" "$local_ub"
      mma_csv="$LOG_DIR/tmp_${RUN_ID}_sweep_pp${bs}_ctx${ctx}_mma_f16.csv"
      mma_tps=$(extract_pp_toks "$mma_csv")

      # Compute delta
      delta_pct="N/A"
      faster="N/A"
      if [[ -n "$bw_tps" && -n "$mma_tps" ]] && \
         (( $(echo "$mma_tps > 0" | bc -l 2>/dev/null || echo 0) )); then
        delta_pct=$(echo "scale=1; ($bw_tps - $mma_tps) / $mma_tps * 100" | bc -l 2>/dev/null || echo "N/A")
        if (( $(echo "$bw_tps > $mma_tps" | bc -l 2>/dev/null || echo 0) )); then
          faster="blackwell"
        elif (( $(echo "$bw_tps < $mma_tps" | bc -l 2>/dev/null || echo 0) )); then
          faster="mma"
        else
          faster="tie"
        fi
      fi

      echo "$bs,$ctx,${bw_tps:-N/A},${mma_tps:-N/A},${delta_pct},${faster}" >> "$SWEEP_CSV"
      SWEEP_RESULTS+=("$bs|$ctx|${bw_tps:-N/A}|${mma_tps:-N/A}|${delta_pct}|${faster}")
    done
  done

  # Print summary table
  echo ""
  echo "========================================"
  echo "  CROSSOVER SUMMARY"
  echo "========================================"
  printf "%-12s %-10s %-18s %-18s %-10s %-10s\n" \
    "n_queries" "context" "blackwell_f16" "mma_f16" "delta%" "faster"
  printf "%-12s %-10s %-18s %-18s %-10s %-10s\n" \
    "----------" "--------" "----------------" "----------------" "--------" "--------"

  crossover_found=""
  prev_faster=""
  for row in "${SWEEP_RESULTS[@]}"; do
    IFS='|' read -r bs ctx bw_tps mma_tps delta_pct faster <<< "$row"
    printf "%-12s %-10s %-18s %-18s %-10s %-10s\n" \
      "$bs" "$ctx" "$bw_tps" "$mma_tps" "$delta_pct" "$faster"

    # Detect crossover: mma -> blackwell transition
    if [[ "$prev_faster" == "mma" && "$faster" == "blackwell" && -z "$crossover_found" ]]; then
      crossover_found="$bs"
    fi
    prev_faster="$faster"
  done

  echo ""
  if [[ -n "$crossover_found" ]]; then
    echo ">>> Crossover detected at n_queries=$crossover_found"
    echo "    Suggested threshold: $crossover_found (blackwell_f16 faster at and above this batch size)"
    echo "    Current threshold in fattn.cu / llama-graph.cpp: 32"
    echo ""
    echo "    To update, change 'Q->ne[1] >= 32' and 'n_queries >= 32' to >= $crossover_found"
  else
    if [[ "$prev_faster" == "blackwell" ]]; then
      echo ">>> blackwell_f16 is faster at all tested batch sizes."
      echo "    Consider lowering the threshold below $(echo "$SWEEP_BATCH_SIZES" | awk '{print $1}')."
    elif [[ "$prev_faster" == "mma" ]]; then
      echo ">>> mma_f16 is faster at all tested batch sizes."
      echo "    Consider raising the threshold above $(echo "$SWEEP_BATCH_SIZES" | awk '{print $NF}')."
    else
      echo ">>> Could not determine crossover (missing data?)."
    fi
  fi

  echo ""
  echo "Sweep CSV: $SWEEP_CSV"
  echo "Full CSV:  $OUT_CSV"
  echo "Log:       $LOG_FILE"

# ============================================================
# Scenarios mode (original behavior)
# ============================================================
else
  run_case "prefill" "$PREFILL_PROMPT" "$PREFILL_GEN" "blackwell_f16" "0"
  run_case "decode"  "$DECODE_PROMPT" "$DECODE_GEN" "blackwell_f16" "0"
  if [[ "$COMBINED_PROMPT" -gt 0 && "$COMBINED_GEN" -gt 0 ]]; then
    run_case "combined" "$COMBINED_PROMPT" "$COMBINED_GEN" "blackwell_f16" "0"
  fi
  if [[ "$LONG_PROMPT" -gt 0 || "$LONG_GEN" -gt 0 ]]; then
    run_case "long" "$LONG_PROMPT" "$LONG_GEN" "blackwell_f16" "0"
  fi
  if [[ "$COMBINED_LONG_PROMPT" -gt 0 && "$COMBINED_LONG_GEN" -gt 0 ]]; then
    run_case "combined_long" "$COMBINED_LONG_PROMPT" "$COMBINED_LONG_GEN" "blackwell_f16" "0"
  fi

  run_case "prefill" "$PREFILL_PROMPT" "$PREFILL_GEN" "mma_f16" "1"
  run_case "decode"  "$DECODE_PROMPT" "$DECODE_GEN" "mma_f16" "1"
  if [[ "$COMBINED_PROMPT" -gt 0 && "$COMBINED_GEN" -gt 0 ]]; then
    run_case "combined" "$COMBINED_PROMPT" "$COMBINED_GEN" "mma_f16" "1"
  fi
  if [[ "$LONG_PROMPT" -gt 0 || "$LONG_GEN" -gt 0 ]]; then
    run_case "long" "$LONG_PROMPT" "$LONG_GEN" "mma_f16" "1"
  fi
  if [[ "$COMBINED_LONG_PROMPT" -gt 0 && "$COMBINED_LONG_GEN" -gt 0 ]]; then
    run_case "combined_long" "$COMBINED_LONG_PROMPT" "$COMBINED_LONG_GEN" "mma_f16" "1"
  fi

  echo ""
  echo "Done. Results: $OUT_CSV"
  echo "Log: $LOG_FILE"
fi
