#!/bin/bash
set -euo pipefail

# Apples-to-apples server comparison (fork vs vanilla) with CSV output.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/server"
mkdir -p "$LOG_DIR"

# Binaries
LLAMA_LITE_BIN="${LLAMA_LITE_BIN:-$ROOT_DIR/build/bin/llama-server}"
LLAMA_VANILLA_BIN="${LLAMA_VANILLA_BIN:-/home/rgilbreth/Desktop/AI-Software/llama.cpp/build/bin/llama-server}"

# Model and runtime params
MODEL="${MODEL:-/media/rgilbreth/AI-M2-2TB/Models/Qwen3/Coder/30B-A3B/Q8_0/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf}"
THREADS="${THREADS:-24}"
THREADS_BATCH="${THREADS_BATCH:-48}"
CTX_SIZE="${CTX_SIZE:-52000}"
NGL="${NGL:-999}"
BATCH="${BATCH:-1024}"
UBATCH="${UBATCH:-1024}"
PORT_LITE="${PORT_LITE:-5001}"
PORT_VANILLA="${PORT_VANILLA:-5002}"
AUTO_PORTS="${AUTO_PORTS:-1}"

# Request params
RUNS_DEFAULT=5
if [[ -z "${RUNS+x}" ]]; then
  RUNS="$RUNS_DEFAULT"
fi
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:-0}"
MIN_P="${MIN_P:-0}"
SEED="${SEED:-1}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT_DEFAULT="Write a concise 3-sentence summary about why kernel launch overhead matters in GPU workloads."
PROFILE_NCU="${PROFILE_NCU:-0}"
NCU_BIN="${NCU_BIN:-/opt/nvidia/nsight-compute/2025.4.0/ncu}"
NCU_KERNEL="${NCU_KERNEL:-regex:.*}"
NCU_METRICS="${NCU_METRICS:-gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__shared_mem_per_block_dynamic}"
NCU_TIMEOUT_S="${NCU_TIMEOUT_S:-300}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-200}"
PROFILE_SERIAL="${PROFILE_SERIAL:-}"
WAIT_READY_TRIES="${WAIT_READY_TRIES:-120}"
WAIT_READY_OK_CODES="${WAIT_READY_OK_CODES:-200}"
REQUEST_RETRIES="${REQUEST_RETRIES:-10}"
REQUEST_RETRY_SLEEP_S="${REQUEST_RETRY_SLEEP_S:-1}"
PROFILE_NCU_WARMUP_S="${PROFILE_NCU_WARMUP_S:-120}"

if [[ -n "$PROMPT_FILE" ]]; then
  PROMPT="$(cat "$PROMPT_FILE")"
else
  PROMPT="$PROMPT_DEFAULT"
fi

if [[ "$PROFILE_NCU" == "1" ]]; then
  if [[ ! -x "$NCU_BIN" && "$(command -v ncu || true)" != "" ]]; then
    NCU_BIN="ncu"
  fi
  if [[ ! -x "$NCU_BIN" ]]; then
    echo "ERROR: NCU not found at $NCU_BIN" >&2
    exit 1
  fi
  PROFILE_NCU_RUNS="${PROFILE_NCU_RUNS:-3}"
  if [[ "$RUNS" != "$PROFILE_NCU_RUNS" ]]; then
    echo "PROFILE_NCU=1 forces RUNS=$PROFILE_NCU_RUNS"
    RUNS="$PROFILE_NCU_RUNS"
  fi
  if [[ -z "$PROFILE_SERIAL" ]]; then
    PROFILE_SERIAL="1"
  fi
  if [[ "$WAIT_READY_TRIES" -lt 240 ]]; then
    WAIT_READY_TRIES=240
  fi
  WAIT_READY_OK_CODES="200,503"
  if [[ "$REQUEST_RETRIES" -lt 120 ]]; then
    REQUEST_RETRIES=120
  fi
  if [[ "$REQUEST_RETRY_SLEEP_S" -lt 2 ]]; then
    REQUEST_RETRY_SLEEP_S=2
  fi
fi

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
CSV_OUT="$LOG_DIR/compare_servers_${RUN_ID}.csv"
LOG_LITE="$LOG_DIR/compare_servers_${RUN_ID}_lite.log"
LOG_VANILLA="$LOG_DIR/compare_servers_${RUN_ID}_vanilla.log"
STARTUP_LITE="$LOG_DIR/compare_servers_${RUN_ID}_lite.startup.log"
STARTUP_VANILLA="$LOG_DIR/compare_servers_${RUN_ID}_vanilla.startup.log"
PROFILE_DIR="$LOG_DIR/compare_servers_${RUN_ID}_profiles"
mkdir -p "$PROFILE_DIR"

echo "run_id,config,server,run_idx,latency_ms,prompt_tokens,completion_tokens,total_tokens,ms_per_completion_token,log_prompt_ms,log_prompt_tokens,log_prompt_ms_per_token,log_eval_ms,log_eval_tokens,log_eval_ms_per_token" > "$CSV_OUT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export GGML_CUDA_FORCE_MMQ="${GGML_CUDA_FORCE_MMQ:-1}"
export GGML_CUDA_F16="${GGML_CUDA_F16:-1}"

# Optional per-server env overrides (space-separated KEY=VAL)
LITE_ENV_VARS="${LITE_ENV_VARS:-}"
VANILLA_ENV_VARS="${VANILLA_ENV_VARS:-}"
CONFIGS_FILE="${CONFIGS_FILE:-}"
CONFIGS_DEFAULT=$(cat <<'EOF'
baseline||
no_blackwell|GGML_CUDA_NO_BLACKWELL_F16=0|GGML_CUDA_NO_BLACKWELL_F16=1
EOF
)

load_env_file() {
  local file="$1"
  local -a vars=()
  if [[ -n "$file" && -f "$file" ]]; then
    while IFS= read -r line; do
      [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
      vars+=("$line")
    done < "$file"
  fi
  echo "${vars[@]}"
}

LITE_ENV_FILE="${LITE_ENV_FILE:-}"
VANILLA_ENV_FILE="${VANILLA_ENV_FILE:-}"
if [[ -n "$LITE_ENV_FILE" ]]; then
  LITE_ENV_VARS="$(load_env_file "$LITE_ENV_FILE") ${LITE_ENV_VARS}"
fi
if [[ -n "$VANILLA_ENV_FILE" ]]; then
  VANILLA_ENV_VARS="$(load_env_file "$VANILLA_ENV_FILE") ${VANILLA_ENV_VARS}"
fi

if [[ ! -x "$LLAMA_LITE_BIN" ]]; then
  echo "ERROR: llama-lite server not found: $LLAMA_LITE_BIN" >&2
  exit 1
fi
if [[ ! -x "$LLAMA_VANILLA_BIN" ]]; then
  echo "ERROR: vanilla server not found: $LLAMA_VANILLA_BIN" >&2
  exit 1
fi

start_server() {
  local bin="$1"
  local port="$2"
  local log_file="$3"
  local startup_log="$4"
  local extra_env="$5"
  local profile_tag="$6"

  local -a cmd
  cmd=( "$bin"
    -m "$MODEL" \
    --threads "$THREADS" \
    --threads-batch "$THREADS_BATCH" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers "$NGL" \
    --flash-attn on \
    --batch-size "$BATCH" \
    --ubatch-size "$UBATCH" \
    --parallel 1 \
    --port "$port" \
    --no-warmup \
    --cache-ram 0 \
    --perf \
    --log-file "$log_file" \
    --log-colors off \
  )

  if [[ "$PROFILE_NCU" == "1" ]]; then
    local profile_out="$PROFILE_DIR/${profile_tag}"
    timeout --preserve-status "$NCU_TIMEOUT_S" env $extra_env "$NCU_BIN" \
      --metrics "$NCU_METRICS" \
      --replay-mode kernel \
      -k "$NCU_KERNEL" \
      --launch-skip "$NCU_LAUNCH_SKIP" \
      --launch-count "$NCU_LAUNCH_COUNT" \
      --target-processes all \
      -o "$profile_out" \
      --force-overwrite \
      "${cmd[@]}" \
      >>"$startup_log" 2>&1 &
  else
    env $extra_env "${cmd[@]}" \
      >>"$startup_log" 2>&1 &
  fi
  echo $!
}

wait_ready() {
  local port="$1"
  local tries="$WAIT_READY_TRIES"
  for _ in $(seq 1 "$tries"); do
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/" || true)
    if [[ ",$WAIT_READY_OK_CODES," == *",$code,"* ]]; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

cleanup() {
  if [[ -n "${PID_LITE:-}" ]]; then
    kill "$PID_LITE" >/dev/null 2>&1 || true
  fi
  if [[ -n "${PID_VANILLA:-}" ]]; then
    kill "$PID_VANILLA" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

request_once() {
  local port="$1"
  local server_label="$2"
  local run_idx="$3"
  local log_file="$4"
  local config_name="$5"

  local t0 t1 latency_ms response http_code payload
  t0=$(date +%s%3N)
  payload=$(cat <<JSON
{
  "model": "local",
  "messages": [
    {"role": "user", "content": $(python3 - <<PY
import json
print(json.dumps("""$PROMPT"""))
PY
    )}
  ],
  "max_tokens": $MAX_TOKENS,
  "temperature": $TEMPERATURE,
  "top_p": $TOP_P,
  "top_k": $TOP_K,
  "min_p": $MIN_P,
  "seed": $SEED
}
JSON
)

  for _ in $(seq 1 "$REQUEST_RETRIES"); do
    response=$(curl -s -w "\n%{http_code}" "http://127.0.0.1:${port}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "$payload")
    http_code="$(echo "$response" | tail -n 1)"
    response="$(echo "$response" | sed '$d')"
    if [[ "$http_code" == "200" ]]; then
      break
    fi
    sleep "$REQUEST_RETRY_SLEEP_S"
  done
  if [[ "$http_code" != "200" ]]; then
    echo "ERROR: request failed after ${REQUEST_RETRIES} retries (HTTP $http_code) on port $port" >&2
    exit 1
  fi

  t1=$(date +%s%3N)
  latency_ms=$((t1 - t0))

  # Extract last prompt/eval timing from server log for this request
  read -r log_prompt_ms log_prompt_tokens log_prompt_mspt log_eval_ms log_eval_tokens log_eval_mspt < <(
    python3 - "$log_file" <<'PY'
import sys
log_file = sys.argv[1]

def parse_line(line: str):
    # Expected format:
    # prompt eval time =  34.55 ms /     9 tokens (    3.84 ms per token,   260.45 tokens per second)
    # eval time =     359.75 ms /    75 tokens (    4.80 ms per token,   208.48 tokens per second)
    try:
        left, right = line.split(" ms /", 1)
        ms = left.split("=")[-1].strip()
        tokens = right.strip().split(" tokens", 1)[0].strip()
        mspt = line.split("(", 1)[1].split(" ms per token", 1)[0].strip()
        return ms, tokens, mspt
    except Exception:
        return "0", "0", "0"

last_prompt = None
last_eval = None
with open(log_file, "r", errors="ignore") as f:
    for line in f:
        if "prompt eval time" in line:
            last_prompt = line
        elif line.lstrip().startswith("eval time"):
            last_eval = line

if last_prompt:
    p_ms, p_tok, p_mspt = parse_line(last_prompt)
else:
    p_ms, p_tok, p_mspt = "0", "0", "0"
if last_eval:
    e_ms, e_tok, e_mspt = parse_line(last_eval)
else:
    e_ms, e_tok, e_mspt = "0", "0", "0"

print(p_ms, p_tok, p_mspt, e_ms, e_tok, e_mspt)
PY
  )

  python3 - <<PY
import json, sys
resp = json.loads("""$response""")
usage = resp.get("usage") or {}
prompt_toks = int(usage.get("prompt_tokens", 0))
comp_toks = int(usage.get("completion_tokens", 0))
total_toks = int(usage.get("total_tokens", 0))
ms = $latency_ms
ms_per = (ms / comp_toks) if comp_toks > 0 else 0.0
print("${RUN_ID},${config_name},${server_label},${run_idx},%d,%d,%d,%d,%.6f,%s,%s,%s,%s,%s,%s" % (
    ms, prompt_toks, comp_toks, total_toks, ms_per,
    "$log_prompt_ms", "$log_prompt_tokens", "$log_prompt_mspt",
    "$log_eval_ms", "$log_eval_tokens", "$log_eval_mspt",
))
PY
}

configs="$CONFIGS_DEFAULT"
if [[ -n "$CONFIGS_FILE" && -f "$CONFIGS_FILE" ]]; then
  configs="$(grep -v '^[[:space:]]*$' "$CONFIGS_FILE" | grep -v '^[[:space:]]*#' || true)"
fi

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  IFS='|' read -r cfg_name cfg_lite_env cfg_vanilla_env <<< "$line"
  cfg_name="${cfg_name:-baseline}"

  LITE_ENV_MERGED="${LITE_ENV_VARS} ${cfg_lite_env}"
  VANILLA_ENV_MERGED="${VANILLA_ENV_VARS} ${cfg_vanilla_env}"

  if [[ "$AUTO_PORTS" == "1" ]]; then
    read -r PORT_LITE PORT_VANILLA < <(
      python3 - <<'PY'
import socket
def pick():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port
p1 = pick()
p2 = pick()
while p2 == p1:
    p2 = pick()
print(p1, p2)
PY
    )
  fi

  if [[ "$PROFILE_SERIAL" == "1" ]]; then
    # Run llama-lite first
    PID_LITE="$(start_server "$LLAMA_LITE_BIN" "$PORT_LITE" "$LOG_LITE" "$STARTUP_LITE" "$LITE_ENV_MERGED" "${cfg_name}_lite")"
    echo "Waiting for servers to become ready... (config: $cfg_name, server: llama_lite)"
    if ! wait_ready "$PORT_LITE"; then
      echo "ERROR: llama-lite server failed to start on port $PORT_LITE" >&2
      if [[ -s "$STARTUP_LITE" ]]; then
        echo "Startup log (llama-lite):"
        tail -n 60 "$STARTUP_LITE"
      fi
      exit 1
    fi
    if [[ "$PROFILE_NCU" == "1" ]]; then
      sleep "$PROFILE_NCU_WARMUP_S"
    fi
    for i in $(seq 1 "$RUNS"); do
      request_once "$PORT_LITE" "llama_lite" "$i" "$LOG_LITE" "$cfg_name" >> "$CSV_OUT"
    done
    kill "$PID_LITE" >/dev/null 2>&1 || true
    PID_LITE=""

    # Then vanilla
    PID_VANILLA="$(start_server "$LLAMA_VANILLA_BIN" "$PORT_VANILLA" "$LOG_VANILLA" "$STARTUP_VANILLA" "$VANILLA_ENV_MERGED" "${cfg_name}_vanilla")"
    echo "Waiting for servers to become ready... (config: $cfg_name, server: vanilla)"
    if ! wait_ready "$PORT_VANILLA"; then
      echo "ERROR: vanilla server failed to start on port $PORT_VANILLA" >&2
      if [[ -s "$STARTUP_VANILLA" ]]; then
        echo "Startup log (vanilla):"
        tail -n 60 "$STARTUP_VANILLA"
      fi
      exit 1
    fi
    if [[ "$PROFILE_NCU" == "1" ]]; then
      sleep "$PROFILE_NCU_WARMUP_S"
    fi
    for i in $(seq 1 "$RUNS"); do
      request_once "$PORT_VANILLA" "vanilla" "$i" "$LOG_VANILLA" "$cfg_name" >> "$CSV_OUT"
    done
    kill "$PID_VANILLA" >/dev/null 2>&1 || true
    PID_VANILLA=""
  else
    PID_LITE="$(start_server "$LLAMA_LITE_BIN" "$PORT_LITE" "$LOG_LITE" "$STARTUP_LITE" "$LITE_ENV_MERGED" "${cfg_name}_lite")"
    PID_VANILLA="$(start_server "$LLAMA_VANILLA_BIN" "$PORT_VANILLA" "$LOG_VANILLA" "$STARTUP_VANILLA" "$VANILLA_ENV_MERGED" "${cfg_name}_vanilla")"

    echo "Waiting for servers to become ready... (config: $cfg_name)"
    if ! wait_ready "$PORT_LITE"; then
      echo "ERROR: llama-lite server failed to start on port $PORT_LITE" >&2
      if [[ -s "$STARTUP_LITE" ]]; then
        echo "Startup log (llama-lite):"
        tail -n 60 "$STARTUP_LITE"
      fi
      exit 1
    fi
    if ! wait_ready "$PORT_VANILLA"; then
      echo "ERROR: vanilla server failed to start on port $PORT_VANILLA" >&2
      if [[ -s "$STARTUP_VANILLA" ]]; then
        echo "Startup log (vanilla):"
        tail -n 60 "$STARTUP_VANILLA"
      fi
      exit 1
    fi

    for i in $(seq 1 "$RUNS"); do
      request_once "$PORT_LITE" "llama_lite" "$i" "$LOG_LITE" "$cfg_name" >> "$CSV_OUT"
      request_once "$PORT_VANILLA" "vanilla" "$i" "$LOG_VANILLA" "$cfg_name" >> "$CSV_OUT"
    done

    kill "$PID_LITE" >/dev/null 2>&1 || true
    kill "$PID_VANILLA" >/dev/null 2>&1 || true
    PID_LITE=""
    PID_VANILLA=""
  fi
done <<< "$configs"

echo "Done. CSV: $CSV_OUT"
echo "Logs: $LOG_LITE | $LOG_VANILLA"
