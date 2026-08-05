#!/usr/bin/env bash
# Measure fixed-length Gemma4 long-prompt TTFT without profiler overhead.

set +e
set -o pipefail
export ASCEND_RT_VISIBLE_DEVICES=0

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <code-root> <run-set-dir>" >&2
  exit 0
fi

CODE_ROOT=$1
RUN_SET_DIR=$2
TARGET_MODEL=${VLLM_ASCEND_GEMMA4_MTP_MODEL:-}
DRAFT_MODEL=${VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL:-}
PORT=${GEMMA4_PREFILL_TTFT_PORT:-8101}
GPU_MEMORY_UTILIZATION=${GEMMA4_PREFILL_TTFT_GPU_MEMORY_UTILIZATION:-0.92}
NUM_PROMPTS=${GEMMA4_PREFILL_TTFT_NUM_PROMPTS:-10}
NUM_WARMUPS=${GEMMA4_PREFILL_TTFT_NUM_WARMUPS:-1}
SERVER_LOG_LEVEL=${GEMMA4_PREFILL_TTFT_LOG_LEVEL:-WARNING}
SERVED_MODEL_NAME=gemma4-prefill
MODES=("target" "mtp")
PROMPT_LENGTHS=(8192 16384 28672)
SERVER_PID=""
SERVER_LOG_TAIL_PID=""

mkdir -p "$RUN_SET_DIR"

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 120); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "$SERVER_PID" ]]; then
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""

  if [[ -n "$SERVER_LOG_TAIL_PID" ]]; then
    kill "$SERVER_LOG_TAIL_PID" 2>/dev/null || true
    wait "$SERVER_LOG_TAIL_PID" 2>/dev/null || true
  fi
  SERVER_LOG_TAIL_PID=""
}

wait_for_health() {
  local log_file=$1
  for attempt in $(seq 1 1800); do
    if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
      return 0
    fi
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server exited before becoming healthy." >&2
      tail -n 200 "$log_file" >&2 || true
      return 1
    fi
    if (( attempt % 30 == 0 )); then
      echo "Waiting for server health: ${attempt}s"
    fi
    sleep 1
  done
  echo "Timed out waiting for the benchmark server." >&2
  tail -n 200 "$log_file" >&2 || true
  return 1
}

build_benchmark_args() {
  BENCH_ARGS=(
    vllm bench serve
    --backend openai
    --base-url "http://127.0.0.1:$PORT"
    --endpoint /v1/completions
    --model "$SERVED_MODEL_NAME"
    --tokenizer "$TARGET_MODEL"
    --dataset-name random
    --random-input-len "$PROMPT_TOKENS"
    --random-output-len 1
    --random-range-ratio '{"input":0.0,"output":0.0}'
    --num-prompts "$NUM_PROMPTS"
    --num-warmups "$NUM_WARMUPS"
    --request-rate inf
    --max-concurrency 1
    --seed 0
    --disable-shuffle
    --disable-tqdm
    --ignore-eos
    --temperature 0
    --percentile-metrics ttft,e2el
    --metric-percentiles 50,90,99
    --save-result
    --save-detailed
    --result-dir "$CASE_DIR"
    --result-filename "$RESULT_FILE"
    --request-id-prefix "$MODE-$PROMPT_TOKENS-"
    --metadata
      "mode=$MODE"
      "prompt_tokens=$PROMPT_TOKENS"
      "num_prompts=$NUM_PROMPTS"
      "max_concurrency=1"
      "output_tokens=1"
  )
}

trap stop_server EXIT

ENVIRONMENT_RC=0
for VARIABLE_NAME in \
  ASCEND_LAUNCH_BLOCKING \
  VLLM_ASCEND_GEMMA4_MTP_DEBUG \
  VLLM_ASCEND_GEMMA4_MTP_ORACLE \
  VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE \
  VLLM_ASCEND_GEMMA4_MTP_ASYNC_UNIPROC_SUBMIT; do
  VARIABLE_VALUE=${!VARIABLE_NAME:-0}
  if [[ "$VARIABLE_VALUE" != "0" ]]; then
    echo "$VARIABLE_NAME must be unset or 0 for TTFT." >&2
    ENVIRONMENT_RC=2
  fi
done

if [[ -z "$TARGET_MODEL" || -z "$DRAFT_MODEL" ]]; then
  echo "Target and draft model environment variables must be set." >&2
  ENVIRONMENT_RC=2
fi

IMPORT_OUTPUT=$(
  PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}" python -c \
    'import pathlib, vllm_ascend; print(pathlib.Path(vllm_ascend.__file__).resolve())' \
    2>&1
)
IMPORT_RC=$?
IMPORT_PATH=$(printf '%s\n' "$IMPORT_OUTPUT" | tail -n 1)
if [[ $IMPORT_RC -ne 0 || "$IMPORT_PATH" != "$CODE_ROOT"/* ]]; then
  echo "Imported vllm_ascend is not from the requested code root." >&2
  printf '%s\n' "$IMPORT_OUTPUT" >&2
  ENVIRONMENT_RC=2
fi

REPO_SHA=$(git -C "$CODE_ROOT" rev-parse HEAD 2>/dev/null)
REPO_SHA_RC=$?
if [[ $REPO_SHA_RC -ne 0 || -z "$REPO_SHA" ]]; then
  echo "Unable to resolve repo SHA from $CODE_ROOT." >&2
  ENVIRONMENT_RC=2
fi

if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "Benchmark port $PORT already has a healthy service." >&2
  ENVIRONMENT_RC=2
fi

if [[ $ENVIRONMENT_RC -ne 0 ]]; then
  {
    echo "environment_exit_code=$ENVIRONMENT_RC"
    echo "import_path=$IMPORT_PATH"
    echo "repo_sha=$REPO_SHA"
  } >"$RUN_SET_DIR/matrix-status.txt"
  exit 0
fi

FAILED_CASES=0
for MODE in "${MODES[@]}"; do
  MODE_DIR="$RUN_SET_DIR/$MODE"
  SERVER_LOG="$MODE_DIR/server.log"
  SERVER_STATUS="$MODE_DIR/server-status.txt"
  SERVER_COMMAND="$MODE_DIR/server-command.txt"
  mkdir -p "$MODE_DIR"

  SERVER_ARGS=(
    vllm serve "$TARGET_MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --max-model-len 32768
    --max-num-seqs 1
    --max-num-batched-tokens 8192
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size 1
    --distributed-executor-backend uni
    --seed 0
    --trust-remote-code
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --language-model-only
    --no-async-scheduling
  )
  if [[ "$MODE" == "mtp" ]]; then
    SPECULATIVE_CONFIG=$(
      python - "$DRAFT_MODEL" <<'PY'
import json
import sys

print(json.dumps({
    "method": "mtp",
    "model": sys.argv[1],
    "num_speculative_tokens": 3,
    "max_model_len": 32768,
}))
PY
    )
    SERVER_ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
  fi

  printf 'PYTHONPATH=%q VLLM_LOGGING_LEVEL=%q ' \
    "$CODE_ROOT:${PYTHONPATH:-}" "$SERVER_LOG_LEVEL" \
    >"$SERVER_COMMAND"
  printf '%q ' "${SERVER_ARGS[@]}" >>"$SERVER_COMMAND"
  printf '\n' >>"$SERVER_COMMAND"

  : >"$SERVER_LOG"
  tail -n +1 -F "$SERVER_LOG" &
  SERVER_LOG_TAIL_PID=$!
  PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}" \
    VLLM_LOGGING_LEVEL="$SERVER_LOG_LEVEL" \
    "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "server pid: $SERVER_PID"

  wait_for_health "$SERVER_LOG"
  HEALTH_RC=$?
  echo "server health exit code: $HEALTH_RC" | tee "$SERVER_STATUS"

  for PROMPT_TOKENS in "${PROMPT_LENGTHS[@]}"; do
    CASE_DIR="$MODE_DIR/$PROMPT_TOKENS"
    RESULT_FILE="${MODE}_${PROMPT_TOKENS}.json"
    CLIENT_LOG="$CASE_DIR/client.log"
    CLIENT_COMMAND="$CASE_DIR/client-command.txt"
    STATUS_FILE="$CASE_DIR/status.txt"
    mkdir -p "$CASE_DIR"

    build_benchmark_args
    printf 'PYTHONPATH=%q ' "$CODE_ROOT:${PYTHONPATH:-}" \
      >"$CLIENT_COMMAND"
    printf '%q ' "${BENCH_ARGS[@]}" >>"$CLIENT_COMMAND"
    printf '\n' >>"$CLIENT_COMMAND"

    if [[ $HEALTH_RC -eq 0 ]]; then
      PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}" \
        "${BENCH_ARGS[@]}" 2>&1 | tee "$CLIENT_LOG"
      BENCH_RC=${PIPESTATUS[0]}
    else
      BENCH_RC=125
      echo "benchmark skipped because server health failed: $HEALTH_RC" |
        tee "$CLIENT_LOG"
    fi
    echo "benchmark exit code: $BENCH_RC" | tee -a "$CLIENT_LOG"
    {
      echo "mode=$MODE"
      echo "prompt_tokens=$PROMPT_TOKENS"
      echo "server_health_exit_code=$HEALTH_RC"
      echo "benchmark_exit_code=$BENCH_RC"
      echo "result_file=$CASE_DIR/$RESULT_FILE"
      echo "client_log=$CLIENT_LOG"
      echo "client_command=$CLIENT_COMMAND"
    } >"$STATUS_FILE"
    if [[ $BENCH_RC -ne 0 ]]; then
      FAILED_CASES=$((FAILED_CASES + 1))
    fi
  done
  stop_server
done

python - "$RUN_SET_DIR/manifest.json" "$RUN_SET_DIR" "$CODE_ROOT" \
  "$REPO_SHA" "$IMPORT_PATH" "$TARGET_MODEL" "$DRAFT_MODEL" \
  "$NUM_PROMPTS" "$NUM_WARMUPS" "$FAILED_CASES" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
run_set_dir = Path(sys.argv[2])
repo_sha = sys.argv[4]
server_commands = {}
benchmark_commands = {}
case_results = []
for mode in ("target", "mtp"):
    server_command_path = run_set_dir / mode / "server-command.txt"
    server_commands[mode] = (
        server_command_path.read_text(encoding="utf-8").strip()
        if server_command_path.is_file()
        else None
    )
    for prompt_tokens in (8192, 16384, 28672):
        case_dir = run_set_dir / mode / str(prompt_tokens)
        command_path = case_dir / "client-command.txt"
        status_path = case_dir / "status.txt"
        case_key = f"{mode}_{prompt_tokens}"
        benchmark_commands[case_key] = (
            command_path.read_text(encoding="utf-8").strip()
            if command_path.is_file()
            else None
        )
        status = {}
        if status_path.is_file():
            for line in status_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator:
                    status[key] = value
        benchmark_exit_code = status.get("benchmark_exit_code")
        server_health_exit_code = status.get("server_health_exit_code")
        case_results.append(
            {
                "mode": mode,
                "prompt_tokens": prompt_tokens,
                "server_health_exit_code": (
                    int(server_health_exit_code)
                    if server_health_exit_code is not None
                    else None
                ),
                "benchmark_exit_code": (
                    int(benchmark_exit_code)
                    if benchmark_exit_code is not None
                    else None
                ),
                "result_file": status.get("result_file"),
                "client_log": status.get("client_log"),
                "client_command": status.get("client_command"),
            }
        )
payload = {
    "workload": "gemma4_dense_target_prefill_ttft",
    "code_root": sys.argv[3],
    "repo_sha": repo_sha,
    "import_path": sys.argv[5],
    "target_model": sys.argv[6],
    "draft_model": sys.argv[7],
    "modes": ["target", "mtp"],
    "prompt_tokens": [8192, 16384, 28672],
    "num_prompts": int(sys.argv[8]),
    "num_warmups": int(sys.argv[9]),
    "max_concurrency": 1,
    "output_tokens": 1,
    "max_model_len": 32768,
    "max_num_batched_tokens": 8192,
    "prefix_caching": False,
    "async_scheduling": False,
    "ascend_rt_visible_devices": "0",
    "server_commands": server_commands,
    "benchmark_commands": benchmark_commands,
    "case_results": case_results,
    "failed_cases": int(sys.argv[10]),
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY
MANIFEST_RC=$?

{
  echo "failed_cases=$FAILED_CASES"
  echo "case_count=6"
  echo "manifest_exit_code=$MANIFEST_RC"
} | tee "$RUN_SET_DIR/matrix-status.txt"
echo "TTFT artifacts: $RUN_SET_DIR"

# Preserve the interactive container even when one or more recorded cases fail.
exit 0
