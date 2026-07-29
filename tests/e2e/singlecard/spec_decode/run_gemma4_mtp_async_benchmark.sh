#!/usr/bin/env bash
# Compare Gemma4 MTP scheduling/executor candidates with an identical workload.

set -euo pipefail

: "${VLLM_ASCEND_GEMMA4_MTP_MODEL:?set the target model path}"
: "${VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL:?set the draft model path}"

if [[ "${ASCEND_LAUNCH_BLOCKING:-0}" != "0" ]]; then
  echo "ASCEND_LAUNCH_BLOCKING must be unset or 0 for the ACL graph benchmark." >&2
  exit 2
fi
if [[ -n "${VLLM_ASCEND_GEMMA4_MTP_DEBUG:-}" ]]; then
  echo "Unset VLLM_ASCEND_GEMMA4_MTP_DEBUG before benchmarking." >&2
  exit 2
fi
if [[ "${VLLM_ASCEND_GEMMA4_MTP_ORACLE:-0}" != "0" ]]; then
  echo "Unset VLLM_ASCEND_GEMMA4_MTP_ORACLE before benchmarking." >&2
  exit 2
fi
if [[ "${VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE:-0}" != "0" ]]; then
  echo "Unset VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE before benchmarking." >&2
  exit 2
fi
if [[ "${HCCL_OP_EXPANSION_MODE:-AIV}" != "AIV" ]]; then
  echo "HCCL_OP_EXPANSION_MODE must be AIV for the fixed benchmark." >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL=${VLLM_ASCEND_GEMMA4_MTP_MODEL}
DRAFT_MODEL=${VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL}
SERVED_MODEL_NAME=${VLLM_ASCEND_GEMMA4_MTP_SERVED_MODEL:-gemma4}
NUM_SPECULATIVE_TOKENS=${VLLM_ASCEND_GEMMA4_MTP_K:-3}
MAX_MODEL_LEN=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_NUM_SEQS:-72}
MAX_BATCHED_TOKENS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_BATCHED_TOKENS:-16384}
GPU_MEMORY_UTILIZATION=${VLLM_ASCEND_GEMMA4_MTP_BENCH_GPU_MEMORY_UTILIZATION:-0.9}
BENCHMARK_PORT=${VLLM_ASCEND_GEMMA4_MTP_BENCH_PORT:-8100}
SERVER_LOG_LEVEL=${VLLM_ASCEND_GEMMA4_MTP_BENCH_LOG_LEVEL:-WARNING}
NUM_PROMPTS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_NUM_PROMPTS:-3000}
INPUT_LEN=${VLLM_ASCEND_GEMMA4_MTP_BENCH_INPUT_LEN:-256}
OUTPUT_LEN=${VLLM_ASCEND_GEMMA4_MTP_BENCH_OUTPUT_LEN:-128}
MAX_CONCURRENCY=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_CONCURRENCY:-72}
NUM_WARMUPS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_NUM_WARMUPS:-8}
REQUEST_RATE=${VLLM_ASCEND_GEMMA4_MTP_BENCH_REQUEST_RATE:-inf}
BENCHMARK_SEED=${VLLM_ASCEND_GEMMA4_MTP_BENCH_SEED:-0}
RESULT_ROOT=${VLLM_ASCEND_GEMMA4_MTP_BENCH_RESULT_ROOT:-"$(pwd)/gemma4_mtp_async_benchmark_$(date +%Y%m%d_%H%M%S)"}
MODES=("sync" "async_uni" "async_mp" "async_candidate")

require_fixed_value() {
  local name=$1
  local actual=$2
  local expected=$3
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${name} must be ${expected} for BENCH-01; got ${actual}." >&2
    exit 2
  fi
}

require_fixed_value "NUM_SPECULATIVE_TOKENS" "${NUM_SPECULATIVE_TOKENS}" "3"
require_fixed_value "MAX_MODEL_LEN" "${MAX_MODEL_LEN}" "32768"
require_fixed_value "MAX_NUM_SEQS" "${MAX_NUM_SEQS}" "72"
require_fixed_value "NUM_PROMPTS" "${NUM_PROMPTS}" "3000"
require_fixed_value "INPUT_LEN" "${INPUT_LEN}" "256"
require_fixed_value "OUTPUT_LEN" "${OUTPUT_LEN}" "128"
require_fixed_value "MAX_CONCURRENCY" "${MAX_CONCURRENCY}" "72"
require_fixed_value "REQUEST_RATE" "${REQUEST_RATE}" "inf"
require_fixed_value "BENCHMARK_SEED" "${BENCHMARK_SEED}" "0"

SPECULATIVE_CONFIG='{"method":"mtp","model":"'${DRAFT_MODEL}'","num_speculative_tokens": '${NUM_SPECULATIVE_TOKENS}'}'
SERVER_PID=""

mkdir -p "${RESULT_ROOT}"

stop_server() {
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT "${SERVER_PID}" || true
    for _ in $(seq 1 60); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -TERM "${SERVER_PID}" || true
    fi
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

wait_for_health() {
  local url=$1
  local log_file=$2
  for _ in $(seq 1 1800); do
    if curl --fail --silent "${url}/health" >/dev/null; then
      return
    fi
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 120 "${log_file}" >&2 || true
      echo "Server exited before becoming healthy." >&2
      exit 1
    fi
    sleep 1
  done
  tail -n 120 "${log_file}" >&2 || true
  echo "Timed out waiting for ${url}/health." >&2
  exit 1
}

run_mode() {
  local mode=$1
  local mode_dir="${RESULT_ROOT}/${mode}"
  local log_file="${mode_dir}/server.log"
  local result_file="${mode}.json"
  local scheduling_arg
  local executor_backend
  local candidate
  local -a server_args
  local -a server_env

  mkdir -p "${mode_dir}"
  case "${mode}" in
    sync)
      scheduling_arg="--no-async-scheduling"
      executor_backend="uni"
      candidate="0"
      ;;
    async_uni)
      scheduling_arg="--async-scheduling"
      executor_backend="uni"
      candidate="0"
      ;;
    async_mp)
      scheduling_arg="--async-scheduling"
      executor_backend="mp"
      candidate="0"
      ;;
    async_candidate)
      scheduling_arg="--async-scheduling"
      executor_backend="uni"
      candidate="1"
      ;;
    *)
      echo "Unknown benchmark mode: ${mode}" >&2
      exit 2
      ;;
  esac

  server_env=(
    env
    "HCCL_OP_EXPANSION_MODE=AIV"
    "VLLM_LOGGING_LEVEL=${SERVER_LOG_LEVEL}"
    "VLLM_BATCH_INVARIANT=0"
    "VLLM_ASCEND_GEMMA4_MTP_ORACLE=0"
    "VLLM_ASCEND_GEMMA4_MTP_DEBUG=0"
    "VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE=0"
    "VLLM_ASCEND_GEMMA4_MTP_ASYNC_UNIPROC_SUBMIT=${candidate}"
  )

  server_args=(
    vllm serve "${MODEL}"
    --host 0.0.0.0
    --port "${BENCHMARK_PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --kv-cache-dtype auto
    --tensor-parallel-size 1
    --distributed-executor-backend "${executor_backend}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --optimization-level 3
    --trust-remote-code
    --enable-prefix-caching
    --enable-chunked-prefill
    --enable-flashinfer-autotune
    --language-model-only
    --performance-mode throughput
    --served-model-name "${SERVED_MODEL_NAME}"
    --speculative-config "${SPECULATIVE_CONFIG}"
    "${scheduling_arg}"
  )

  printf '%q ' "${server_env[@]}" "${server_args[@]}" >"${mode_dir}/server-command.txt"
  printf '\n' >>"${mode_dir}/server-command.txt"
  "${server_env[@]}" "${server_args[@]}" >"${log_file}" 2>&1 &
  SERVER_PID=$!
  wait_for_health "http://127.0.0.1:${BENCHMARK_PORT}" "${log_file}"

  vllm bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:${BENCHMARK_PORT}" \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer "${MODEL}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --random-range-ratio 0.0 \
    --num-prompts "${NUM_PROMPTS}" \
    --num-warmups "${NUM_WARMUPS}" \
    --request-rate "${REQUEST_RATE}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --seed "${BENCHMARK_SEED}" \
    --disable-shuffle \
    --disable-tqdm \
    --ignore-eos \
    --temperature 0 \
    --percentile-metrics ttft,e2el \
    --metric-percentiles 50,95 \
    --save-result \
    --save-detailed \
    --result-dir "${mode_dir}" \
    --result-filename "${result_file}" \
    --metadata "mode=${mode}" "executor_backend=${executor_backend}" \
      "candidate=${candidate}" "k=${NUM_SPECULATIVE_TOKENS}" \
      "max_model_len=${MAX_MODEL_LEN}" "input_len=${INPUT_LEN}" \
      "output_len=${OUTPUT_LEN}" "max_concurrency=${MAX_CONCURRENCY}" \
      "max_num_seqs=${MAX_NUM_SEQS}" "num_prompts=${NUM_PROMPTS}" \
      "request_rate=${REQUEST_RATE}" "temperature=0" \
      "seed=${BENCHMARK_SEED}" "hccl_op_expansion_mode=AIV" \
    | tee "${mode_dir}/client.log"

  curl --fail --silent --show-error "http://127.0.0.1:${BENCHMARK_PORT}/metrics" \
    >"${mode_dir}/metrics.prom"
  stop_server
}

trap stop_server EXIT

if curl --fail --silent "http://127.0.0.1:${BENCHMARK_PORT}/health" >/dev/null; then
  echo "Benchmark port ${BENCHMARK_PORT} already has a healthy service." >&2
  exit 2
fi

printf 'model=%s\ndraft_model=%s\nmodes=%s\nk=%s\nmax_model_len=%s\nmax_num_seqs=%s\nnum_prompts=%s\ninput_len=%s\noutput_len=%s\nmax_concurrency=%s\nrequest_rate=%s\ntemperature=0\nseed=%s\nhccl_op_expansion_mode=AIV\n' \
  "${MODEL}" "${DRAFT_MODEL}" "${MODES[*]}" "${NUM_SPECULATIVE_TOKENS}" \
  "${MAX_MODEL_LEN}" "${MAX_NUM_SEQS}" "${NUM_PROMPTS}" "${INPUT_LEN}" \
  "${OUTPUT_LEN}" "${MAX_CONCURRENCY}" \
  "${REQUEST_RATE}" "${BENCHMARK_SEED}" >"${RESULT_ROOT}/manifest.txt"

for mode in "${MODES[@]}"; do
  run_mode "${mode}"
done

python3 "${SCRIPT_DIR}/summarize_gemma4_mtp_async_benchmark.py" \
  "${RESULT_ROOT}/summary.md" "${RESULT_ROOT}"

echo "Benchmark artifacts: ${RESULT_ROOT}"
