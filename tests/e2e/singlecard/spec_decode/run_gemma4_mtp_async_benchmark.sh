#!/usr/bin/env bash
# Compare Gemma4 MTP sync and async scheduling with an identical workload.

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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL=${VLLM_ASCEND_GEMMA4_MTP_MODEL}
DRAFT_MODEL=${VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL}
SERVED_MODEL_NAME=${VLLM_ASCEND_GEMMA4_MTP_SERVED_MODEL:-gemma4}
NUM_SPECULATIVE_TOKENS=${VLLM_ASCEND_GEMMA4_MTP_K:-3}
MAX_MODEL_LEN=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_NUM_SEQS:-64}
MAX_BATCHED_TOKENS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_BATCHED_TOKENS:-16384}
GPU_MEMORY_UTILIZATION=${VLLM_ASCEND_GEMMA4_MTP_BENCH_GPU_MEMORY_UTILIZATION:-0.9}
BENCHMARK_PORT=${VLLM_ASCEND_GEMMA4_MTP_BENCH_PORT:-8100}
NUM_PROMPTS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_NUM_PROMPTS:-64}
INPUT_LEN=${VLLM_ASCEND_GEMMA4_MTP_BENCH_INPUT_LEN:-256}
OUTPUT_LEN=${VLLM_ASCEND_GEMMA4_MTP_BENCH_OUTPUT_LEN:-128}
MAX_CONCURRENCY=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_CONCURRENCY:-16}
NUM_WARMUPS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_NUM_WARMUPS:-8}
REQUEST_RATE=${VLLM_ASCEND_GEMMA4_MTP_BENCH_REQUEST_RATE:-inf}
BENCHMARK_SEED=${VLLM_ASCEND_GEMMA4_MTP_BENCH_SEED:-0}
RESULT_ROOT=${VLLM_ASCEND_GEMMA4_MTP_BENCH_RESULT_ROOT:-"$(pwd)/gemma4_mtp_async_benchmark_$(date +%Y%m%d_%H%M%S)"}
MODES=("sync" "async")

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
    if curl --fail --silent --show-error "${url}/health" >/dev/null; then
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
  local -a server_args

  mkdir -p "${mode_dir}"
  if [[ "${mode}" == "sync" ]]; then
    scheduling_arg="--no-async-scheduling"
  else
    scheduling_arg="--async-scheduling"
  fi

  server_args=(
    vllm serve "${MODEL}"
    --host 0.0.0.0
    --port "${BENCHMARK_PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --kv-cache-dtype auto
    --tensor-parallel-size 1
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

  printf '%q ' "${server_args[@]}" >"${mode_dir}/server-command.txt"
  printf '\n' >>"${mode_dir}/server-command.txt"
  "${server_args[@]}" >"${log_file}" 2>&1 &
  SERVER_PID=$!
  wait_for_health "http://127.0.0.1:${BENCHMARK_PORT}" "${log_file}"

  vllm bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:${BENCHMARK_PORT}" \
    --model "${SERVED_MODEL_NAME}" \
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
    --ignore-eos \
    --temperature 0 \
    --percentile-metrics ttft,e2el \
    --metric-percentiles 50,95 \
    --save-result \
    --save-detailed \
    --result-dir "${mode_dir}" \
    --result-filename "${result_file}" \
    --metadata "mode=${mode}" "k=${NUM_SPECULATIVE_TOKENS}" \
      "max_model_len=${MAX_MODEL_LEN}" "input_len=${INPUT_LEN}" \
      "output_len=${OUTPUT_LEN}" "max_concurrency=${MAX_CONCURRENCY}" \
      "num_prompts=${NUM_PROMPTS}" "seed=${BENCHMARK_SEED}" \
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

printf 'model=%s\ndraft_model=%s\nk=%s\nmax_model_len=%s\nnum_prompts=%s\ninput_len=%s\noutput_len=%s\nmax_concurrency=%s\nrequest_rate=%s\nseed=%s\n' \
  "${MODEL}" "${DRAFT_MODEL}" "${NUM_SPECULATIVE_TOKENS}" "${MAX_MODEL_LEN}" \
  "${NUM_PROMPTS}" "${INPUT_LEN}" "${OUTPUT_LEN}" "${MAX_CONCURRENCY}" \
  "${REQUEST_RATE}" "${BENCHMARK_SEED}" >"${RESULT_ROOT}/manifest.txt"

for mode in "${MODES[@]}"; do
  run_mode "${mode}"
done

python3 "${SCRIPT_DIR}/summarize_gemma4_mtp_async_benchmark.py" \
  "${RESULT_ROOT}/sync/sync.json" "${RESULT_ROOT}/async/async.json" \
  "${RESULT_ROOT}/summary.md"

echo "Benchmark artifacts: ${RESULT_ROOT}"
