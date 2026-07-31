#!/usr/bin/env bash
# Compare no-MTP UniProc, MTP UniProc, and MTP multiprocessing on one NPU.

set -euo pipefail

: "${VLLM_ASCEND_GEMMA4_MTP_MODEL:?set the target model path}"
: "${VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL:?set the draft model path}"

if [[ "${ASCEND_LAUNCH_BLOCKING:-0}" != "0" ]]; then
  echo "ASCEND_LAUNCH_BLOCKING must be unset or 0 for the benchmark." >&2
  exit 2
fi
if [[ "${VLLM_ASCEND_GEMMA4_MTP_ORACLE:-0}" != "0" ]]; then
  echo "Unset VLLM_ASCEND_GEMMA4_MTP_ORACLE before benchmarking." >&2
  exit 2
fi
if [[ "${VLLM_ASCEND_GEMMA4_MTP_DEBUG:-0}" != "0" ]]; then
  echo "Unset VLLM_ASCEND_GEMMA4_MTP_DEBUG before benchmarking." >&2
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
MAX_MODEL_LEN=${VLLM_ASCEND_GEMMA4_MTP_AB_MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${VLLM_ASCEND_GEMMA4_MTP_AB_MAX_NUM_SEQS:-24}
MAX_BATCHED_TOKENS=${VLLM_ASCEND_GEMMA4_MTP_AB_MAX_BATCHED_TOKENS:-16384}
GPU_MEMORY_UTILIZATION=${VLLM_ASCEND_GEMMA4_MTP_AB_GPU_MEMORY_UTILIZATION:-0.9}
BENCHMARK_PORT=${VLLM_ASCEND_GEMMA4_MTP_AB_PORT:-8100}
SERVER_LOG_LEVEL=${VLLM_ASCEND_GEMMA4_MTP_AB_LOG_LEVEL:-WARNING}
NUM_PROMPTS=${VLLM_ASCEND_GEMMA4_MTP_AB_NUM_PROMPTS:-200}
INPUT_LEN=${VLLM_ASCEND_GEMMA4_MTP_AB_INPUT_LEN:-12500}
OUTPUT_LEN=${VLLM_ASCEND_GEMMA4_MTP_AB_OUTPUT_LEN:-1024}
MAX_CONCURRENCY=${VLLM_ASCEND_GEMMA4_MTP_AB_MAX_CONCURRENCY:-24}
NUM_WARMUPS=${VLLM_ASCEND_GEMMA4_MTP_AB_NUM_WARMUPS:-16}
REQUEST_RATE=${VLLM_ASCEND_GEMMA4_MTP_AB_REQUEST_RATE:-inf}
BENCHMARK_SEED=${VLLM_ASCEND_GEMMA4_MTP_AB_SEED:-0}
COOLDOWN_SECONDS=${VLLM_ASCEND_GEMMA4_MTP_AB_COOLDOWN_SECONDS:-60}
RESULT_ROOT=${VLLM_ASCEND_GEMMA4_MTP_AB_RESULT_ROOT:-"$(pwd)/gemma4_mtp_ab_$(date +%Y%m%d_%H%M%S)"}
RANGE_RATIO='{"input":0.2,"output":0.0}'
MODES=("no_mtp_uni" "mtp_uni" "mtp_mp")

require_fixed_value() {
  local name=$1
  local actual=$2
  local expected=$3
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${name} must be ${expected} for the fixed A/B benchmark; got ${actual}." >&2
    exit 2
  fi
}

require_fixed_value "NUM_SPECULATIVE_TOKENS" "${NUM_SPECULATIVE_TOKENS}" "3"
require_fixed_value "MAX_MODEL_LEN" "${MAX_MODEL_LEN}" "32768"
require_fixed_value "MAX_NUM_SEQS" "${MAX_NUM_SEQS}" "24"
require_fixed_value "MAX_BATCHED_TOKENS" "${MAX_BATCHED_TOKENS}" "16384"
require_fixed_value "NUM_PROMPTS" "${NUM_PROMPTS}" "200"
require_fixed_value "INPUT_LEN" "${INPUT_LEN}" "12500"
require_fixed_value "OUTPUT_LEN" "${OUTPUT_LEN}" "1024"
require_fixed_value "MAX_CONCURRENCY" "${MAX_CONCURRENCY}" "24"
require_fixed_value "REQUEST_RATE" "${REQUEST_RATE}" "inf"
require_fixed_value "BENCHMARK_SEED" "${BENCHMARK_SEED}" "0"

SPECULATIVE_CONFIG='{"method":"mtp","model":"'${DRAFT_MODEL}'","num_speculative_tokens":'${NUM_SPECULATIVE_TOKENS}'}'
SERVER_PID=""

mkdir -p "${RESULT_ROOT}"

stop_server() {
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT "${SERVER_PID}" || true
    for _ in $(seq 1 120); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

wait_for_health() {
  local log_file=$1
  for _ in $(seq 1 1800); do
    if curl --fail --silent "http://127.0.0.1:${BENCHMARK_PORT}/health" >/dev/null; then
      return
    fi
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 200 "${log_file}" >&2 || true
      echo "Server exited before becoming healthy." >&2
      exit 1
    fi
    sleep 1
  done
  tail -n 200 "${log_file}" >&2 || true
  echo "Timed out waiting for the benchmark server." >&2
  exit 1
}

run_mode() {
  local round=$1
  local mode=$2
  local mode_dir="${RESULT_ROOT}/round_${round}/${mode}"
  local log_file="${mode_dir}/server.log"
  local result_file="${mode}.json"
  local executor_backend
  local use_mtp
  local k
  local -a server_args
  local -a server_env

  case "${mode}" in
    no_mtp_uni)
      executor_backend="uni"
      use_mtp="0"
      k="0"
      ;;
    mtp_uni)
      executor_backend="uni"
      use_mtp="1"
      k="${NUM_SPECULATIVE_TOKENS}"
      ;;
    mtp_mp)
      executor_backend="mp"
      use_mtp="1"
      k="${NUM_SPECULATIVE_TOKENS}"
      ;;
    *)
      echo "Unknown benchmark mode: ${mode}" >&2
      exit 2
      ;;
  esac

  mkdir -p "${mode_dir}"
  server_env=(
    env
    "ASCEND_RT_VISIBLE_DEVICES=0"
    "ASCEND_LAUNCH_BLOCKING=0"
    "HCCL_OP_EXPANSION_MODE=AIV"
    "PYTHONHASHSEED=0"
    "VLLM_LOGGING_LEVEL=${SERVER_LOG_LEVEL}"
    "VLLM_BATCH_INVARIANT=0"
    "VLLM_ASCEND_GEMMA4_MTP_ORACLE=0"
    "VLLM_ASCEND_GEMMA4_MTP_DEBUG=0"
    "VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE=0"
    "VLLM_ASCEND_GEMMA4_MTP_ASYNC_UNIPROC_SUBMIT=0"
  )
  server_args=(
    vllm serve "${MODEL}"
    --host 0.0.0.0
    --port "${BENCHMARK_PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --kv-cache-dtype auto
    --tensor-parallel-size 1
    --distributed-executor-backend "${executor_backend}"
    --seed 0
    --optimization-level 3
    --trust-remote-code
    --enable-prefix-caching
    --enable-chunked-prefill
    --enable-flashinfer-autotune
    --language-model-only
    --performance-mode throughput
    --async-scheduling
  )
  if [[ "${use_mtp}" == "1" ]]; then
    server_args+=(--speculative-config "${SPECULATIVE_CONFIG}")
  fi

  printf '%q ' "${server_env[@]}" "${server_args[@]}" >"${mode_dir}/server-command.txt"
  printf '\n' >>"${mode_dir}/server-command.txt"
  "${server_env[@]}" "${server_args[@]}" >"${log_file}" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "${SERVER_PID}" >"${mode_dir}/server.pid"
  wait_for_health "${log_file}"

  npu-smi info >"${mode_dir}/npu-smi-before.txt" 2>&1 || true
  vllm bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:${BENCHMARK_PORT}" \
    --endpoint /v1/completions \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer "${MODEL}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --random-range-ratio "${RANGE_RATIO}" \
    --num-prompts "${NUM_PROMPTS}" \
    --num-warmups "${NUM_WARMUPS}" \
    --request-rate "${REQUEST_RATE}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --seed "${BENCHMARK_SEED}" \
    --disable-shuffle \
    --disable-tqdm \
    --ignore-eos \
    --temperature 0 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --result-dir "${mode_dir}" \
    --result-filename "${result_file}" \
    --request-id-prefix "r${round}-${mode}-" \
    --metadata "round=${round}" "mode=${mode}" \
      "executor_backend=${executor_backend}" "use_mtp=${use_mtp}" "k=${k}" \
      "device=0" "tensor_parallel_size=1" "max_model_len=${MAX_MODEL_LEN}" \
      "max_num_seqs=${MAX_NUM_SEQS}" "max_batched_tokens=${MAX_BATCHED_TOKENS}" \
      "input_len=${INPUT_LEN}" "input_min=10000" "input_max=15000" \
      "output_len=${OUTPUT_LEN}" "max_concurrency=${MAX_CONCURRENCY}" \
      "num_prompts=${NUM_PROMPTS}" "request_rate=${REQUEST_RATE}" \
      "temperature=0" "seed=${BENCHMARK_SEED}" \
    | tee "${mode_dir}/client.log"

  curl --fail --silent --show-error "http://127.0.0.1:${BENCHMARK_PORT}/metrics" \
    >"${mode_dir}/metrics.prom"
  npu-smi info >"${mode_dir}/npu-smi-after.txt" 2>&1 || true
  stop_server
}

run_round() {
  local round=$1
  shift
  local mode
  for mode in "$@"; do
    run_mode "${round}" "${mode}"
    sleep "${COOLDOWN_SECONDS}"
  done
}

trap stop_server EXIT

if curl --fail --silent "http://127.0.0.1:${BENCHMARK_PORT}/health" >/dev/null; then
  echo "Benchmark port ${BENCHMARK_PORT} already has a healthy service." >&2
  exit 2
fi

printf 'model=%s\ndraft_model=%s\nmodes=%s\ndevice=0\ntensor_parallel_size=1\nmtp_k=%s\nmax_model_len=%s\nmax_num_seqs=%s\nmax_batched_tokens=%s\nnum_prompts=%s\ninput_len=%s\ninput_min=10000\ninput_max=15000\noutput_len=%s\nmax_concurrency=%s\nrequest_rate=%s\nseed=%s\n' \
  "${MODEL}" "${DRAFT_MODEL}" "${MODES[*]}" "${NUM_SPECULATIVE_TOKENS}" \
  "${MAX_MODEL_LEN}" "${MAX_NUM_SEQS}" "${MAX_BATCHED_TOKENS}" \
  "${NUM_PROMPTS}" "${INPUT_LEN}" "${OUTPUT_LEN}" "${MAX_CONCURRENCY}" \
  "${REQUEST_RATE}" "${BENCHMARK_SEED}" >"${RESULT_ROOT}/manifest.txt"

run_round 1 no_mtp_uni mtp_uni mtp_mp
run_round 2 mtp_mp mtp_uni no_mtp_uni
run_round 3 no_mtp_uni mtp_mp mtp_uni

python3 "${SCRIPT_DIR}/summarize_gemma4_mtp_ab_benchmark.py" \
  "${RESULT_ROOT}/summary.md" \
  "${RESULT_ROOT}/round_1" "${RESULT_ROOT}/round_2" "${RESULT_ROOT}/round_3"

echo "Benchmark artifacts: ${RESULT_ROOT}"
