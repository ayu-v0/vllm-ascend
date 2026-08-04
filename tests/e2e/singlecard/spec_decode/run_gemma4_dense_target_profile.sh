#!/usr/bin/env bash
# Run one Gemma4 dense-target profile without terminating an interactive shell.

set +e
set -o pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <label> <code-root> <profile-root>" >&2
  exit 0
fi

LABEL=$1
CODE_ROOT=$2
PROFILE_ROOT=$3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_RUNNER="${SCRIPT_DIR}/profile_gemma4_dense_target_decode.py"
TARGET_MODEL=${VLLM_ASCEND_GEMMA4_MTP_MODEL:-}
DRAFT_MODEL=${VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL:-}
TP_SIZE=${GEMMA4_DENSE_PROFILE_TP_SIZE:-1}
NUM_SPECULATIVE_TOKENS=${VLLM_ASCEND_GEMMA4_MTP_K:-3}
RUN_DIR="${PROFILE_ROOT}/${LABEL}"
TRACE_ROOT="${RUN_DIR}/profile"
LOG_FILE="${RUN_DIR}/run.log"
MANIFEST_FILE="${RUN_DIR}/manifest.json"
TOKEN_IDS_FILE="${RUN_DIR}/token_ids.json"
STATUS_FILE="${RUN_DIR}/status.txt"

mkdir -p "${RUN_DIR}" "${TRACE_ROOT}"

for VARIABLE_NAME in \
  ASCEND_LAUNCH_BLOCKING \
  VLLM_ASCEND_GEMMA4_MTP_ORACLE \
  VLLM_ASCEND_GEMMA4_MTP_DEBUG \
  VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE; do
  VARIABLE_VALUE=${!VARIABLE_NAME:-0}
  if [[ "${VARIABLE_VALUE}" != "0" ]]; then
    {
      echo "${VARIABLE_NAME} must be unset or 0 for profiling."
      echo "profile run exit code: 2"
    } | tee "${LOG_FILE}"
    exit 0
  fi
done

if [[ -z "${TARGET_MODEL}" || -z "${DRAFT_MODEL}" ]]; then
  {
    echo "VLLM_ASCEND_GEMMA4_MTP_MODEL and"
    echo "VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL must be set."
    echo "profile run exit code: 2"
  } | tee "${LOG_FILE}"
  exit 0
fi

IMPORT_OUTPUT=$(
  PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}" python -c \
    'import pathlib, vllm_ascend; print(pathlib.Path(vllm_ascend.__file__).resolve())'
)
IMPORT_RC=$?
IMPORT_PATH=$(printf '%s\n' "${IMPORT_OUTPUT}" | tail -n 1)
{
  echo "label: ${LABEL}"
  echo "code root: ${CODE_ROOT}"
  echo "imported vllm_ascend: ${IMPORT_PATH}"
  echo "target model: ${TARGET_MODEL}"
  echo "draft model: ${DRAFT_MODEL}"
  echo "TP: ${TP_SIZE} K: ${NUM_SPECULATIVE_TOKENS}"
  echo "profile directory: ${TRACE_ROOT}"
} | tee "${LOG_FILE}"

RUN_RC=0
if [[ ${IMPORT_RC} -ne 0 || "${IMPORT_PATH}" != "${CODE_ROOT}"/* ]]; then
  echo "Imported vllm_ascend is not from the requested code root." \
    | tee -a "${LOG_FILE}"
  RUN_RC=2
else
  PYTHONUNBUFFERED=1 PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}" \
    python "${PYTHON_RUNNER}" \
      --target-model "${TARGET_MODEL}" \
      --draft-model "${DRAFT_MODEL}" \
      --tp "${TP_SIZE}" \
      --k "${NUM_SPECULATIVE_TOKENS}" \
      --profile-dir "${TRACE_ROOT}" \
      --manifest-out "${MANIFEST_FILE}" \
      --output-token-ids-out "${TOKEN_IDS_FILE}" \
      2>&1 | tee -a "${LOG_FILE}"
  RUN_RC=${PIPESTATUS[0]}
fi

echo "profile run exit code: ${RUN_RC}" | tee -a "${LOG_FILE}"

ANALYSIS_RC=0
TRACE_COUNT=0
if [[ ${RUN_RC} -eq 0 ]]; then
  while IFS= read -r TRACE_DIR; do
    [[ -z "${TRACE_DIR}" ]] && continue
    TRACE_COUNT=$((TRACE_COUNT + 1))
    echo "Analysing: ${TRACE_DIR}" | tee -a "${LOG_FILE}"
    PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}" python - "${TRACE_DIR}" \
      2>&1 <<'PY' | tee -a "${LOG_FILE}"
import sys
from torch_npu.profiler.profiler import analyse

analyse(sys.argv[1])
PY
    CURRENT_RC=${PIPESTATUS[0]}
    if [[ ${CURRENT_RC} -ne 0 ]]; then
      ANALYSIS_RC=${CURRENT_RC}
    fi
  done < <(find "${TRACE_ROOT}" -type d -name '*_ascend_pt' | sort)
fi

if [[ ${RUN_RC} -eq 0 && ${TRACE_COUNT} -eq 0 ]]; then
  echo "No Ascend trace directory was generated." | tee -a "${LOG_FILE}"
  ANALYSIS_RC=3
fi

echo "Ascend trace directories: ${TRACE_COUNT}" | tee -a "${LOG_FILE}"
echo "profile analysis exit code: ${ANALYSIS_RC}" | tee -a "${LOG_FILE}"
echo "Generated profiler reports:" | tee -a "${LOG_FILE}"
find "${TRACE_ROOT}" -type f \
  \( -name op_statistic.csv \
     -o -name operator_details.csv \
     -o -name kernel_details.csv \) \
  | sort | tee -a "${LOG_FILE}"

{
  echo "label=${LABEL}"
  echo "code_root=${CODE_ROOT}"
  echo "import_path=${IMPORT_PATH}"
  echo "profile_run_exit_code=${RUN_RC}"
  echo "profile_analysis_exit_code=${ANALYSIS_RC}"
  echo "trace_count=${TRACE_COUNT}"
  echo "log_file=${LOG_FILE}"
} >"${STATUS_FILE}"

echo "status file: ${STATUS_FILE}" | tee -a "${LOG_FILE}"
echo "log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# Keep an interactive container alive even when the recorded run failed.
exit 0
