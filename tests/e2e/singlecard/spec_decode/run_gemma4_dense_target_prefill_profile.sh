#!/usr/bin/env bash
# Run the fixed Gemma4 long-prompt prefill profiler matrix.

set +e
set -o pipefail
export ASCEND_RT_VISIBLE_DEVICES=0

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <code-root> <run-set-dir>" >&2
  exit 0
fi

CODE_ROOT=$1
RUN_SET_DIR=$2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_RUNNER="$SCRIPT_DIR/profile_gemma4_dense_target_prefill.py"
TARGET_MODEL=${VLLM_ASCEND_GEMMA4_MTP_MODEL:-}
DRAFT_MODEL=${VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL:-}
TP_SIZE=${GEMMA4_PREFILL_PROFILE_TP_SIZE:-1}
MTP_K=${GEMMA4_PREFILL_PROFILE_K:-3}
MAX_MODEL_LEN=${GEMMA4_PREFILL_MAX_MODEL_LEN:-32768}
MAX_BATCHED_TOKENS=${GEMMA4_PREFILL_MAX_BATCHED_TOKENS:-8192}
MATRIX_STATUS="$RUN_SET_DIR/matrix-status.txt"
RUNS=(
  "target compiled 8192"
  "target compiled 16384"
  "target compiled 28672"
  "target eager 4096"
  "mtp compiled 28672"
)

mkdir -p "$RUN_SET_DIR"

ENVIRONMENT_RC=0
for VARIABLE_NAME in \
  ASCEND_LAUNCH_BLOCKING \
  VLLM_ASCEND_GEMMA4_MTP_DEBUG \
  VLLM_ASCEND_GEMMA4_MTP_ORACLE \
  VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE \
  VLLM_ASCEND_GEMMA4_MTP_ASYNC_UNIPROC_SUBMIT; do
  VARIABLE_VALUE=${!VARIABLE_NAME:-0}
  if [[ "$VARIABLE_VALUE" != "0" ]]; then
    echo "$VARIABLE_NAME must be unset or 0 for profiling." >&2
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

{
  echo "code_root=$CODE_ROOT"
  echo "run_set_dir=$RUN_SET_DIR"
  echo "import_path=$IMPORT_PATH"
  echo "environment_exit_code=$ENVIRONMENT_RC"
} >"$MATRIX_STATUS"

if [[ $ENVIRONMENT_RC -ne 0 ]]; then
  echo "profile matrix environment exit code: $ENVIRONMENT_RC" |
    tee -a "$MATRIX_STATUS"
  exit 0
fi

FAILED_CASES=0
for RUN_SPEC in "${RUNS[@]}"; do
  read -r MODE EXECUTION PROMPT_TOKENS <<<"$RUN_SPEC"
  CASE_NAME="${MODE}_${EXECUTION}_${PROMPT_TOKENS}"
  RUN_DIR="$RUN_SET_DIR/$CASE_NAME"
  TRACE_ROOT="$RUN_DIR/profile"
  LOG_FILE="$RUN_DIR/run.log"
  MANIFEST_FILE="$RUN_DIR/manifest.json"
  TOKEN_IDS_FILE="$RUN_DIR/output_token_ids.json"
  STATUS_FILE="$RUN_DIR/status.txt"
  K=0
  if [[ "$MODE" == "mtp" ]]; then
    K=$MTP_K
  fi

  mkdir -p "$RUN_DIR" "$TRACE_ROOT"
  {
    echo "case: $CASE_NAME"
    echo "mode: $MODE"
    echo "execution: $EXECUTION"
    echo "prompt tokens: $PROMPT_TOKENS"
    echo "target model: $TARGET_MODEL"
    echo "draft model: $DRAFT_MODEL"
    echo "TP: $TP_SIZE K: $K"
    echo "profile directory: $TRACE_ROOT"
  } | tee "$LOG_FILE"

  PYTHONUNBUFFERED=1 PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}" \
    python "$PYTHON_RUNNER" \
      --target-model "$TARGET_MODEL" \
      --draft-model "$DRAFT_MODEL" \
      --mode "$MODE" \
      --execution "$EXECUTION" \
      --tp "$TP_SIZE" \
      --k "$K" \
      --prompt-tokens "$PROMPT_TOKENS" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
      --profile-dir "$TRACE_ROOT" \
      --manifest-out "$MANIFEST_FILE" \
      --output-token-ids-out "$TOKEN_IDS_FILE" \
      2>&1 | tee -a "$LOG_FILE"
  RUN_RC=${PIPESTATUS[0]}
  echo "profile run exit code: $RUN_RC" | tee -a "$LOG_FILE"

  ANALYSIS_RC=0
  TRACE_COUNT=0
  if [[ $RUN_RC -eq 0 ]]; then
    while IFS= read -r TRACE_DIR; do
      [[ -z "$TRACE_DIR" ]] && continue
      TRACE_COUNT=$((TRACE_COUNT + 1))
      echo "Analysing: $TRACE_DIR" | tee -a "$LOG_FILE"
      PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}" \
        python - "$TRACE_DIR" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import sys
from torch_npu.profiler.profiler import analyse

analyse(sys.argv[1])
PY
      CURRENT_RC=${PIPESTATUS[0]}
      if [[ $CURRENT_RC -ne 0 ]]; then
        ANALYSIS_RC=$CURRENT_RC
      fi
    done < <(find "$TRACE_ROOT" -type d -name '*_ascend_pt' | sort)
  fi

  if [[ $RUN_RC -eq 0 && $TRACE_COUNT -eq 0 ]]; then
    echo "No Ascend trace directory was generated." | tee -a "$LOG_FILE"
    ANALYSIS_RC=3
  fi

  echo "profile analysis exit code: $ANALYSIS_RC" | tee -a "$LOG_FILE"
  echo "Ascend trace directories: $TRACE_COUNT" | tee -a "$LOG_FILE"
  echo "Generated profiler reports:" | tee -a "$LOG_FILE"
  find "$TRACE_ROOT" -type f \
    \( -name op_statistic.csv \
       -o -name operator_details.csv \
       -o -name kernel_details.csv \
       -o -name trace_view.json \) \
    | sort | tee -a "$LOG_FILE"

  {
    echo "case=$CASE_NAME"
    echo "mode=$MODE"
    echo "execution=$EXECUTION"
    echo "prompt_tokens=$PROMPT_TOKENS"
    echo "profile_run_exit_code=$RUN_RC"
    echo "profile_analysis_exit_code=$ANALYSIS_RC"
    echo "trace_count=$TRACE_COUNT"
    echo "manifest_file=$MANIFEST_FILE"
    echo "log_file=$LOG_FILE"
  } >"$STATUS_FILE"

  if [[ $RUN_RC -ne 0 || $ANALYSIS_RC -ne 0 ]]; then
    FAILED_CASES=$((FAILED_CASES + 1))
  fi
done

{
  echo "failed_cases=$FAILED_CASES"
  echo "case_count=${#RUNS[@]}"
} | tee -a "$MATRIX_STATUS"
echo "matrix status: $MATRIX_STATUS"

# Preserve the interactive container even when one or more recorded cases fail.
exit 0
