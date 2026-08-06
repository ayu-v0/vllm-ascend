#!/usr/bin/env bash
# Run Gemma4 prefill attention profiler and three-round TTFT A/B.

set +e
set -o pipefail
export ASCEND_RT_VISIBLE_DEVICES=0

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <code-root> <ab-root>" >&2
  exit 0
fi

CODE_ROOT=$1
AB_ROOT=$2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROFILE_RUNNER="$SCRIPT_DIR/run_gemma4_dense_target_prefill_profile.sh"
TTFT_RUNNER="$SCRIPT_DIR/run_gemma4_dense_target_prefill_ttft.sh"
SUMMARIZER="$SCRIPT_DIR/summarize_gemma4_prefill_attention.py"
ROUND_1=("reference" "windowed")
ROUND_2=("windowed" "reference")
ROUND_3=("reference" "windowed")

mkdir -p "$AB_ROOT/profiles" "$AB_ROOT/ttft"

run_profile() {
  local impl=$1
  local run_dir="$AB_ROOT/profiles/$impl"
  local log_file="$run_dir/wrapper.log"
  mkdir -p "$run_dir"
  echo "Starting attention profile impl=$impl" | tee "$log_file"
  VLLM_ASCEND_W4A16_LINEAR_IMPL=reference \
    VLLM_ASCEND_GEMMA4_PREFILL_ATTENTION_IMPL="$impl" \
    "$PROFILE_RUNNER" "$CODE_ROOT" "$run_dir" \
    2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  echo "profile wrapper exit code: $rc" | tee -a "$log_file"
}

run_profile reference
run_profile windowed

PROFILE_JSON="$AB_ROOT/profiles/comparison.json"
PROFILE_MD="$AB_ROOT/profiles/comparison.md"
PYTHONUNBUFFERED=1 python "$SUMMARIZER" \
  --reference-profile-root "$AB_ROOT/profiles/reference" \
  --windowed-profile-root "$AB_ROOT/profiles/windowed" \
  --output-json "$PROFILE_JSON" \
  --output-markdown "$PROFILE_MD" \
  2>&1 | tee "$AB_ROOT/profiles/summary.log"
PROFILE_RC=${PIPESTATUS[0]}
echo "profile comparison exit code: $PROFILE_RC" |
  tee -a "$AB_ROOT/profiles/summary.log"

run_ttft() {
  local round=$1
  local impl=$2
  local run_dir="$AB_ROOT/ttft/round_${round}/${impl}"
  local log_file="$run_dir/wrapper.log"
  mkdir -p "$run_dir"
  echo "Starting TTFT round=$round impl=$impl" | tee "$log_file"
  VLLM_ASCEND_W4A16_LINEAR_IMPL=reference \
    VLLM_ASCEND_GEMMA4_PREFILL_ATTENTION_IMPL="$impl" \
    "$TTFT_RUNNER" "$CODE_ROOT" "$run_dir" \
    2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  {
    echo "round=$round"
    echo "gemma4_prefill_attention_impl=$impl"
    echo "runner_exit_code=$rc"
  } | tee "$run_dir/ab-status.txt"
}

for ROUND in 1 2 3; do
  case "$ROUND" in
    1) ORDER=("${ROUND_1[@]}") ;;
    2) ORDER=("${ROUND_2[@]}") ;;
    3) ORDER=("${ROUND_3[@]}") ;;
  esac
  for IMPL in "${ORDER[@]}"; do
    run_ttft "$ROUND" "$IMPL"
  done
done

PYTHONUNBUFFERED=1 python - "$AB_ROOT" "$PROFILE_JSON" \
  2>&1 <<'PY' | tee "$AB_ROOT/ttft/summary.log"
import json
import statistics
import sys
from pathlib import Path


root = Path(sys.argv[1])
profile_path = Path(sys.argv[2])
records = []
for round_index in (1, 2, 3):
    for impl in ("reference", "windowed"):
        run_dir = root / "ttft" / f"round_{round_index}" / impl
        manifest_path = run_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else None
        )
        indexed = {
            (item["mode"], int(item["prompt_tokens"])): item
            for item in (manifest or {}).get("case_results", [])
        }
        for mode in ("target", "mtp"):
            for prompt_tokens in (8192, 16384, 28672):
                status = indexed.get((mode, prompt_tokens), {})
                result_path = status.get("result_file")
                result = None
                if result_path and Path(result_path).is_file():
                    result = json.loads(
                        Path(result_path).read_text(encoding="utf-8")
                    )
                expected = int((manifest or {}).get("num_prompts", 10))
                success = (
                    manifest is not None
                    and manifest.get("w4a16_linear_impl") == "reference"
                    and manifest.get("gemma4_prefill_attention_impl") == impl
                    and status.get("server_health_exit_code") == 0
                    and status.get("benchmark_exit_code") == 0
                    and result is not None
                    and int(result.get("completed", 0)) == expected
                    and int(result.get("failed", 0)) == 0
                )
                records.append({
                    "round": round_index,
                    "impl": impl,
                    "mode": mode,
                    "prompt_tokens": prompt_tokens,
                    "success": success,
                    "median_ttft_ms": (
                        float(result["median_ttft_ms"]) if success else None
                    ),
                    "p90_ttft_ms": (
                        float(result["p90_ttft_ms"]) if success else None
                    ),
                    "result_file": result_path,
                })


def aggregate(impl, mode, prompt_tokens):
    selected = [
        record for record in records
        if record["impl"] == impl
        and record["mode"] == mode
        and record["prompt_tokens"] == prompt_tokens
        and record["success"]
    ]
    if len(selected) != 3:
        return None
    return {
        "round_median_ttft_ms": [
            record["median_ttft_ms"] for record in selected
        ],
        "round_p90_ttft_ms": [
            record["p90_ttft_ms"] for record in selected
        ],
        "median_of_round_medians_ms": statistics.median(
            record["median_ttft_ms"] for record in selected
        ),
        "median_of_round_p90_ms": statistics.median(
            record["p90_ttft_ms"] for record in selected
        ),
    }


aggregates = {
    f"{impl}_{mode}_{tokens}": aggregate(impl, mode, tokens)
    for impl in ("reference", "windowed")
    for mode in ("target", "mtp")
    for tokens in (8192, 16384, 28672)
}


def improvement(mode, tokens, metric):
    reference = aggregates[f"reference_{mode}_{tokens}"]
    windowed = aggregates[f"windowed_{mode}_{tokens}"]
    if reference is None or windowed is None:
        return None
    return (
        1.0 - float(windowed[metric]) / float(reference[metric])
    ) * 100.0


improvements = {
    "target_28672_median": improvement(
        "target", 28672, "median_of_round_medians_ms"
    ),
    "target_28672_p90": improvement(
        "target", 28672, "median_of_round_p90_ms"
    ),
    "target_16384_median": improvement(
        "target", 16384, "median_of_round_medians_ms"
    ),
    "target_8192_median": improvement(
        "target", 8192, "median_of_round_medians_ms"
    ),
    "mtp_28672_median": improvement(
        "mtp", 28672, "median_of_round_medians_ms"
    ),
}
profile = (
    json.loads(profile_path.read_text(encoding="utf-8"))
    if profile_path.is_file()
    else {}
)
successful = sum(record["success"] for record in records)
expected = 36
gates = {
    "request_success_rate": successful == expected,
    "profile_gate": bool(profile.get("accept_candidate")),
    "target_28k_median_gain": (
        improvements["target_28672_median"] is not None
        and improvements["target_28672_median"] >= 5.0
    ),
    "target_28k_p90_no_regression": (
        improvements["target_28672_p90"] is not None
        and improvements["target_28672_p90"] >= 0.0
    ),
    "target_16k_median_gain": (
        improvements["target_16384_median"] is not None
        and improvements["target_16384_median"] >= 0.0
    ),
    "target_8k_median_regression": (
        improvements["target_8192_median"] is not None
        and improvements["target_8192_median"] >= -2.0
    ),
    "mtp_28k_median_no_regression": (
        improvements["mtp_28672_median"] is not None
        and improvements["mtp_28672_median"] >= 0.0
    ),
}
summary = {
    "records": records,
    "aggregates": aggregates,
    "improvements_percent": improvements,
    "successful_cases": successful,
    "expected_cases": expected,
    "gates": gates,
    "accept_candidate": all(gates.values()),
}
(root / "ttft" / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
raise SystemExit(0 if summary["accept_candidate"] else 1)
PY
SUMMARY_RC=${PIPESTATUS[0]}
echo "TTFT A/B summary exit code: $SUMMARY_RC" |
  tee -a "$AB_ROOT/ttft/summary.log"
echo "Attention A/B artifacts: $AB_ROOT" |
  tee -a "$AB_ROOT/ttft/summary.log"

# Record failures in artifacts but preserve the interactive container shell.
exit 0
