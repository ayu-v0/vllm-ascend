#!/usr/bin/env bash
# Run three interleaved Gemma4 W4A16 TTFT rounds with process restarts.

set +e
set -o pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <code-root> <ab-root>" >&2
  exit 0
fi

CODE_ROOT=$1
AB_ROOT=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TTFT_RUNNER="$SCRIPT_DIR/run_gemma4_dense_target_prefill_ttft.sh"
PROFILE_COMPARISON_JSON=${GEMMA4_W4A16_PROFILE_COMPARISON_JSON:-}
ROUND_1=("reference" "candidate")
ROUND_2=("candidate" "reference")
ROUND_3=("reference" "candidate")
CURRENT_CHILD_PID=""

mkdir -p "$AB_ROOT"

cleanup() {
  if [[ -n "$CURRENT_CHILD_PID" ]] && \
      kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
    kill -TERM "$CURRENT_CHILD_PID" 2>/dev/null || true
    wait "$CURRENT_CHILD_PID" 2>/dev/null || true
  fi
  CURRENT_CHILD_PID=""
}
trap cleanup EXIT

run_one() {
  local round=$1
  local impl=$2
  local run_dir="$AB_ROOT/round_${round}/${impl}"
  local log_file="$run_dir/runner.log"
  mkdir -p "$run_dir"

  echo "Starting TTFT round=$round impl=$impl" | tee "$log_file"
  VLLM_ASCEND_W4A16_LINEAR_IMPL="$impl" \
    VLLM_ASCEND_ENABLE_NZ=1 \
    "$TTFT_RUNNER" "$CODE_ROOT" "$run_dir" \
    2>&1 | tee -a "$log_file"
  local run_rc=${PIPESTATUS[0]}
  {
    echo "round=$round"
    echo "w4a16_linear_impl=$impl"
    echo "runner_exit_code=$run_rc"
  } | tee "$run_dir/ab-status.txt"
}

for ROUND in 1 2 3; do
  case "$ROUND" in
    1) ORDER=("${ROUND_1[@]}") ;;
    2) ORDER=("${ROUND_2[@]}") ;;
    3) ORDER=("${ROUND_3[@]}") ;;
  esac
  for IMPL in "${ORDER[@]}"; do
    run_one "$ROUND" "$IMPL"
  done
done

PYTHONUNBUFFERED=1 python - "$AB_ROOT" "$PROFILE_COMPARISON_JSON" \
  2>&1 <<'PY' | tee "$AB_ROOT/summary.log"
import json
import statistics
import sys
from pathlib import Path


root = Path(sys.argv[1])
profile_comparison_path = Path(sys.argv[2]) if sys.argv[2] else None
expected_cases = 3 * 2 * 2 * 3
case_records = []

for round_index in (1, 2, 3):
    for impl in ("reference", "candidate"):
        run_dir = root / f"round_{round_index}" / impl
        manifest_path = run_dir / "manifest.json"
        manifest = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        indexed_cases = {}
        if manifest is not None:
            indexed_cases = {
                (item["mode"], int(item["prompt_tokens"])): item
                for item in manifest.get("case_results", [])
            }
        for mode in ("target", "mtp"):
            for prompt_tokens in (8192, 16384, 28672):
                status = indexed_cases.get((mode, prompt_tokens), {})
                result_path = status.get("result_file")
                result = None
                if result_path and Path(result_path).is_file():
                    result = json.loads(
                        Path(result_path).read_text(encoding="utf-8")
                    )
                expected_prompts = (
                    int(manifest.get("num_prompts", 10))
                    if manifest is not None
                    else 10
                )
                success = (
                    manifest is not None
                    and manifest.get("w4a16_linear_impl") == impl
                    and status.get("server_health_exit_code") == 0
                    and status.get("benchmark_exit_code") == 0
                    and result is not None
                    and int(result.get("completed", 0)) == expected_prompts
                    and int(result.get("failed", 0)) == 0
                )
                case_records.append(
                    {
                        "round": round_index,
                        "impl": impl,
                        "mode": mode,
                        "prompt_tokens": prompt_tokens,
                        "success": success,
                        "median_ttft_ms": (
                            float(result["median_ttft_ms"])
                            if success
                            else None
                        ),
                        "p90_ttft_ms": (
                            float(result["p90_ttft_ms"])
                            if success
                            else None
                        ),
                        "result_file": result_path,
                    }
                )


def aggregate(impl, mode, prompt_tokens):
    selected = [
        record
        for record in case_records
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
        "round_p90_ttft_ms": [record["p90_ttft_ms"] for record in selected],
        "median_of_round_medians_ms": statistics.median(
            record["median_ttft_ms"] for record in selected
        ),
        "median_of_round_p90_ms": statistics.median(
            record["p90_ttft_ms"] for record in selected
        ),
    }


aggregates = {}
for impl in ("reference", "candidate"):
    for mode in ("target", "mtp"):
        for prompt_tokens in (8192, 16384, 28672):
            aggregates[f"{impl}_{mode}_{prompt_tokens}"] = aggregate(
                impl,
                mode,
                prompt_tokens,
            )


def improvement(reference_key, candidate_key, metric):
    reference = aggregates.get(reference_key)
    candidate = aggregates.get(candidate_key)
    if reference is None or candidate is None:
        return None
    reference_value = float(reference[metric])
    candidate_value = float(candidate[metric])
    return (1.0 - candidate_value / reference_value) * 100.0


target_28k_median_gain = improvement(
    "reference_target_28672",
    "candidate_target_28672",
    "median_of_round_medians_ms",
)
target_28k_p90_gain = improvement(
    "reference_target_28672",
    "candidate_target_28672",
    "median_of_round_p90_ms",
)
target_8k_median_gain = improvement(
    "reference_target_8192",
    "candidate_target_8192",
    "median_of_round_medians_ms",
)
mtp_28k_median_gain = improvement(
    "reference_mtp_28672",
    "candidate_mtp_28672",
    "median_of_round_medians_ms",
)

profile_comparison = None
if profile_comparison_path is not None and profile_comparison_path.is_file():
    profile_comparison = json.loads(
        profile_comparison_path.read_text(encoding="utf-8")
    )
profile_gates = (
    profile_comparison.get("gates", {})
    if isinstance(profile_comparison, dict)
    else {}
)
weight_quant_device_time_gain = bool(
    profile_gates.get("target_28k_w4a16_gain")
    and profile_gates.get("target_28k_total_device_gain")
)

successful_cases = sum(record["success"] for record in case_records)
gates = {
    "request_success_rate": successful_cases == expected_cases,
    "target_28k_median_gain": (
        target_28k_median_gain is not None
        and target_28k_median_gain >= 5.0
    ),
    "target_28k_p90_no_regression": (
        target_28k_p90_gain is not None and target_28k_p90_gain >= 0.0
    ),
    "target_8k_median_regression": (
        target_8k_median_gain is not None and target_8k_median_gain >= -2.0
    ),
    "mtp_28k_median_no_regression": (
        mtp_28k_median_gain is not None and mtp_28k_median_gain >= 0.0
    ),
    "weight_quant_device_time_gain": weight_quant_device_time_gain,
}
summary = {
    "ab_root": str(root.resolve()),
    "profile_comparison_json": (
        str(profile_comparison_path.resolve())
        if profile_comparison_path is not None
        else None
    ),
    "expected_cases": expected_cases,
    "successful_cases": successful_cases,
    "success_rate": successful_cases / expected_cases,
    "case_records": case_records,
    "aggregates": aggregates,
    "improvements_percent": {
        "target_28k_median": target_28k_median_gain,
        "target_28k_p90": target_28k_p90_gain,
        "target_8k_median": target_8k_median_gain,
        "mtp_28k_median": mtp_28k_median_gain,
    },
    "gates": gates,
    "accept_candidate": all(gates.values()),
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
)

markdown = [
    "# Gemma4 W4A16 prefill GEMM TTFT A/B",
    "",
    f"Successful cases: {successful_cases}/{expected_cases}",
    "",
    "## Gates",
    "",
]
markdown.extend(f"- `{name}`: `{passed}`" for name, passed in gates.items())
markdown.extend(["", "## Improvements", ""])
markdown.extend(
    f"- `{name}`: `{value}`"
    for name, value in summary["improvements_percent"].items()
)
(root / "summary.md").write_text(
    "\n".join(markdown) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
raise SystemExit(0 if summary["accept_candidate"] else 1)
PY
SUMMARY_RC=${PIPESTATUS[0]}
echo "A/B summary exit code: $SUMMARY_RC" | tee -a "$AB_ROOT/summary.log"
echo "A/B artifacts: $AB_ROOT" | tee -a "$AB_ROOT/summary.log"

# The harness records failures but never exits the interactive container.
exit 0
