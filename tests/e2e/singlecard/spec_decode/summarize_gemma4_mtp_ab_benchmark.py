#!/usr/bin/env python3
"""Validate and summarize three-mode Gemma4 MTP A/B benchmark runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EXPECTED_MODES = {
    "no_mtp_uni": {"executor_backend": "uni", "use_mtp": "0", "k": "0"},
    "mtp_uni": {"executor_backend": "uni", "use_mtp": "1", "k": "3"},
    "mtp_mp": {"executor_backend": "mp", "use_mtp": "1", "k": "3"},
}
EXPECTED_FIXED_METADATA = {
    "device": "0",
    "tensor_parallel_size": "1",
    "max_model_len": "32768",
    "max_num_seqs": "24",
    "max_batched_tokens": "16384",
    "input_len": "12500",
    "input_min": "10000",
    "input_max": "15000",
    "output_len": "1024",
    "max_concurrency": "24",
    "num_prompts": "200",
    "request_rate": "inf",
    "temperature": "0",
    "seed": "0",
}
COMMON_METRICS = (
    "completed",
    "failed",
    "request_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p95_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
)
SPECULATIVE_METRICS = (
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
    "spec_decode_num_drafts",
    "spec_decode_draft_tokens",
    "spec_decode_accepted_tokens",
)
HIGHER_IS_BETTER = {"request_throughput", "output_throughput"}
COMPARISON_METRICS = (
    "request_throughput",
    "output_throughput",
    "median_ttft_ms",
    "p95_ttft_ms",
    "median_tpot_ms",
    "p95_tpot_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _improvement_percent(metric: str, baseline: float, candidate: float) -> float:
    if baseline == 0:
        raise SystemExit(f"cannot compare zero baseline for {metric}")
    if metric in HIGHER_IS_BETTER:
        return (candidate / baseline - 1.0) * 100.0
    return (baseline / candidate - 1.0) * 100.0


def _validate_result(path: Path, run_index: int, mode: str, result: dict) -> None:
    expected_metadata = {
        "round": str(run_index),
        "mode": mode,
        **EXPECTED_MODES[mode],
        **EXPECTED_FIXED_METADATA,
    }
    mismatches = [
        key
        for key, value in expected_metadata.items()
        if str(result.get(key)) != value
    ]
    if mismatches:
        raise SystemExit(
            f"{path} has invalid benchmark metadata: {', '.join(mismatches)}"
        )

    missing_metrics = [key for key in COMMON_METRICS if key not in result]
    if missing_metrics:
        raise SystemExit(
            f"{path} is missing required metrics: {', '.join(missing_metrics)}"
        )
    if int(result.get("completed", -1)) != 200:
        raise SystemExit(f"{path} did not complete all 200 requests")
    if int(result.get("failed", -1)) != 0:
        raise SystemExit(f"{path} reported failed requests")

    input_lens = result.get("input_lens")
    output_lens = result.get("output_lens")
    if not isinstance(input_lens, list) or len(input_lens) != 200:
        raise SystemExit(f"{path} does not contain 200 detailed input lengths")
    if any(not 10000 <= int(length) <= 15000 for length in input_lens):
        raise SystemExit(f"{path} contains input lengths outside 10000-15000")
    if not isinstance(output_lens, list) or len(output_lens) != 200:
        raise SystemExit(f"{path} does not contain 200 detailed output lengths")
    if any(int(length) != 1024 for length in output_lens):
        raise SystemExit(f"{path} contains outputs that are not 1024 tokens")

    missing_spec = [key for key in SPECULATIVE_METRICS if key not in result]
    if EXPECTED_MODES[mode]["use_mtp"] == "1" and missing_spec:
        raise SystemExit(
            f"{path} is missing MTP metrics: {', '.join(missing_spec)}"
        )
    if EXPECTED_MODES[mode]["use_mtp"] == "0" and not missing_spec:
        raise SystemExit(f"{path} unexpectedly contains speculative metrics")


def _validate_command(path: Path, mode: str, command: str) -> None:
    executor = EXPECTED_MODES[mode]["executor_backend"]
    required = (
        "ASCEND_RT_VISIBLE_DEVICES=0",
        "--tensor-parallel-size 1",
        f"--distributed-executor-backend {executor}",
        "--async-scheduling",
    )
    missing = [token for token in required if token not in command]
    if missing:
        raise SystemExit(
            f"{path} is missing single-card command evidence: {', '.join(missing)}"
        )
    has_speculative_config = "--speculative-config" in command
    expects_mtp = EXPECTED_MODES[mode]["use_mtp"] == "1"
    if has_speculative_config != expects_mtp:
        raise SystemExit(f"{path} has an invalid speculative-config boundary")


def _read_run(
    result_root: Path, run_index: int
) -> tuple[dict[str, dict], dict[str, str]]:
    results = {}
    commands = {}
    for mode in EXPECTED_MODES:
        mode_dir = result_root / mode
        result_path = mode_dir / f"{mode}.json"
        command_path = mode_dir / "server-command.txt"
        if not result_path.is_file():
            raise SystemExit(f"missing benchmark result: {result_path}")
        if not command_path.is_file():
            raise SystemExit(f"missing server command: {command_path}")
        result = _read_json(result_path)
        command = command_path.read_text(encoding="utf-8").strip()
        _validate_result(result_path, run_index, mode, result)
        _validate_command(command_path, mode, command)
        results[mode] = result
        commands[mode] = command
    return results, commands


def _median_results(runs: list[dict[str, dict]]) -> dict[str, dict[str, float]]:
    medians = {}
    for mode in EXPECTED_MODES:
        metric_names = list(COMMON_METRICS)
        if EXPECTED_MODES[mode]["use_mtp"] == "1":
            metric_names.extend(SPECULATIVE_METRICS)
        medians[mode] = {
            key: statistics.median(run[mode][key] for run in runs)
            for key in metric_names
        }
    return medians


def _append_metric_table(lines: list[str], results: dict[str, dict]) -> None:
    modes = list(EXPECTED_MODES)
    lines.extend(
        [
            "| Metric | " + " | ".join(f"`{mode}`" for mode in modes) + " |",
            "| --- | " + " | ".join("---:" for _ in modes) + " |",
        ]
    )
    for key in COMMON_METRICS + SPECULATIVE_METRICS:
        values = " | ".join(_format(results[mode].get(key)) for mode in modes)
        lines.append(f"| `{key}` | {values} |")


def _paired_improvements(
    runs: list[dict[str, dict]], baseline_mode: str, candidate_mode: str
) -> dict[str, list[float]]:
    return {
        metric: [
            _improvement_percent(
                metric,
                float(run[baseline_mode][metric]),
                float(run[candidate_mode][metric]),
            )
            for run in runs
        ]
        for metric in COMPARISON_METRICS
    }


def _append_comparison_table(
    lines: list[str],
    heading: str,
    medians: dict[str, dict[str, float]],
    baseline_mode: str,
    candidate_modes: tuple[str, ...],
) -> None:
    lines.extend(
        [
            "",
            heading,
            "",
            "| Metric | "
            + " | ".join(f"`{mode}` improvement" for mode in candidate_modes)
            + " |",
            "| --- | " + " | ".join("---:" for _ in candidate_modes) + " |",
        ]
    )
    for metric in COMPARISON_METRICS:
        values = []
        for candidate_mode in candidate_modes:
            improvement = _improvement_percent(
                metric,
                float(medians[baseline_mode][metric]),
                float(medians[candidate_mode][metric]),
            )
            values.append(f"{improvement:+.2f}%")
        lines.append(f"| `{metric}` | " + " | ".join(values) + " |")


def _classify(runs: list[dict[str, dict]], candidate_mode: str) -> tuple[str, dict]:
    improvements = _paired_improvements(runs, "no_mtp_uni", candidate_mode)
    values = {
        metric: statistics.median(metric_values)
        for metric, metric_values in improvements.items()
    }
    positive_output_rounds = sum(
        value > 0 for value in improvements["output_throughput"]
    )
    positive_tpot_rounds = sum(
        value > 0 for value in improvements["median_tpot_ms"]
    )

    clear_improvement = (
        values["output_throughput"] >= 5.0
        and values["median_tpot_ms"] >= 5.0
        and values["median_e2el_ms"] >= 3.0
        and values["p95_e2el_ms"] >= -2.0
        and values["p95_ttft_ms"] >= -10.0
        and positive_output_rounds >= 2
        and positive_tpot_rounds >= 2
    )
    regression = (
        values["output_throughput"] <= -2.0
        or values["median_tpot_ms"] <= -2.0
        or values["p95_e2el_ms"] <= -5.0
    )
    no_gain = (
        abs(values["output_throughput"]) < 2.0
        and abs(values["median_tpot_ms"]) < 2.0
    )
    if clear_improvement:
        decision = "clear improvement"
    elif regression:
        decision = "regression"
    elif no_gain:
        decision = "no measurable gain"
    else:
        decision = "inconclusive"
    return decision, values


def summarize(result_roots: list[Path], output_path: Path) -> None:
    if len(result_roots) != 3:
        raise SystemExit("exactly three result roots are required")

    runs = []
    commands = []
    for run_index, result_root in enumerate(result_roots, start=1):
        run_results, run_commands = _read_run(result_root, run_index)
        runs.append(run_results)
        commands.append(run_commands)

    medians = _median_results(runs)
    lines = [
        "# Gemma4 MTP Single-NPU A/B Benchmark",
        "",
        "## Fixed Configuration",
        "",
    ]
    lines.extend(
        f"- `{key}`: `{value}`" for key, value in EXPECTED_FIXED_METADATA.items()
    )
    lines.extend(
        [
            "",
            "## Mode Matrix",
            "",
            "| Mode | MTP | Executor | K | Device | TP |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
            "| `no_mtp_uni` | 0 | uni | 0 | 0 | 1 |",
            "| `mtp_uni` | 1 | uni | 3 | 0 | 1 |",
            "| `mtp_mp` | 1 | mp | 3 | 0 | 1 |",
        ]
    )

    for run_index, (result_root, run) in enumerate(
        zip(result_roots, runs, strict=True), start=1
    ):
        lines.extend(["", f"## Run {run_index}: `{result_root}`", ""])
        _append_metric_table(lines, run)

    lines.extend(["", "## Three-Run Median", ""])
    _append_metric_table(lines, medians)
    _append_comparison_table(
        lines,
        "## MTP vs No-MTP",
        medians,
        "no_mtp_uni",
        ("mtp_uni", "mtp_mp"),
    )
    _append_comparison_table(
        lines,
        "## MTP MP vs Uni",
        medians,
        "mtp_uni",
        ("mtp_mp",),
    )

    lines.extend(["", "## Decisions", ""])
    for candidate_mode in ("mtp_uni", "mtp_mp"):
        decision, values = _classify(runs, candidate_mode)
        lines.extend(
            [
                f"- `{candidate_mode}`: {decision}",
                f"  - output throughput: {values['output_throughput']:+.2f}%",
                f"  - median TPOT: {values['median_tpot_ms']:+.2f}%",
                f"  - median E2EL: {values['median_e2el_ms']:+.2f}%",
                f"  - P95 E2EL: {values['p95_e2el_ms']:+.2f}%",
                f"  - P95 TTFT: {values['p95_ttft_ms']:+.2f}%",
            ]
        )

    lines.extend(["", "## Server Commands"])
    for run_index, (result_root, run_commands) in enumerate(
        zip(result_roots, commands, strict=True), start=1
    ):
        lines.extend(["", f"### Run {run_index}: `{result_root}`"])
        for mode, command in run_commands.items():
            lines.extend(["", f"#### `{mode}`", "", "```bash", command, "```"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_md", type=Path)
    parser.add_argument("result_roots", nargs="+", type=Path)
    args = parser.parse_args()
    summarize(args.result_roots, args.output_md)


if __name__ == "__main__":
    main()
