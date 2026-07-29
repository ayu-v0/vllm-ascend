#!/usr/bin/env python3
"""Summarize one or more four-mode Gemma4 MTP benchmark runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EXPECTED_MODES = {
    "sync": {"executor_backend": "uni", "candidate": "0"},
    "async_uni": {"executor_backend": "uni", "candidate": "0"},
    "async_mp": {"executor_backend": "mp", "candidate": "0"},
    "async_candidate": {"executor_backend": "uni", "candidate": "1"},
}
COMPARISON_KEYS = (
    "k",
    "max_model_len",
    "max_num_seqs",
    "input_len",
    "output_len",
    "max_concurrency",
    "num_prompts",
    "request_rate",
    "temperature",
    "seed",
    "hccl_op_expansion_mode",
)
EXPECTED_FIXED_METADATA = {
    "k": "3",
    "max_model_len": "32768",
    "max_num_seqs": "72",
    "input_len": "256",
    "output_len": "128",
    "max_concurrency": "72",
    "num_prompts": "3000",
    "request_rate": "inf",
    "temperature": "0",
    "seed": "0",
    "hccl_op_expansion_mode": "AIV",
}
METRIC_KEYS = (
    "completed",
    "failed",
    "request_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
    "spec_decode_draft_tokens",
    "spec_decode_accepted_tokens",
)
CANDIDATE_COMPARISON_KEYS = (
    "median_ttft_ms",
    "output_throughput",
    "request_throughput",
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
)


def _read_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _delta(baseline: object, candidate: object) -> str:
    if not isinstance(baseline, (int, float)) or not isinstance(
        candidate, (int, float)
    ):
        return "n/a"
    if baseline == 0:
        return "n/a"
    return f"{(candidate - baseline) / baseline * 100:.2f}%"


def _validate_result(path: Path, mode: str, result: dict) -> None:
    expected = EXPECTED_MODES[mode]
    expected_metadata = {"mode": mode, **expected, **EXPECTED_FIXED_METADATA}
    mismatches = [
        key
        for key, value in expected_metadata.items()
        if str(result.get(key)) != value
    ]
    if mismatches:
        raise SystemExit(
            f"{path} has invalid mode metadata: {', '.join(mismatches)}"
        )

    missing_metrics = [key for key in METRIC_KEYS if key not in result]
    if missing_metrics:
        raise SystemExit(
            f"{path} is missing required benchmark metrics: "
            f"{', '.join(missing_metrics)}"
        )
    if result.get("failed", 0):
        raise SystemExit(f"{path} reported failed requests")
    if int(result.get("completed", -1)) != int(result.get("num_prompts", -2)):
        raise SystemExit(f"{path} did not complete every requested prompt")


def _read_run(result_root: Path) -> tuple[dict[str, dict], dict[str, str]]:
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
        result = _read_result(result_path)
        _validate_result(result_path, mode, result)
        command = command_path.read_text(encoding="utf-8").strip()
        if not command:
            raise SystemExit(f"empty server command: {command_path}")
        results[mode] = result
        commands[mode] = command
    return results, commands


def _validate_fixed_configuration(runs: list[dict[str, dict]]) -> dict:
    reference = runs[0]["sync"]
    for run_index, run in enumerate(runs, start=1):
        for mode, result in run.items():
            mismatches = [
                key
                for key in COMPARISON_KEYS
                if str(result.get(key)) != str(reference.get(key))
            ]
            if mismatches:
                raise SystemExit(
                    "refusing comparison because benchmark metadata differs "
                    f"for run {run_index} mode {mode}: {', '.join(mismatches)}"
                )
    return reference


def _median_results(runs: list[dict[str, dict]]) -> dict[str, dict[str, float]]:
    return {
        mode: {
            key: statistics.median(run[mode][key] for run in runs)
            for key in METRIC_KEYS
        }
        for mode in EXPECTED_MODES
    }


def _append_metric_table(lines: list[str], results: dict[str, dict]) -> None:
    modes = list(EXPECTED_MODES)
    lines.extend(
        [
            "| Metric | " + " | ".join(f"`{mode}`" for mode in modes) + " |",
            "| --- | " + " | ".join("---:" for _ in modes) + " |",
        ]
    )
    for key in METRIC_KEYS:
        values = " | ".join(_format(results[mode][key]) for mode in modes)
        lines.append(f"| `{key}` | {values} |")


def summarize(result_roots: list[Path], output_path: Path) -> None:
    if not result_roots:
        raise SystemExit("at least one result root is required")

    runs = []
    commands = []
    for result_root in result_roots:
        run_results, run_commands = _read_run(result_root)
        runs.append(run_results)
        commands.append(run_commands)

    reference = _validate_fixed_configuration(runs)
    medians = _median_results(runs)
    lines = [
        "# Gemma4 MTP Async Scheduling Benchmark",
        "",
        "## Fixed Configuration",
        "",
    ]
    lines.extend(f"- `{key}`: `{reference.get(key)}`" for key in COMPARISON_KEYS)
    lines.extend(["", "## Mode Matrix", ""])
    lines.extend(
        [
            "| Mode | Scheduling | Executor | Candidate |",
            "| --- | --- | --- | ---: |",
            "| `sync` | sync | uni | 0 |",
            "| `async_uni` | async | uni | 0 |",
            "| `async_mp` | async | mp | 0 |",
            "| `async_candidate` | async | uni | 1 |",
        ]
    )

    for run_index, (result_root, run) in enumerate(
        zip(result_roots, runs, strict=True), start=1
    ):
        lines.extend(
            [
                "",
                f"## Run {run_index}: `{result_root}`",
                "",
            ]
        )
        _append_metric_table(lines, run)

    median_heading = (
        "## Three-Run Median"
        if len(runs) == 3
        else f"## Across-Run Median ({len(runs)} Run{'s' if len(runs) != 1 else ''})"
    )
    lines.extend(["", median_heading, ""])
    _append_metric_table(lines, medians)

    lines.extend(
        [
            "",
            "## Candidate Comparison",
            "",
            "| Metric | `async_uni` | `async_mp` | `async_mp` vs uni | "
            "`async_candidate` | candidate vs uni |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in CANDIDATE_COMPARISON_KEYS:
        async_uni = medians["async_uni"][key]
        async_mp = medians["async_mp"][key]
        candidate = medians["async_candidate"][key]
        lines.append(
            f"| `{key}` | {_format(async_uni)} | {_format(async_mp)} | "
            f"{_delta(async_uni, async_mp)} | {_format(candidate)} | "
            f"{_delta(async_uni, candidate)} |"
        )
    if len(runs) != 3:
        lines.extend(
            [
                "",
                "Candidate retention is not evaluated until exactly three runs are "
                "available.",
            ]
        )
    lines.extend(
        [
            "",
            "Correctness preflight evidence remains required for every run before "
            "applying the candidate retention gates.",
            "",
            "## Server Commands",
        ]
    )
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
