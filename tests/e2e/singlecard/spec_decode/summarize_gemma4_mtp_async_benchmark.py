#!/usr/bin/env python3
"""Create a comparable Markdown summary from paired vLLM serving results."""

from __future__ import annotations

import json
import sys
from pathlib import Path


COMPARISON_KEYS = (
    "k",
    "max_model_len",
    "input_len",
    "output_len",
    "max_concurrency",
    "num_prompts",
    "seed",
)
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


def _read_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _delta(sync_value: object, async_value: object) -> str:
    if not isinstance(sync_value, (int, float)) or not isinstance(async_value, (int, float)):
        return "n/a"
    if sync_value == 0:
        return "n/a"
    return f"{(async_value - sync_value) / sync_value * 100:.2f}%"


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: summarize_gemma4_mtp_async_benchmark.py SYNC_JSON ASYNC_JSON OUTPUT_MD")

    sync_path, async_path, output_path = map(Path, sys.argv[1:])
    sync = _read_result(sync_path)
    async_result = _read_result(async_path)

    mismatches = [
        key
        for key in COMPARISON_KEYS
        if str(sync.get(key)) != str(async_result.get(key))
    ]
    if mismatches:
        raise SystemExit(f"refusing comparison because benchmark metadata differs: {', '.join(mismatches)}")
    if sync.get("mode") != "sync" or async_result.get("mode") != "async":
        raise SystemExit("result metadata does not identify paired sync and async runs")
    if sync.get("failed", 0) or async_result.get("failed", 0):
        raise SystemExit("refusing comparison because one benchmark reported failed requests")

    lines = [
        "# Gemma4 MTP Async Scheduling Benchmark",
        "",
        "## Fixed Configuration",
        "",
    ]
    lines.extend(f"- `{key}`: `{sync.get(key)}`" for key in COMPARISON_KEYS)
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Metric | Sync | Async | Async vs Sync |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in METRIC_KEYS:
        sync_value = sync.get(key, "n/a")
        async_value = async_result.get(key, "n/a")
        lines.append(
            f"| `{key}` | {_format(sync_value)} | {_format(async_value)} | "
            f"{_delta(sync_value, async_value)} |"
        )
    lines.extend(
        [
            "",
            "The comparison holds K and workload metadata constant. It does not compare K tuning results.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
