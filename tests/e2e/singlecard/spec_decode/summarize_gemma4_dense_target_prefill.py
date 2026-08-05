#!/usr/bin/env python3
"""Summarize Gemma4 dense-target long-prompt prefill evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CATEGORY_NAMES = (
    "weight_quant_gemm",
    "dense_matmul",
    "attention",
    "norm",
    "activation_elementwise",
    "kv_cache_update",
    "data_movement",
    "other",
)
HOST_PHASE_NAMES = (
    "prepare input",
    "forward",
    "post process",
    "sample_token",
)
COMPARABLE_MANIFEST_FIELDS = (
    "target_model",
    "draft_model",
    "mode",
    "execution",
    "tensor_parallel_size",
    "num_speculative_tokens",
    "prompt_tokens",
    "profile_prompt_sha256",
    "engine",
    "sampling",
    "profiler",
)
SOURCE_PATHS = {
    "host_prepare": "vllm_ascend/worker/model_runner_v1.py",
    "weight_quant_gemm": "vllm_ascend/quantization/",
    "dense_matmul": "vllm/model_executor/models/gemma4.py",
    "attention": "vllm_ascend/attention/attention_v1.py",
    "norm": "vllm/model_executor/models/gemma4.py",
    "activation_elementwise": "vllm/model_executor/models/gemma4.py",
    "kv_cache_update": "vllm_ascend/attention/attention_v1.py",
    "data_movement": "vllm_ascend/worker/model_runner_v1.py",
    "graph_fragmentation": "vllm_ascend/compilation/",
}
NEXT_PLAN_FILENAMES = {
    "host_prepare": (
        "gemma4-ascend-prefill-runner-data-movement-development-plan.md"
    ),
    "weight_quant_gemm": (
        "gemma4-ascend-w4a16-prefill-gemm-optimization-development-plan.md"
    ),
    "dense_matmul": (
        "gemma4-dense-target-ple-prefill-optimization-development-plan.md"
    ),
    "attention": (
        "gemma4-ascend-dense-target-prefill-attention-optimization-"
        "development-plan.md"
    ),
    "norm": "gemma4-dense-target-ple-prefill-optimization-development-plan.md",
    "activation_elementwise": (
        "gemma4-dense-target-ple-prefill-optimization-development-plan.md"
    ),
    "kv_cache_update": (
        "gemma4-ascend-dense-target-prefill-kv-cache-update-"
        "development-plan.md"
    ),
    "data_movement": (
        "gemma4-ascend-prefill-runner-data-movement-development-plan.md"
    ),
    "graph_fragmentation": (
        "gemma4-ascend-prefill-graph-fragmentation-development-plan.md"
    ),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _number(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "").strip().strip('"')
    return float(raw) if raw else 0.0


def normalize_op_type(op_type: str) -> str:
    if op_type.startswith("WeightQuantBatchMatmulV2"):
        return "WeightQuantBatchMatmulV2"
    return op_type


def classify_op_type(op_type: str) -> str:
    lowered = op_type.casefold()
    if lowered.startswith("weightquantbatchmatmulv2"):
        return "weight_quant_gemm"
    if any(
        name in lowered
        for name in (
            "flashattentionscore",
            "fusedinferattentionscore",
            "pagedattention",
        )
    ):
        return "attention"
    if "rmsnorm" in lowered or "layernorm" in lowered:
        return "norm"
    if any(
        name in lowered
        for name in ("reshapeandcache", "scatter", "indexput", "cacheupdate")
    ):
        return "kv_cache_update"
    if "matmul" in lowered:
        return "dense_matmul"
    if any(
        name in lowered
        for name in ("gelu", "silu", "sigmoid", "tanh", "add", "mul")
    ):
        return "activation_elementwise"
    if any(
        name in lowered
        for name in (
            "gather",
            "index",
            "transpose",
            "copy",
            "cast",
            "slice",
        )
    ):
        return "data_movement"
    return "other"


def analyze_profile(
    report_dir: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    op_rows = _read_csv(report_dir / "op_statistic.csv")
    operator_rows = _read_csv(report_dir / "operator_details.csv")
    kernel_rows = _read_csv(report_dir / "kernel_details.csv")

    raw_ops: dict[str, dict[str, object]] = {}
    categories: dict[str, dict[str, float | int]] = {
        category: {"count": 0, "total_device_us": 0.0}
        for category in CATEGORY_NAMES
    }
    for row in op_rows:
        op_type = row.get("OP Type", "").strip()
        count = int(_number(row, "Count"))
        total_us = _number(row, "Total Time(us)")
        category = classify_op_type(op_type)
        categories[category]["count"] += count
        categories[category]["total_device_us"] += total_us

        aggregate = raw_ops.setdefault(
            op_type,
            {
                "op_type": op_type,
                "normalized_op_type": normalize_op_type(op_type),
                "category": category,
                "count": 0,
                "total_device_us": 0.0,
            },
        )
        aggregate["count"] += count
        aggregate["total_device_us"] += total_us

    total_device_us = sum(
        float(metrics["total_device_us"]) for metrics in categories.values()
    )
    prompt_tokens = int(manifest.get("prompt_tokens", 0))
    for metrics in categories.values():
        count = int(metrics["count"])
        total_us = float(metrics["total_device_us"])
        metrics["ratio_of_profile_device_time"] = (
            total_us / total_device_us if total_device_us else 0.0
        )
        metrics["avg_device_us"] = total_us / count if count else 0.0
        metrics["device_us_per_prompt_token"] = (
            total_us / prompt_tokens if prompt_tokens else 0.0
        )

    top_raw_ops = []
    for aggregate in raw_ops.values():
        count = int(aggregate["count"])
        total_us = float(aggregate["total_device_us"])
        aggregate["avg_device_us"] = total_us / count if count else 0.0
        aggregate["ratio_of_profile_device_time"] = (
            total_us / total_device_us if total_device_us else 0.0
        )
        top_raw_ops.append(aggregate)
    top_raw_ops.sort(
        key=lambda item: float(item["total_device_us"]),
        reverse=True,
    )

    kernels: dict[tuple[str, str], dict[str, object]] = {}
    kernel_wait_us = 0.0
    for row in kernel_rows:
        name = row.get("Name", "").strip()
        kernel_type = row.get("Type", "").strip()
        duration_us = _number(row, "Duration(us)")
        wait_us = _number(row, "Wait Time(us)")
        kernel_wait_us += wait_us
        aggregate = kernels.setdefault(
            (name, kernel_type),
            {
                "name": name,
                "type": kernel_type,
                "count": 0,
                "total_duration_us": 0.0,
                "total_wait_us": 0.0,
            },
        )
        aggregate["count"] += 1
        aggregate["total_duration_us"] += duration_us
        aggregate["total_wait_us"] += wait_us
    top_kernels = sorted(
        kernels.values(),
        key=lambda item: float(item["total_duration_us"]),
        reverse=True,
    )[:30]

    host_phases = {phase: 0.0 for phase in HOST_PHASE_NAMES}
    for row in operator_rows:
        name = row.get("Name", "").strip().casefold().replace("_", " ")
        for phase in HOST_PHASE_NAMES:
            if name == phase.replace("_", " "):
                host_phases[phase] += _number(row, "Host Total Duration(us)")
                break
    host_phase_total_us = sum(host_phases.values())
    host_prepare_ratio = (
        host_phases["prepare input"] / host_phase_total_us
        if host_phase_total_us
        else 0.0
    )

    label = "_".join(
        (
            str(manifest.get("mode", "unknown")),
            str(manifest.get("execution", "unknown")),
            str(prompt_tokens),
        )
    )
    return {
        "label": label,
        "report_dir": str(report_dir.resolve()),
        "manifest": manifest,
        "total_device_us": total_device_us,
        "device_us_per_prompt_token": (
            total_device_us / prompt_tokens if prompt_tokens else 0.0
        ),
        "categories": categories,
        "top_raw_ops": top_raw_ops[:30],
        "top_kernels": top_kernels,
        "kernel_wait_us": kernel_wait_us,
        "host_phases": host_phases,
        "host_phase_total_us": host_phase_total_us,
        "host_prepare_ratio": host_prepare_ratio,
    }


def validate_comparable_manifests(
    manifests: list[dict[str, object]],
) -> None:
    if len(manifests) < 2:
        return
    baseline = manifests[0]
    for index, manifest in enumerate(manifests[1:], start=1):
        mismatches = [
            field
            for field in COMPARABLE_MANIFEST_FIELDS
            if manifest.get(field) != baseline.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"manifest {index} is not comparable: {', '.join(mismatches)}"
            )


def read_ttft_result(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    required = (
        "mode",
        "prompt_tokens",
        "completed",
        "failed",
        "request_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
    )
    missing = [field for field in required if field not in result]
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    expected_prompts = int(result.get("num_prompts", 10))
    completed = int(result["completed"])
    failed = int(result["failed"])
    if failed != 0 or completed != expected_prompts:
        raise ValueError(
            f"{path} reported failed or incomplete requests: "
            f"completed={completed} failed={failed} expected={expected_prompts}"
        )
    return {
        "path": str(path.resolve()),
        "mode": str(result["mode"]),
        "prompt_tokens": int(result["prompt_tokens"]),
        "completed": completed,
        "failed": failed,
        "request_throughput": float(result["request_throughput"]),
        "mean_ttft_ms": float(result["mean_ttft_ms"]),
        "median_ttft_ms": float(result["median_ttft_ms"]),
        "p90_ttft_ms": float(result["p90_ttft_ms"]),
        "p99_ttft_ms": float(result["p99_ttft_ms"]),
    }


def select_decision(profile: dict[str, object]) -> dict[str, object]:
    host_ratio = float(profile.get("host_prepare_ratio", 0.0))
    categories = profile.get("categories", {})
    if not isinstance(categories, dict):
        raise TypeError("profile categories must be a dict")

    if host_ratio >= 0.15:
        primary = "host_prepare"
        device_ratio = 0.0
    else:
        candidates = [
            (name, float(metrics.get("ratio_of_profile_device_time", 0.0)))
            for name, metrics in categories.items()
            if name != "other" and isinstance(metrics, dict)
        ]
        primary, device_ratio = max(
            candidates,
            key=lambda item: item[1],
            default=("graph_fragmentation", 0.0),
        )
        if device_ratio < 0.10:
            primary = "graph_fragmentation"

    rejected = sorted(
        name
        for name in categories
        if name not in {primary, "other"}
    )
    return {
        "primary_hotspot": primary,
        "device_ratio": device_ratio,
        "host_ratio": host_ratio,
        "prompt_scaling": profile.get("prompt_scaling", {}),
        "mapped_source_path": SOURCE_PATHS[primary],
        "rejected_candidates": rejected,
        "next_plan_filename": NEXT_PLAN_FILENAMES[primary],
    }


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_profiles(profile_root: Path) -> list[dict[str, object]]:
    profiles = []
    seen_cases: set[tuple[str, str, int]] = set()
    for manifest_path in sorted(profile_root.rglob("manifest.json")):
        manifest = _read_json(manifest_path)
        if manifest.get("workload") != "gemma4_dense_target_prefill":
            continue
        case_key = (
            str(manifest.get("mode")),
            str(manifest.get("execution")),
            int(manifest.get("prompt_tokens", 0)),
        )
        if case_key in seen_cases:
            raise ValueError(f"duplicate profile case: {case_key}")
        seen_cases.add(case_key)

        report_dirs = sorted(
            path.parent
            for path in manifest_path.parent.rglob("op_statistic.csv")
            if (path.parent / "operator_details.csv").is_file()
            and (path.parent / "kernel_details.csv").is_file()
        )
        if len(report_dirs) != 1:
            raise ValueError(
                f"{manifest_path.parent} must contain exactly one complete "
                f"Ascend report, found {len(report_dirs)}"
            )
        profiles.append(analyze_profile(report_dirs[0], manifest))
    if not profiles:
        raise FileNotFoundError(
            f"No Gemma4 prefill manifests found below {profile_root}"
        )
    return profiles


def _load_ttft_results(ttft_root: Path) -> list[dict[str, object]]:
    results = []
    seen_cases: set[tuple[str, int]] = set()
    for path in sorted(ttft_root.rglob("*.json")):
        raw = _read_json(path)
        if "completed" not in raw or "prompt_tokens" not in raw:
            continue
        result = read_ttft_result(path)
        case_key = (str(result["mode"]), int(result["prompt_tokens"]))
        if case_key in seen_cases:
            raise ValueError(f"duplicate TTFT case: {case_key}")
        seen_cases.add(case_key)
        results.append(result)
    if not results:
        raise FileNotFoundError(f"No TTFT results found below {ttft_root}")
    return results


def _profile_map(
    profiles: list[dict[str, object]],
) -> dict[tuple[str, str, int], dict[str, object]]:
    return {
        (
            str(profile["manifest"]["mode"]),
            str(profile["manifest"]["execution"]),
            int(profile["manifest"]["prompt_tokens"]),
        ): profile
        for profile in profiles
    }


def _ttft_map(
    results: list[dict[str, object]],
) -> dict[tuple[str, int], dict[str, object]]:
    return {
        (str(result["mode"]), int(result["prompt_tokens"])): result
        for result in results
    }


def _require_cases(
    actual: set[tuple],
    required: set[tuple],
    *,
    kind: str,
) -> None:
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        raise ValueError(
            f"invalid {kind} matrix: missing={missing} extra={extra}"
        )


def build_summary(
    profile_root: Path,
    ttft_root: Path,
) -> dict[str, object]:
    profile_root = profile_root.resolve()
    ttft_root = ttft_root.resolve()
    profiles = _load_profiles(profile_root)
    ttft_results = _load_ttft_results(ttft_root)
    profiles_by_case = _profile_map(profiles)
    ttft_by_case = _ttft_map(ttft_results)

    required_profiles = {
        ("target", "compiled", 8192),
        ("target", "compiled", 16384),
        ("target", "compiled", 28672),
        ("target", "eager", 4096),
        ("mtp", "compiled", 28672),
    }
    required_ttft = {
        (mode, prompt_tokens)
        for mode in ("target", "mtp")
        for prompt_tokens in (8192, 16384, 28672)
    }
    _require_cases(
        set(profiles_by_case),
        required_profiles,
        kind="profile",
    )
    _require_cases(set(ttft_by_case), required_ttft, kind="TTFT")

    target_8k = profiles_by_case[("target", "compiled", 8192)]
    target_16k = profiles_by_case[("target", "compiled", 16384)]
    target_28k = profiles_by_case[("target", "compiled", 28672)]
    mtp_28k = profiles_by_case[("mtp", "compiled", 28672)]
    scaling = {
        "8192_to_16384_total_device_ratio": (
            float(target_16k["total_device_us"])
            / float(target_8k["total_device_us"])
        ),
        "8192_to_28672_total_device_ratio": (
            float(target_28k["total_device_us"])
            / float(target_8k["total_device_us"])
        ),
        "8192_to_28672_per_token_ratio": (
            float(target_28k["device_us_per_prompt_token"])
            / float(target_8k["device_us_per_prompt_token"])
        ),
    }
    target_vs_mtp = {
        "target_total_device_us": float(target_28k["total_device_us"]),
        "mtp_total_device_us": float(mtp_28k["total_device_us"]),
        "mtp_device_time_change_percent": (
            float(mtp_28k["total_device_us"])
            / float(target_28k["total_device_us"])
            - 1.0
        )
        * 100.0,
        "target_median_ttft_ms": float(
            ttft_by_case[("target", 28672)]["median_ttft_ms"]
        ),
        "mtp_median_ttft_ms": float(
            ttft_by_case[("mtp", 28672)]["median_ttft_ms"]
        ),
    }
    decision_input = dict(target_28k)
    decision_input["prompt_scaling"] = scaling
    decision = select_decision(decision_input)
    return {
        "profile_root": str(profile_root),
        "ttft_root": str(ttft_root),
        "profiles": sorted(
            profiles,
            key=lambda item: (
                str(item["manifest"]["mode"]),
                str(item["manifest"]["execution"]),
                int(item["manifest"]["prompt_tokens"]),
            ),
        ),
        "ttft_results": sorted(
            ttft_results,
            key=lambda item: (str(item["mode"]), int(item["prompt_tokens"])),
        ),
        "scaling": scaling,
        "target_vs_mtp_28672": target_vs_mtp,
        "decision": decision,
    }


def _format_float(value: object) -> str:
    return f"{float(value):.3f}"


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Gemma4 Dense Target Long-Prompt Prefill Hotspot Summary",
        "",
        "## Workload contract",
        "",
        "- TP=1, PCP=1, executor=uni, async_scheduling=False.",
        "- max_model_len=32768, max_num_batched_tokens=8192.",
        "- Prefix caching is disabled and every request generates one token.",
        "- Profiler elapsed time is diagnostic evidence, not TTFT.",
        "",
        "## TTFT baseline",
        "",
        "| Mode | Prompt tokens | Completed | Failed | Median TTFT ms | "
        "P90 TTFT ms | P99 TTFT ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in summary["ttft_results"]:
        lines.append(
            "| {mode} | {tokens} | {completed} | {failed} | {median} | "
            "{p90} | {p99} |".format(
                mode=result["mode"],
                tokens=result["prompt_tokens"],
                completed=result["completed"],
                failed=result["failed"],
                median=_format_float(result["median_ttft_ms"]),
                p90=_format_float(result["p90_ttft_ms"]),
                p99=_format_float(result["p99_ttft_ms"]),
            )
        )

    lines.extend(
        [
            "",
            "## Compiled profile",
            "",
            "| Mode | Prompt tokens | Device us | Device us/token | "
            "Host prepare ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile in summary["profiles"]:
        manifest = profile["manifest"]
        if manifest["execution"] != "compiled":
            continue
        lines.append(
            "| {mode} | {tokens} | {device} | {per_token} | {host:.2%} |".format(
                mode=manifest["mode"],
                tokens=manifest["prompt_tokens"],
                device=_format_float(profile["total_device_us"]),
                per_token=_format_float(profile["device_us_per_prompt_token"]),
                host=float(profile["host_prepare_ratio"]),
            )
        )

    eager_profiles = [
        profile
        for profile in summary["profiles"]
        if profile["manifest"]["execution"] == "eager"
    ]
    lines.extend(["", "## Eager attribution", ""])
    for profile in eager_profiles:
        top_ops = ", ".join(
            str(item["normalized_op_type"])
            for item in profile["top_raw_ops"][:5]
        )
        lines.append(
            f"- {profile['label']}: top normalized ops: {top_ops}."
        )

    comparison = summary["target_vs_mtp_28672"]
    lines.extend(
        [
            "",
            "## Target versus MTP",
            "",
            "- 28K device time change: "
            f"{_format_float(comparison['mtp_device_time_change_percent'])}%.",
            "- Target median TTFT: "
            f"{_format_float(comparison['target_median_ttft_ms'])} ms.",
            "- MTP median TTFT: "
            f"{_format_float(comparison['mtp_median_ttft_ms'])} ms.",
            "",
            "## Scaling",
            "",
        ]
    )
    for name, value in summary["scaling"].items():
        lines.append(f"- {name}: {_format_float(value)}.")

    decision = summary["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- primary_hotspot: {decision['primary_hotspot']}",
            f"- device_ratio: {_format_float(decision['device_ratio'])}",
            f"- host_ratio: {_format_float(decision['host_ratio'])}",
            f"- mapped_source_path: {decision['mapped_source_path']}",
            f"- next_plan_filename: {decision['next_plan_filename']}",
            "- rejected_candidates: "
            + ", ".join(decision["rejected_candidates"]),
            "",
            "## Evidence limitations",
            "",
            "- The eager trace is used only for source attribution.",
            "- Profiler wall time is not reported as TTFT.",
            "- This report selects an optimization target; it does not prove "
            "an optimization gain.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--ttft-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.profile_root, args.ttft_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    args.output_markdown.write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    print(f"profiles: {len(summary['profiles'])}")
    print(f"TTFT results: {len(summary['ttft_results'])}")
    print(f"decision: {summary['decision']['primary_hotspot']}")
    print(f"json: {args.output_json}")
    print(f"markdown: {args.output_markdown}")


if __name__ == "__main__":
    main()
