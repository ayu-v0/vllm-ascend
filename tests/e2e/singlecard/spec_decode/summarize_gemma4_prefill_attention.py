#!/usr/bin/env python3
"""Summarize Gemma4 long-prompt prefill attention evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ATTENTION_OP_FRAGMENTS = (
    "flashattentionscore",
    "fusedinferattentionscore",
    "promptflashattention",
)
KV_GATHER_OP_FRAGMENTS = (
    "indexselect",
    "gatherv2",
    "gather",
    "stridedslice",
    "slice",
    "contiguous",
    "transdata",
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
    "vllm_ascend_enable_nz",
    "engine",
    "sampling",
    "profiler",
)


def _comparable_manifest_value(
    manifest: dict[str, object],
    field: str,
) -> object:
    value = manifest.get(field)
    if field == "profiler" and isinstance(value, dict):
        value = dict(value)
        value.pop("torch_profiler_dir", None)
    return value


def _number(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        raw = row.get(key, "").strip().strip('"')
        if raw:
            return float(raw)
    return 0.0


def _basename(path: object) -> str:
    return str(path or "").replace("\\", "/").rsplit("/", 1)[-1]


def summarize_attention_rows(
    rows: list[dict[str, str]],
) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "attention_us": 0.0,
        "attention_count": 0,
        "kv_gather_us": 0.0,
        "kv_gather_count": 0,
        "weight_quant_us": 0.0,
        "weight_quant_count": 0,
        "total_device_us": 0.0,
    }
    for row in rows:
        op_type = (
            row.get("OP Type", "")
            or row.get("Type", "")
            or row.get("Name", "")
        ).strip()
        lowered = op_type.casefold().replace("_", "")
        count = int(_number(row, "Count")) or 1
        total_us = _number(
            row,
            "Total Time(us)",
            "Task Duration(us)",
            "Duration(us)",
        )
        result["total_device_us"] += total_us
        if any(fragment in lowered for fragment in ATTENTION_OP_FRAGMENTS):
            result["attention_us"] += total_us
            result["attention_count"] += count
        elif any(
            fragment in lowered for fragment in KV_GATHER_OP_FRAGMENTS
        ):
            result["kv_gather_us"] += total_us
            result["kv_gather_count"] += count
        elif "weightquantbatchmatmulv2" in lowered:
            result["weight_quant_us"] += total_us
            result["weight_quant_count"] += count
    result["attention_plus_gather_us"] = (
        float(result["attention_us"]) + float(result["kv_gather_us"])
    )
    return result


def _improvement(reference: float, candidate: float) -> float:
    if reference <= 0:
        raise ValueError(
            f"reference duration must be positive, got {reference}"
        )
    return (1.0 - candidate / reference) * 100.0


def compare_attention_runs(
    reference: dict[str, object],
    windowed: dict[str, object],
) -> dict[str, object]:
    attention_gain = _improvement(
        float(reference["attention_plus_gather_us"]),
        float(windowed["attention_plus_gather_us"]),
    )
    total_gain = _improvement(
        float(reference["total_device_us"]),
        float(windowed["total_device_us"]),
    )
    reference_weight_count = int(reference.get("weight_quant_count", 0))
    windowed_weight_count = int(windowed.get("weight_quant_count", 0))
    gates = {
        "attention_plus_gather_gain": attention_gain >= 15.0,
        "total_device_gain": total_gain >= 3.0,
        "weight_quant_count_matches": (
            reference_weight_count == windowed_weight_count
        ),
    }
    return {
        "reference": reference,
        "windowed": windowed,
        "attention_plus_gather_improvement_percent": attention_gain,
        "total_device_improvement_percent": total_gain,
        "weight_quant_improvement_percent": _improvement(
            float(reference["weight_quant_us"]),
            float(windowed["weight_quant_us"]),
        ),
        "gates": gates,
        "accept_candidate": all(gates.values()),
    }


def validate_w4a16_no_go_evidence(
    log_summary: dict[str, object],
    json_summary: dict[str, object],
) -> dict[str, object]:
    same_run = _basename(log_summary.get("result_path")) == _basename(
        json_summary.get("result_path")
    )
    same_candidate_format = log_summary.get(
        "candidate_format"
    ) == json_summary.get("candidate_format")
    return {
        "same_run": same_run,
        "same_candidate_format": same_candidate_format,
        "matches": same_run and same_candidate_format,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_profile_root(root: Path) -> dict[str, object]:
    reports = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = _read_json(manifest_path)
        if manifest.get("workload") != "gemma4_dense_target_prefill":
            continue
        report_paths = sorted(manifest_path.parent.rglob("op_statistic.csv"))
        if len(report_paths) != 1:
            raise ValueError(
                f"{manifest_path.parent} must contain one op_statistic.csv"
            )
        metrics = summarize_attention_rows(_read_csv(report_paths[0]))
        prompt_tokens = int(manifest.get("prompt_tokens", 0))
        metrics.update(
            {
                "mode": manifest.get("mode"),
                "execution": manifest.get("execution"),
                "prompt_tokens": prompt_tokens,
                "device_us_per_prompt_token": (
                    float(metrics["total_device_us"]) / prompt_tokens
                    if prompt_tokens
                    else 0.0
                ),
                "manifest": manifest,
                "report_dir": str(report_paths[0].parent.resolve()),
            }
        )
        reports.append(metrics)
    if not reports:
        raise FileNotFoundError(f"No Gemma4 profile reports below {root}")
    return {"profile_root": str(root.resolve()), "reports": reports}


def _case_map(summary: dict[str, object]) -> dict[tuple[str, str, int], dict]:
    return {
        (
            str(report["mode"]),
            str(report["execution"]),
            int(report["prompt_tokens"]),
        ): report
        for report in summary["reports"]
    }


def _validate_ab_case_manifests(
    reference: dict[str, object],
    windowed: dict[str, object],
) -> None:
    reference_manifest = reference.get("manifest")
    windowed_manifest = windowed.get("manifest")
    if not isinstance(reference_manifest, dict) or not isinstance(
        windowed_manifest,
        dict,
    ):
        raise ValueError("A/B reports require manifest dictionaries")
    if (
        reference_manifest.get("w4a16_linear_impl") != "reference"
        or windowed_manifest.get("w4a16_linear_impl") != "reference"
    ):
        raise ValueError(
            "Gemma4 attention A/B requires W4A16 reference mode"
        )
    if (
        reference_manifest.get("gemma4_prefill_attention_impl")
        != "reference"
        or windowed_manifest.get("gemma4_prefill_attention_impl")
        != "windowed"
    ):
        raise ValueError(
            "A/B manifests must use reference and windowed attention modes"
        )
    mismatches = [
        field
        for field in COMPARABLE_MANIFEST_FIELDS
        if _comparable_manifest_value(reference_manifest, field)
        != _comparable_manifest_value(windowed_manifest, field)
    ]
    if mismatches:
        raise ValueError(
            "reference/windowed manifests are not comparable: "
            + ", ".join(mismatches)
        )


def build_ab_comparison(
    reference: dict[str, object],
    windowed: dict[str, object],
) -> dict[str, object]:
    reference_cases = _case_map(reference)
    windowed_cases = _case_map(windowed)
    if set(reference_cases) != set(windowed_cases):
        raise ValueError("reference/windowed profile matrices differ")
    cases = []
    for key in sorted(reference_cases):
        _validate_ab_case_manifests(
            reference_cases[key],
            windowed_cases[key],
        )
        comparison = compare_attention_runs(
            reference_cases[key],
            windowed_cases[key],
        )
        cases.append({"case": key, **comparison})
    target_28k = next(
        (
            case
            for case in cases
            if case["case"] == ("target", "compiled", 28672)
        ),
        None,
    )
    return {
        "reference": reference,
        "windowed": windowed,
        "cases": cases,
        "target_28k": target_28k,
        "accept_candidate": bool(
            target_28k and target_28k["accept_candidate"]
        ),
    }


def _render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Gemma4 Prefill Attention Summary",
        "",
        f"- accept_candidate: `{summary.get('accept_candidate')}`",
        "",
        "## Cases",
        "",
    ]
    for case in summary.get("cases", summary.get("reports", [])):
        lines.append(f"- `{case.get('case', case.get('prompt_tokens'))}`: `{case}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path)
    parser.add_argument("--reference-profile-root", type=Path)
    parser.add_argument("--windowed-profile-root", type=Path)
    parser.add_argument("--ttft-root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    if args.profile_root is not None:
        if (
            args.reference_profile_root is not None
            or args.windowed_profile_root is not None
        ):
            raise ValueError(
                "--profile-root cannot be combined with A/B profile roots"
            )
        summary = _load_profile_root(args.profile_root)
    else:
        if (
            args.reference_profile_root is None
            or args.windowed_profile_root is None
        ):
            raise ValueError(
                "provide --profile-root or both A/B profile roots"
            )
        summary = build_ab_comparison(
            _load_profile_root(args.reference_profile_root),
            _load_profile_root(args.windowed_profile_root),
        )

    if args.ttft_root is not None:
        summary["ttft_root"] = str(args.ttft_root.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    args.output_markdown.write_text(
        _render_markdown(summary),
        encoding="utf-8",
    )
    print(f"json: {args.output_json}")
    print(f"markdown: {args.output_markdown}")
    if "accept_candidate" in summary:
        raise SystemExit(0 if summary["accept_candidate"] else 1)


if __name__ == "__main__":
    main()
