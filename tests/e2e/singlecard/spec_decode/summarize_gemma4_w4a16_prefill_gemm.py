#!/usr/bin/env python3
"""Attribute Gemma4 W4A16 prefill GEMMs in Ascend profiler reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


WEIGHT_QUANT_OP = "WeightQuantBatchMatmulV2"
LAYOUT_OVERHEAD_OPS = ("Cast", "TransData", "Contiguous")
REQUIRED_IMPROVEMENT_PERCENT = 8.0
MAX_ADDED_LAYOUT_OVERHEAD_RATIO = 0.01


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _number(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "").strip().strip('"')
    if not raw:
        return 0.0
    return float(raw)


def normalize_weight_quant_name(name: str) -> str:
    name = name.strip()
    if name.startswith(WEIGHT_QUANT_OP):
        return WEIGHT_QUANT_OP
    return name


def _is_weight_quant_kernel(row: dict[str, str]) -> bool:
    return any(
        normalize_weight_quant_name(row.get(field, "")) == WEIGHT_QUANT_OP
        for field in ("Name", "Type")
    )


def _is_layout_overhead_op(op_type: str) -> bool:
    normalized = op_type.strip()
    return any(normalized.startswith(name) for name in LAYOUT_OVERHEAD_OPS)


def analyze_w4a16_report(
    report_dir: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    report_dir = report_dir.resolve()
    op_rows = _read_csv(report_dir / "op_statistic.csv")
    kernel_rows = _read_csv(report_dir / "kernel_details.csv")
    total_device_us = sum(_number(row, "Total Time(us)") for row in op_rows)
    weight_quant_rows = [
        row
        for row in op_rows
        if normalize_weight_quant_name(row.get("OP Type", ""))
        == WEIGHT_QUANT_OP
    ]
    total_w4a16_us = sum(
        _number(row, "Total Time(us)") for row in weight_quant_rows
    )
    op_w4a16_count = sum(
        int(_number(row, "Count")) for row in weight_quant_rows
    )
    layout_overhead_us = sum(
        _number(row, "Total Time(us)")
        for row in op_rows
        if _is_layout_overhead_op(row.get("OP Type", ""))
    )

    grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in kernel_rows:
        if not _is_weight_quant_kernel(row):
            continue
        key = (
            row.get("Input Shapes", ""),
            row.get("Input Data Types", ""),
            row.get("Input Formats", ""),
            row.get("Output Shapes", ""),
        )
        item = grouped.setdefault(
            key,
            {"durations_us": [], "kernels": set()},
        )
        item["durations_us"].append(_number(row, "Duration(us)"))
        kernel_name = row.get("Name", "") or row.get("Type", "")
        item["kernels"].add(kernel_name)

    shapes: list[dict[str, object]] = []
    for key, item in grouped.items():
        durations = [float(value) for value in item["durations_us"]]
        count = len(durations)
        total_us = sum(durations)
        shapes.append(
            {
                "shape_id": f"w4a16_shape_{len(shapes):02d}",
                "input_shapes": key[0],
                "input_dtypes": key[1],
                "input_formats": key[2],
                "output_shapes": key[3],
                "count": count,
                "total_us": total_us,
                "avg_us": total_us / count,
                "min_us": min(durations),
                "max_us": max(durations),
                "ratio_of_profile_device_time": (
                    total_us / total_device_us if total_device_us else 0.0
                ),
                "kernels": sorted(item["kernels"]),
            }
        )
    shapes.sort(key=lambda item: float(item["total_us"]), reverse=True)
    for index, item in enumerate(shapes):
        item["shape_id"] = f"w4a16_shape_{index:02d}"

    raw_w4a16_count = sum(int(item["count"]) for item in shapes)
    if op_w4a16_count and raw_w4a16_count != op_w4a16_count:
        raise ValueError(
            "WeightQuant raw count differs between op_statistic.csv and "
            f"kernel_details.csv: op={op_w4a16_count} raw={raw_w4a16_count}"
        )
    weighted_w4a16_us = sum(float(item["total_us"]) for item in shapes)
    prompt_tokens = int(manifest.get("prompt_tokens", 0) or 0)
    return {
        "report_dir": str(report_dir),
        "manifest": manifest,
        "total_device_us": total_device_us,
        "total_w4a16_us": total_w4a16_us,
        "weighted_w4a16_us": weighted_w4a16_us,
        "raw_w4a16_count": raw_w4a16_count,
        "layout_overhead_us": layout_overhead_us,
        "weight_quant_ratio": (
            total_w4a16_us / total_device_us if total_device_us else 0.0
        ),
        "weighted_w4a16_us_per_token": (
            weighted_w4a16_us / prompt_tokens if prompt_tokens else None
        ),
        "shapes": shapes,
    }


def compare_w4a16_reports(
    reference: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    reference_us = float(reference["weighted_w4a16_us"])
    candidate_us = float(candidate["weighted_w4a16_us"])
    if reference_us <= 0:
        raise ValueError("reference weighted_w4a16_us must be positive")
    return {
        "reference_weighted_w4a16_us": reference_us,
        "candidate_weighted_w4a16_us": candidate_us,
        "improvement_percent": (1.0 - candidate_us / reference_us) * 100.0,
    }


def select_candidate(
    reference: dict[str, object],
    candidate: dict[str, object],
    *,
    oracle_passed: bool,
) -> dict[str, object]:
    comparison = compare_w4a16_reports(reference, candidate)
    improvement = float(comparison["improvement_percent"])
    raw_count_matches = int(reference["raw_w4a16_count"]) == int(
        candidate["raw_w4a16_count"]
    )
    added_layout_overhead_us = max(
        0.0,
        float(candidate.get("layout_overhead_us", 0.0))
        - float(reference.get("layout_overhead_us", 0.0)),
    )
    candidate_total_us = float(candidate["total_device_us"])
    added_layout_overhead_ratio = (
        added_layout_overhead_us / candidate_total_us
        if candidate_total_us > 0
        else float("inf")
    )
    overhead_passed = (
        added_layout_overhead_ratio <= MAX_ADDED_LAYOUT_OVERHEAD_RATIO
    )
    return {
        **comparison,
        "oracle_passed": oracle_passed,
        "raw_count_matches": raw_count_matches,
        "added_layout_overhead_us": added_layout_overhead_us,
        "added_layout_overhead_ratio": added_layout_overhead_ratio,
        "required_improvement_percent": REQUIRED_IMPROVEMENT_PERCENT,
        "max_added_layout_overhead_ratio": MAX_ADDED_LAYOUT_OVERHEAD_RATIO,
        "accept_candidate": (
            oracle_passed
            and raw_count_matches
            and overhead_passed
            and improvement >= REQUIRED_IMPROVEMENT_PERCENT
        ),
    }


def _find_manifest(report_dir: Path) -> dict[str, object]:
    for parent in (report_dir, *report_dir.parents):
        path = parent / "manifest.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _find_report_dirs(profile_root: Path) -> list[Path]:
    if (profile_root / "op_statistic.csv").is_file():
        return [profile_root]
    report_dirs = sorted(
        path.parent
        for path in profile_root.rglob("op_statistic.csv")
        if (path.parent / "kernel_details.csv").is_file()
    )
    if not report_dirs:
        raise FileNotFoundError(
            f"No Ascend profiler reports were found below {profile_root}"
        )
    return report_dirs


def build_summary(profile_root: Path) -> dict[str, object]:
    reports = [
        analyze_w4a16_report(report_dir, _find_manifest(report_dir))
        for report_dir in _find_report_dirs(profile_root)
    ]
    scaling = {
        str(report["manifest"].get("prompt_tokens")): report[
            "weighted_w4a16_us_per_token"
        ]
        for report in reports
        if report["manifest"].get("prompt_tokens")
        and report["weighted_w4a16_us_per_token"] is not None
    }
    return {
        "profile_root": str(profile_root.resolve()),
        "reports": reports,
        "per_token_device_us_by_prompt_tokens": scaling,
    }


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Gemma4 W4A16 prefill GEMM shapes",
        "",
        "| Prompt tokens | Shape ID | Count | Total us | Avg us | Input shapes | Input formats |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for report in summary["reports"]:
        prompt_tokens = report["manifest"].get("prompt_tokens", "unknown")
        for shape in report["shapes"]:
            lines.append(
                "| {prompt} | {shape_id} | {count} | {total:.3f} | "
                "{avg:.3f} | `{inputs}` | `{formats}` |".format(
                    prompt=prompt_tokens,
                    shape_id=shape["shape_id"],
                    count=shape["count"],
                    total=float(shape["total_us"]),
                    avg=float(shape["avg_us"]),
                    inputs=shape["input_shapes"],
                    formats=shape["input_formats"],
                )
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(args.profile_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote JSON summary: {args.output_json}", flush=True)
    print(f"Wrote Markdown summary: {args.output_markdown}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
