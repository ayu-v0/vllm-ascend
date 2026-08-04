#!/usr/bin/env python3
"""Summarize paired Gemma4 dense-target Ascend profiler reports."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


WEIGHT_QUANT_OP = "WeightQuantBatchMatmulV2"
WEIGHT_QUANT_CALLS_PER_TARGET_STEP = 60 * 4
TRACKED_OP_TYPES = (
    "Index",
    "NonZero",
    "Range",
    "Less",
    "MemSet",
    "Cast",
    "GatherV3",
)
KERNEL_GROUPS = (
    "legacy_block_gather",
    "legacy_boolean_index",
    "compact_sliding_gather",
    "compact_full_gather",
)


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


def _classify_kernel(row: dict[str, str]) -> str | None:
    op_type = row.get("Type", "").strip()
    shapes = row.get("Input Shapes", "").replace('"', "").replace(" ", "")
    is_sliding = ",16,256;" in shapes
    is_full = ",4,512;" in shapes
    has_block_size_dimension = (
        ",128,16,256;" in shapes or ",128,4,512;" in shapes
    )

    if op_type == "GatherV3" and has_block_size_dimension:
        return "legacy_block_gather"
    if op_type == "Index" and (is_sliding or is_full) and shapes.startswith("1,"):
        return "legacy_boolean_index"
    if op_type == "GatherV3" and is_sliding and not has_block_size_dimension:
        return "compact_sliding_gather"
    if op_type == "GatherV3" and is_full and not has_block_size_dimension:
        return "compact_full_gather"
    return None


def _find_manifest(report_dir: Path) -> dict[str, object] | None:
    for parent in (report_dir, *report_dir.parents):
        manifest_path = parent / "manifest.json"
        if manifest_path.is_file():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
    return None


def analyze_report(report_dir: Path, *, label: str) -> dict[str, object]:
    report_dir = report_dir.resolve()
    op_rows = _read_csv(report_dir / "op_statistic.csv")
    kernel_rows = _read_csv(report_dir / "kernel_details.csv")

    weight_count = sum(
        int(_number(row, "Count"))
        for row in op_rows
        if row.get("OP Type", "").startswith(WEIGHT_QUANT_OP)
    )
    if (
        weight_count <= 0
        or weight_count % WEIGHT_QUANT_CALLS_PER_TARGET_STEP != 0
    ):
        raise ValueError(
            "WeightQuantBatchMatmulV2 count does not identify an integral "
            f"target step count: count={weight_count} "
            f"calls_per_step={WEIGHT_QUANT_CALLS_PER_TARGET_STEP}"
        )
    target_steps = weight_count // WEIGHT_QUANT_CALLS_PER_TARGET_STEP
    total_device_us = sum(_number(row, "Total Time(us)") for row in op_rows)

    op_types: dict[str, dict[str, float | int]] = {}
    for op_type in TRACKED_OP_TYPES:
        rows = [row for row in op_rows if row.get("OP Type") == op_type]
        count = sum(int(_number(row, "Count")) for row in rows)
        total_us = sum(_number(row, "Total Time(us)") for row in rows)
        op_types[op_type] = {
            "count": count,
            "total_us": total_us,
            "count_per_step": count / target_steps,
            "us_per_step": total_us / target_steps,
        }

    kernel_groups: dict[str, dict[str, float | int]] = {
        name: {
            "count": 0,
            "total_us": 0.0,
            "count_per_step": 0.0,
            "us_per_step": 0.0,
        }
        for name in KERNEL_GROUPS
    }
    for row in kernel_rows:
        group = _classify_kernel(row)
        if group is None:
            continue
        kernel_groups[group]["count"] += 1
        kernel_groups[group]["total_us"] += _number(row, "Duration(us)")

    for metrics in kernel_groups.values():
        metrics["count_per_step"] = metrics["count"] / target_steps
        metrics["us_per_step"] = metrics["total_us"] / target_steps

    paged_kv_us_per_step = sum(
        float(kernel_groups[name]["us_per_step"])
        for name in KERNEL_GROUPS
    )
    mask_overhead_us_per_step = sum(
        float(op_types[op_type]["us_per_step"])
        for op_type in ("NonZero", "Range", "Less", "MemSet")
    )
    return {
        "label": label,
        "report_dir": str(report_dir),
        "manifest": _find_manifest(report_dir),
        "target_steps": target_steps,
        "total_device_us": total_device_us,
        "total_device_us_per_step": total_device_us / target_steps,
        "paged_kv_classified_us_per_step": paged_kv_us_per_step,
        "paged_kv_estimated_us_per_step": (
            paged_kv_us_per_step + mask_overhead_us_per_step
        ),
        "op_types": op_types,
        "kernel_groups": kernel_groups,
    }


def _find_reports(profile_root: Path) -> list[Path]:
    if (profile_root / "op_statistic.csv").is_file():
        return [profile_root]
    reports = sorted(
        path.parent
        for path in profile_root.rglob("op_statistic.csv")
        if (path.parent / "kernel_details.csv").is_file()
    )
    if not reports:
        raise FileNotFoundError(
            f"No Ascend profiler reports were found below {profile_root}"
        )
    return reports


def _median_for_prefix(
    reports: list[dict[str, object]],
    prefix: str,
) -> dict[str, float] | None:
    selected = [
        report
        for report in reports
        if str(report["label"]).lower().startswith(prefix)
    ]
    if not selected:
        return None
    metrics = (
        "total_device_us_per_step",
        "paged_kv_classified_us_per_step",
        "paged_kv_estimated_us_per_step",
    )
    result = {
        metric: statistics.median(float(report[metric]) for report in selected)
        for metric in metrics
    }
    for group in KERNEL_GROUPS:
        result[f"{group}_us_per_step"] = statistics.median(
            float(report["kernel_groups"][group]["us_per_step"])
            for report in selected
        )
        result[f"{group}_count_per_step"] = statistics.median(
            float(report["kernel_groups"][group]["count_per_step"])
            for report in selected
        )
    return result


def _build_summary(reports: list[dict[str, object]]) -> dict[str, object]:
    manifests = [report["manifest"] for report in reports]
    available_manifests = [manifest for manifest in manifests if manifest is not None]
    if available_manifests and len(available_manifests) != len(manifests):
        raise ValueError("Some profiler reports are missing manifest.json")
    if available_manifests:
        comparable_fields = (
            "target_model",
            "draft_model",
            "tensor_parallel_size",
            "num_speculative_tokens",
            "prompt_sha256",
            "profile_max_tokens",
            "generated_token_count",
            "output_token_ids",
            "engine",
            "sampling",
            "profiler",
        )
        baseline = available_manifests[0]
        for manifest in available_manifests[1:]:
            mismatches = [
                field
                for field in comparable_fields
                if manifest.get(field) != baseline.get(field)
            ]
            if mismatches:
                raise ValueError(
                    "Profiler manifests are not comparable: "
                    + ", ".join(mismatches)
                )

    medians = {
        "old": _median_for_prefix(reports, "old"),
        "new": _median_for_prefix(reports, "new"),
    }
    comparison = None
    if medians["old"] is not None and medians["new"] is not None:
        comparison = {}
        for metric in (
            "total_device_us_per_step",
            "paged_kv_classified_us_per_step",
            "paged_kv_estimated_us_per_step",
        ):
            old_value = medians["old"][metric]
            new_value = medians["new"][metric]
            comparison[f"{metric}_change_percent"] = (
                (new_value / old_value - 1.0) * 100.0
                if old_value
                else None
            )
    return {
        "reports": reports,
        "medians": medians,
        "comparison": comparison,
    }


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Gemma4 Dense Target Profiler Summary",
        "",
        "| Label | Target steps | Device us/step | "
        "Estimated paged-KV us/step | Legacy Index/step | "
        "Compact sliding gather/step | Compact full gather/step |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in summary["reports"]:
        groups = report["kernel_groups"]
        lines.append(
            "| {label} | {steps} | {device:.3f} | {paged:.3f} | "
            "{legacy:.3f} | {sliding:.3f} | {full:.3f} |".format(
                label=report["label"],
                steps=report["target_steps"],
                device=report["total_device_us_per_step"],
                paged=report["paged_kv_estimated_us_per_step"],
                legacy=groups["legacy_boolean_index"]["count_per_step"],
                sliding=groups["compact_sliding_gather"]["count_per_step"],
                full=groups["compact_full_gather"]["count_per_step"],
            )
        )

    lines.extend(["", "## Paired medians", ""])
    for mode in ("old", "new"):
        median = summary["medians"][mode]
        if median is None:
            lines.append(f"- `{mode}`: no reports")
        else:
            lines.append(
                f"- `{mode}` device: "
                f"{median['total_device_us_per_step']:.3f} us/step; "
                "estimated paged-KV: "
                f"{median['paged_kv_estimated_us_per_step']:.3f} us/step"
            )
    comparison = summary["comparison"]
    if comparison is not None:
        lines.extend(
            [
                "",
                "## New versus old",
                "",
                "- Device time change: "
                f"{comparison['total_device_us_per_step_change_percent']:.3f}%",
                "- Estimated paged-KV time change: "
                f"{comparison['paged_kv_estimated_us_per_step_change_percent']:.3f}%",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    profile_root = args.profile_root.resolve()
    reports = [
        analyze_report(
            report_dir,
            label=report_dir.relative_to(profile_root).parts[0]
            if report_dir != profile_root
            else profile_root.name,
        )
        for report_dir in _find_reports(profile_root)
    ]
    summary = _build_summary(reports)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    args.output_markdown.write_text(_markdown(summary), encoding="utf-8")
    print(f"reports: {len(reports)}")
    print(f"json: {args.output_json}")
    print(f"markdown: {args.output_markdown}")


if __name__ == "__main__":
    main()
