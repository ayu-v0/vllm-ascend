#!/usr/bin/env python3
"""Benchmark reference and candidate W4A16 GEMMs on real Gemma4 shapes."""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import groupby
from pathlib import Path
from typing import Any


REQUIRED_IMPROVEMENT_PERCENT = 8.0
CORRECTNESS_ATOL = 2e-2
CORRECTNESS_RTOL = 2e-2
FIXED_M_VALUES = (1, 4096, 8192)


def _parse_matrix_shape(raw: object) -> tuple[int, int]:
    cleaned = str(raw).strip().strip('"')
    dimensions = cleaned.split(",")
    if len(dimensions) != 2:
        raise ValueError(f"Expected a two-dimensional shape, got {raw!r}")
    return int(dimensions[0]), int(dimensions[1])


def _parse_w4a16_shapes(raw: object) -> tuple[int, int, int, int]:
    inputs = str(raw).strip().strip('"').split(";")
    if len(inputs) < 3:
        raise ValueError(f"Expected x, weight, and scale shapes, got {raw!r}")
    m, k = _parse_matrix_shape(inputs[0])
    weight_k, n = _parse_matrix_shape(inputs[1])
    scale_groups, scale_n = _parse_matrix_shape(inputs[2])
    if weight_k != k or scale_n != n or scale_groups <= 0:
        raise ValueError(f"Inconsistent W4A16 input shapes: {raw!r}")
    if k % scale_groups:
        raise ValueError(f"Cannot derive integral group_size from {raw!r}")
    return m, k, n, k // scale_groups


def _baseline_reports(summary: dict[str, object]) -> list[dict[str, object]]:
    reports = list(summary.get("reports", []))
    if not reports:
        raise ValueError("Shape manifest contains no reports")
    target_compiled = [
        report
        for report in reports
        if report.get("manifest", {}).get("mode", "target") == "target"
        and report.get("manifest", {}).get("execution", "compiled")
        == "compiled"
    ]
    selected = target_compiled or reports
    max_prompt_tokens = max(
        int(report.get("manifest", {}).get("prompt_tokens", 0) or 0)
        for report in selected
    )
    if max_prompt_tokens:
        selected = [
            report
            for report in selected
            if int(
                report.get("manifest", {}).get("prompt_tokens", 0) or 0
            )
            == max_prompt_tokens
        ]
    return selected


def build_workloads(summary: dict[str, object]) -> list[dict[str, object]]:
    profile_shapes: dict[tuple[int, int, int], dict[int, int]] = {}
    for report in _baseline_reports(summary):
        for shape in report.get("shapes", []):
            m, k, n, group_size = _parse_w4a16_shapes(
                shape["input_shapes"]
            )
            counts = profile_shapes.setdefault((k, n, group_size), {})
            counts[m] = counts.get(m, 0) + int(shape["count"])

    workloads: list[dict[str, object]] = []
    for shape_index, ((k, n, group_size), source_shapes) in enumerate(
        sorted(profile_shapes.items())
    ):
        shape_id = f"shape-{shape_index}"
        for m in sorted(set(FIXED_M_VALUES) | set(source_shapes)):
            workloads.append(
                {
                    "shape_id": shape_id,
                    "m": m,
                    "k": k,
                    "n": n,
                    "group_size": group_size,
                    "source_count": source_shapes.get(m, 0),
                    "is_profile_shape": m in source_shapes,
                }
            )
    if not workloads:
        raise ValueError("Shape manifest contains no W4A16 shapes")
    return workloads


def summarize_results(
    results: list[dict[str, object]],
) -> dict[str, object]:
    correctness_passed = all(bool(result["passed"]) for result in results)
    profile_results = [
        result
        for result in results
        if bool(result["is_profile_shape"])
        and "reference_us" in result
        and "candidate_us" in result
    ]
    weighted_reference_us = sum(
        float(result["reference_us"]) * int(result["source_count"])
        for result in profile_results
    )
    weighted_candidate_us = sum(
        float(result["candidate_us"]) * int(result["source_count"])
        for result in profile_results
    )
    improvement = None
    if weighted_reference_us > 0:
        improvement = (
            1.0 - weighted_candidate_us / weighted_reference_us
        ) * 100.0
    performance_gate_passed = (
        improvement is not None
        and improvement >= REQUIRED_IMPROVEMENT_PERCENT
    )
    return {
        "weighted_reference_us": weighted_reference_us,
        "weighted_candidate_us": weighted_candidate_us,
        "weighted_improvement_percent": improvement,
        "required_improvement_percent": REQUIRED_IMPROVEMENT_PERCENT,
        "correctness_passed": correctness_passed,
        "performance_gate_passed": performance_gate_passed,
        "passed": correctness_passed and performance_gate_passed,
        "conclusion": (
            "candidate reached the W4A16 GEMM gate"
            if performance_gate_passed
            else "W4A16 GEMM is dominant but not proven inefficient"
        ),
    }


def _time_apply(
    torch,
    apply_fn,
    x,
    weight,
    scale,
    group_size: int,
    iterations: int,
) -> tuple[float, Any]:
    torch.npu.synchronize()
    started = time.perf_counter_ns()
    output = None
    for _ in range(iterations):
        output = apply_fn(x, weight, scale, group_size, None)
    torch.npu.synchronize()
    elapsed_us = (time.perf_counter_ns() - started) / 1000 / iterations
    return elapsed_us, output


def _warmup(
    apply_fn,
    x,
    weight,
    scale,
    group_size: int,
    warmups: int,
) -> None:
    for _ in range(warmups):
        apply_fn(x, weight, scale, group_size, None)


def _run_case(
    torch,
    torch_npu,
    apply_reference,
    apply_candidate,
    workload: dict[str, object],
    reference_weight,
    candidate_weight,
    scale,
    *,
    warmups: int,
    iterations: int,
    case_index: int,
) -> dict[str, object]:
    m = int(workload["m"])
    k = int(workload["k"])
    n = int(workload["n"])
    group_size = int(workload["group_size"])
    x = torch.randn((m, k), dtype=torch.bfloat16, device="npu")
    functions = (
        (("reference", apply_reference), ("candidate", apply_candidate))
        if case_index % 2 == 0
        else (("candidate", apply_candidate), ("reference", apply_reference))
    )
    timings: dict[str, float] = {}
    outputs: dict[str, Any] = {}
    for name, apply_fn in functions:
        weight = reference_weight if name == "reference" else candidate_weight
        _warmup(apply_fn, x, weight, scale, group_size, warmups)
        timings[name], outputs[name] = _time_apply(
            torch,
            apply_fn,
            x,
            weight,
            scale,
            group_size,
            iterations,
        )

    reference = outputs["reference"]
    candidate = outputs["candidate"]
    finite = torch.isfinite(reference).all() & torch.isfinite(candidate).all()
    absolute_error = (reference - candidate).abs()
    relative_error = absolute_error / reference.abs().clamp_min(1e-6)
    torch.npu.synchronize()
    finite_outputs = bool(finite.item())
    max_abs_error = float(absolute_error.amax().item())
    max_rel_error = float(relative_error.amax().item())
    correctness_passed = finite_outputs and not (
        max_abs_error > CORRECTNESS_ATOL
        and max_rel_error > CORRECTNESS_RTOL
    )
    reference_format = int(torch_npu.get_npu_format(reference_weight))
    candidate_format = int(torch_npu.get_npu_format(candidate_weight))
    reference_us = timings["reference"]
    candidate_us = timings["candidate"]
    return {
        **workload,
        "reference_us": reference_us,
        "candidate_us": candidate_us,
        "improvement_percent": (
            1.0 - candidate_us / reference_us
        ) * 100.0,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "finite_outputs": finite_outputs,
        "reference_format": reference_format,
        "candidate_format": candidate_format,
        "format_changed": candidate_format != reference_format,
        "passed": correctness_passed,
        "input_shape": [m, k],
        "output_shape": [m, n],
        "scale_dtype": str(scale.dtype),
    }


def run_benchmark(
    workloads: list[dict[str, object]],
    *,
    warmups: int,
    iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    import torch
    import torch_npu

    from vllm_ascend.quantization.methods.w4a16 import (
        _apply_w4a16_candidate,
        _apply_w4a16_reference,
    )
    from vllm_ascend.utils import maybe_trans_nz

    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    sorted_workloads = sorted(
        workloads,
        key=lambda item: (
            int(item["k"]),
            int(item["n"]),
            int(item["group_size"]),
            int(item["m"]),
        ),
    )
    results: list[dict[str, object]] = []
    case_index = 0
    for group, cases in groupby(
        sorted_workloads,
        key=lambda item: (
            int(item["k"]),
            int(item["n"]),
            int(item["group_size"]),
        ),
    ):
        k, n, group_size = group
        unpacked_weight = torch.randint(
            -8,
            8,
            (k, n),
            dtype=torch.int32,
            device="npu",
        )
        reference_weight = torch_npu.npu_convert_weight_to_int4pack(
            unpacked_weight
        )
        candidate_weight = maybe_trans_nz(reference_weight.clone())
        scale = torch.rand(
            (k // group_size, n),
            dtype=torch.bfloat16,
            device="npu",
        ).contiguous()
        for workload in cases:
            print(
                "Benchmarking "
                f"shape={workload['shape_id']} m={workload['m']} "
                f"k={k} n={n} group_size={group_size}",
                flush=True,
            )
            try:
                result = _run_case(
                    torch,
                    torch_npu,
                    _apply_w4a16_reference,
                    _apply_w4a16_candidate,
                    workload,
                    reference_weight,
                    candidate_weight,
                    scale,
                    warmups=warmups,
                    iterations=iterations,
                    case_index=case_index,
                )
            except Exception as error:
                result = {
                    **workload,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            case_index += 1
        del unpacked_weight, reference_weight, candidate_weight, scale
        torch.npu.empty_cache()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-manifest", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    shape_manifest = json.loads(
        args.shape_manifest.read_text(encoding="utf-8")
    )
    workloads = build_workloads(shape_manifest)
    results = run_benchmark(
        workloads,
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed,
    )
    summary = summarize_results(results)
    output = {
        "shape_manifest": str(args.shape_manifest.resolve()),
        "warmups": args.warmups,
        "iterations": args.iterations,
        "seed": args.seed,
        "w4a16_linear_impl": os.environ.get(
            "VLLM_ASCEND_W4A16_LINEAR_IMPL",
            "reference",
        ),
        "vllm_ascend_enable_nz": os.environ.get(
            "VLLM_ASCEND_ENABLE_NZ",
            "1",
        ),
        "results": results,
        "summary": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Wrote microbenchmark result: {args.output_json}", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
