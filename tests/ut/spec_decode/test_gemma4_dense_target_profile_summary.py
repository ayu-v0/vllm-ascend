import csv
import importlib.util
import tempfile
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[3]
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
    / "summarize_gemma4_dense_target_profile.py"
)
PROFILE_RUNNER = SCRIPT.with_name("profile_gemma4_dense_target_decode.py")


def _load_summarizer():
    spec = importlib.util.spec_from_file_location(
        "gemma4_dense_target_profile_summary",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_report(report_dir: Path, *, weight_count: int = 480) -> None:
    report_dir.mkdir(parents=True)
    with (report_dir / "op_statistic.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Device_id",
                "OP Type",
                "Core Type",
                "Count",
                "Total Time(us)",
                "Min Time(us)",
                "Avg Time(us)",
                "Max Time(us)",
                "Ratio(%)",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "Device_id": "0",
                    "OP Type": "WeightQuantBatchMatmulV2",
                    "Core Type": "MIX_AIC",
                    "Count": str(weight_count),
                    "Total Time(us)": "2000",
                    "Min Time(us)": "1",
                    "Avg Time(us)": "1",
                    "Max Time(us)": "1",
                    "Ratio(%)": "50",
                },
                {
                    "Device_id": "0",
                    "OP Type": "Index",
                    "Core Type": "AI_VECTOR_CORE",
                    "Count": "4",
                    "Total Time(us)": "80",
                    "Min Time(us)": "20",
                    "Avg Time(us)": "20",
                    "Max Time(us)": "20",
                    "Ratio(%)": "2",
                },
                {
                    "Device_id": "0",
                    "OP Type": "GatherV3",
                    "Core Type": "AI_VECTOR_CORE",
                    "Count": "4",
                    "Total Time(us)": "40",
                    "Min Time(us)": "10",
                    "Avg Time(us)": "10",
                    "Max Time(us)": "10",
                    "Ratio(%)": "1",
                },
            ]
        )

    fields = [
        "Device_id",
        "Model ID",
        "Task ID",
        "Stream ID",
        "Name",
        "Type",
        "OP State",
        "Accelerator Core",
        "Start Time(us)",
        "Duration(us)",
        "Wait Time(us)",
        "Block Num",
        "Mix Block Num",
        "HF32 Eligible",
        "Input Shapes",
        "Input Data Types",
        "Input Formats",
        "Output Shapes",
        "Output Data Types",
        "Output Formats",
        "Context ID",
    ]
    with (report_dir / "kernel_details.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for op_type, shape, duration in (
            ("GatherV3", "1830,128,16,256;4;1", "12"),
            ("Index", "1,512,16,256;2;3;421;421", "40"),
            ("GatherV3", "234240,16,256;4;1", "8"),
            ("GatherV3", "468480,4,512;4;1", "7"),
        ):
            row = dict.fromkeys(fields, "")
            row.update(
                {
                    "Device_id": "0",
                    "Name": op_type,
                    "Type": op_type,
                    "Duration(us)": duration,
                    "Input Shapes": shape,
                }
            )
            writer.writerow(row)


def test_profile_summary_classifies_legacy_and_compact_kernels():
    module = _load_summarizer()
    with tempfile.TemporaryDirectory() as temp_dir:
        report_dir = Path(temp_dir) / "ASCEND_PROFILER_OUTPUT"
        _write_report(report_dir)

        result = module.analyze_report(report_dir, label="synthetic")

    assert result["target_steps"] == 2
    assert result["op_types"]["Index"]["count_per_step"] == 2
    assert result["paged_kv_estimated_us_per_step"] == 33.5
    assert result["kernel_groups"]["legacy_block_gather"]["count"] == 1
    assert result["kernel_groups"]["legacy_boolean_index"]["count"] == 1
    assert result["kernel_groups"]["compact_sliding_gather"]["count"] == 1
    assert result["kernel_groups"]["compact_full_gather"]["count"] == 1


def test_profile_summary_rejects_non_integral_target_steps():
    module = _load_summarizer()
    with tempfile.TemporaryDirectory() as temp_dir:
        report_dir = Path(temp_dir) / "ASCEND_PROFILER_OUTPUT"
        _write_report(report_dir, weight_count=481)

        with pytest.raises(ValueError, match="target step"):
            module.analyze_report(report_dir, label="invalid")


def test_profile_summary_rejects_old_new_token_mismatch():
    module = _load_summarizer()
    base = {
        "target_model": "target",
        "draft_model": "draft",
        "tensor_parallel_size": 1,
        "num_speculative_tokens": 3,
        "prompt_sha256": "abc",
        "profile_max_tokens": 64,
        "generated_token_count": 2,
        "output_token_ids": [1, 2],
        "engine": {"seed": 0},
        "profiler": {"max_iterations": 0},
    }
    reports = [
        {"label": "old_r1", "manifest": base},
        {
            "label": "new_r1",
            "manifest": {**base, "output_token_ids": [1, 3]},
        },
    ]

    with pytest.raises(ValueError, match="output_token_ids"):
        module._build_summary(reports)


def test_profile_runner_manifest_records_sampling_and_full_profiler_config():
    source = PROFILE_RUNNER.read_text(encoding="utf-8")
    manifest_source = source[
        source.index("manifest = {") : source.index("manifest_out.write_text")
    ]

    assert '"sampling"' in manifest_source
    for field in (
        '"temperature"',
        '"seed"',
        '"ignore_eos"',
        '"max_tokens"',
        '"torch_profiler_with_stack"',
        '"torch_profiler_with_flops"',
        '"torch_profiler_use_gzip"',
        '"torch_profiler_dump_cuda_time_total"',
        '"torch_profiler_record_shapes"',
        '"torch_profiler_with_memory"',
        '"ignore_frontend"',
    ):
        assert field in manifest_source


def test_profile_summary_rejects_old_new_sampling_mismatch():
    module = _load_summarizer()
    base = {
        "target_model": "target",
        "draft_model": "draft",
        "tensor_parallel_size": 1,
        "num_speculative_tokens": 3,
        "prompt_sha256": "abc",
        "profile_max_tokens": 64,
        "generated_token_count": 2,
        "output_token_ids": [1, 2],
        "engine": {"seed": 0},
        "sampling": {"temperature": 0, "seed": 0, "ignore_eos": True},
        "profiler": {"max_iterations": 0},
    }
    metrics = {
        "total_device_us_per_step": 1.0,
        "paged_kv_classified_us_per_step": 1.0,
        "paged_kv_estimated_us_per_step": 1.0,
        "kernel_groups": {
            group: {"us_per_step": 0.0, "count_per_step": 0.0}
            for group in module.KERNEL_GROUPS
        },
    }
    reports = [
        {"label": "old_r1", "manifest": base, **metrics},
        {
            "label": "new_r1",
            "manifest": {
                **base,
                "sampling": {
                    "temperature": 0,
                    "seed": 1,
                    "ignore_eos": True,
                },
            },
            **metrics,
        },
    ]

    with pytest.raises(ValueError, match="sampling"):
        module._build_summary(reports)


if __name__ == "__main__":
    test_profile_summary_classifies_legacy_and_compact_kernels()
    test_profile_summary_rejects_non_integral_target_steps()
    test_profile_summary_rejects_old_new_token_mismatch()
    test_profile_runner_manifest_records_sampling_and_full_profiler_config()
    test_profile_summary_rejects_old_new_sampling_mismatch()
