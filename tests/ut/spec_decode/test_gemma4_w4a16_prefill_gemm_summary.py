import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[3]
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
    / "summarize_gemma4_w4a16_prefill_gemm.py"
)
BENCHMARK = SCRIPT.with_name("benchmark_gemma4_w4a16_prefill_gemm.py")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "gemma4_w4a16_prefill_gemm_summary",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "gemma4_w4a16_prefill_gemm_benchmark",
        BENCHMARK,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_synthetic_report(report_dir: Path) -> None:
    _write_csv(
        report_dir / "op_statistic.csv",
        ["OP Type", "Count", "Total Time(us)"],
        [
            {
                "OP Type": "WeightQuantBatchMatmulV2",
                "Count": "3",
                "Total Time(us)": "300",
            },
            {
                "OP Type": "FlashAttentionScore",
                "Count": "1",
                "Total Time(us)": "198",
            },
            {"OP Type": "Cast", "Count": "1", "Total Time(us)": "2"},
        ],
    )
    kernel_fields = [
        "Name",
        "Type",
        "Duration(us)",
        "Input Shapes",
        "Input Data Types",
        "Input Formats",
        "Output Shapes",
    ]
    common = {
        "Type": "WeightQuantBatchMatmulV2",
        "Input Data Types": "BF16;INT4;BF16",
        "Input Formats": "ND;FRACTAL_NZ;ND",
        "Output Shapes": "8192,4096",
    }
    _write_csv(
        report_dir / "kernel_details.csv",
        kernel_fields,
        [
            {
                **common,
                "Name": "WeightQuantBatchMatmulV2",
                "Duration(us)": "100",
                "Input Shapes": "8192,4096;4096,4096;4096,32",
            },
            {
                **common,
                "Name": (
                    "WeightQuantBatchMatmulV2_bf16_int4_bf16_"
                    "high_performance_123"
                ),
                "Duration(us)": "140",
                "Input Shapes": "8192,4096;4096,4096;4096,32",
            },
            {
                **common,
                "Name": "unrelated_kernel_name",
                "Type": "WeightQuantBatchMatmulV2_bf16_int4_bf16_456",
                "Duration(us)": "60",
                "Input Shapes": "8192,4096;4096,16384;16384,32",
                "Output Shapes": "8192,16384",
            },
        ],
    )


class W4A16SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_groups_normalized_kernel_by_shape_and_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir)
            _write_synthetic_report(report_dir)
            result = self.module.analyze_w4a16_report(
                report_dir,
                {"prompt_tokens": 8192},
            )

        self.assertEqual(result["total_w4a16_us"], 300.0)
        self.assertEqual(result["weighted_w4a16_us"], 300.0)
        self.assertAlmostEqual(result["weight_quant_ratio"], 0.6)
        self.assertEqual(result["raw_w4a16_count"], 3)
        self.assertEqual(result["shapes"][0]["count"], 2)
        self.assertEqual(result["shapes"][0]["total_us"], 240.0)
        self.assertEqual(result["shapes"][0]["min_us"], 100.0)
        self.assertEqual(result["shapes"][0]["max_us"], 140.0)
        self.assertEqual(len(result["shapes"][0]["kernels"]), 2)

    def test_candidate_gate_requires_eight_percent_weighted_gemm_gain(self):
        decision = self.module.select_candidate(
            reference={
                "weighted_w4a16_us": 1000.0,
                "raw_w4a16_count": 10,
                "total_device_us": 2000.0,
                "layout_overhead_us": 0.0,
            },
            candidate={
                "weighted_w4a16_us": 925.0,
                "raw_w4a16_count": 10,
                "total_device_us": 1900.0,
                "layout_overhead_us": 0.0,
            },
            oracle_passed=True,
        )

        self.assertFalse(decision["accept_candidate"])
        self.assertEqual(decision["required_improvement_percent"], 8.0)
        self.assertAlmostEqual(decision["improvement_percent"], 7.5)

    def test_candidate_gate_rejects_count_mismatch_or_added_layout_overhead(self):
        reference = {
            "weighted_w4a16_us": 1000.0,
            "raw_w4a16_count": 10,
            "total_device_us": 2000.0,
            "layout_overhead_us": 1.0,
        }
        candidate = {
            "weighted_w4a16_us": 900.0,
            "raw_w4a16_count": 9,
            "total_device_us": 1800.0,
            "layout_overhead_us": 30.0,
        }

        decision = self.module.select_candidate(
            reference,
            candidate,
            oracle_passed=True,
        )

        self.assertFalse(decision["accept_candidate"])
        self.assertFalse(decision["raw_count_matches"])
        self.assertGreater(decision["added_layout_overhead_ratio"], 0.01)


class W4A16BenchmarkContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_benchmark()

    def test_builds_fixed_m_matrix_from_profile_shape(self):
        summary = {
            "reports": [
                {
                    "manifest": {"prompt_tokens": 28672},
                    "shapes": [
                        {
                            "shape_id": "w4a16_shape_00",
                            "input_shapes": (
                                '"123,5376;5376,16384;42,16384"'
                            ),
                            "count": 60,
                        }
                    ],
                }
            ]
        }

        workloads = self.module.build_workloads(summary)

        self.assertEqual(
            [item["m"] for item in workloads],
            [1, 123, 4096, 8192],
        )
        self.assertTrue(
            all(item["k"] == 5376 and item["n"] == 16384 for item in workloads)
        )
        tail = next(item for item in workloads if item["is_profile_shape"])
        self.assertEqual(tail["source_count"], 60)

    def test_weighted_gate_uses_only_profile_shape_counts(self):
        results = [
            {
                "reference_us": 100.0,
                "candidate_us": 90.0,
                "source_count": 60,
                "is_profile_shape": True,
                "passed": True,
            },
            {
                "reference_us": 1000.0,
                "candidate_us": 2000.0,
                "source_count": 60,
                "is_profile_shape": False,
                "passed": True,
            },
        ]

        summary = self.module.summarize_results(results)

        self.assertEqual(summary["weighted_reference_us"], 6000.0)
        self.assertEqual(summary["weighted_candidate_us"], 5400.0)
        self.assertAlmostEqual(summary["weighted_improvement_percent"], 10.0)
        self.assertTrue(summary["passed"])

    def test_failed_operator_case_preserves_a_fail_closed_summary(self):
        results = [
            {
                "shape_id": "shape-0",
                "m": 8192,
                "source_count": 60,
                "is_profile_shape": True,
                "passed": False,
                "error": "RuntimeError: candidate format unsupported",
            }
        ]

        summary = self.module.summarize_results(results)

        self.assertFalse(summary["correctness_passed"])
        self.assertFalse(summary["performance_gate_passed"])
        self.assertFalse(summary["passed"])
        self.assertIn("not proven inefficient", summary["conclusion"])


if __name__ == "__main__":
    unittest.main()
