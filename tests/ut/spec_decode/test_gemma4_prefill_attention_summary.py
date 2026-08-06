import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[3]
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
    / "summarize_gemma4_prefill_attention.py"
)
AB_SHELL = SCRIPT.with_name("run_gemma4_prefill_attention_ab.sh")
NPU_TEST = SCRIPT.with_name("test_gemma4_prefill_attention.py")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "gemma4_prefill_attention_summary",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Gemma4PrefillAttentionSummaryTests(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT.is_file())

    def test_attention_summary_separates_attention_and_kv_gather(self):
        module = _load_module()
        rows = [
            {
                "OP Type": "FlashAttentionScore",
                "Count": "2",
                "Total Time(us)": "800",
            },
            {
                "OP Type": "IndexSelect",
                "Count": "4",
                "Total Time(us)": "200",
            },
            {
                "OP Type": "WeightQuantBatchMatmulV2",
                "Count": "10",
                "Total Time(us)": "3000",
            },
        ]

        summary = module.summarize_attention_rows(rows)

        self.assertEqual(summary["attention_us"], 800.0)
        self.assertEqual(summary["attention_count"], 2)
        self.assertEqual(summary["kv_gather_us"], 200.0)
        self.assertEqual(summary["kv_gather_count"], 4)
        self.assertEqual(summary["attention_plus_gather_us"], 1000.0)
        self.assertEqual(summary["weight_quant_us"], 3000.0)

    def test_result_identity_requires_matching_run_and_format(self):
        module = _load_module()
        result = module.validate_w4a16_no_go_evidence(
            log_summary={
                "result_path": (
                    "/data/gemma4_w4a16_gemm/"
                    "20260805_102442_microbenchmark_internal_format.json"
                ),
                "candidate_format": 29,
            },
            json_summary={
                "result_path": "20260805_093628_microbenchmark.json",
                "candidate_format": 2,
            },
        )

        self.assertFalse(result["same_run"])
        self.assertFalse(result["same_candidate_format"])
        self.assertFalse(result["matches"])

    def test_profile_gates_require_attention_and_total_device_gain(self):
        module = _load_module()
        reference = {
            "attention_us": 800.0,
            "kv_gather_us": 200.0,
            "attention_plus_gather_us": 1000.0,
            "total_device_us": 5000.0,
            "weight_quant_us": 3000.0,
            "weight_quant_count": 100,
        }
        windowed = {
            "attention_us": 680.0,
            "kv_gather_us": 120.0,
            "attention_plus_gather_us": 800.0,
            "total_device_us": 4800.0,
            "weight_quant_us": 2990.0,
            "weight_quant_count": 100,
        }

        comparison = module.compare_attention_runs(reference, windowed)

        self.assertAlmostEqual(
            comparison["attention_plus_gather_improvement_percent"],
            20.0,
        )
        self.assertAlmostEqual(
            comparison["total_device_improvement_percent"],
            4.0,
        )
        self.assertTrue(comparison["gates"]["attention_plus_gather_gain"])
        self.assertTrue(comparison["gates"]["total_device_gain"])
        self.assertTrue(comparison["gates"]["weight_quant_count_matches"])
        self.assertTrue(comparison["accept_candidate"])

    def test_ab_comparison_rejects_non_reference_w4a16(self):
        module = _load_module()

        def report(attention_impl, w4a16_impl):
            return {
                "mode": "target",
                "execution": "compiled",
                "prompt_tokens": 28672,
                "attention_us": 800.0,
                "kv_gather_us": 200.0,
                "attention_plus_gather_us": 1000.0,
                "total_device_us": 5000.0,
                "weight_quant_us": 3000.0,
                "weight_quant_count": 100,
                "manifest": {
                    "target_model": "target",
                    "draft_model": None,
                    "mode": "target",
                    "execution": "compiled",
                    "tensor_parallel_size": 1,
                    "prompt_tokens": 28672,
                    "w4a16_linear_impl": w4a16_impl,
                    "vllm_ascend_enable_nz": 1,
                    "gemma4_prefill_attention_impl": attention_impl,
                    "engine": {"max_num_batched_tokens": 8192},
                    "sampling": {"max_tokens": 1},
                    "profiler": {"active_iterations": 5},
                },
            }

        with self.assertRaisesRegex(ValueError, "W4A16 reference"):
            module.build_ab_comparison(
                {"reports": [report("reference", "candidate")]},
                {"reports": [report("windowed", "candidate")]},
            )

    def test_ab_comparison_ignores_profiler_output_directory(self):
        module = _load_module()

        def report(attention_impl, profile_dir, total_device_us):
            return {
                "mode": "target",
                "execution": "compiled",
                "prompt_tokens": 28672,
                "attention_us": 800.0,
                "kv_gather_us": 200.0,
                "attention_plus_gather_us": 1000.0,
                "total_device_us": total_device_us,
                "weight_quant_us": 3000.0,
                "weight_quant_count": 100,
                "manifest": {
                    "target_model": "target",
                    "draft_model": None,
                    "mode": "target",
                    "execution": "compiled",
                    "tensor_parallel_size": 1,
                    "num_speculative_tokens": 0,
                    "prompt_tokens": 28672,
                    "profile_prompt_sha256": "prompt",
                    "w4a16_linear_impl": "reference",
                    "vllm_ascend_enable_nz": 1,
                    "gemma4_prefill_attention_impl": attention_impl,
                    "engine": {"max_num_batched_tokens": 8192},
                    "sampling": {"max_tokens": 1},
                    "profiler": {
                        "active_iterations": 5,
                        "torch_profiler_dir": profile_dir,
                    },
                },
            }

        try:
            comparison = module.build_ab_comparison(
                {
                    "reports": [
                        report(
                            "reference",
                            "/profiles/reference",
                            5000.0,
                        )
                    ]
                },
                {
                    "reports": [
                        report(
                            "windowed",
                            "/profiles/windowed",
                            4800.0,
                        )
                    ]
                },
            )
        except ValueError as error:
            self.fail(str(error))

        self.assertEqual(len(comparison["cases"]), 1)


class Gemma4PrefillAttentionHarnessContractTests(unittest.TestCase):
    def test_ab_shell_interleaves_modes_and_preserves_container(self):
        self.assertTrue(AB_SHELL.is_file())
        source = AB_SHELL.read_text(encoding="utf-8")
        for token in (
            "set +e",
            "set -o pipefail",
            'ROUND_1=("reference" "windowed")',
            'ROUND_2=("windowed" "reference")',
            'ROUND_3=("reference" "windowed")',
            "VLLM_ASCEND_W4A16_LINEAR_IMPL=reference",
            "VLLM_ASCEND_GEMMA4_PREFILL_ATTENTION_IMPL",
            "run_gemma4_dense_target_prefill_profile.sh",
            "run_gemma4_dense_target_prefill_ttft.sh",
            "summarize_gemma4_prefill_attention.py",
            "2>&1 | tee",
            "PIPESTATUS[0]",
            "statistics.median",
            "exit 0",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_npu_test_has_configurable_models_and_oracle_cases(self):
        self.assertTrue(NPU_TEST.is_file())
        source = NPU_TEST.read_text(encoding="utf-8")
        for token in (
            "VLLM_TEST_GEMMA4_TARGET_MODEL",
            "VLLM_TEST_GEMMA4_DRAFT_MODEL",
            "VLLM_TEST_GEMMA4_TP_SIZE",
            "VLLM_TEST_GEMMA4_NUM_SPEC_TOKENS",
            'VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"',
            "AutoConfig.from_pretrained",
            "npu_fusion_attention",
            "test_synthetic_windowed_attention_oracle",
            "test_target_model_windowed_attention_smoke",
            "test_mtp_windowed_attention_smoke",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_npu_test_has_explicit_reference_rollback_smoke(self):
        self.assertTrue(NPU_TEST.is_file())
        source = NPU_TEST.read_text(encoding="utf-8")
        for token in (
            "test_target_model_reference_attention_smoke",
            'attention_impl="reference"',
            '"prefill_attention_impl": attention_impl',
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
