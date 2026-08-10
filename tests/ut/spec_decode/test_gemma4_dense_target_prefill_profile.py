import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = (
    Path(__file__).parents[3]
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
)
PROFILE_RUNNER = SCRIPT_DIR / "profile_gemma4_dense_target_prefill.py"
SUMMARIZER = SCRIPT_DIR / "summarize_gemma4_dense_target_prefill.py"
PROFILE_SHELL = SCRIPT_DIR / "run_gemma4_dense_target_prefill_profile.sh"
TTFT_SHELL = SCRIPT_DIR / "run_gemma4_dense_target_prefill_ttft.sh"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    bos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        return [11, 12, 13] if "warmup" in text else [21, 22]


def _args(**overrides) -> argparse.Namespace:
    values = {
        "mode": "target",
        "execution": "compiled",
        "target_model": "target",
        "draft_model": None,
        "tp": 1,
        "k": 0,
        "prompt_tokens": 8192,
        "output_tokens": 1,
        "max_model_len": 32768,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.92,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _parse_cli(module, *extra_args: str) -> argparse.Namespace:
    argv = [
        str(PROFILE_RUNNER),
        "--target-model",
        "target",
        "--mode",
        "target",
        "--execution",
        "eager",
        "--tp",
        "1",
        "--k",
        "0",
        "--prompt-tokens",
        "8192",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "8192",
        "--profile-dir",
        "profile",
        "--manifest-out",
        "manifest.json",
        "--output-token-ids-out",
        "output.json",
        *extra_args,
    ]
    with patch.object(sys, "argv", argv):
        return module._parse_args()


class ProfileRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module(PROFILE_RUNNER, "gemma4_prefill_runner")

    def test_exact_prompts_have_requested_shape_and_different_content(self):
        tokenizer = FakeTokenizer()
        warmup = self.module.build_exact_prompt_ids(
            tokenizer,
            prompt_tokens=17,
            variant="warmup",
        )
        profile = self.module.build_exact_prompt_ids(
            tokenizer,
            prompt_tokens=17,
            variant="profile",
        )

        self.assertEqual(len(warmup), 17)
        self.assertEqual(len(profile), 17)
        self.assertEqual(warmup[0], tokenizer.bos_token_id)
        self.assertEqual(profile[0], tokenizer.bos_token_id)
        self.assertNotEqual(warmup, profile)

    def test_exact_prompt_rejects_too_few_tokens(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            self.module.build_exact_prompt_ids(
                FakeTokenizer(),
                prompt_tokens=1,
                variant="profile",
            )

    def test_validate_args_rejects_prompt_plus_output_overflow(self):
        with self.assertRaisesRegex(ValueError, "max_model_len"):
            self.module.validate_args(
                _args(prompt_tokens=32768, max_model_len=32768)
            )

    def test_validate_args_rejects_prompt_plus_configured_output_overflow(self):
        with self.assertRaisesRegex(ValueError, "max_model_len"):
            self.module.validate_args(
                _args(
                    prompt_tokens=32760,
                    output_tokens=8,
                    max_model_len=32767,
                )
            )

    def test_validate_args_requires_draft_for_mtp(self):
        with self.assertRaisesRegex(ValueError, "draft_model"):
            self.module.validate_args(_args(mode="mtp", k=3))

    def test_validate_args_requires_zero_k_for_target(self):
        with self.assertRaisesRegex(ValueError, "k must be 0"):
            self.module.validate_args(_args(k=3))

    def test_validate_args_rejects_unknown_execution(self):
        with self.assertRaisesRegex(ValueError, "execution"):
            self.module.validate_args(_args(execution="graph"))

    def test_validate_args_rejects_invalid_gpu_memory_utilization(self):
        for value in (0, -0.1, 1.01):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "gpu_memory_utilization",
                ):
                    self.module.validate_args(
                        _args(gpu_memory_utilization=value)
                    )

    def test_validate_args_rejects_invalid_output_tokens(self):
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "output_tokens"):
                    self.module.validate_args(_args(output_tokens=value))

    def test_validate_args_accepts_target_and_mtp_contracts(self):
        self.module.validate_args(_args())
        self.module.validate_args(_args(gpu_memory_utilization=0.85))
        self.module.validate_args(_args(mode="mtp", draft_model="draft", k=3))

    def test_parse_args_gpu_memory_utilization_default(self):
        args = _parse_cli(self.module)

        self.assertEqual(args.gpu_memory_utilization, 0.92)

    def test_parse_args_gpu_memory_utilization_override(self):
        args = _parse_cli(
            self.module,
            "--gpu-memory-utilization",
            "0.85",
        )

        self.assertEqual(args.gpu_memory_utilization, 0.85)

    def test_parse_args_output_tokens_default(self):
        args = _parse_cli(self.module)

        self.assertEqual(args.output_tokens, 1)

    def test_parse_args_output_tokens_override(self):
        args = _parse_cli(self.module, "--output-tokens", "8")

        self.assertEqual(args.output_tokens, 8)

    def test_profiler_kwargs_capture_prefill_from_first_iteration(self):
        compiled = self.module.build_profiler_kwargs(
            execution="compiled",
            profile_dir=Path("profile"),
        )
        eager = self.module.build_profiler_kwargs(
            execution="eager",
            profile_dir=Path("profile"),
        )

        self.assertEqual(compiled["delay_iterations"], 0)
        self.assertEqual(compiled["max_iterations"], 0)
        self.assertTrue(compiled["torch_profiler_record_shapes"])
        self.assertFalse(compiled["torch_profiler_with_stack"])
        self.assertTrue(eager["torch_profiler_with_stack"])

    def test_target_engine_args_disable_prefix_cache_and_speculation(self):
        engine_args = self.module.build_engine_args(
            _args(),
            profiler_config="profiler",
        )

        self.assertFalse(engine_args["enable_prefix_caching"])
        self.assertTrue(engine_args["enable_chunked_prefill"])
        self.assertFalse(engine_args["async_scheduling"])
        self.assertFalse(engine_args["enforce_eager"])
        self.assertNotIn("speculative_config", engine_args)

    def test_engine_and_manifest_use_gpu_memory_utilization(self):
        args = _args(gpu_memory_utilization=0.85)
        engine_args = self.module.build_engine_args(
            args,
            profiler_config="profiler",
        )
        engine_manifest = self.module.build_engine_manifest(args)

        self.assertEqual(engine_args["gpu_memory_utilization"], 0.85)
        self.assertEqual(engine_manifest["gpu_memory_utilization"], 0.85)

    def test_mtp_engine_args_record_only_controlled_speculation_change(self):
        args = _args(mode="mtp", draft_model="draft", k=3)
        engine_args = self.module.build_engine_args(
            args,
            profiler_config="profiler",
        )

        self.assertEqual(
            engine_args["speculative_config"],
            {
                "method": "mtp",
                "model": "draft",
                "num_speculative_tokens": 3,
                "max_model_len": 32768,
            },
        )

    def test_manifest_distinguishes_offline_elapsed_from_ttft(self):
        args = _args()
        manifest = self.module.build_manifest(
            args=args,
            repo_root=Path("repo"),
            repo_sha="abc",
            imported_file=Path("repo/vllm_ascend/__init__.py"),
            profile_dir=Path("profile"),
            warmup_ids=[2, 11, 12],
            profile_ids=[2, 21, 22],
            output_token_ids=[7],
            offline_request_elapsed_ms=12.5,
            engine_args={"enable_prefix_caching": False},
            profiler_kwargs={"delay_iterations": 0},
            w4a16_linear_impl="reference",
            vllm_ascend_enable_nz=1,
            gemma4_prefill_attention_impl="windowed",
        )

        self.assertEqual(manifest["workload"], "gemma4_dense_target_prefill")
        self.assertEqual(manifest["generated_token_count"], 1)
        self.assertEqual(manifest["offline_request_elapsed_ms"], 12.5)
        self.assertEqual(manifest["w4a16_linear_impl"], "reference")
        self.assertEqual(manifest["vllm_ascend_enable_nz"], 1)
        self.assertEqual(
            manifest["gemma4_prefill_attention_impl"],
            "windowed",
        )
        self.assertNotIn("ttft", manifest)
        self.assertNotEqual(
            manifest["warmup_prompt_sha256"],
            manifest["profile_prompt_sha256"],
        )

    def test_manifest_records_configured_output_tokens(self):
        args = _args(output_tokens=8)
        output_token_ids = list(range(8))
        manifest = self.module.build_manifest(
            args=args,
            repo_root=Path("repo"),
            repo_sha="abc",
            imported_file=Path("repo/vllm_ascend/__init__.py"),
            profile_dir=Path("profile"),
            warmup_ids=[2, 11, 12],
            profile_ids=[2, 21, 22],
            output_token_ids=output_token_ids,
            offline_request_elapsed_ms=12.5,
            engine_args={"enable_prefix_caching": False},
            profiler_kwargs={"delay_iterations": 0},
            w4a16_linear_impl="reference",
            vllm_ascend_enable_nz=1,
            gemma4_prefill_attention_impl="oracle",
        )

        self.assertEqual(manifest["generated_token_count"], 8)
        self.assertEqual(manifest["sampling"]["max_tokens"], 8)
        self.assertEqual(manifest["sampling"]["min_tokens"], 8)

    def test_validate_generated_token_count_uses_configured_value(self):
        self.module.validate_generated_token_count(list(range(8)), 8)
        with self.assertRaisesRegex(
            RuntimeError,
            "expected=8 actual=7",
        ):
            self.module.validate_generated_token_count(list(range(7)), 8)


def _write_synthetic_profile(report_dir: Path) -> None:
    report_dir.mkdir(parents=True)
    with (report_dir / "op_statistic.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "Device_id",
            "OP Type",
            "Core Type",
            "Count",
            "Total Time(us)",
            "Min Time(us)",
            "Avg Time(us)",
            "Max Time(us)",
            "Ratio(%)",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for op_type, count, total_us in (
            (
                "WeightQuantBatchMatmulV2_bf16_int4_bf16_"
                "high_performance_1",
                2,
                100,
            ),
            ("MatMulV2", 1, 50),
            ("FlashAttentionScore", 1, 25),
            ("RmsNorm", 1, 10),
            ("Add", 1, 5),
            ("Scatter", 1, 4),
            ("aclnnGatherV3", 1, 3),
            ("MysteryOp", 1, 3),
        ):
            writer.writerow(
                {
                    "Device_id": "0",
                    "OP Type": op_type,
                    "Core Type": "AI_CORE",
                    "Count": str(count),
                    "Total Time(us)": str(total_us),
                    "Min Time(us)": "1",
                    "Avg Time(us)": str(total_us / count),
                    "Max Time(us)": str(total_us),
                    "Ratio(%)": "0",
                }
            )

    with (report_dir / "operator_details.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "Name",
            "Input Shapes",
            "Call Stack",
            "Host Self Duration(us)",
            "Host Total Duration(us)",
            "Device Self Duration(us)",
            "Device Total Duration(us)",
            "Device Self Duration With AICore(us)",
            "Device Total Duration With AICore(us)",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name, host_us in (
            ("prepare input", 10),
            ("forward", 80),
            ("post process", 5),
            ("sample_token", 5),
        ):
            row = dict.fromkeys(fields, "0")
            row["Name"] = name
            row["Host Total Duration(us)"] = str(host_us)
            writer.writerow(row)

    with (report_dir / "kernel_details.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = ["Name", "Type", "Duration(us)", "Wait Time(us)"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "Name": "weight_quant_kernel",
                "Type": "WeightQuantBatchMatmulV2",
                "Duration(us)": "80",
                "Wait Time(us)": "7",
            }
        )
        writer.writerow(
            {
                "Name": "flash_attention_kernel",
                "Type": "FlashAttentionScore",
                "Duration(us)": "20",
                "Wait Time(us)": "3",
            }
        )


def _write_profile_case(
    profile_root: Path,
    *,
    mode: str,
    execution: str,
    prompt_tokens: int,
) -> None:
    case_dir = profile_root / f"{mode}_{execution}_{prompt_tokens}"
    report_dir = case_dir / "profile" / "trace" / "ASCEND_PROFILER_OUTPUT"
    _write_synthetic_profile(report_dir)
    manifest = {
        "workload": "gemma4_dense_target_prefill",
        "mode": mode,
        "execution": execution,
        "target_model": "target",
        "draft_model": "draft" if mode == "mtp" else None,
        "tensor_parallel_size": 1,
        "num_speculative_tokens": 3 if mode == "mtp" else 0,
        "prompt_tokens": prompt_tokens,
        "profile_prompt_sha256": f"{mode}-{execution}-{prompt_tokens}",
        "offline_request_elapsed_ms": 100.0,
        "engine": {"enable_prefix_caching": False},
        "sampling": {"max_tokens": 1},
        "profiler": {"delay_iterations": 0},
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _write_ttft_case(
    ttft_root: Path,
    *,
    mode: str,
    prompt_tokens: int,
) -> None:
    case_dir = ttft_root / mode / str(prompt_tokens)
    case_dir.mkdir(parents=True)
    result = {
        "mode": mode,
        "prompt_tokens": str(prompt_tokens),
        "num_prompts": "10",
        "completed": 10,
        "failed": 0,
        "request_throughput": 0.5,
        "mean_ttft_ms": prompt_tokens / 10,
        "median_ttft_ms": prompt_tokens / 11,
        "p90_ttft_ms": prompt_tokens / 9,
        "p99_ttft_ms": prompt_tokens / 8,
    }
    (case_dir / f"{mode}_{prompt_tokens}.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )


class ProfileSummaryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module(SUMMARIZER, "gemma4_prefill_summary")

    def test_classifies_suffixed_and_common_prefill_ops(self):
        cases = {
            "WeightQuantBatchMatmulV2_bf16_int4_bf16_1": (
                "weight_quant_gemm"
            ),
            "MatMulV2": "dense_matmul",
            "FlashAttentionScore": "attention",
            "RmsNorm": "norm",
            "Gelu": "activation_elementwise",
            "Scatter": "kv_cache_update",
            "aclnnGatherV3": "data_movement",
            "MysteryOp": "other",
        }
        for op_type, expected in cases.items():
            with self.subTest(op_type=op_type):
                self.assertEqual(
                    self.module.classify_op_type(op_type),
                    expected,
                )

    def test_analyzes_synthetic_profile_and_host_phases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "ASCEND_PROFILER_OUTPUT"
            _write_synthetic_profile(report_dir)
            result = self.module.analyze_profile(
                report_dir,
                {"prompt_tokens": 100, "mode": "target", "execution": "compiled"},
            )

        self.assertEqual(result["total_device_us"], 200)
        self.assertEqual(
            result["categories"]["weight_quant_gemm"]["total_device_us"],
            100,
        )
        self.assertAlmostEqual(
            sum(
                category["ratio_of_profile_device_time"]
                for category in result["categories"].values()
            ),
            1.0,
        )
        self.assertTrue(
            result["top_raw_ops"][0]["op_type"].startswith(
                "WeightQuantBatchMatmulV2_"
            )
        )
        self.assertEqual(
            result["top_raw_ops"][0]["normalized_op_type"],
            "WeightQuantBatchMatmulV2",
        )
        self.assertEqual(result["kernel_wait_us"], 10)
        self.assertEqual(result["host_phases"]["prepare input"], 10)
        self.assertAlmostEqual(result["host_prepare_ratio"], 0.1)

    def test_rejects_incomparable_manifests(self):
        base = {
            "target_model": "target",
            "draft_model": None,
            "mode": "target",
            "execution": "compiled",
            "tensor_parallel_size": 1,
            "num_speculative_tokens": 0,
            "prompt_tokens": 8192,
            "profile_prompt_sha256": "abc",
            "engine": {"enable_prefix_caching": False},
            "sampling": {"max_tokens": 1},
            "profiler": {"delay_iterations": 0},
        }
        changed = {**base, "prompt_tokens": 16384}
        with self.assertRaisesRegex(ValueError, "prompt_tokens"):
            self.module.validate_comparable_manifests([base, changed])

    def test_ab_manifests_allow_impl_change_but_require_same_nz_mode(self):
        base = {
            "target_model": "target",
            "mode": "target",
            "execution": "compiled",
            "w4a16_linear_impl": "reference",
            "vllm_ascend_enable_nz": 1,
        }
        candidate = {
            **base,
            "w4a16_linear_impl": "candidate",
        }
        self.module.validate_comparable_manifests([base, candidate])

        with self.assertRaisesRegex(
            ValueError,
            "vllm_ascend_enable_nz",
        ):
            self.module.validate_comparable_manifests(
                [base, {**candidate, "vllm_ascend_enable_nz": 2}]
            )

    def test_reads_successful_ttft_result_and_rejects_failures(self):
        result = {
            "mode": "target",
            "prompt_tokens": "8192",
            "completed": 10,
            "failed": 0,
            "request_throughput": 0.5,
            "mean_ttft_ms": 100.0,
            "median_ttft_ms": 90.0,
            "p90_ttft_ms": 120.0,
            "p99_ttft_ms": 140.0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            parsed = self.module.read_ttft_result(path)
            self.assertEqual(parsed["prompt_tokens"], 8192)

            path.write_text(
                json.dumps({**result, "completed": 9, "failed": 1}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "failed"):
                self.module.read_ttft_result(path)

    def test_decision_prefers_host_then_largest_device_category(self):
        categories = {
            "weight_quant_gemm": {"ratio_of_profile_device_time": 0.5},
            "attention": {"ratio_of_profile_device_time": 0.2},
        }
        host = self.module.select_decision(
            {"host_prepare_ratio": 0.16, "categories": categories}
        )
        device = self.module.select_decision(
            {"host_prepare_ratio": 0.1, "categories": categories}
        )

        self.assertEqual(host["primary_hotspot"], "host_prepare")
        self.assertEqual(device["primary_hotspot"], "weight_quant_gemm")
        self.assertEqual(
            device["next_plan_filename"],
            "gemma4-ascend-w4a16-prefill-gemm-optimization-development-plan.md",
        )

    def test_builds_complete_summary_and_markdown_from_fixed_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profile_root = root / "profiles"
            ttft_root = root / "ttft"
            for mode, execution, prompt_tokens in (
                ("target", "compiled", 8192),
                ("target", "compiled", 16384),
                ("target", "compiled", 28672),
                ("target", "eager", 4096),
                ("mtp", "compiled", 28672),
            ):
                _write_profile_case(
                    profile_root,
                    mode=mode,
                    execution=execution,
                    prompt_tokens=prompt_tokens,
                )
            for mode in ("target", "mtp"):
                for prompt_tokens in (8192, 16384, 28672):
                    _write_ttft_case(
                        ttft_root,
                        mode=mode,
                        prompt_tokens=prompt_tokens,
                    )

            summary = self.module.build_summary(profile_root, ttft_root)
            markdown = self.module.render_markdown(summary)

        self.assertEqual(len(summary["profiles"]), 5)
        self.assertEqual(len(summary["ttft_results"]), 6)
        self.assertEqual(
            summary["decision"]["primary_hotspot"],
            "weight_quant_gemm",
        )
        self.assertIn("8192_to_28672_total_device_ratio", summary["scaling"])
        for heading in (
            "## Workload contract",
            "## TTFT baseline",
            "## Compiled profile",
            "## Eager attribution",
            "## Target versus MTP",
            "## Scaling",
            "## Decision",
            "## Evidence limitations",
        ):
            self.assertIn(heading, markdown)


class ShellContractTests(unittest.TestCase):
    def test_profile_shell_preserves_container_and_records_real_status(self):
        source = PROFILE_SHELL.read_text(encoding="utf-8")
        self.assertIn("export ASCEND_RT_VISIBLE_DEVICES=0", source)
        for token in (
            "set +e",
            "set -o pipefail",
            "2>&1 | tee",
            "PIPESTATUS[0]",
            "profile run exit code:",
            "profile analysis exit code:",
            "trace_count=",
            "exit 0",
        ):
            self.assertIn(token, source)
        for case in (
            "target compiled 8192",
            "target compiled 16384",
            "target compiled 28672",
            "target eager 4096",
            "mtp compiled 28672",
        ):
            self.assertIn(case, source)
        for variable in (
            "ASCEND_LAUNCH_BLOCKING",
            "VLLM_ASCEND_GEMMA4_MTP_DEBUG",
            "VLLM_ASCEND_GEMMA4_MTP_ORACLE",
            "VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE",
            "VLLM_ASCEND_GEMMA4_MTP_ASYNC_UNIPROC_SUBMIT",
        ):
            self.assertIn(variable, source)

    def test_ttft_shell_fixes_long_prompt_matrix_and_cleans_server(self):
        source = TTFT_SHELL.read_text(encoding="utf-8")
        self.assertIn("export ASCEND_RT_VISIBLE_DEVICES=0", source)
        for token in (
            "set +e",
            "set -o pipefail",
            "trap stop_server EXIT",
            'MODES=("target" "mtp")',
            "PROMPT_LENGTHS=(8192 16384 28672)",
            "--random-output-len 1",
            "--max-concurrency 1",
            "--random-range-ratio",
            "{\"input\":0.0,\"output\":0.0}",
            "--no-async-scheduling",
            "--no-enable-prefix-caching",
            "benchmark exit code:",
            "PIPESTATUS[0]",
            "exit 0",
        ):
            self.assertIn(token, source)
        self.assertNotIn("--enable-prefix-caching", source)
        self.assertIn("json.dumps", source)
        self.assertIn("kill -INT", source)
        self.assertIn("kill -TERM", source)

    def test_ttft_manifest_records_reproducibility_contract(self):
        source = TTFT_SHELL.read_text(encoding="utf-8")
        for token in (
            '"repo_sha"',
            '"server_commands"',
            '"benchmark_commands"',
            '"case_results"',
            '"w4a16_linear_impl"',
            '"vllm_ascend_enable_nz"',
            '"gemma4_prefill_attention_impl"',
            "W4A16_LINEAR_IMPL",
            "VLLM_ASCEND_ENABLE_NZ",
            "PREFILL_ATTENTION_IMPL",
            "VLLM_ASCEND_GEMMA4_PREFILL_ATTENTION_IMPL",
            "server-command.txt",
            "client-command.txt",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
