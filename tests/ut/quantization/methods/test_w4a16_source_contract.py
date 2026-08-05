import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[4]
W4A16 = ROOT / "vllm_ascend" / "quantization" / "methods" / "w4a16.py"
ENVS = ROOT / "vllm_ascend" / "envs.py"
BENCHMARK = (
    ROOT
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
    / "benchmark_gemma4_w4a16_prefill_gemm.py"
)


class W4A16SourceContractTests(unittest.TestCase):
    def test_central_env_defaults_to_reference(self):
        source = ENVS.read_text(encoding="utf-8")
        self.assertIn('"VLLM_ASCEND_W4A16_LINEAR_IMPL"', source)
        self.assertIn(
            'os.getenv("VLLM_ASCEND_W4A16_LINEAR_IMPL", "reference")',
            source,
        )

    def test_dense_method_exposes_three_fixed_runtime_modes(self):
        source = W4A16.read_text(encoding="utf-8")
        tree = ast.parse(source)
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "AscendW4A16LinearMethod"
        )
        methods = {
            node.name
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(
            {"_apply_reference", "_apply_candidate", "_apply_oracle"}
            <= methods
        )
        self.assertIn("_VALID_W4A16_LINEAR_IMPLS", source)
        self.assertNotIn("os.getenv(", ast.get_source_segment(source, class_node))

    def test_reference_branch_cannot_call_nz_conversion(self):
        source = W4A16.read_text(encoding="utf-8")
        tree = ast.parse(source)
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "AscendW4A16LinearMethod"
        )
        process = next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "process_weights_after_loading"
        )
        reference_branches = [
            node
            for node in ast.walk(process)
            if isinstance(node, ast.If)
            and "reference" in (ast.get_source_segment(source, node.test) or "")
        ]
        self.assertTrue(reference_branches)
        reference_branch = reference_branches[0]
        branch_source = "\n".join(
            ast.get_source_segment(source, statement) or ""
            for statement in reference_branch.body
        )
        self.assertNotIn("maybe_trans_nz", branch_source)

    def test_oracle_failure_message_attributes_layer_shape_and_formats(self):
        source = W4A16.read_text(encoding="utf-8")
        for token in (
            "layer_prefix=",
            "input=",
            "weight=",
            "reference_format=",
            "candidate_format=",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_standalone_benchmark_does_not_require_initialized_ascend_config(self):
        w4a16_source = W4A16.read_text(encoding="utf-8")
        benchmark_source = BENCHMARK.read_text(encoding="utf-8")
        self.assertIn("def _prepare_w4a16_candidate_weight", w4a16_source)
        self.assertIn("nz_mode=", benchmark_source)
        self.assertNotIn("maybe_trans_nz(reference_weight", benchmark_source)

    def test_benchmark_matches_runtime_internal_format_and_releases_cases(self):
        source = BENCHMARK.read_text(encoding="utf-8")
        enable_internal_format = (
            "torch.npu.config.allow_internal_format = True"
        )
        self.assertIn(enable_internal_format, source)
        self.assertLess(
            source.index(enable_internal_format),
            source.index("_prepare_w4a16_candidate_weight("),
        )
        self.assertGreaterEqual(source.count("torch.npu.empty_cache()"), 2)


if __name__ == "__main__":
    unittest.main()
