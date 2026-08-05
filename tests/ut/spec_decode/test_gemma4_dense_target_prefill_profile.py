import argparse
import importlib.util
import unittest
from pathlib import Path


SCRIPT_DIR = (
    Path(__file__).parents[3]
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
)
PROFILE_RUNNER = SCRIPT_DIR / "profile_gemma4_dense_target_prefill.py"


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
        "max_model_len": 32768,
        "max_num_batched_tokens": 8192,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


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

    def test_validate_args_requires_draft_for_mtp(self):
        with self.assertRaisesRegex(ValueError, "draft_model"):
            self.module.validate_args(_args(mode="mtp", k=3))

    def test_validate_args_requires_zero_k_for_target(self):
        with self.assertRaisesRegex(ValueError, "k must be 0"):
            self.module.validate_args(_args(k=3))

    def test_validate_args_rejects_unknown_execution(self):
        with self.assertRaisesRegex(ValueError, "execution"):
            self.module.validate_args(_args(execution="graph"))

    def test_validate_args_accepts_target_and_mtp_contracts(self):
        self.module.validate_args(_args())
        self.module.validate_args(_args(mode="mtp", draft_model="draft", k=3))

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
        )

        self.assertEqual(manifest["workload"], "gemma4_dense_target_prefill")
        self.assertEqual(manifest["generated_token_count"], 1)
        self.assertEqual(manifest["offline_request_elapsed_ms"], 12.5)
        self.assertNotIn("ttft", manifest)
        self.assertNotEqual(
            manifest["warmup_prompt_sha256"],
            manifest["profile_prompt_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
