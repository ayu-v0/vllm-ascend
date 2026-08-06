import ast
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
ATTENTION_SOURCE = (
    REPO_ROOT / "vllm_ascend" / "attention" / "attention_v1.py"
)
ENVS_SOURCE = REPO_ROOT / "vllm_ascend" / "envs.py"


def _module_function_source(function_name: str) -> str:
    text = ATTENTION_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def _class_method_source(class_name: str, method_name: str) -> str:
    text = ATTENTION_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return "\n".join(lines[method_node.lineno - 1 : method_node.end_lineno])


class Gemma4PrefillAttentionSourceContractTests(unittest.TestCase):
    def test_environment_defaults_to_reference(self):
        source = ENVS_SOURCE.read_text(encoding="utf-8")

        self.assertIn(
            '"VLLM_ASCEND_GEMMA4_PREFILL_ATTENTION_IMPL"',
            source,
        )
        self.assertRegex(
            source,
            re.compile(
                r'os\.getenv\(\s*'
                r'"VLLM_ASCEND_GEMMA4_PREFILL_ATTENTION_IMPL",\s*'
                r'"reference"\s*\)',
                re.MULTILINE,
            ),
        )

    def test_mode_normalizer_is_fail_closed(self):
        source = _module_function_source(
            "_normalize_gemma4_prefill_attention_impl"
        )

        for value in ("oracle", "reference", "windowed"):
            self.assertIn(f'"{value}"', ATTENTION_SOURCE.read_text(
                encoding="utf-8"
            ))
        self.assertIn("raise ValueError", source)
        self.assertIn("Unsupported Gemma4 prefill attention impl", source)

    def test_impl_reads_mode_once_during_construction(self):
        source = _class_method_source(
            "AscendAttentionBackendImpl",
            "__init__",
        )

        self.assertIn(
            "envs_ascend.VLLM_ASCEND_GEMMA4_PREFILL_ATTENTION_IMPL",
            source,
        )
        self.assertIn(
            "self.gemma4_prefill_attention_impl",
            source,
        )

    def test_window_selection_is_a_zero_copy_tail_view(self):
        source = _module_function_source(
            "_select_single_request_sliding_kv_view"
        )

        self.assertIn("physical_slots[window_start:seq_len]", source)
        for forbidden in (
            ".clone(",
            "torch.cat",
            "torch.stack",
            "repeat_interleave",
            "torch.arange",
            ".index_select(",
            ".cpu(",
            ".to(",
        ):
            self.assertNotIn(forbidden, source)

    def test_windowed_guard_preserves_unsupported_paths(self):
        source = _class_method_source(
            "AscendAttentionBackendImpl",
            "_can_use_single_request_windowed_prefill",
        )

        for token in (
            '!= "reference"',
            "not _EXTRA_CTX.is_draft_model",
            'getattr(self.vllm_config, "model_config", None)',
            "_can_use_singlecard_compact_paged_kv",
            "not self.enable_c8_quant",
            "self.sliding_window is not None",
            "AscendAttentionState.ChunkedPrefill",
            "attn_metadata.num_decodes == 0",
            "attn_metadata.num_prefills == 1",
            "len(attn_metadata.seq_lens_list) == 1",
            "len(attn_metadata.actual_seq_lengths_q) == 1",
            "attn_metadata.causal",
            'attn_metadata.model_runner_type == "generate"',
            'in {"gemma4", "gemma4_text"}',
            "_GEMMA4_SPLITFUSE_CAUSAL_MASK_SHAPE",
            'getattr(attn_metadata, "mm_prefix_range", None) is None',
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_reference_windowed_and_oracle_are_explicit(self):
        source = _class_method_source(
            "AscendAttentionBackendImpl",
            "_forward_large_head_prefill_attention",
        )

        self.assertIn(
            "_can_use_single_request_windowed_prefill",
            source,
        )
        self.assertIn(
            'if self.gemma4_prefill_attention_impl == "reference"',
            source,
        )
        self.assertIn(
            "_get_single_request_windowed_prefill_kv",
            source,
        )
        self.assertIn(
            'self.gemma4_prefill_attention_impl == "oracle"',
            source,
        )
        self.assertIn("_assert_windowed_prefill_oracle_equal", source)


if __name__ == "__main__":
    unittest.main()
