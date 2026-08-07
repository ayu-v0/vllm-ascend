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
            "not _has_nonempty_mm_prefix_range(",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_multimodal_prefix_guard_is_empty_mapping_aware(self):
        source = ATTENTION_SOURCE.read_text(encoding="utf-8")
        if "def _has_nonempty_mm_prefix_range" not in source:
            self.fail("multimodal prefix helper is missing")
        helper_source = _module_function_source(
            "_has_nonempty_mm_prefix_range"
        )

        for token in (
            "mm_prefix_range is None",
            "isinstance(mm_prefix_range, dict)",
            "mm_prefix_range.values()",
            "return True",
            "return False",
        ):
            with self.subTest(token=token):
                self.assertIn(token, helper_source)

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

    def test_windowed_fallback_diagnostics_cover_guard_inputs(self):
        source = ATTENTION_SOURCE.read_text(encoding="utf-8")
        if "def _log_windowed_prefill_fallback_once" not in source:
            self.fail("windowed fallback diagnostic method is missing")
        forward_source = _class_method_source(
            "AscendAttentionBackendImpl",
            "_forward_large_head_prefill_attention",
        )
        diagnostic_source = _class_method_source(
            "AscendAttentionBackendImpl",
            "_log_windowed_prefill_fallback_once",
        )

        self.assertIn(
            "_log_windowed_prefill_fallback_once",
            forward_source,
        )
        for token in (
            "logger.info_once",
            "Gemma4 windowed prefill attention fallback",
            "attn_state",
            "num_decodes",
            "num_prefills",
            "model_type",
            "attn_mask_dtype",
            "attn_mask_shape",
            "key_cache_dtype",
            "value_cache_dtype",
        ):
            with self.subTest(token=token):
                self.assertIn(token, diagnostic_source)

    def test_windowed_routing_diagnostics_precede_branch_selection(self):
        source = ATTENTION_SOURCE.read_text(encoding="utf-8")
        if "def _log_windowed_prefill_routing_once" not in source:
            self.fail("windowed routing diagnostic method is missing")
        forward_source = _class_method_source(
            "AscendAttentionBackendImpl",
            "forward_impl",
        )
        diagnostic_source = _class_method_source(
            "AscendAttentionBackendImpl",
            "_log_windowed_prefill_routing_once",
        )

        self.assertIn(
            "_log_windowed_prefill_routing_once",
            forward_source,
        )
        self.assertLess(
            forward_source.index("_log_windowed_prefill_routing_once"),
            forward_source.index("if shared_kv_prefill:"),
        )
        for token in (
            "logger.warning_once",
            "windowed_guard",
            "head_size",
            "large_head_fallback",
            "shared_kv_prefill",
            "query_shape",
            "key_shape",
            "attn_state",
            "num_decodes",
            "num_prefills",
            "seq_lens_count",
            "actual_q_count",
            "causal",
            "model_runner_type",
            "capturing",
            "kv_sharing_target",
            "key_cache_available",
            "key_cache_dtype",
            "value_cache_dtype",
            "mm_prefix_range",
        ):
            with self.subTest(token=token):
                self.assertIn(token, diagnostic_source)

    def test_oracle_success_evidence_uses_warning_level(self):
        selection_source = _class_method_source(
            "AscendAttentionBackendImpl",
            "_get_single_request_windowed_prefill_kv",
        )
        oracle_source = _class_method_source(
            "AscendAttentionBackendImpl",
            "_assert_windowed_prefill_oracle_equal",
        )

        self.assertIn(
            'self.gemma4_prefill_attention_impl == "oracle"',
            selection_source,
        )
        self.assertIn("logger.warning", selection_source)
        self.assertIn(
            "Gemma4 windowed prefill attention enabled",
            selection_source,
        )
        self.assertIn("logger.warning", oracle_source)
        self.assertIn(
            "Gemma4 windowed prefill oracle passed",
            oracle_source,
        )


if __name__ == "__main__":
    unittest.main()
