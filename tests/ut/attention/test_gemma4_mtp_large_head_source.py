import ast
from pathlib import Path


SOURCE = Path(__file__).parents[3] / "vllm_ascend" / "attention" / "attention_v1.py"


def _method_source(method_name: str) -> str:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return ast.unparse(node)
    raise AssertionError(f"{method_name} was not found")


def test_spec_decoding_large_head_uses_paged_attention_fallback():
    source = _method_source("forward_impl")

    assert "AscendAttentionState.SpecDecoding" in source
    assert "self.sliding_window is None" in source
    assert "self._should_use_large_head_attention_fallback()" in source
    assert "self.forward_paged_attention(query, attn_metadata, output)" in source


def test_shared_kv_large_head_path_has_gemma4_diagnostics():
    source = SOURCE.read_text(encoding="utf-8")

    assert "VLLM_ASCEND_GEMMA4_MTP_DEBUG" in source
    assert "Gemma4 MTP debug: shared_kv_prefill enter" in source
    assert "Gemma4 MTP debug: shared_kv_prefill dense_kv" in source
    assert "Gemma4 MTP debug: shared_kv_prefill output" in source
    assert "shared_cache_available" in source


if __name__ == "__main__":
    test_spec_decoding_large_head_uses_paged_attention_fallback()
    test_shared_kv_large_head_path_has_gemma4_diagnostics()
