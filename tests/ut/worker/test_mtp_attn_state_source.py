import ast
from pathlib import Path


SOURCE = Path(__file__).parents[3] / "vllm_ascend" / "worker" / "model_runner_v1.py"


def _method_source(method_name: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{method_name} was not found")


def test_mtp_runtime_keeps_spec_decoding_state_for_non_mla_models():
    source = _method_source("_build_attn_state")

    assert "`AscendAttentionState.SpecDecoding` is only designed for mla" not in source
    assert "self.speculative_config and self.speculative_config.method == \"mtp\"" in source
    assert "attn_state = AscendAttentionState.SpecDecoding" in source
    assert "and not self.vllm_config.model_config.use_mla" not in source


if __name__ == "__main__":
    test_mtp_runtime_keeps_spec_decoding_state_for_non_mla_models()
