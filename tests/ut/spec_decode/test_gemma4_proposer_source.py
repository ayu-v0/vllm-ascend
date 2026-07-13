import ast
from pathlib import Path


SOURCE = Path(__file__).parents[3] / "vllm_ascend" / "spec_decode" / "gemma4_proposer.py"
BASE_SOURCE = Path(__file__).parents[3] / "vllm_ascend" / "spec_decode" / "llm_base_proposer.py"


def _method_source(method_name: str) -> str:
    return _method_source_from(SOURCE, method_name)


def _method_source_from(source: Path, method_name: str) -> str:
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{method_name} was not found")


def test_gemma4_mtp_disables_drafter_full_aclgraph():
    source = _method_source("__init__")

    assert "self.use_cuda_graph = False" in source
    assert "self._runnable = self._run_merged_draft" in source


def test_ascend_base_uses_per_group_metadata_builder_for_gemma4_mtp():
    source = _method_source_from(BASE_SOURCE, "_propose")

    assert "use_gemma4_mtp" in source
    assert "self.build_per_group_and_layer_attn_metadata(common_attn_metadata)" in source
    assert "per_layer_attn_metadata[self.attn_layer_names[0]]" in source


def test_gemma4_mtp_diagnostic_stage_logs_are_present():
    base_source = BASE_SOURCE.read_text(encoding="utf-8")
    runner_source = (
        Path(__file__).parents[3]
        / "vllm_ascend"
        / "worker"
        / "model_runner_v1.py"
    ).read_text(encoding="utf-8")

    for marker in [
        "Gemma4 MTP debug: draft_token stage",
        "Gemma4 MTP debug: propose_draft_token_ids enter",
        "Gemma4 MTP debug: _propose enter",
        "Gemma4 MTP debug: run_draft start",
        "Gemma4 MTP debug: _run_merged_draft model forward start",
        "Gemma4 MTP debug: _run_merged_draft logits start",
        "Gemma4 MTP debug: _copy_draft_token_ids_to_cpu",
        "Gemma4 MTP debug: bookkeeping_sync done",
        "Gemma4 MTP debug: parsed sampled tokens",
        "Gemma4 MTP debug: finalize_kv_connector start",
        "Gemma4 MTP debug: async output construct start",
        "Gemma4 MTP debug: async output return",
    ]:
        assert marker in base_source or marker in runner_source

    assert "_gemma4_mtp_debug_enabled" in base_source
    assert "_gemma4_mtp_debug_enabled" in runner_source
    assert "VLLM_ASCEND_GEMMA4_MTP_DEBUG" in (
        Path(__file__).parents[3]
        / "vllm_ascend"
        / "envs.py"
    ).read_text(encoding="utf-8")


def test_gemma4_mtp_draft_logits_logs_token_preview_and_topk():
    base_source = BASE_SOURCE.read_text(encoding="utf-8")

    assert "draft_token_ids_preview=%s" in base_source
    assert "draft_top_ids=%s" in base_source
    assert "draft_top_values=%s" in base_source


if __name__ == "__main__":
    test_gemma4_mtp_disables_drafter_full_aclgraph()
    test_ascend_base_uses_per_group_metadata_builder_for_gemma4_mtp()
    test_gemma4_mtp_diagnostic_stage_logs_are_present()
    test_gemma4_mtp_draft_logits_logs_token_preview_and_topk()
