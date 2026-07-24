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


def test_ascend_gemma4_uses_sparse_top_tokens_without_cuda_graphs():
    source = _method_source("_greedy_sample")

    assert "masked_embedding" in source
    assert "self.model.get_top_tokens(hidden_states)" in source
    assert "super()._greedy_sample(hidden_states)" in source


def test_ascend_gemma4_syncs_kv_sharing_target_into_backend_impl():
    source = _method_source("_setup_gemma4_kv_sharing")

    assert "super()._setup_gemma4_kv_sharing(target_attn_layer_names)" in source
    assert 'impl = getattr(attn, "impl", None)' in source
    assert "impl.kv_sharing_target_layer_name = target_layer_name" in source


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
        "Gemma4 MTP debug: async output construct done",
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
    assert "draft_hidden_finite=%s" in base_source
    assert "draft_sparse_top_ids=%s" in base_source
    assert "draft_sparse_top_values=%s" in base_source


def test_gemma4_mtp_uses_specialized_greedy_sampling_for_draft_tokens():
    source = _method_source_from(BASE_SOURCE, "_run_merged_draft")

    assert "if use_gemma4_mtp:" in source
    assert "draft_token_ids = self._greedy_sample(sample_hidden_states)" in source
    gemma4_branch = source.split("if use_gemma4_mtp:", 1)[1].split("else:", 1)[0]
    assert "self.model.compute_logits(sample_hidden_states)" not in gemma4_branch
    assert "logits.argmax(dim=-1)" not in gemma4_branch


def test_run_merged_draft_initializes_gemma4_mtp_flag_before_use():
    source = _method_source_from(BASE_SOURCE, "_run_merged_draft")

    init = "use_gemma4_mtp = _use_gemma4_mtp(self.speculative_config)"
    use = "if use_gemma4_mtp:"
    assert init in source
    assert source.index(init) < source.index(use)


def test_gemma4_mtp_uses_constant_position_multistep_metadata_helper():
    base_propose = _method_source_from(BASE_SOURCE, "_propose")
    gemma_helper = _method_source("build_constant_position_multi_step_metadata")

    assert "if use_gemma4_mtp:" in base_propose
    assert "build_constant_position_multi_step_metadata" in base_propose
    assert "for layer_name in attn_group.layer_names:" in gemma_helper
    assert "keep_positions_and_seq_lens=True" in gemma_helper


def test_gemma4_mtp_constant_metadata_does_not_mutate_position_input():
    source = _method_source_from(BASE_SOURCE, "attn_update_stack_num_spec_norm")

    assert "keep_positions_and_seq_lens: bool = False" in source
    assert "next_positions = used_update_positions.clone()" in source
    assert "used_update_positions += 1" not in source


def test_gemma4_mtp_reinitializes_metadata_shape_for_every_draft_step():
    source = _method_source_from(BASE_SOURCE, "attn_update_stack_num_spec_norm")

    assert "if draft_step == 1 or keep_positions_and_seq_lens:" in source


def test_gemma4_mtp_merged_draft_keeps_model_positions_constant():
    source = _method_source_from(BASE_SOURCE, "_run_merged_draft")

    assert "if not self.constant_draft_positions:" in source
    constant_position_guard = source.index("if not self.constant_draft_positions:")
    position_increment = source.index("positions += 1")
    assert constant_position_guard < position_increment


if __name__ == "__main__":
    test_gemma4_mtp_disables_drafter_full_aclgraph()
    test_ascend_gemma4_uses_sparse_top_tokens_without_cuda_graphs()
    test_ascend_gemma4_syncs_kv_sharing_target_into_backend_impl()
    test_ascend_base_uses_per_group_metadata_builder_for_gemma4_mtp()
    test_gemma4_mtp_diagnostic_stage_logs_are_present()
    test_gemma4_mtp_draft_logits_logs_token_preview_and_topk()
    test_gemma4_mtp_uses_specialized_greedy_sampling_for_draft_tokens()
    test_run_merged_draft_initializes_gemma4_mtp_flag_before_use()
    test_gemma4_mtp_uses_constant_position_multistep_metadata_helper()
    test_gemma4_mtp_constant_metadata_does_not_mutate_position_input()
    test_gemma4_mtp_reinitializes_metadata_shape_for_every_draft_step()
    test_gemma4_mtp_merged_draft_keeps_model_positions_constant()
