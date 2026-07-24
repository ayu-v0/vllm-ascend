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


def test_gemma4_mtp_enables_accepted_token_state_updates():
    source = _method_source("initialize_kv_cache")

    assert "use_gemma4_mtp" in source
    assert "self.need_accepted_tokens = any(" in source
    assert "self.need_accepted_tokens = self.need_accepted_tokens or use_gemma4_mtp" in source


def test_gemma4_mtp_keeps_async_output_async_by_default():
    source = _method_source("sample_tokens")

    assert "use_gemma4_mtp_debug = _gemma4_mtp_debug_enabled(self.drafter)" in source
    assert "return async_output.get_output()" not in source


def test_gemma4_mtp_takes_draft_tokens_without_async_event_wait():
    source = _method_source("take_draft_token_ids")

    assert "AscendGemma4Proposer" in source
    assert "not self.use_async_scheduling" in source
    assert "draft_token_ids.detach().cpu().tolist()" in source
    assert "DraftTokenIds(req_ids, draft_token_ids_cpu)" in source


def test_gemma4_mtp_logs_target_verification_inputs_and_logits():
    text = SOURCE.read_text(encoding="utf-8")

    assert "Gemma4 MTP debug: target_verify_inputs" in text
    assert "input_ids_preview=%s" in text
    assert "positions_preview=%s" in text
    assert "metadata_draft_preview=%s" in text
    assert "attn_metadata=%s" in text
    assert "Gemma4 MTP debug: target_verify_logits" in text
    assert "target_hidden_finite=%s" in text
    assert "bonus_hidden_finite=%s" in text
    assert "target_logits_finite=%s" in text
    assert "bonus_logits_finite=%s" in text
    assert "target_top_ids=%s" in text
    assert "target_top_values=%s" in text


def test_gemma4_mtp_logs_async_state_handoff_boundaries():
    sample_source = _method_source("sample_tokens")
    count_copy_source = _method_source("_copy_valid_sampled_token_count")

    assert "Gemma4 MTP async trace: sampling path" in sample_source
    assert "use_padded_batch=%s" in sample_source
    assert "Gemma4 MTP async trace: valid-count copy" in count_copy_source
    assert "trace_id=%s" in count_copy_source
    assert "counts_cpu_id=%s" in count_copy_source
    assert "count_event_id=%s" in count_copy_source
    assert "output_count_cpu_id=%s" in sample_source


def test_gemma4_mtp_logs_paged_kv_gather_bounds():
    source = (SOURCE.parents[1] / "attention" / "attention_v1.py").read_text(encoding="utf-8")

    assert "Gemma4 MTP debug: paged_kv_gather" in source
    assert "key_cache_shape=%s" in source
    assert "value_cache_shape=%s" in source
    assert "block_table_shape=%s" in source
    assert "flat_block_id_min=%s" in source
    assert "flat_block_id_max=%s" in source
    assert "cache_block_capacity=%s" in source
    assert "paged KV block id is out of range" in source


def test_gemma4_mtp_gather_owns_block_id_tensor():
    source = (SOURCE.parents[1] / "attention" / "attention_v1.py").read_text(encoding="utf-8")

    assert "block_table = block_table[: len(seq_lens), :num_blocks].long().clone()" in source


def test_gemma4_mtp_passes_current_count_to_async_output_only():
    source = _method_source("sample_tokens")

    assert "valid_sampled_token_count = (" in source
    assert "valid_sampled_token_count=valid_sampled_token_count" in source
    assert "isinstance(self.drafter, AscendGemma4Proposer)" in source
    assert "self.valid_sampled_token_count_gpu" in source


def test_gemma4_mtp_collects_async_state_snapshots_without_early_host_reads():
    text = SOURCE.read_text(encoding="utf-8")

    assert "_gemma4_mtp_async_debug_tensors" in text
    assert '"prev_positions"' in text
    assert '"prev_num_draft_tokens"' in text
    assert '"num_computed_before"' in text
    assert '"num_computed_after"' in text
    assert '"num_accepted_tokens"' in text
    assert '"draft_token_ids"' in text
    assert 'f"group_{kv_cache_gid}_block_table"' in text
    assert 'f"group_{kv_cache_gid}_slot_mapping"' in text
    assert "debug_tensors=debug_tensors" in text
    assert "debug_context=debug_context" in text

    prepare_source = _method_source("_prepare_inputs")
    snapshot_start = prepare_source.index("debug_async_state =")
    snapshot_source = prepare_source[snapshot_start:]
    assert ".tolist()" not in snapshot_source
    assert ".cpu()" not in snapshot_source


if __name__ == "__main__":
    test_gemma4_mtp_enables_accepted_token_state_updates()
    test_gemma4_mtp_keeps_async_output_async_by_default()
    test_gemma4_mtp_takes_draft_tokens_without_async_event_wait()
    test_gemma4_mtp_logs_target_verification_inputs_and_logits()
    test_gemma4_mtp_logs_async_state_handoff_boundaries()
    test_gemma4_mtp_logs_paged_kv_gather_bounds()
    test_gemma4_mtp_gather_owns_block_id_tensor()
    test_gemma4_mtp_passes_current_count_to_async_output_only()
    test_gemma4_mtp_collects_async_state_snapshots_without_early_host_reads()
