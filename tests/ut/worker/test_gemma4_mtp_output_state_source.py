import ast
from pathlib import Path


SOURCE = Path(__file__).parents[3] / "vllm_ascend" / "worker" / "model_runner_v1.py"
REJECTION_SOURCE = (
    Path(__file__).parents[3]
    / "vllm_ascend"
    / "sample"
    / "rejection_sampler.py"
)


def _method_source(method_name: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{method_name} was not found")


def _class_method_source(class_name: str, method_name: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
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
    snapshot_start = prepare_source.index("trace_async_state =")
    snapshot_source = prepare_source[snapshot_start:]
    assert ".tolist()" not in snapshot_source
    assert ".cpu()" not in snapshot_source


def test_gemma4_oracle_snapshot_is_instance_owned_and_has_one_shot_handoff():
    source = REJECTION_SOURCE.read_text(encoding="utf-8")

    assert "_GEMMA4_MTP_ORACLE" in source
    assert "self._gemma4_oracle_tensors" in source
    assert "self._gemma4_oracle_context" in source
    assert "def take_gemma4_oracle_snapshot(" in source
    assert "self._gemma4_oracle_tensors = {}" in source
    assert "self._gemma4_oracle_context = None" in source


def test_gemma4_oracle_snapshot_has_no_early_host_reads():
    source = REJECTION_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    rejection_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendRejectionSampler"
    )
    forward = next(
        node
        for node in rejection_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    body = "\n".join(lines[forward.lineno - 1 : forward.end_lineno])
    snapshot = body[body.index("capture_gemma4_oracle =") :]

    assert '"oracle_target_argmax"' in snapshot
    assert '"oracle_target_top2_ids"' in snapshot
    assert '"oracle_sampled_token_ids"' in snapshot
    assert ".cpu()" not in snapshot
    assert ".tolist()" not in snapshot
    assert ".item()" not in snapshot
    assert "synchronize()" not in snapshot


def test_gemma4_async_oracle_runs_after_base_copy_and_parse():
    source = _class_method_source(
        "_Gemma4OracleAsyncGPUModelRunnerOutput", "get_output"
    )

    assert "output = super().get_output()" in source
    assert "validate_greedy_oracle_snapshot(" in source
    assert "validate_async_state_snapshot(" in source
    assert source.index("output = super().get_output()") < source.index(
        "validate_greedy_oracle_snapshot("
    )


def test_gemma4_runner_transports_oracle_without_async_host_reads():
    source = _method_source("sample_tokens")

    assert "use_gemma4_mtp_oracle = _gemma4_mtp_oracle_enabled" in source
    assert "self.speculative_config.use_gemma4_mtp()" in source
    assert "Gemma4 oracle requires AscendGemma4Proposer" in source
    assert "take_gemma4_oracle_snapshot()" in source
    assert "oracle_debug_tensors" in source
    assert "oracle_debug_context" in source
    assert "_Gemma4OracleAsyncGPUModelRunnerOutput" in source
    async_start = source.index("debug_tensors = None")
    async_source = source[async_start:]
    assert ".cpu()" not in async_source


def test_gemma4_oracle_state_snapshot_owns_previous_correction_inputs():
    source = _method_source("_prepare_inputs")

    assert "_gemma4_mtp_oracle_enabled(self.drafter)" in source
    assert "if state_correction_applied or trace_async_state:" in source
    assert "self.prev_positions.copy_to_gpu(num_reqs)" in source
    assert "self.prev_num_draft_tokens.copy_to_gpu()" in source
    assert '"prev_valid_sampled_token_count"' in source
    assert '"cpu_num_computed_tokens"' in source
    assert '"state_correction_applied"' in source
    assert '"oracle_kv_group_ids"' in source
    assert '"oracle_kv_group_shapes"' in source


def test_gemma4_oracle_collects_per_group_kv_state_for_async_validation():
    source = _method_source("_build_attention_metadata")

    assert "_gemma4_mtp_oracle_enabled(self.drafter)" in source
    assert "and self.use_async_spec_decode" in source
    assert "oracle_kv_group_ids" in source
    assert "oracle_kv_group_shapes" in source
    assert "duplicate Gemma4 oracle KV group" in source
    assert "block_table_expected_shape = (" in source
    assert "slot_mapping_expected_shape = (" in source
    assert "tuple(block_table_snapshot.shape)" in source
    assert "tuple(slot_mapping_snapshot.shape)" in source
    assert 'f"group_{kv_cache_gid}_block_table"' in source
    assert 'f"group_{kv_cache_gid}_slot_mapping"' in source


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
