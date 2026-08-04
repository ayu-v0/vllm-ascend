import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[3]
    / "vllm_ascend"
    / "attention"
    / "attention_v1.py"
)


def _module_function_source(function_name: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


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


def _class_source(class_name: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return "\n".join(lines[class_node.lineno - 1 : class_node.end_lineno])


def test_compact_builder_owns_one_group_snapshot_and_vectorizes_slots():
    source = _module_function_source("_build_compact_paged_kv_metadata")

    assert ".long().clone()" in source
    assert "repeat_interleave" in source
    assert "output_size=total_tokens" in source
    assert "flat_block_table_indices" in source
    assert ".index_select(" in source
    assert "physical_slots" in source
    assert ".max().item()" not in source


def test_ascend_metadata_owns_forward_scoped_compact_group_cache():
    source = _class_source("AscendMetadata")

    assert "compact_paged_kv: CompactPagedKVMetadata | None = None" in source


def test_compact_gather_reads_flat_cache_without_rectangular_mask():
    source = _class_method_source(
        "AscendAttentionBackendImpl",
        "_gather_paged_kv_to_dense_compact",
    )

    assert "compact.physical_slots" in source
    assert ".view(" in source
    assert ".index_select(0, compact.physical_slots)" in source
    for forbidden in (
        "dense_shape",
        "max_tokens_padded",
        "valid_mask",
        "[valid_mask]",
        ".max().item()",
    ):
        assert forbidden not in source


def test_singlecard_dispatch_has_fail_closed_legacy_boundary():
    source = _class_method_source(
        "AscendAttentionBackendImpl",
        "_gather_paged_kv_to_dense",
    )

    assert "_can_use_singlecard_compact_paged_kv" in source
    assert "_gather_paged_kv_to_dense_compact" in source
    assert "_gather_paged_kv_to_dense_legacy" in source
    assert "except" not in source

    guard = _class_method_source(
        "AscendAttentionBackendImpl",
        "_can_use_singlecard_compact_paged_kv",
    )
    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        assert f"{field} == 1" in guard


def test_same_forward_oracle_is_test_only_and_uses_reference_snapshot():
    source = _class_method_source(
        "AscendAttentionBackendImpl",
        "_gather_paged_kv_to_dense_compact",
    )

    assert "_GEMMA4_MTP_ORACLE" in source
    assert "_gather_paged_kv_snapshot_to_dense_reference" in source
    assert "compact.block_table_snapshot" in source


def test_large_head_prefill_reuses_cached_actual_sequence_lengths():
    source = _class_method_source(
        "AscendAttentionBackendImpl",
        "_get_large_head_prefill_kv",
    )

    assert "return self._gather_paged_kv_to_dense(" in source
    assert "torch.tensor(seq_lens" not in source
    assert ".cumsum(" not in source
