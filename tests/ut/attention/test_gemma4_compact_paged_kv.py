from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.attention.backend import AttentionType

import vllm_ascend.attention.attention_v1 as attention_v1
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendMetadata,
    _build_compact_paged_kv_metadata,
)


def _make_config(**parallel_overrides):
    parallel = dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    parallel.update(parallel_overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        kv_transfer_config=None,
        quant_config=None,
    )


def _make_impl(
    monkeypatch,
    *,
    num_kv_heads=2,
    head_size=3,
    sliding_window=1024,
    **parallel_overrides,
):
    config = _make_config(**parallel_overrides)
    monkeypatch.setattr(
        attention_v1,
        "get_current_vllm_config",
        lambda: config,
    )
    impl = AscendAttentionBackendImpl(
        num_heads=num_kv_heads,
        head_size=head_size,
        scale=1.0,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="float16",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
    )
    impl._layer_name = "language_model.model.layers.0.self_attn.attn"
    return impl


def _make_direct_gather_case(monkeypatch):
    block_size = 2
    impl = _make_impl(monkeypatch)
    key_cache = torch.arange(
        4 * block_size * 2 * 3,
        dtype=torch.float32,
    ).to(torch.float16).reshape(4, block_size, 2, 3)
    value_cache = key_cache + 1000
    metadata = AscendMetadata(
        block_tables=torch.tensor(
            [[1, 3], [0, 2]],
            dtype=torch.int32,
        ),
        seq_lens_list=[3, 4],
    )
    return impl, key_cache, value_cache, metadata


def test_build_compact_slots_sequence_major_across_blocks():
    block_table = torch.tensor(
        [[3, 1, 99], [2, 0, 99]],
        dtype=torch.int32,
    )

    result = _build_compact_paged_kv_metadata(
        block_table,
        [3, 4],
        block_size=2,
        cache_block_capacity=4,
    )

    assert result.physical_slots.tolist() == [6, 7, 2, 4, 5, 0, 1]
    assert result.actual_seq_lengths_kv == [3, 7]
    assert result.block_table_snapshot.tolist() == [[3, 1], [2, 0]]
    assert result.block_table_snapshot.is_contiguous()
    assert result.block_table_snapshot.data_ptr() != block_table.data_ptr()


def test_build_compact_slots_does_not_select_padding_sentinel():
    block_table = torch.tensor(
        [[0, 99], [1, 2]],
        dtype=torch.int32,
    )

    result = _build_compact_paged_kv_metadata(
        block_table,
        [1, 3],
        block_size=2,
        cache_block_capacity=3,
    )

    assert result.physical_slots.tolist() == [0, 2, 3, 4]
    assert 99 * 2 not in result.physical_slots.tolist()


@pytest.mark.parametrize("seq_lens", [[], [0], [0, 0, 0]])
def test_build_compact_slots_empty(seq_lens):
    block_table = torch.empty(
        (len(seq_lens), 4),
        dtype=torch.int32,
    )

    result = _build_compact_paged_kv_metadata(
        block_table,
        seq_lens,
        block_size=128,
        cache_block_capacity=16,
    )

    assert result.physical_slots.dtype == torch.long
    assert result.physical_slots.numel() == 0
    assert result.block_table_snapshot.shape == (len(seq_lens), 0)


@pytest.mark.parametrize(
    ("block_table", "seq_lens", "error"),
    [
        (torch.zeros(1, 1, dtype=torch.float32), [1], TypeError),
        (torch.zeros(1, 1, dtype=torch.int32), [-1], ValueError),
        (torch.zeros(1, 1, dtype=torch.int32), [True], TypeError),
        (torch.zeros(1, dtype=torch.int32), [1], ValueError),
        (torch.zeros(0, 1, dtype=torch.int32), [1], ValueError),
        (torch.zeros(1, 0, dtype=torch.int32), [1], ValueError),
    ],
)
def test_build_compact_slots_rejects_invalid_metadata(
    block_table,
    seq_lens,
    error,
):
    with pytest.raises(error):
        _build_compact_paged_kv_metadata(
            block_table,
            seq_lens,
            block_size=2,
            cache_block_capacity=4,
        )


def test_compact_slots_reused_within_group_and_isolated_between_groups(
    monkeypatch,
):
    calls = 0
    real_builder = attention_v1._build_compact_paged_kv_metadata

    def counting_builder(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(
        attention_v1,
        "_build_compact_paged_kv_metadata",
        counting_builder,
    )
    sliding_metadata = AscendMetadata(
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_lens_list=[3],
    )
    full_metadata = AscendMetadata(
        block_tables=torch.tensor([[2, 3]], dtype=torch.int32),
        seq_lens_list=[3],
    )
    sliding_a = _make_impl(monkeypatch)
    sliding_b = _make_impl(monkeypatch)
    full = _make_impl(
        monkeypatch,
        num_kv_heads=1,
        head_size=6,
        sliding_window=None,
    )
    sliding_cache = torch.empty(4, 2, 2, 3, dtype=torch.float16)
    full_cache = torch.empty(4, 2, 1, 6, dtype=torch.float16)

    sliding_first = sliding_a._get_or_build_compact_paged_kv_metadata(
        sliding_metadata,
        sliding_cache,
    )
    sliding_second = sliding_b._get_or_build_compact_paged_kv_metadata(
        sliding_metadata,
        sliding_cache,
    )
    full_first = full._get_or_build_compact_paged_kv_metadata(
        full_metadata,
        full_cache,
    )

    assert calls == 2
    assert sliding_first is sliding_second
    assert (
        sliding_first.physical_slots.data_ptr()
        == sliding_second.physical_slots.data_ptr()
    )
    assert sliding_first is not full_first
    assert (
        sliding_first.physical_slots.data_ptr()
        != full_first.physical_slots.data_ptr()
    )


def test_compact_slots_reject_reuse_after_signature_change(monkeypatch):
    metadata = AscendMetadata(
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_lens_list=[3],
    )
    impl = _make_impl(monkeypatch)
    cache = torch.empty(4, 2, 2, 3, dtype=torch.float16)
    impl._get_or_build_compact_paged_kv_metadata(metadata, cache)
    metadata.seq_lens_list = [2]

    with pytest.raises(RuntimeError, match="compact paged KV metadata"):
        impl._get_or_build_compact_paged_kv_metadata(metadata, cache)


def test_compact_slots_do_not_cross_new_metadata_step(monkeypatch):
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    first_metadata = AscendMetadata(
        block_tables=block_table,
        seq_lens_list=[3],
    )
    next_metadata = AscendMetadata(
        block_tables=block_table,
        seq_lens_list=[3],
    )
    impl = _make_impl(monkeypatch)
    cache = torch.empty(4, 2, 2, 3, dtype=torch.float16)

    first = impl._get_or_build_compact_paged_kv_metadata(
        first_metadata,
        cache,
    )
    next_step = impl._get_or_build_compact_paged_kv_metadata(
        next_metadata,
        cache,
    )

    assert first is not next_step
    assert first.physical_slots.data_ptr() != next_step.physical_slots.data_ptr()


def test_compact_gather_reads_flat_cache_by_physical_slot(monkeypatch):
    impl, key_cache, value_cache, metadata = _make_direct_gather_case(
        monkeypatch
    )

    dense_key, dense_value, actual_seq_lengths = (
        impl._gather_paged_kv_to_dense_compact(
            key_cache,
            value_cache,
            metadata,
        )
    )

    expected_key = torch.cat(
        (key_cache[1], key_cache[3, :1], key_cache[0], key_cache[2]),
        dim=0,
    )
    assert torch.equal(dense_key, expected_key)
    assert torch.equal(dense_value, expected_key + 1000)
    assert actual_seq_lengths == [3, 7]


def test_compact_gather_rejects_non_float16_cache(monkeypatch):
    impl, key_cache, value_cache, metadata = _make_direct_gather_case(
        monkeypatch
    )

    with pytest.raises(TypeError, match="only BF16/FP16"):
        impl._gather_paged_kv_to_dense_compact(
            key_cache.float(),
            value_cache.float(),
            metadata,
        )


def test_compact_gather_rejects_noncontiguous_cache(monkeypatch):
    impl, key_cache, value_cache, metadata = _make_direct_gather_case(
        monkeypatch
    )
    key_cache = key_cache.transpose(0, 1)
    value_cache = value_cache.transpose(0, 1)

    with pytest.raises(ValueError, match="must be contiguous"):
        impl._gather_paged_kv_to_dense_compact(
            key_cache,
            value_cache,
            metadata,
        )


@pytest.mark.parametrize(
    "parallel_field",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_non_singlecard_dispatch_uses_legacy(monkeypatch, parallel_field):
    impl, key_cache, value_cache, metadata = _make_direct_gather_case(
        monkeypatch
    )
    setattr(impl.vllm_config.parallel_config, parallel_field, 2)
    compact = Mock(side_effect=AssertionError("compact must not run"))
    legacy = Mock(return_value=(key_cache[:1], value_cache[:1], [1]))
    monkeypatch.setattr(impl, "_gather_paged_kv_to_dense_compact", compact)
    monkeypatch.setattr(impl, "_gather_paged_kv_to_dense_legacy", legacy)

    impl._gather_paged_kv_to_dense(key_cache, value_cache, metadata)

    compact.assert_not_called()
    legacy.assert_called_once_with(key_cache, value_cache, metadata)


def test_singlecard_dispatch_uses_compact(monkeypatch):
    impl, key_cache, value_cache, metadata = _make_direct_gather_case(
        monkeypatch
    )
    compact = Mock(return_value=(key_cache[:1], value_cache[:1], [1]))
    legacy = Mock(side_effect=AssertionError("legacy must not run"))
    monkeypatch.setattr(impl, "_gather_paged_kv_to_dense_compact", compact)
    monkeypatch.setattr(impl, "_gather_paged_kv_to_dense_legacy", legacy)

    impl._gather_paged_kv_to_dense(key_cache, value_cache, metadata)

    compact.assert_called_once_with(key_cache, value_cache, metadata)
    legacy.assert_not_called()


def test_oracle_reference_not_called_when_disabled(monkeypatch):
    impl, key_cache, value_cache, metadata = _make_direct_gather_case(
        monkeypatch
    )
    monkeypatch.setattr(attention_v1, "_GEMMA4_MTP_ORACLE", False)
    reference = Mock(side_effect=AssertionError("reference must not run"))
    monkeypatch.setattr(
        impl,
        "_gather_paged_kv_snapshot_to_dense_reference",
        reference,
    )

    impl._gather_paged_kv_to_dense_compact(
        key_cache,
        value_cache,
        metadata,
    )

    reference.assert_not_called()


def test_same_forward_oracle_accepts_exact_match(monkeypatch):
    impl, key_cache, value_cache, metadata = _make_direct_gather_case(
        monkeypatch
    )
    monkeypatch.setattr(attention_v1, "_GEMMA4_MTP_ORACLE", True)

    dense_key, dense_value, actual_seq_lengths = (
        impl._gather_paged_kv_to_dense_compact(
            key_cache,
            value_cache,
            metadata,
        )
    )

    assert dense_key.shape[0] == 7
    assert dense_value.shape[0] == 7
    assert actual_seq_lengths == [3, 7]


def test_same_forward_oracle_reports_first_mismatch(monkeypatch):
    impl, key_cache, value_cache, metadata = _make_direct_gather_case(
        monkeypatch
    )
    monkeypatch.setattr(attention_v1, "_GEMMA4_MTP_ORACLE", True)
    original_reference = impl._gather_paged_kv_snapshot_to_dense_reference

    def corrupted_reference(*args, **kwargs):
        key, value = original_reference(*args, **kwargs)
        key = key.clone()
        key.view(-1)[0] += 1
        return key, value

    monkeypatch.setattr(
        impl,
        "_gather_paged_kv_snapshot_to_dense_reference",
        corrupted_reference,
    )

    with pytest.raises(
        RuntimeError,
        match="first_mismatch_flat_index=0",
    ):
        impl._gather_paged_kv_to_dense_compact(
            key_cache,
            value_cache,
            metadata,
        )
