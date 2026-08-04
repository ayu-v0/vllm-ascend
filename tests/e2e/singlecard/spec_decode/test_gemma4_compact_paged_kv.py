from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backend import AttentionType

import vllm_ascend.attention.attention_v1 as attention_v1
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendMetadata,
    _build_compact_paged_kv_metadata,
)


torch_npu = pytest.importorskip("torch_npu")


def _require_npu():
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    torch.npu.set_device(0)


def _make_impl(
    monkeypatch,
    *,
    num_kv_heads: int,
    head_size: int,
    sliding_window: int | None,
):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=None,
        quant_config=None,
    )
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
    impl._layer_name = "gemma4.compact.synthetic"
    return impl


def _valid_block_table(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 4, 0],
            [5, 6, 7],
        ],
        dtype=torch.int32,
        device=device,
    )


@pytest.mark.parametrize(
    ("num_kv_heads", "head_size", "sliding_window"),
    [(16, 256, 1024), (4, 512, None)],
    ids=["sliding-16x256", "full-4x512"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compact_paged_kv_matches_legacy_on_npu(
    monkeypatch,
    num_kv_heads,
    head_size,
    sliding_window,
    dtype,
):
    _require_npu()
    device = torch.device("npu:0")
    block_size = 128
    cache_block_capacity = 8
    seq_lens = [0, 1, 127, 128, 129, 259]
    block_table = _valid_block_table(device)
    key_cache = torch.randn(
        cache_block_capacity,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    metadata = AscendMetadata(
        block_tables=block_table,
        seq_lens_list=seq_lens,
    )
    impl = _make_impl(
        monkeypatch,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        sliding_window=sliding_window,
    )
    monkeypatch.setattr(attention_v1, "_GEMMA4_MTP_ORACLE", True)

    new_key, new_value, actual_seq_lengths = (
        impl._gather_paged_kv_to_dense_compact(
            key_cache,
            value_cache,
            metadata,
        )
    )
    compact = metadata.compact_paged_kv
    assert compact is not None
    old_key, old_value = impl._gather_paged_kv_snapshot_to_dense_reference(
        key_cache,
        value_cache,
        compact.block_table_snapshot,
        seq_lens,
    )

    assert torch.equal(new_key, old_key)
    assert torch.equal(new_value, old_value)
    assert actual_seq_lengths == [0, 1, 128, 256, 385, 644]
    assert compact.physical_slots.numel() == sum(seq_lens)
    assert compact.block_table_snapshot.data_ptr() != block_table.data_ptr()

    second_impl = _make_impl(
        monkeypatch,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        sliding_window=sliding_window,
    )
    reused = second_impl._get_or_build_compact_paged_kv_metadata(
        metadata,
        key_cache,
    )
    assert reused is compact
    assert reused.physical_slots.data_ptr() == compact.physical_slots.data_ptr()


def test_compact_slots_skip_invalid_padding_columns_on_npu():
    _require_npu()
    device = torch.device("npu:0")
    block_size = 128
    cache_block_capacity = 8
    sentinel = cache_block_capacity + 17
    seq_lens = [1, 129, 259]
    block_table = torch.tensor(
        [
            [0, sentinel, sentinel],
            [1, 2, sentinel],
            [3, 4, 5],
        ],
        dtype=torch.int32,
        device=device,
    )

    compact = _build_compact_paged_kv_metadata(
        block_table,
        seq_lens,
        block_size=block_size,
        cache_block_capacity=cache_block_capacity,
    )

    assert compact.physical_slots.numel() == sum(seq_lens)
    assert bool(
        torch.all(
            compact.physical_slots
            < cache_block_capacity * block_size
        ).item()
    )


def test_compact_metadata_isolated_between_sliding_and_full_groups_on_npu(
    monkeypatch,
):
    _require_npu()
    device = torch.device("npu:0")
    block_size = 128
    seq_lens = [1, 129]
    block_table = torch.tensor(
        [[0, 0], [1, 2]],
        dtype=torch.int32,
        device=device,
    )
    sliding_metadata = AscendMetadata(
        block_tables=block_table.clone(),
        seq_lens_list=seq_lens,
    )
    full_metadata = AscendMetadata(
        block_tables=block_table.clone(),
        seq_lens_list=seq_lens,
    )
    sliding_impl = _make_impl(
        monkeypatch,
        num_kv_heads=16,
        head_size=256,
        sliding_window=1024,
    )
    full_impl = _make_impl(
        monkeypatch,
        num_kv_heads=4,
        head_size=512,
        sliding_window=None,
    )
    sliding_cache = torch.empty(
        3,
        block_size,
        16,
        256,
        dtype=torch.bfloat16,
        device=device,
    )
    full_cache = torch.empty(
        3,
        block_size,
        4,
        512,
        dtype=torch.bfloat16,
        device=device,
    )

    sliding = sliding_impl._get_or_build_compact_paged_kv_metadata(
        sliding_metadata,
        sliding_cache,
    )
    full = full_impl._get_or_build_compact_paged_kv_metadata(
        full_metadata,
        full_cache,
    )

    assert sliding is not full
    assert (
        sliding.block_table_snapshot.data_ptr()
        != full.block_table_snapshot.data_ptr()
    )
    assert sliding.physical_slots.data_ptr() != full.physical_slots.data_ptr()
