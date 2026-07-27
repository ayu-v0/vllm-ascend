import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[3]
    / "vllm_ascend"
    / "spec_decode"
    / "gemma4_oracle.py"
)
SPEC = importlib.util.spec_from_file_location("gemma4_oracle_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

reconstruct_greedy_output = MODULE.reconstruct_greedy_output
validate_async_state_snapshot = MODULE.validate_async_state_snapshot
validate_greedy_oracle_snapshot = MODULE.validate_greedy_oracle_snapshot


def _oracle_tensors(
    *,
    target: list[int],
    draft: list[int],
    bonus: list[int],
    sampled: list[list[int]],
    top2_ids: list[list[int]] | None = None,
    top2_values: list[list[float]] | None = None,
) -> dict[str, object]:
    if top2_ids is None:
        top2_ids = [[token, token + 1] for token in target]
    if top2_values is None:
        top2_values = [[1.0, 0.0] for _ in target]
    return {
        "oracle_target_argmax": target,
        "oracle_target_top2_ids": top2_ids,
        "oracle_target_top2_values": top2_values,
        "oracle_draft_token_ids": draft,
        "oracle_bonus_token_ids": bonus,
        "oracle_sampled_token_ids": sampled,
    }


@pytest.mark.parametrize(
    ("target", "draft", "bonus", "counts", "max_spec_len", "expected"),
    [
        ([10], [10], [20], [1], 1, [[10, 20]]),
        ([10], [11], [20], [1], 1, [[10, -1]]),
        ([10, 11, 12], [10, 11, 12], [20], [3], 3, [[10, 11, 12, 20]]),
        ([10, 11, 12], [99, 11, 12], [20], [3], 3, [[10, -1, -1, -1]]),
        ([10, 11, 12], [10, 99, 12], [20], [3], 3, [[10, 11, -1, -1]]),
    ],
)
def test_reconstruct_greedy_output(
    target, draft, bonus, counts, max_spec_len, expected
):
    actual = reconstruct_greedy_output(
        target_argmax=target,
        draft_token_ids=draft,
        bonus_token_ids=bonus,
        num_draft_tokens=counts,
        max_spec_len=max_spec_len,
    )

    assert actual == expected


def test_reconstruct_greedy_output_mixed_batch_preserves_token_zero():
    actual = reconstruct_greedy_output(
        target_argmax=[0, 11, 12, 20, 21, 22],
        draft_token_ids=[99, 11, 12, 20, 99, 22],
        bonus_token_ids=[30, 31, 32],
        num_draft_tokens=[1, 2, 3],
        max_spec_len=3,
    )

    assert actual == [
        [0, -1, -1, -1],
        [11, 12, 31, -1],
        [20, 21, -1, -1],
    ]


def test_validate_greedy_oracle_accepts_near_tie_when_output_matches_argmax():
    tensors = _oracle_tensors(
        target=[10],
        draft=[11],
        bonus=[20],
        sampled=[[10, -1]],
        top2_ids=[[10, 11]],
        top2_values=[[1.000001, 1.0]],
    )

    validate_greedy_oracle_snapshot(
        tensors=tensors,
        context={
            "trace_id": 7,
            "oracle_num_draft_tokens": [1],
            "oracle_max_spec_len": 1,
            "oracle_vocab_size": 100,
        },
        parsed_token_ids=[[10]],
        valid_sampled_token_count=[1],
    )


def test_validate_greedy_oracle_rejects_sampled_token_mismatch():
    tensors = _oracle_tensors(
        target=[10],
        draft=[11],
        bonus=[20],
        sampled=[[11, -1]],
    )

    with pytest.raises(AssertionError, match="greedy oracle mismatch"):
        validate_greedy_oracle_snapshot(
            tensors=tensors,
            context={
                "trace_id": 8,
                "oracle_num_draft_tokens": [1],
                "oracle_max_spec_len": 1,
                "oracle_vocab_size": 100,
            },
            parsed_token_ids=[[11]],
            valid_sampled_token_count=[1],
        )


def test_validate_greedy_oracle_rejects_valid_count_mismatch():
    tensors = _oracle_tensors(
        target=[10],
        draft=[10],
        bonus=[20],
        sampled=[[10, 20]],
    )

    with pytest.raises(AssertionError, match="valid count"):
        validate_greedy_oracle_snapshot(
            tensors=tensors,
            context={
                "trace_id": 9,
                "oracle_num_draft_tokens": [1],
                "oracle_max_spec_len": 1,
                "oracle_vocab_size": 100,
            },
            parsed_token_ids=[[10, 20]],
            valid_sampled_token_count=[1],
        )


def test_validate_async_state_snapshot_handles_reorder_and_new_request():
    tensors = {
        "prev_positions": [2, 0, -1, 1],
        "prev_num_draft_tokens": [3, 3, 0],
        "prev_valid_sampled_token_count": [1, 3, 2],
        "cpu_num_computed_tokens": [301, 104, 7, 204],
        "num_computed_before": [100, 200, 300],
        "num_computed_after": [301, 101, 7, 203],
        "num_accepted_tokens": [1, 1, 1, 3],
        "group_0_block_table": [[1, 2]],
        "group_0_slot_mapping": [8, 9],
    }
    context = {
        "req_ids": ["c", "a", "new", "b"],
        "prev_req_id_to_index": {"a": 0, "b": 1, "c": 2},
        "state_correction_applied": True,
    }

    validate_async_state_snapshot(tensors=tensors, context=context)


def test_validate_async_state_snapshot_uses_cpu_value_without_previous_batch():
    tensors = {
        "prev_positions": [-1],
        "prev_num_draft_tokens": [],
        "prev_valid_sampled_token_count": [],
        "cpu_num_computed_tokens": [7],
        "num_computed_before": [],
        "num_computed_after": [7],
        "num_accepted_tokens": [1],
    }
    context = {
        "req_ids": ["new"],
        "prev_req_id_to_index": {},
        "state_correction_applied": False,
    }

    validate_async_state_snapshot(tensors=tensors, context=context)


def test_validate_async_state_snapshot_rejects_unpaired_kv_group():
    tensors = {
        "prev_positions": [-1],
        "prev_num_draft_tokens": [],
        "prev_valid_sampled_token_count": [],
        "cpu_num_computed_tokens": [7],
        "num_computed_before": [],
        "num_computed_after": [7],
        "num_accepted_tokens": [1],
        "group_0_block_table": [[1]],
    }
    context = {
        "req_ids": ["new"],
        "prev_req_id_to_index": {},
        "state_correction_applied": False,
    }

    with pytest.raises(AssertionError, match="KV group snapshot is incomplete"):
        validate_async_state_snapshot(tensors=tensors, context=context)


def test_validate_async_state_snapshot_rejects_request_mapping_mismatch():
    tensors = {
        "prev_positions": [1],
        "prev_num_draft_tokens": [3],
        "prev_valid_sampled_token_count": [1],
        "cpu_num_computed_tokens": [104],
        "num_computed_before": [100],
        "num_computed_after": [101],
        "num_accepted_tokens": [1],
    }
    context = {
        "req_ids": ["a"],
        "prev_req_id_to_index": {"a": 0},
        "state_correction_applied": True,
    }

    with pytest.raises(AssertionError, match="request mapping mismatch"):
        validate_async_state_snapshot(tensors=tensors, context=context)
