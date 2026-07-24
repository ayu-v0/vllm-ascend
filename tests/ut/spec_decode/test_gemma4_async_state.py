# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.spec_decode.utils import (
    update_num_computed_tokens_for_batch_change,
)


pytestmark = pytest.mark.skip_global_cleanup


def test_async_state_correction_handles_reorder_reject_and_new_request():
    num_computed = torch.tensor([100, 200, 300, 0], dtype=torch.int32)
    num_accepted = torch.ones(4, dtype=torch.int32)
    prev_positions = torch.tensor([2, 0, -1, 1], dtype=torch.int64)
    valid_counts = torch.tensor([1, 3, 2, 4], dtype=torch.int32)
    prev_num_drafts = torch.tensor([3, 3, 0, 0], dtype=torch.int32)
    cpu_values = torch.tensor([301, 104, 7, 204], dtype=torch.int32)

    update_num_computed_tokens_for_batch_change(
        num_computed,
        num_accepted,
        prev_positions,
        valid_counts,
        prev_num_drafts,
        cpu_values,
    )

    assert num_computed.tolist() == [301, 101, 7, 203]
    assert num_accepted.tolist() == [1, 1, 1, 3]


def test_async_state_correction_rolls_back_k3_to_one_valid_token():
    num_computed = torch.tensor([512], dtype=torch.int32)
    num_accepted = torch.ones(1, dtype=torch.int32)
    prev_positions = torch.tensor([0], dtype=torch.int64)
    valid_counts = torch.tensor([1], dtype=torch.int32)
    prev_num_drafts = torch.tensor([3], dtype=torch.int32)
    cpu_values = torch.tensor([516], dtype=torch.int32)

    update_num_computed_tokens_for_batch_change(
        num_computed,
        num_accepted,
        prev_positions,
        valid_counts,
        prev_num_drafts,
        cpu_values,
    )

    assert num_computed.tolist() == [513]
    assert num_accepted.tolist() == [1]


def test_async_state_correction_mixed_accept_reject_rows():
    num_computed = torch.tensor([100, 200, 300], dtype=torch.int32)
    num_accepted = torch.ones(3, dtype=torch.int32)
    prev_positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    valid_counts = torch.tensor([1, 2, 4], dtype=torch.int32)
    prev_num_drafts = torch.tensor([3, 3, 3], dtype=torch.int32)
    cpu_values = torch.tensor([104, 204, 304], dtype=torch.int32)

    update_num_computed_tokens_for_batch_change(
        num_computed,
        num_accepted,
        prev_positions,
        valid_counts,
        prev_num_drafts,
        cpu_values,
    )

    assert num_computed.tolist() == [101, 202, 304]
    assert num_accepted.tolist() == [1, 2, 4]
