import copy
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

import vllm_ascend.spec_decode as spec_decode


class _FakeGemma4Proposer:
    def __init__(self, vllm_config, device, runner):
        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner


class _FakeEagleProposer:
    def __init__(self, vllm_config, device, runner):
        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner


def _vllm_config_for_gemma4_mtp(enabled: bool):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            use_gemma4_mtp=lambda: enabled,
        )
    )


def test_gemma4_mtp_routes_to_dedicated_proposer(monkeypatch):
    monkeypatch.setattr(spec_decode, "AscendGemma4Proposer", _FakeGemma4Proposer, raising=False)
    monkeypatch.setattr(spec_decode, "AscendEagleProposer", _FakeEagleProposer)

    proposer = spec_decode.get_spec_decode_method(
        "mtp",
        _vllm_config_for_gemma4_mtp(True),
        device="npu",
        runner="runner",
    )

    assert isinstance(proposer, _FakeGemma4Proposer)
    assert proposer.device == "npu"
    assert proposer.runner == "runner"


def test_regular_mtp_still_routes_to_eagle_proposer(monkeypatch):
    monkeypatch.setattr(spec_decode, "AscendGemma4Proposer", _FakeGemma4Proposer, raising=False)
    monkeypatch.setattr(spec_decode, "AscendEagleProposer", _FakeEagleProposer)

    proposer = spec_decode.get_spec_decode_method(
        "mtp",
        _vllm_config_for_gemma4_mtp(False),
        device="npu",
        runner="runner",
    )

    assert isinstance(proposer, _FakeEagleProposer)


def test_ascend_gemma4_proposer_reports_tuple_outputs():
    from vllm_ascend.spec_decode.gemma4_proposer import AscendGemma4Proposer

    proposer = object.__new__(AscendGemma4Proposer)

    assert proposer.model_returns_tuple() is True


class _FakeAttentionGroup:
    def __init__(self, group_id: int, layer_names: list[str]):
        self.kv_cache_group_id = group_id
        self.layer_names = layer_names


def _make_metadata_test_proposer(num_speculative_tokens: int = 4):
    from vllm_ascend.spec_decode.gemma4_proposer import AscendGemma4Proposer

    proposer = object.__new__(AscendGemma4Proposer)
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.constant_draft_positions = True
    proposer.pcp_size = 1
    proposer.dcp_size = 1
    proposer.use_cuda_graph = False
    proposer.use_compress = False
    proposer.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )
    )
    proposer.draft_attn_groups = [
        _FakeAttentionGroup(0, ["layers.0.self_attn"]),
        _FakeAttentionGroup(1, ["layers.1.self_attn"]),
    ]
    proposer._per_group_block_tables = {
        0: torch.tensor([[10]], dtype=torch.int32),
        1: torch.tensor([[20]], dtype=torch.int32),
    }
    proposer.shallow_copy_metadata = copy.copy
    return proposer


def _make_common_metadata():
    return SimpleNamespace(
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([27], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([27], dtype=torch.int32),
    )


def _install_fake_metadata_update(proposer):
    calls = []

    def fake_update(
        draft_step,
        _old_attn_metadata,
        common_attn_metadata,
        _batch_size,
        _num_input_tokens,
        positions,
        _aclgraph_runtime_mode,
        _ori_seq_len,
        slot_indices,
        mtp_slot_mapping,
        *,
        attn_group,
        keep_positions_and_seq_lens,
    ):
        assert keep_positions_and_seq_lens is True
        calls.append(
            {
                "draft_step": draft_step,
                "group_id": attn_group.kv_cache_group_id,
                "positions": positions,
                "slot_indices": slot_indices,
                "mtp_slot_mapping": mtp_slot_mapping,
            }
        )
        metadata = SimpleNamespace(
            group_id=attn_group.kv_cache_group_id,
            positions=positions.clone(),
            seq_lens=common_attn_metadata.seq_lens.clone(),
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu.clone(),
            slot_mapping=common_attn_metadata.block_table_tensor.view(-1),
        )
        return common_attn_metadata, metadata

    proposer.attn_update_stack_num_spec_norm = fake_update
    return calls


def _build_metadata(proposer, **kwargs):
    common_attn_metadata = _make_common_metadata()
    initial_metadata = {
        "layers.0.self_attn": object(),
        "layers.1.self_attn": object(),
    }
    target_positions = torch.tensor([26], dtype=torch.int32)
    steps = proposer.build_constant_position_multi_step_metadata(
        common_attn_metadata,
        initial_metadata,
        batch_size=1,
        num_input_tokens=1,
        used_update_positions=target_positions,
        aclgraph_runtime_mode=kwargs.pop(
            "aclgraph_runtime_mode", CUDAGraphMode.NONE
        ),
        **kwargs,
    )
    return steps, common_attn_metadata, initial_metadata, target_positions


def test_gemma4_singlecard_reuses_followup_metadata_per_group():
    proposer = _make_metadata_test_proposer(num_speculative_tokens=4)
    calls = _install_fake_metadata_update(proposer)

    steps, common, initial, target_positions = _build_metadata(proposer)

    assert len(steps) == 3
    assert [(call["draft_step"], call["group_id"]) for call in calls] == [
        (1, 0),
        (1, 1),
    ]
    assert all(call["positions"] is target_positions for call in calls)

    assert steps[0] is steps[1]
    assert steps[1] is steps[2]

    group0 = steps[0]["layers.0.self_attn"]
    group1 = steps[0]["layers.1.self_attn"]
    assert all(step["layers.0.self_attn"] is group0 for step in steps)
    assert all(step["layers.1.self_attn"] is group1 for step in steps)
    assert group0 is not group1

    assert group0.slot_mapping.tolist() == [10]
    assert group1.slot_mapping.tolist() == [20]
    assert group0.slot_mapping.data_ptr() != group1.slot_mapping.data_ptr()
    assert (
        group0.slot_mapping.data_ptr()
        != proposer._per_group_block_tables[0].data_ptr()
    )
    assert (
        group1.slot_mapping.data_ptr()
        != proposer._per_group_block_tables[1].data_ptr()
    )

    assert target_positions.tolist() == [26]
    assert common.block_table_tensor.tolist() == [[0]]
    assert common.seq_lens.tolist() == [27]
    assert common.seq_lens_cpu.tolist() == [27]
    assert set(initial) == {
        "layers.0.self_attn",
        "layers.1.self_attn",
    }
    assert all(
        metadata.positions.tolist() == [26]
        for metadata in (group0, group1)
    )
    assert all(
        metadata.seq_lens.tolist() == [27]
        for metadata in (group0, group1)
    )


def test_gemma4_k1_builds_no_followup_metadata():
    proposer = _make_metadata_test_proposer(num_speculative_tokens=1)
    calls = _install_fake_metadata_update(proposer)

    steps, common, _initial, target_positions = _build_metadata(proposer)

    assert steps == []
    assert calls == []
    assert common.seq_lens.tolist() == [27]
    assert target_positions.tolist() == [26]


def test_gemma4_singlecard_metadata_reuse_guard_accepts_exact_scope():
    proposer = _make_metadata_test_proposer()

    assert proposer._can_reuse_singlecard_multistep_metadata(
        CUDAGraphMode.NONE,
        ori_seq_len=None,
        slot_indices=None,
        mtp_slot_mapping=None,
    )


@pytest.mark.parametrize(
    "case",
    [
        "non_constant_positions",
        "no_attention_groups",
        "pcp2",
        "dcp2",
        "tp2",
        "pp2",
        "dp2",
        "proposer_graph",
        "runtime_full_graph",
        "compressed_attention",
        "cp_original_seq_len",
        "dynamic_slot_indices",
        "prebuilt_mtp_slot_mapping",
    ],
)
def test_gemma4_metadata_reuse_guard_rejects_unsupported_paths(case):
    proposer = _make_metadata_test_proposer()
    runtime_mode = CUDAGraphMode.NONE
    ori_seq_len = None
    slot_indices = None
    mtp_slot_mapping = None

    if case == "non_constant_positions":
        proposer.constant_draft_positions = False
    elif case == "no_attention_groups":
        proposer.draft_attn_groups = []
    elif case == "pcp2":
        proposer.pcp_size = 2
    elif case == "dcp2":
        proposer.dcp_size = 2
    elif case == "tp2":
        proposer.vllm_config.parallel_config.tensor_parallel_size = 2
    elif case == "pp2":
        proposer.vllm_config.parallel_config.pipeline_parallel_size = 2
    elif case == "dp2":
        proposer.vllm_config.parallel_config.data_parallel_size = 2
    elif case == "proposer_graph":
        proposer.use_cuda_graph = True
    elif case == "runtime_full_graph":
        runtime_mode = CUDAGraphMode.FULL
    elif case == "compressed_attention":
        proposer.use_compress = True
    elif case == "cp_original_seq_len":
        ori_seq_len = torch.tensor([27], dtype=torch.int32)
    elif case == "dynamic_slot_indices":
        slot_indices = torch.tensor([0], dtype=torch.int64)
    elif case == "prebuilt_mtp_slot_mapping":
        mtp_slot_mapping = torch.tensor([10], dtype=torch.int32)

    assert not proposer._can_reuse_singlecard_multistep_metadata(
        runtime_mode,
        ori_seq_len=ori_seq_len,
        slot_indices=slot_indices,
        mtp_slot_mapping=mtp_slot_mapping,
    )


def test_gemma4_tp2_falls_back_to_per_step_metadata_construction():
    proposer = _make_metadata_test_proposer(num_speculative_tokens=3)
    proposer.vllm_config.parallel_config.tensor_parallel_size = 2
    calls = _install_fake_metadata_update(proposer)

    steps, _common, _initial, target_positions = _build_metadata(proposer)

    assert len(steps) == 2
    assert [(call["draft_step"], call["group_id"]) for call in calls] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    assert steps[0] is not steps[1]
    assert (
        steps[0]["layers.0.self_attn"]
        is not steps[1]["layers.0.self_attn"]
    )
    assert (
        steps[0]["layers.1.self_attn"]
        is not steps[1]["layers.1.self_attn"]
    )
    assert all(call["positions"] is not target_positions for call in calls)
    assert target_positions.tolist() == [26]
