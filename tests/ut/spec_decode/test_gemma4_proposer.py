import copy
from types import SimpleNamespace

import torch
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


def test_gemma4_multistep_metadata_keeps_group_state_and_layer_ownership():
    from vllm_ascend.spec_decode.gemma4_proposer import AscendGemma4Proposer

    proposer = object.__new__(AscendGemma4Proposer)
    proposer.num_speculative_tokens = 3
    proposer.pcp_size = 1
    proposer.draft_attn_groups = [
        _FakeAttentionGroup(0, ["layers.0.self_attn"]),
        _FakeAttentionGroup(1, ["layers.1.self_attn"]),
    ]
    proposer._per_group_block_tables = {
        0: torch.tensor([[10]], dtype=torch.int32),
        1: torch.tensor([[20]], dtype=torch.int32),
    }
    proposer.shallow_copy_metadata = copy.copy

    update_positions = []

    def fake_update(
        draft_step,
        _old_attn_metadata,
        common_attn_metadata,
        _batch_size,
        _num_input_tokens,
        positions,
        _aclgraph_runtime_mode,
        _ori_seq_len,
        _slot_indices,
        _mtp_slot_mapping,
        *,
        attn_group,
        keep_positions_and_seq_lens,
    ):
        assert keep_positions_and_seq_lens is True
        update_positions.append(positions.clone())
        metadata = SimpleNamespace(
            group_id=attn_group.kv_cache_group_id,
            positions=positions.clone(),
            seq_lens=common_attn_metadata.seq_lens.clone(),
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu.clone(),
            slot_mapping=common_attn_metadata.block_table_tensor.view(-1),
        )
        return common_attn_metadata, metadata

    proposer.attn_update_stack_num_spec_norm = fake_update
    common_attn_metadata = SimpleNamespace(
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([27], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([27], dtype=torch.int32),
    )
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
        aclgraph_runtime_mode=None,
    )

    assert len(steps) == 2
    assert [metadata.group_id for metadata in steps[0].values()] == [0, 1]
    assert [metadata.group_id for metadata in steps[1].values()] == [0, 1]
    assert steps[0]["layers.0.self_attn"].slot_mapping.tolist() == [10]
    assert steps[0]["layers.1.self_attn"].slot_mapping.tolist() == [20]
    assert (
        steps[0]["layers.0.self_attn"].slot_mapping.data_ptr()
        != steps[0]["layers.1.self_attn"].slot_mapping.data_ptr()
    )
    assert all(positions.tolist() == [26] for positions in update_positions)
    assert target_positions.tolist() == [26]
    for step in steps:
        for metadata in step.values():
            assert metadata.positions.tolist() == [26]
            assert metadata.seq_lens.tolist() == [27]
            assert metadata.seq_lens_cpu.tolist() == [27]
