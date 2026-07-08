from types import SimpleNamespace

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
