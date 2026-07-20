import importlib.util
from pathlib import Path


ENVS_SOURCE = Path(__file__).parents[3] / "vllm_ascend" / "envs.py"
PLATFORM_SOURCE = Path(__file__).parents[3] / "vllm_ascend" / "platform.py"
E2E_SOURCE = (
    Path(__file__).parents[3]
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
    / "test_gemma4_mtp_async_scheduling.py"
)


def _load_envs_module():
    spec = importlib.util.spec_from_file_location("test_vllm_ascend_envs", ENVS_SOURCE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gemma4_mtp_async_test_flag_defaults_off_and_can_be_enabled(monkeypatch):
    envs = _load_envs_module()

    monkeypatch.delenv("VLLM_ASCEND_ENABLE_GEMMA4_MTP_ASYNC", raising=False)
    assert envs.VLLM_ASCEND_ENABLE_GEMMA4_MTP_ASYNC is False

    monkeypatch.setenv("VLLM_ASCEND_ENABLE_GEMMA4_MTP_ASYNC", "1")
    assert envs.VLLM_ASCEND_ENABLE_GEMMA4_MTP_ASYNC is True


def test_platform_keeps_the_default_guard_and_has_an_explicit_test_only_bypass():
    source = PLATFORM_SOURCE.read_text(encoding="utf-8")

    assert "not envs.VLLM_ASCEND_ENABLE_GEMMA4_MTP_ASYNC" in source
    assert "experimental " in source
    assert "validation path" in source
    assert "scheduler_config.async_scheduling = False" in source


def test_gemma4_mtp_async_e2e_uses_the_expected_network_and_context_window():
    source = E2E_SOURCE.read_text(encoding="utf-8")

    assert '"--host",\n        "0.0.0.0"' in source
    assert '"--max-model-len",\n        "32768"' in source
    assert '"--language-model-only"' in source
