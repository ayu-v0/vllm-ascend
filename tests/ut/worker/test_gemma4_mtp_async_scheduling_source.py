import ast
from pathlib import Path


SOURCE = Path(__file__).parents[3] / "vllm_ascend" / "platform.py"
E2E_SOURCE = (
    Path(__file__).parents[3]
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
    / "test_gemma4_mtp_async_scheduling.py"
)
BENCHMARK_SCRIPT = (
    Path(__file__).parents[3]
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
    / "run_gemma4_mtp_async_benchmark.sh"
)
MODEL_RUNNER_SOURCE = (
    Path(__file__).parents[3]
    / "vllm_ascend"
    / "worker"
    / "model_runner_v1.py"
)
ASCEND_GEMMA4_PROPOSER_SOURCE = (
    Path(__file__).parents[3]
    / "vllm_ascend"
    / "spec_decode"
    / "gemma4_proposer.py"
)


def _method_source(method_name: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{method_name} was not found")


def test_gemma4_mtp_keeps_requested_async_scheduling_on_ascend():
    source = _method_source("_fix_incompatible_config")

    assert "use_gemma4_mtp" not in source
    assert "scheduler_config.async_scheduling = False" not in source
    assert "Gemma4 MTP does not support async scheduling on Ascend" not in source
    assert "VLLM_ASCEND_ENABLE_GEMMA4_MTP_ASYNC" not in source


def test_gemma4_mtp_e2e_starts_explicit_sync_and_async_servers():
    source = E2E_SOURCE.read_text(encoding="utf-8")

    assert "async_scheduling: bool" in source
    assert 'args.append("--async-scheduling")' in source
    assert 'args.append("--no-async-scheduling")' in source
    assert "VLLM_ASCEND_ENABLE_GEMMA4_MTP_ASYNC" not in source
    assert '"--host",\n        "0.0.0.0"' in source
    assert '"--max-model-len",\n        "32768"' in source
    assert '"--language-model-only"' in source
    assert "if close_early and delta_content:" in source


def test_gemma4_mtp_e2e_validates_cancelled_request_resource_release():
    source = E2E_SOURCE.read_text(encoding="utf-8")

    assert 'server.url_for("v1/responses")' in source
    assert '"background": True' in source
    assert 'f"v1/responses/{response_id}/cancel"' in source
    assert "vllm:num_requests_running" in source
    assert "vllm:num_requests_waiting" in source
    assert "_poll_until" in source


def test_gemma4_mtp_benchmark_keeps_k_and_workload_constant():
    source = BENCHMARK_SCRIPT.read_text(encoding="utf-8")

    assert 'MODES=("sync" "async")' in source
    assert '"--no-async-scheduling"' in source
    assert '"--async-scheduling"' in source
    assert "SPECULATIVE_CONFIG" in source
    assert '"k=${NUM_SPECULATIVE_TOKENS}"' in source
    assert '--tokenizer "${MODEL}"' in source
    for argument, value in (
        ("--seed", "${BENCHMARK_SEED}"),
        ("--num-prompts", "${NUM_PROMPTS}"),
        ("--random-input-len", "${INPUT_LEN}"),
        ("--random-output-len", "${OUTPUT_LEN}"),
        ("--max-concurrency", "${MAX_CONCURRENCY}"),
    ):
        assert argument in source
        assert value in source


def test_gemma4_mtp_per_group_metadata_keeps_slot_mapping_with_block_table():
    runner_source = MODEL_RUNNER_SOURCE.read_text(encoding="utf-8")
    proposer_source = ASCEND_GEMMA4_PROPOSER_SOURCE.read_text(encoding="utf-8")

    assert "self.drafter.set_per_group_attention_metadata(" in runner_source
    assert "self._per_group_slot_mappings: dict[int, torch.Tensor] = {}" in proposer_source
    assert "cm.block_table_tensor" in runner_source
    assert "cm.slot_mapping" in runner_source


if __name__ == "__main__":
    test_gemma4_mtp_keeps_requested_async_scheduling_on_ascend()
    test_gemma4_mtp_e2e_starts_explicit_sync_and_async_servers()
    test_gemma4_mtp_e2e_validates_cancelled_request_resource_release()
    test_gemma4_mtp_benchmark_keeps_k_and_workload_constant()
    test_gemma4_mtp_per_group_metadata_keeps_slot_mapping_with_block_table()
