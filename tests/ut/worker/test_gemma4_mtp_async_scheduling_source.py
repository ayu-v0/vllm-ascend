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
ASCEND_ENVS_SOURCE = Path(__file__).parents[3] / "vllm_ascend" / "envs.py"


def _find_vllm_engine_core_source() -> Path:
    ascend_root = Path(__file__).parents[3]
    suffix = Path("vllm") / "v1" / "engine" / "core.py"
    direct_candidate = ascend_root.parent / "vllm" / suffix
    if direct_candidate.is_file():
        return direct_candidate

    versioned_ascend_parent = ascend_root.parent
    versioned_vllm_parent = versioned_ascend_parent.parent / versioned_ascend_parent.name.replace(
        "vllm-ascend-", "vllm-", 1
    )
    versioned_candidate = versioned_vllm_parent / "vllm" / suffix
    if versioned_candidate.is_file():
        return versioned_candidate
    raise AssertionError("Unable to locate the sibling vLLM EngineCore source")


VLLM_ENGINE_CORE_SOURCE = _find_vllm_engine_core_source()


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


def test_gemma4_mtp_benchmark_defaults_support_the_target_load():
    source = BENCHMARK_SCRIPT.read_text(encoding="utf-8")

    assert 'MAX_NUM_SEQS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_NUM_SEQS:-72}' in source
    assert 'NUM_PROMPTS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_NUM_PROMPTS:-3000}' in source
    assert 'MAX_CONCURRENCY=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_CONCURRENCY:-72}' in source


def test_gemma4_mtp_benchmark_uses_low_noise_logging_without_disabling_metrics():
    source = BENCHMARK_SCRIPT.read_text(encoding="utf-8")

    assert 'SERVER_LOG_LEVEL=${VLLM_ASCEND_GEMMA4_MTP_BENCH_LOG_LEVEL:-WARNING}' in source
    assert 'VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE' in source
    assert '"${server_env[@]}" "${server_args[@]}"' in source
    assert '--disable-tqdm' in source
    assert '--disable-log-stats' not in source
    assert '--enable-log-requests' not in source


def test_gemma4_mtp_per_group_metadata_keeps_slot_mapping_with_block_table():
    runner_source = MODEL_RUNNER_SOURCE.read_text(encoding="utf-8")
    proposer_source = ASCEND_GEMMA4_PROPOSER_SOURCE.read_text(encoding="utf-8")

    assert "self.drafter.set_per_group_attention_metadata(" in runner_source
    assert "self._per_group_slot_mappings: dict[int, torch.Tensor] = {}" in proposer_source
    assert "cm.block_table_tensor" in runner_source
    assert "cm.slot_mapping" in runner_source


def test_gemma4_mtp_async_profile_is_sampled_and_passed_to_async_output():
    runner_source = MODEL_RUNNER_SOURCE.read_text(encoding="utf-8")
    envs_source = ASCEND_ENVS_SOURCE.read_text(encoding="utf-8")

    assert "VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE" in envs_source
    assert "VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE_EVERY" in envs_source
    assert "Gemma4 MTP async profile: worker" in runner_source
    assert "profile_context=profile_context" in runner_source
    assert "profile_iteration <= 8 or profile_iteration % profile_every == 0" in runner_source


def test_gemma4_mtp_completed_head_ttft_fix_is_scoped_and_nonblocking():
    core_source = VLLM_ENGINE_CORE_SOURCE.read_text(encoding="utf-8")
    envs_source = ASCEND_ENVS_SOURCE.read_text(encoding="utf-8")

    assert "VLLM_ASCEND_GEMMA4_MTP_COMPLETED_HEAD_TTFT_FIX" in envs_source
    assert "_should_deliver_completed_gemma4_mtp_batch_head" in core_source
    assert "_consume_batch_queue_output" in core_source
    assert "future.done()" in core_source
    assert 'device_type == "npu"' in core_source
    assert 'method == "mtp"' in core_source
    assert 'model_type == "gemma4"' in core_source


def test_gemma4_mtp_completed_head_ttft_fix_is_enabled_only_for_async_validation():
    e2e_source = E2E_SOURCE.read_text(encoding="utf-8")
    benchmark_source = BENCHMARK_SCRIPT.read_text(encoding="utf-8")

    assert "VLLM_ASCEND_GEMMA4_MTP_COMPLETED_HEAD_TTFT_FIX" in e2e_source
    assert "if async_scheduling" in e2e_source
    assert '"VLLM_ASCEND_GEMMA4_MTP_COMPLETED_HEAD_TTFT_FIX=1"' in benchmark_source
    assert '"VLLM_ASCEND_GEMMA4_MTP_COMPLETED_HEAD_TTFT_FIX=0"' in benchmark_source


if __name__ == "__main__":
    test_gemma4_mtp_keeps_requested_async_scheduling_on_ascend()
    test_gemma4_mtp_e2e_starts_explicit_sync_and_async_servers()
    test_gemma4_mtp_e2e_validates_cancelled_request_resource_release()
    test_gemma4_mtp_benchmark_keeps_k_and_workload_constant()
    test_gemma4_mtp_benchmark_defaults_support_the_target_load()
    test_gemma4_mtp_benchmark_uses_low_noise_logging_without_disabling_metrics()
    test_gemma4_mtp_per_group_metadata_keeps_slot_mapping_with_block_table()
    test_gemma4_mtp_async_profile_is_sampled_and_passed_to_async_output()
    test_gemma4_mtp_completed_head_ttft_fix_is_scoped_and_nonblocking()
    test_gemma4_mtp_completed_head_ttft_fix_is_enabled_only_for_async_validation()
