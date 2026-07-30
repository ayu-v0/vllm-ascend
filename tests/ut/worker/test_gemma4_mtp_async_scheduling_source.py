import ast
import json
import subprocess
import sys
import tempfile
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
SUMMARIZER_SCRIPT = (
    Path(__file__).parents[3]
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
    / "summarize_gemma4_mtp_async_benchmark.py"
)
AB_BENCHMARK_SCRIPT = (
    Path(__file__).parents[3]
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
    / "run_gemma4_mtp_ab_benchmark.sh"
)
AB_SUMMARIZER_SCRIPT = (
    Path(__file__).parents[3]
    / "tests"
    / "e2e"
    / "singlecard"
    / "spec_decode"
    / "summarize_gemma4_mtp_ab_benchmark.py"
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


def _find_vllm_uniproc_executor_source() -> Path:
    ascend_root = Path(__file__).parents[3]
    suffix = Path("vllm") / "v1" / "executor" / "uniproc_executor.py"
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
    raise AssertionError("Unable to locate the sibling vLLM UniProc source")


VLLM_ENGINE_CORE_SOURCE = _find_vllm_engine_core_source()
VLLM_UNIPROC_EXECUTOR_SOURCE = _find_vllm_uniproc_executor_source()


def _method_source(method_name: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{method_name} was not found")


def _function_node(
    source_path: Path, function_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            return node
    raise AssertionError(f"{function_name} was not found in {source_path}")


def _gemma4_server_calls(function_name: str) -> list[ast.Call]:
    function = _function_node(E2E_SOURCE, function_name)
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "gemma4_server"
    ]


def _oracle_debug_call_owners(source: str) -> set[str]:
    owners = set()
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(node):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "gemma4_server"
            ):
                continue
            if any(
                keyword.arg == "oracle_debug"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in call.keywords
            ):
                owners.add(node.name)
    return owners


def test_gemma4_mtp_keeps_requested_async_scheduling_on_ascend():
    source = _method_source("_fix_incompatible_config")

    assert "use_gemma4_mtp" not in source
    assert "scheduler_config.async_scheduling = False" not in source
    assert "Gemma4 MTP does not support async scheduling on Ascend" not in source
    assert "VLLM_ASCEND_ENABLE_GEMMA4_MTP_ASYNC" not in source


def test_gemma4_mtp_oracle_has_a_dedicated_default_off_switch():
    source = ASCEND_ENVS_SOURCE.read_text(encoding="utf-8")

    assert '"VLLM_ASCEND_GEMMA4_MTP_ORACLE"' in source
    assert 'os.getenv("VLLM_ASCEND_GEMMA4_MTP_ORACLE", "0")' in source
    assert '"VLLM_ASCEND_GEMMA4_MTP_DEBUG"' in source


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


def test_gemma4_mtp_oracle_debug_is_scoped_to_correctness_gate():
    server = _function_node(E2E_SOURCE, "gemma4_server")
    keyword_defaults = dict(
        zip(server.args.kwonlyargs, server.args.kw_defaults, strict=True)
    )
    oracle_argument = next(
        argument for argument in keyword_defaults if argument.arg == "oracle_debug"
    )
    oracle_default = keyword_defaults[oracle_argument]
    assert isinstance(oracle_argument.annotation, ast.Name)
    assert oracle_argument.annotation.id == "bool"
    assert isinstance(oracle_default, ast.Constant)
    assert oracle_default.value is False

    source = E2E_SOURCE.read_text(encoding="utf-8")
    assert '"VLLM_ASCEND_GEMMA4_MTP_ORACLE": (' in source
    assert '"VLLM_ASCEND_GEMMA4_MTP_DEBUG": "0"' in source
    assert '"VLLM_BATCH_INVARIANT": "0"' in source

    gate_name = "test_greedy_target_sync_async_diagnostic_and_oracle"
    gate_calls = _gemma4_server_calls(gate_name)
    mtp_calls = [
        call
        for call in gate_calls
        if any(
            keyword.arg == "use_mtp"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
    ]
    assert mtp_calls
    assert all(
        any(
            keyword.arg == "oracle_debug"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in mtp_calls
    )

    default_tests = {
        "test_greedy_boundaries_without_padding",
        "test_streaming_matches_non_streaming_and_cancel_releases_resources",
        "test_mixed_prefill_decode_requests_complete_without_padding_or_reordering",
    }
    for test_name in default_tests:
        calls = _gemma4_server_calls(test_name)
        assert calls
        assert all(
            not (
                keyword.arg == "oracle_debug"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            )
            for call in calls
            for keyword in call.keywords
        )

    owners = _oracle_debug_call_owners(source)
    assert owners == {gate_name}


def test_oracle_debug_owner_scan_covers_sync_async_helpers_and_ignores_false():
    owners = _oracle_debug_call_owners(
        """
def helper_escape():
    gemma4_server(oracle_debug=True)

async def async_escape():
    gemma4_server(oracle_debug=True)

def helper_explicit_false():
    gemma4_server(oracle_debug=False)
"""
    )

    assert owners == {"helper_escape", "async_escape"}


def test_gemma4_mtp_benchmark_keeps_k_and_workload_constant():
    source = BENCHMARK_SCRIPT.read_text(encoding="utf-8")

    assert 'MODES=("sync" "async_uni" "async_mp" "async_candidate")' in source
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


def test_gemma4_mtp_benchmark_records_executor_and_candidate_per_mode():
    source = BENCHMARK_SCRIPT.read_text(encoding="utf-8")

    assert 'executor_backend="uni"' in source
    assert 'executor_backend="mp"' in source
    assert 'candidate="0"' in source
    assert 'candidate="1"' in source
    assert '--distributed-executor-backend' in source
    assert '"VLLM_ASCEND_GEMMA4_MTP_ASYNC_UNIPROC_SUBMIT=${candidate}"' in source
    assert '"HCCL_OP_EXPANSION_MODE=AIV"' in source
    assert 'printf \'%q \' "${server_env[@]}" "${server_args[@]}"' in source
    assert '"executor_backend=${executor_backend}"' in source
    assert '"candidate=${candidate}"' in source


def test_gemma4_mtp_benchmark_summarizer_aggregates_three_runs():
    mode_config = {
        "sync": ("uni", "0"),
        "async_uni": ("uni", "0"),
        "async_mp": ("mp", "0"),
        "async_candidate": ("uni", "1"),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        run_roots = []
        for run_index in range(3):
            run_root = root / f"run-{run_index + 1}"
            run_roots.append(run_root)
            for mode_index, (mode, (executor, candidate)) in enumerate(
                mode_config.items()
            ):
                mode_dir = run_root / mode
                mode_dir.mkdir(parents=True)
                result = {
                    "mode": mode,
                    "executor_backend": executor,
                    "candidate": candidate,
                    "k": "3",
                    "max_model_len": "32768",
                    "max_num_seqs": "72",
                    "input_len": "256",
                    "output_len": "128",
                    "max_concurrency": "72",
                    "num_prompts": "3000",
                    "request_rate": "inf",
                    "temperature": "0",
                    "seed": "0",
                    "hccl_op_expansion_mode": "AIV",
                    "completed": 3000,
                    "failed": 0,
                    "request_throughput": 10.0 + run_index + mode_index,
                    "output_throughput": 100.0 + run_index + mode_index,
                    "mean_ttft_ms": 20.0 + run_index + mode_index,
                    "median_ttft_ms": 18.0 + run_index + mode_index,
                    "p95_ttft_ms": 30.0 + run_index + mode_index,
                    "mean_e2el_ms": 200.0 + run_index + mode_index,
                    "median_e2el_ms": 190.0 + run_index + mode_index,
                    "p95_e2el_ms": 250.0 + run_index + mode_index,
                    "spec_decode_acceptance_rate": 80.0 + mode_index,
                    "spec_decode_acceptance_length": 3.0 + mode_index / 10,
                    "spec_decode_draft_tokens": 9000,
                    "spec_decode_accepted_tokens": 7200,
                }
                (mode_dir / f"{mode}.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
                (mode_dir / "server-command.txt").write_text(
                    f"env backend={executor} candidate={candidate} vllm serve\n",
                    encoding="utf-8",
                )

        output_path = root / "summary.md"
        subprocess.run(
            [
                sys.executable,
                str(SUMMARIZER_SCRIPT),
                str(output_path),
                *(str(run_root) for run_root in run_roots),
            ],
            check=True,
        )
        summary = output_path.read_text(encoding="utf-8")

    assert "## Three-Run Median" in summary
    assert "`async_uni`" in summary
    assert "`async_mp`" in summary
    assert "`async_candidate`" in summary
    assert "## Server Commands" in summary
    assert "backend=mp candidate=0" in summary


def test_gemma4_mtp_benchmark_defaults_support_the_target_load():
    source = BENCHMARK_SCRIPT.read_text(encoding="utf-8")

    assert 'MAX_NUM_SEQS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_NUM_SEQS:-72}' in source
    assert 'NUM_PROMPTS=${VLLM_ASCEND_GEMMA4_MTP_BENCH_NUM_PROMPTS:-3000}' in source
    assert 'MAX_CONCURRENCY=${VLLM_ASCEND_GEMMA4_MTP_BENCH_MAX_CONCURRENCY:-72}' in source
    for name, value in (
        ("NUM_SPECULATIVE_TOKENS", "3"),
        ("MAX_MODEL_LEN", "32768"),
        ("MAX_NUM_SEQS", "72"),
        ("NUM_PROMPTS", "3000"),
        ("INPUT_LEN", "256"),
        ("OUTPUT_LEN", "128"),
        ("MAX_CONCURRENCY", "72"),
        ("REQUEST_RATE", "inf"),
        ("BENCHMARK_SEED", "0"),
    ):
        assert f'require_fixed_value "{name}" "${{{name}}}" "{value}"' in source


def test_gemma4_mtp_benchmark_uses_low_noise_logging_without_disabling_metrics():
    source = BENCHMARK_SCRIPT.read_text(encoding="utf-8")

    assert 'SERVER_LOG_LEVEL=${VLLM_ASCEND_GEMMA4_MTP_BENCH_LOG_LEVEL:-WARNING}' in source
    assert 'VLLM_ASCEND_GEMMA4_MTP_ORACLE' in source
    assert 'VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE' in source
    assert '"VLLM_BATCH_INVARIANT=0"' in source
    assert '"VLLM_ASCEND_GEMMA4_MTP_ORACLE=0"' in source
    assert '"VLLM_ASCEND_GEMMA4_MTP_DEBUG=0"' in source
    assert '"VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE=0"' in source
    assert '"${server_env[@]}" "${server_args[@]}"' in source
    assert '--disable-tqdm' in source
    assert '--disable-log-stats' not in source
    assert '--enable-log-requests' not in source


def test_gemma4_mtp_ab_benchmark_defines_three_single_card_modes():
    assert AB_BENCHMARK_SCRIPT.is_file()
    source = AB_BENCHMARK_SCRIPT.read_text(encoding="utf-8")

    assert 'MODES=("no_mtp_uni" "mtp_uni" "mtp_mp")' in source
    assert '"ASCEND_RT_VISIBLE_DEVICES=0"' in source
    assert '--tensor-parallel-size 1' in source
    assert '--async-scheduling' in source
    assert 'MAX_NUM_SEQS=${VLLM_ASCEND_GEMMA4_MTP_AB_MAX_NUM_SEQS:-16}' in source
    assert 'NUM_PROMPTS=${VLLM_ASCEND_GEMMA4_MTP_AB_NUM_PROMPTS:-500}' in source
    assert 'MAX_CONCURRENCY=${VLLM_ASCEND_GEMMA4_MTP_AB_MAX_CONCURRENCY:-16}' in source
    assert 'INPUT_LEN=${VLLM_ASCEND_GEMMA4_MTP_AB_INPUT_LEN:-12500}' in source
    assert 'OUTPUT_LEN=${VLLM_ASCEND_GEMMA4_MTP_AB_OUTPUT_LEN:-1024}' in source
    assert 'RANGE_RATIO=\'{"input":0.2,"output":0.0}\'' in source
    assert '--random-range-ratio "${RANGE_RATIO}"' in source
    assert 'no_mtp_uni)\n      executor_backend="uni"\n      use_mtp="0"' in source
    assert 'mtp_uni)\n      executor_backend="uni"\n      use_mtp="1"' in source
    assert 'mtp_mp)\n      executor_backend="mp"\n      use_mtp="1"' in source
    assert 'if [[ "${use_mtp}" == "1" ]]; then' in source
    assert '--distributed-executor-backend "${executor_backend}"' in source
    assert '--speculative-config' in source
    assert 'run_round 1 no_mtp_uni mtp_uni mtp_mp' in source
    assert 'run_round 2 mtp_mp mtp_uni no_mtp_uni' in source
    assert 'run_round 3 no_mtp_uni mtp_mp mtp_uni' in source


def test_gemma4_mtp_ab_summarizer_aggregates_three_modes():
    assert AB_SUMMARIZER_SCRIPT.is_file()
    mode_config = {
        "no_mtp_uni": ("uni", "0", "0"),
        "mtp_uni": ("uni", "1", "3"),
        "mtp_mp": ("mp", "1", "3"),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        run_roots = []
        for run_index in range(3):
            run_root = root / f"round_{run_index + 1}"
            run_roots.append(run_root)
            for mode_index, (mode, (executor, use_mtp, k)) in enumerate(
                mode_config.items()
            ):
                mode_dir = run_root / mode
                mode_dir.mkdir(parents=True)
                throughput_multiplier = (1.0, 1.08, 1.10)[mode_index]
                latency_multiplier = (1.0, 0.90, 0.88)[mode_index]
                result = {
                    "round": str(run_index + 1),
                    "mode": mode,
                    "executor_backend": executor,
                    "use_mtp": use_mtp,
                    "k": k,
                    "device": "0",
                    "tensor_parallel_size": "1",
                    "max_model_len": "32768",
                    "max_num_seqs": "16",
                    "max_batched_tokens": "16384",
                    "input_len": "12500",
                    "input_min": "10000",
                    "input_max": "15000",
                    "output_len": "1024",
                    "max_concurrency": "16",
                    "num_prompts": "500",
                    "request_rate": "inf",
                    "temperature": "0",
                    "seed": "0",
                    "completed": 500,
                    "failed": 0,
                    "input_lens": [10000 + i * 10 for i in range(500)],
                    "output_lens": [1024] * 500,
                    "request_throughput": (1.0 + run_index / 100)
                    * throughput_multiplier,
                    "output_throughput": (100.0 + run_index)
                    * throughput_multiplier,
                    "mean_ttft_ms": (1000.0 + run_index) * latency_multiplier,
                    "median_ttft_ms": (900.0 + run_index) * latency_multiplier,
                    "p95_ttft_ms": (1200.0 + run_index) * latency_multiplier,
                    "mean_tpot_ms": (10.0 + run_index / 100) * latency_multiplier,
                    "median_tpot_ms": (9.0 + run_index / 100) * latency_multiplier,
                    "p95_tpot_ms": (12.0 + run_index / 100) * latency_multiplier,
                    "mean_itl_ms": (10.5 + run_index / 100) * latency_multiplier,
                    "median_itl_ms": (9.5 + run_index / 100) * latency_multiplier,
                    "p95_itl_ms": (12.5 + run_index / 100) * latency_multiplier,
                    "mean_e2el_ms": (12000.0 + run_index) * latency_multiplier,
                    "median_e2el_ms": (11000.0 + run_index) * latency_multiplier,
                    "p95_e2el_ms": (15000.0 + run_index) * latency_multiplier,
                }
                if use_mtp == "1":
                    result.update(
                        {
                            "spec_decode_acceptance_rate": 20.0 + mode_index,
                            "spec_decode_acceptance_length": 1.6 + mode_index / 10,
                            "spec_decode_num_drafts": 100000,
                            "spec_decode_draft_tokens": 300000,
                            "spec_decode_accepted_tokens": 60000,
                        }
                    )
                (mode_dir / f"{mode}.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
                command = (
                    f"env ASCEND_RT_VISIBLE_DEVICES=0 backend={executor} "
                    "vllm serve --tensor-parallel-size 1 "
                    f"--distributed-executor-backend {executor} "
                    "--async-scheduling"
                )
                if use_mtp == "1":
                    command += " --speculative-config mtp-config"
                (mode_dir / "server-command.txt").write_text(
                    command + "\n", encoding="utf-8"
                )

        output_path = root / "summary.md"
        subprocess.run(
            [
                sys.executable,
                str(AB_SUMMARIZER_SCRIPT),
                str(output_path),
                *(str(run_root) for run_root in run_roots),
            ],
            check=True,
        )
        summary = output_path.read_text(encoding="utf-8")

    assert "## Three-Run Median" in summary
    assert "`no_mtp_uni`" in summary
    assert "`mtp_uni`" in summary
    assert "`mtp_mp`" in summary
    assert "## MTP vs No-MTP" in summary
    assert "## MTP MP vs Uni" in summary
    assert "## Decisions" in summary
    assert "`mtp_uni`: clear improvement" in summary
    assert "`mtp_mp`: clear improvement" in summary
    assert "## Server Commands" in summary


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


def test_gemma4_mtp_completed_head_experiment_is_removed_from_core_and_envs():
    core_source = VLLM_ENGINE_CORE_SOURCE.read_text(encoding="utf-8")
    envs_source = ASCEND_ENVS_SOURCE.read_text(encoding="utf-8")
    removed_env = "VLLM_ASCEND_GEMMA4_MTP_" + "COMPLETED_HEAD_" + "TTFT_FIX"

    assert removed_env not in envs_source
    assert "_gemma4_mtp_completed_head_ttft_fix_enabled" not in core_source
    assert "_should_deliver_completed_gemma4_mtp_batch_head" not in core_source
    assert "phase=completed_head_check" not in core_source
    assert "phase=completed_head" not in core_source
    assert "_consume_batch_queue_output" in core_source


def test_gemma4_mtp_completed_head_experiment_is_removed_from_validation():
    e2e_source = E2E_SOURCE.read_text(encoding="utf-8")
    benchmark_source = BENCHMARK_SCRIPT.read_text(encoding="utf-8")
    removed_env = "VLLM_ASCEND_GEMMA4_MTP_" + "COMPLETED_HEAD_" + "TTFT_FIX"

    assert removed_env not in e2e_source
    assert removed_env not in benchmark_source


def test_gemma4_mtp_async_uniproc_submit_gate_is_scoped_and_default_off():
    envs_source = ASCEND_ENVS_SOURCE.read_text(encoding="utf-8")
    uniproc_source = VLLM_UNIPROC_EXECUTOR_SOURCE.read_text(encoding="utf-8")

    assert '"VLLM_ASCEND_GEMMA4_MTP_ASYNC_UNIPROC_SUBMIT"' in envs_source
    assert (
        'os.getenv("VLLM_ASCEND_GEMMA4_MTP_ASYNC_UNIPROC_SUBMIT", "0")'
        in envs_source
    )
    for token in (
        "_should_enable_gemma4_mtp_async_uniproc_submit",
        'device_type == "npu"',
        'method == "mtp"',
        'model_type == "gemma4"',
        "type(self) is UniProcExecutor",
        "max_workers=1",
        "initializer=current_platform.set_device",
        "command_future.add_done_callback(command_done)",
        "output_future.add_done_callback(complete_public_future)",
    ):
        assert token in uniproc_source

    assert "self.worker_command_thread: ThreadPoolExecutor | None = None" in uniproc_source
    command_shutdown = uniproc_source.index("command_thread.shutdown(wait=True)")
    output_shutdown = uniproc_source.index("output_thread.shutdown(wait=True)")
    worker_shutdown = uniproc_source.index("worker.shutdown()")
    assert command_shutdown < output_shutdown < worker_shutdown


if __name__ == "__main__":
    test_gemma4_mtp_keeps_requested_async_scheduling_on_ascend()
    test_gemma4_mtp_oracle_has_a_dedicated_default_off_switch()
    test_gemma4_mtp_e2e_starts_explicit_sync_and_async_servers()
    test_gemma4_mtp_e2e_validates_cancelled_request_resource_release()
    test_gemma4_mtp_oracle_debug_is_scoped_to_correctness_gate()
    test_oracle_debug_owner_scan_covers_sync_async_helpers_and_ignores_false()
    test_gemma4_mtp_benchmark_keeps_k_and_workload_constant()
    test_gemma4_mtp_benchmark_records_executor_and_candidate_per_mode()
    test_gemma4_mtp_benchmark_summarizer_aggregates_three_runs()
    test_gemma4_mtp_benchmark_defaults_support_the_target_load()
    test_gemma4_mtp_benchmark_uses_low_noise_logging_without_disabling_metrics()
    test_gemma4_mtp_ab_benchmark_defines_three_single_card_modes()
    test_gemma4_mtp_ab_summarizer_aggregates_three_modes()
    test_gemma4_mtp_per_group_metadata_keeps_slot_mapping_with_block_table()
    test_gemma4_mtp_async_profile_is_sampled_and_passed_to_async_output()
    test_gemma4_mtp_completed_head_experiment_is_removed_from_core_and_envs()
    test_gemma4_mtp_completed_head_experiment_is_removed_from_validation()
    test_gemma4_mtp_async_uniproc_submit_gate_is_scoped_and_default_off()
