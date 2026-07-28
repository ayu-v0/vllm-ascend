# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU-only regression gate for the Gemma4 MTP async validation path.

Set VLLM_ASCEND_GEMMA4_MTP_MODEL and VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL
to local model paths before running this test. The correctness gate enables
the same-forward target oracle only for its sync/async MTP server processes.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Iterator

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer


MODEL = os.getenv("VLLM_ASCEND_GEMMA4_MTP_MODEL")
DRAFT_MODEL = os.getenv("VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL")
SERVED_MODEL_NAME = os.getenv("VLLM_ASCEND_GEMMA4_MTP_SERVED_MODEL", "gemma4")
NUM_SPECULATIVE_TOKENS = int(os.getenv("VLLM_ASCEND_GEMMA4_MTP_K", "3"))

pytestmark = pytest.mark.skipif(
    not MODEL or not DRAFT_MODEL,
    reason=(
        "requires VLLM_ASCEND_GEMMA4_MTP_MODEL and "
        "VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL local paths"
    ),
)


def _server_args(
    port: int,
    *,
    async_scheduling: bool,
    use_mtp: bool,
    num_speculative_tokens: int,
    executor_backend: str = "uni",
) -> list[str]:
    args = [
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--max-model-len",
        "32768",
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--trust-remote-code",
        "--language-model-only",
        "--enable-chunked-prefill",
        "--distributed-executor-backend",
        executor_backend,
    ]
    if use_mtp:
        args.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "mtp",
                        "model": DRAFT_MODEL,
                        "num_speculative_tokens": num_speculative_tokens,
                    }
                ),
            ]
        )
    if async_scheduling:
        args.append("--async-scheduling")
    else:
        args.append("--no-async-scheduling")
    return args


@contextmanager
def gemma4_server(
    *,
    async_scheduling: bool,
    use_mtp: bool,
    num_speculative_tokens: int = NUM_SPECULATIVE_TOKENS,
    executor_backend: str = "uni",
    oracle_debug: bool = False,
    completed_head: bool = False,
    async_uniproc_submit: bool = False,
    async_profile: bool = False,
    enable_responses_store: bool = False,
) -> Iterator[RemoteOpenAIServer]:
    assert MODEL is not None
    port = get_open_port()
    env_dict = {
        "VLLM_BATCH_INVARIANT": "0",
        "VLLM_ASCEND_GEMMA4_MTP_ORACLE": (
            "1" if oracle_debug else "0"
        ),
        "VLLM_ASCEND_GEMMA4_MTP_DEBUG": "0",
        "VLLM_ASCEND_GEMMA4_MTP_COMPLETED_HEAD_TTFT_FIX": (
            "1" if completed_head else "0"
        ),
        "VLLM_ASCEND_GEMMA4_MTP_ASYNC_UNIPROC_SUBMIT": (
            "1" if async_uniproc_submit else "0"
        ),
        "VLLM_ASCEND_GEMMA4_MTP_ASYNC_PROFILE": "1" if async_profile else "0",
    }
    if enable_responses_store:
        env_dict["VLLM_ENABLE_RESPONSES_API_STORE"] = "1"
    with RemoteOpenAIServer(
        MODEL,
        _server_args(
            port,
            async_scheduling=async_scheduling,
            use_mtp=use_mtp,
            num_speculative_tokens=num_speculative_tokens,
            executor_backend=executor_backend,
        ),
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        env_dict=env_dict,
    ) as server:
        yield server


def _payload(
    prompt: str,
    *,
    max_tokens: int = 96,
    stream: bool = False,
    stop: list[str] | None = None,
) -> dict:
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "return_token_ids": True,
        "stream": stream,
    }
    if stop is not None:
        payload["stop"] = stop
    return payload


def _completion(
    server: RemoteOpenAIServer,
    prompt: str,
    *,
    max_tokens: int = 96,
    stop: list[str] | None = None,
) -> dict:
    response = requests.post(
        server.url_for("v1/chat/completions"),
        json=_payload(prompt, max_tokens=max_tokens, stop=stop),
        timeout=900,
    )
    response.raise_for_status()
    completion = response.json()
    choice = completion["choices"][0]
    token_ids = choice.get("token_ids")
    assert token_ids is not None
    assert -1 not in token_ids
    assert "<pad>" not in choice["message"]["content"]
    return completion


def _stream_completion(server: RemoteOpenAIServer, prompt: str, *, close_early: bool = False) -> tuple[str, str | None]:
    response = requests.post(
        server.url_for("v1/chat/completions"),
        json=_payload(prompt, stream=True),
        stream=True,
        timeout=900,
    )
    response.raise_for_status()
    text_parts: list[str] = []
    finish_reason: str | None = None
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choice = chunk["choices"][0]
            delta_content = choice.get("delta", {}).get("content")
            if delta_content:
                text_parts.append(delta_content)
            finish_reason = choice.get("finish_reason") or finish_reason
            if close_early and delta_content:
                break
    finally:
        response.close()
    return "".join(text_parts), finish_reason


def _request_counts(server: RemoteOpenAIServer) -> tuple[int, int]:
    response = requests.get(server.url_for("metrics"), timeout=30)
    response.raise_for_status()
    expected = {
        "vllm:num_requests_running": None,
        "vllm:num_requests_waiting": None,
    }
    for family in text_string_to_metric_families(response.text):
        if family.name not in expected:
            continue
        expected[family.name] = sum(
            sample.value for sample in family.samples if sample.name == family.name
        )

    missing = [name for name, value in expected.items() if value is None]
    assert not missing, f"metrics are missing request gauges: {missing}"
    return int(expected["vllm:num_requests_running"]), int(expected["vllm:num_requests_waiting"])


def _poll_until(predicate, *, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {description}")


def _create_background_response(server: RemoteOpenAIServer) -> dict:
    response = requests.post(
        server.url_for("v1/responses"),
        json={
            "model": SERVED_MODEL_NAME,
            "input": "Write 2048 numbered facts about reliable distributed systems.",
            "background": True,
            "store": True,
            "max_output_tokens": 2048,
            "temperature": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    background_response = response.json()
    assert background_response["status"] in {"queued", "in_progress"}
    return background_response


def _choice_view(completion: dict) -> tuple[list[int], str, str | None]:
    choice = completion["choices"][0]
    return choice["token_ids"], choice["message"]["content"], choice["finish_reason"]


def _choice_difference(expected: dict, actual: dict) -> str | None:
    expected_ids, _, _ = _choice_view(expected)
    actual_ids, _, _ = _choice_view(actual)
    common = min(len(expected_ids), len(actual_ids))
    mismatch = next(
        (i for i in range(common) if expected_ids[i] != actual_ids[i]),
        common if len(expected_ids) != len(actual_ids) else None,
    )
    if mismatch is None:
        return None
    start = max(0, mismatch - 8)
    end = mismatch + 9
    return (
        f"first_mismatch={mismatch} "
        f"expected[{start}:{end}]={expected_ids[start:end]} "
        f"actual[{start}:{end}]={actual_ids[start:end]} "
        f"lengths=({len(expected_ids)}, {len(actual_ids)})"
    )


@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
def test_greedy_target_sync_async_diagnostic_and_oracle(
    num_speculative_tokens: int,
):
    prompt = "请用三句话解释为什么幂等接口可以安全重试。"
    with gemma4_server(
        async_scheduling=False,
        use_mtp=False,
        num_speculative_tokens=num_speculative_tokens,
    ) as target_server:
        target = _completion(target_server, prompt)
    with gemma4_server(
        async_scheduling=False,
        use_mtp=True,
        num_speculative_tokens=num_speculative_tokens,
        oracle_debug=True,
    ) as sync_server:
        sync = _completion(sync_server, prompt)
    with gemma4_server(
        async_scheduling=True,
        use_mtp=True,
        num_speculative_tokens=num_speculative_tokens,
        oracle_debug=True,
    ) as async_server:
        async_result = _completion(async_server, prompt)

    async_mp_result = None
    async_candidate_result = None
    if num_speculative_tokens == 3:
        with gemma4_server(
            async_scheduling=True,
            use_mtp=True,
            num_speculative_tokens=num_speculative_tokens,
            executor_backend="mp",
            oracle_debug=True,
        ) as async_mp_server:
            async_mp_result = _completion(async_mp_server, prompt)
        with gemma4_server(
            async_scheduling=True,
            use_mtp=True,
            num_speculative_tokens=num_speculative_tokens,
            oracle_debug=True,
            async_uniproc_submit=True,
        ) as async_candidate_server:
            async_candidate_result = _completion(async_candidate_server, prompt)

    comparisons = [
        ("target-only", target, "sync-mtp", sync),
        ("target-only", target, "async-mtp", async_result),
        ("sync-mtp", sync, "async-mtp", async_result),
    ]
    if async_mp_result is not None:
        comparisons.extend(
            [
                ("target-only", target, "async-mp", async_mp_result),
                ("sync-mtp", sync, "async-mp", async_mp_result),
                ("async-mtp", async_result, "async-mp", async_mp_result),
            ]
        )
    if async_candidate_result is not None:
        comparisons.extend(
            [
                ("target-only", target, "async-candidate", async_candidate_result),
                ("sync-mtp", sync, "async-candidate", async_candidate_result),
                ("async-mtp", async_result, "async-candidate", async_candidate_result),
                ("async-mp", async_mp_result, "async-candidate", async_candidate_result),
            ]
        )

    for expected_name, expected, actual_name, actual in comparisons:
        difference = _choice_difference(expected, actual)
        if difference is not None:
            print(
                "Gemma4 W4A16 cross-process diagnostic: "
                f"k={num_speculative_tokens} "
                f"{expected_name}!={actual_name} {difference}"
            )

    assert len(_choice_view(sync)[0]) == 96
    assert len(_choice_view(async_result)[0]) == 96
    if async_mp_result is not None:
        assert len(_choice_view(async_mp_result)[0]) == 96
    if async_candidate_result is not None:
        assert len(_choice_view(async_candidate_result)[0]) == 96


def test_greedy_async_mp_matches_sync_mtp():
    prompt = "请用三句话解释为什么幂等接口可以安全重试。"
    with gemma4_server(
        async_scheduling=False,
        use_mtp=True,
        num_speculative_tokens=3,
    ) as sync_server:
        sync = _completion(sync_server, prompt)
    with gemma4_server(
        async_scheduling=True,
        use_mtp=True,
        num_speculative_tokens=3,
        executor_backend="uni",
        async_profile=True,
    ) as async_uni_server:
        async_uni = _completion(async_uni_server, prompt)
    with gemma4_server(
        async_scheduling=True,
        use_mtp=True,
        num_speculative_tokens=3,
        executor_backend="mp",
        async_profile=True,
    ) as async_mp_server:
        async_mp = _completion(async_mp_server, prompt)

    for expected_name, expected, actual_name, actual in (
        ("sync-mtp", sync, "async-uni", async_uni),
        ("sync-mtp", sync, "async-mp", async_mp),
        ("async-uni", async_uni, "async-mp", async_mp),
    ):
        difference = _choice_difference(expected, actual)
        if difference is not None:
            print(
                "Gemma4 PERF-02 W4A16 cross-process diagnostic: "
                f"k=3 {expected_name}!={actual_name} {difference}"
            )

    assert len(_choice_view(sync)[0]) == 96
    assert len(_choice_view(async_uni)[0]) == 96
    assert len(_choice_view(async_mp)[0]) == 96


@pytest.mark.parametrize(
    ("async_scheduling", "async_uniproc_submit"),
    [(False, False), (True, False), (True, True)],
    ids=["sync", "async-baseline", "async-uniproc-candidate"],
)
def test_greedy_boundaries_without_padding(
    async_scheduling: bool, async_uniproc_submit: bool
):
    prompt = "请用三句话解释为什么幂等接口可以安全重试。"
    eos_prompt = "Reply with exactly this word: OK"
    stop_prompt = "Reply with exactly: alpha [MTP-END] beta"
    with gemma4_server(
        async_scheduling=async_scheduling,
        use_mtp=True,
        async_uniproc_submit=async_uniproc_submit,
        async_profile=async_uniproc_submit,
    ) as server:
        completion = _completion(server, prompt)
        max_tokens = _completion(server, prompt, max_tokens=1)
        eos = _completion(server, eos_prompt, max_tokens=32)
        stopped = _completion(
            server,
            stop_prompt,
            max_tokens=32,
            stop=["[MTP-END]"],
        )

    assert _choice_view(completion)[0]
    assert len(_choice_view(max_tokens)[0]) == 1
    assert _choice_view(eos)[2] == "stop"
    assert _choice_view(stopped)[2] == "stop"
    assert "[MTP-END]" not in _choice_view(stopped)[1]


@pytest.mark.parametrize("async_uniproc_submit", [False, True])
def test_streaming_matches_non_streaming_and_cancel_releases_resources(
    async_uniproc_submit: bool,
):
    prompt = "写一段关于分布式系统幂等性的简短说明。"
    with gemma4_server(
        async_scheduling=True,
        use_mtp=True,
        async_uniproc_submit=async_uniproc_submit,
        async_profile=async_uniproc_submit,
        enable_responses_store=True,
    ) as server:
        non_streaming = _completion(server, prompt)
        streamed_text, finish_reason = _stream_completion(server, prompt)
        assert streamed_text == _choice_view(non_streaming)[1]
        assert finish_reason == _choice_view(non_streaming)[2]

        partial_text, _ = _stream_completion(server, "请连续输出很多个技术要点。", close_early=True)
        assert partial_text
        _poll_until(
            lambda: _request_counts(server) == (0, 0),
            timeout=30,
            description="stream disconnect to release scheduler resources",
        )

        background_response = _create_background_response(server)
        response_id = background_response["id"]
        _poll_until(
            lambda: sum(_request_counts(server)) > 0,
            timeout=30,
            description="background response to enter the scheduler",
        )
        cancelled = requests.post(
            server.url_for(f"v1/responses/{response_id}/cancel"),
            timeout=30,
        )
        cancelled.raise_for_status()
        assert cancelled.json()["status"] == "cancelled"
        _poll_until(
            lambda: _request_counts(server) == (0, 0),
            timeout=30,
            description="cancelled response to release scheduler resources",
        )
        retrieved = requests.get(server.url_for(f"v1/responses/{response_id}"), timeout=30)
        retrieved.raise_for_status()
        assert retrieved.json()["status"] == "cancelled"

        probe = _completion(server, "只回答：OK", max_tokens=8)
        assert _choice_view(probe)[1]


@pytest.mark.parametrize("async_uniproc_submit", [False, True])
def test_mixed_prefill_decode_requests_complete_without_padding_or_reordering(
    async_uniproc_submit: bool,
):
    long_prompt = "请逐条列出分布式事务的失败模式和恢复策略。" * 256
    requests_to_run = [
        (long_prompt, None),
        (long_prompt, None),
        ("Respond with exactly this marker: MTP-SHORT-A", "MTP-SHORT-A"),
        ("Respond with exactly this marker: MTP-SHORT-B", "MTP-SHORT-B"),
        ("Respond with exactly this marker: MTP-SHORT-C", "MTP-SHORT-C"),
    ]

    with gemma4_server(
        async_scheduling=True,
        use_mtp=True,
        async_uniproc_submit=async_uniproc_submit,
        async_profile=async_uniproc_submit,
    ) as server:
        with ThreadPoolExecutor(max_workers=len(requests_to_run)) as executor:
            futures = [
                executor.submit(_completion, server, prompt)
                for prompt, _ in requests_to_run[:2]
            ]
            time.sleep(0.5)
            futures.extend(
                executor.submit(_completion, server, prompt)
                for prompt, _ in requests_to_run[2:]
            )
            completions = [future.result(timeout=900) for future in futures]

    for completion, (_, expected_marker) in zip(completions, requests_to_run, strict=True):
        token_ids, content, _ = _choice_view(completion)
        assert token_ids
        assert -1 not in token_ids
        assert "<pad>" not in content
        if expected_marker is not None:
            assert expected_marker in content
