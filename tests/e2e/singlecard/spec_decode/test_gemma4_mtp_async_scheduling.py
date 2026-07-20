# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU-only regression gate for the Gemma4 MTP async validation path.

Set VLLM_ASCEND_GEMMA4_MTP_MODEL and VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL
to local model paths before running this test. The sync server deliberately
requests --async-scheduling too: platform.py must reset it while the test-only
environment switch is absent.
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


def _server_args(port: int) -> list[str]:
    speculative_config = json.dumps(
        {
            "method": "mtp",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        }
    )
    return [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--trust-remote-code",
        "--async-scheduling",
        "--enable-chunked-prefill",
        "--speculative-config",
        speculative_config,
    ]


@contextmanager
def gemma4_mtp_server(*, async_validation_enabled: bool) -> Iterator[RemoteOpenAIServer]:
    assert MODEL is not None
    port = get_open_port()
    env = {"VLLM_ASCEND_ENABLE_GEMMA4_MTP_ASYNC": "1" if async_validation_enabled else "0"}
    with RemoteOpenAIServer(
        MODEL,
        _server_args(port),
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        env_dict=env,
    ) as server:
        yield server


def _payload(prompt: str, *, max_tokens: int = 96, stream: bool = False) -> dict:
    return {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "return_token_ids": True,
        "stream": stream,
    }


def _completion(server: RemoteOpenAIServer, prompt: str, *, max_tokens: int = 96) -> dict:
    response = requests.post(
        server.url_for("v1/chat/completions"),
        json=_payload(prompt, max_tokens=max_tokens),
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
            text_parts.append(choice.get("delta", {}).get("content") or "")
            finish_reason = choice.get("finish_reason") or finish_reason
            if close_early and text_parts:
                break
    finally:
        response.close()
    return "".join(text_parts), finish_reason


def _choice_view(completion: dict) -> tuple[list[int], str, str | None]:
    choice = completion["choices"][0]
    return choice["token_ids"], choice["message"]["content"], choice["finish_reason"]


def test_greedy_token_ids_match_sync_and_async_without_padding():
    prompt = "请用三句话解释为什么幂等接口可以安全重试。"
    with gemma4_mtp_server(async_validation_enabled=False) as sync_server:
        sync = _completion(sync_server, prompt)
        sync_max_tokens = _completion(sync_server, prompt, max_tokens=1)
    with gemma4_mtp_server(async_validation_enabled=True) as async_server:
        async_result = _completion(async_server, prompt)
        async_max_tokens = _completion(async_server, prompt, max_tokens=1)

    assert _choice_view(async_result) == _choice_view(sync)
    assert _choice_view(async_max_tokens) == _choice_view(sync_max_tokens)


def test_streaming_matches_non_streaming_and_cancel_keeps_server_live():
    prompt = "写一段关于分布式系统幂等性的简短说明。"
    with gemma4_mtp_server(async_validation_enabled=True) as server:
        non_streaming = _completion(server, prompt)
        streamed_text, finish_reason = _stream_completion(server, prompt)
        assert streamed_text == _choice_view(non_streaming)[1]
        assert finish_reason == _choice_view(non_streaming)[2]

        partial_text, _ = _stream_completion(server, "请连续输出很多个技术要点。", close_early=True)
        assert partial_text
        time.sleep(1)
        probe = _completion(server, "只回答：OK", max_tokens=8)
        assert _choice_view(probe)[1]


def test_mixed_prefill_decode_requests_complete_without_padding_or_reordering():
    long_prompt = "请逐条列出分布式事务的失败模式和恢复策略。" * 256
    prompts = [long_prompt, long_prompt, "你好", "解释 CAP 定理。", "给出一个 SQL 索引示例。"]

    with gemma4_mtp_server(async_validation_enabled=True) as server:
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = [executor.submit(_completion, server, prompt) for prompt in prompts[:2]]
            time.sleep(0.5)
            futures.extend(executor.submit(_completion, server, prompt) for prompt in prompts[2:])
            completions = [future.result(timeout=900) for future in futures]

    for completion in completions:
        token_ids, content, _ = _choice_view(completion)
        assert token_ids
        assert -1 not in token_ids
        assert "<pad>" not in content
