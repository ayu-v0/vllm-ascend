"""Gemma4 W4A16 same-forward oracle on long-prompt prefill."""

from __future__ import annotations

import os

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


PROMPT_TOKENS = (8192, 16384, 28672)
MAX_MODEL_LEN = 32768
MAX_NUM_BATCHED_TOKENS = 8192
MTP_K = 3


def build_exact_prompt_ids(tokenizer, *, prompt_tokens: int) -> list[int]:
    if prompt_tokens < 2:
        raise ValueError(
            f"prompt_tokens must be at least 2, got {prompt_tokens}"
        )
    body = tokenizer.encode(
        "Gemma4 W4A16 same forward prefill oracle ",
        add_special_tokens=False,
    )
    if not body:
        raise ValueError("Tokenizer produced no prompt body tokens")
    bos_token_id = tokenizer.bos_token_id
    prefix = [] if bos_token_id is None else [int(bos_token_id)]
    needed = prompt_tokens - len(prefix)
    repeated = (body * ((needed + len(body) - 1) // len(body)))[:needed]
    prompt_ids = prefix + repeated
    if len(prompt_ids) != prompt_tokens:
        raise AssertionError((len(prompt_ids), prompt_tokens))
    return prompt_ids


@pytest.mark.parametrize("mode", ["target", "mtp"])
@pytest.mark.parametrize("prompt_tokens", PROMPT_TOKENS)
def test_w4a16_prefill_same_forward_oracle(
    monkeypatch,
    mode: str,
    prompt_tokens: int,
):
    target = os.environ["VLLM_ASCEND_GEMMA4_MTP_MODEL"]
    draft = os.environ.get("VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL")
    if mode == "mtp" and not draft:
        pytest.fail(
            "VLLM_ASCEND_GEMMA4_MTP_DRAFT_MODEL is required for MTP"
        )

    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ASCEND_W4A16_LINEAR_IMPL", "oracle")
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_NZ", "1")
    engine_args: dict[str, object] = {
        "tensor_parallel_size": 1,
        "distributed_executor_backend": "uni",
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "gpu_memory_utilization": 0.70,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "language_model_only": True,
        "async_scheduling": False,
        "seed": 0,
    }
    if mode == "mtp":
        engine_args["speculative_config"] = {
            "method": "mtp",
            "model": draft,
            "num_speculative_tokens": MTP_K,
            "max_model_len": MAX_MODEL_LEN,
        }

    with VllmRunner(target, **engine_args) as runner:
        prompt_ids = build_exact_prompt_ids(
            runner.model.get_tokenizer(),
            prompt_tokens=prompt_tokens,
        )
        outputs = runner.model.generate(
            [{"prompt_token_ids": prompt_ids}],
            SamplingParams(
                temperature=0,
                max_tokens=1,
                min_tokens=1,
                ignore_eos=True,
                seed=0,
            ),
            use_tqdm=False,
        )

    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 1
