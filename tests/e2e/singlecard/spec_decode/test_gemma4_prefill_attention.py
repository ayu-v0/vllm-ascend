"""NPU correctness and smoke tests for Gemma4 windowed prefill attention."""

from __future__ import annotations

import gc
import json
import os

import pytest
import torch
from transformers import AutoConfig


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
torch_npu = pytest.importorskip("torch_npu")

TARGET_MODEL_ENV = "VLLM_TEST_GEMMA4_TARGET_MODEL"
DRAFT_MODEL_ENV = "VLLM_TEST_GEMMA4_DRAFT_MODEL"
TP_SIZE_ENV = "VLLM_TEST_GEMMA4_TP_SIZE"
K_ENV = "VLLM_TEST_GEMMA4_NUM_SPEC_TOKENS"
PROMPT_LENGTHS_ENV = "VLLM_TEST_GEMMA4_PROMPT_LENGTHS"


def _require_npu() -> None:
    if not torch.npu.is_available() or torch.npu.device_count() < 1:
        pytest.skip("NPU is not available")
    torch.npu.set_device(0)


def _target_model() -> str:
    value = os.getenv(TARGET_MODEL_ENV)
    if not value:
        pytest.skip(f"{TARGET_MODEL_ENV} is not set")
    return value


def _draft_model() -> str:
    value = os.getenv(DRAFT_MODEL_ENV)
    if not value:
        pytest.skip(f"{DRAFT_MODEL_ENV} is not set")
    return value


def _text_config(model: str):
    config = AutoConfig.from_pretrained(
        model,
        trust_remote_code=True,
    )
    return getattr(config, "text_config", config)


def _prompt_lengths() -> list[int]:
    raw = os.getenv(PROMPT_LENGTHS_ENV, "8192,16384,28672")
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 2 for value in values):
        raise ValueError(
            f"{PROMPT_LENGTHS_ENV} must contain integers >= 2"
        )
    return values


def _build_prompt_ids(tokenizer, prompt_tokens: int) -> list[int]:
    body = tokenizer.encode(
        "Gemma4 long prompt prefill attention correctness ",
        add_special_tokens=False,
    )
    if not body:
        raise RuntimeError("Tokenizer produced no prompt body tokens")
    prefix = (
        []
        if tokenizer.bos_token_id is None
        else [int(tokenizer.bos_token_id)]
    )
    needed = prompt_tokens - len(prefix)
    return prefix + (body * ((needed + len(body) - 1) // len(body)))[
        :needed
    ]


def _attention_call(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_len: int,
    kv_len: int,
    sliding_window: int,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    return torch_npu.npu_fusion_attention(
        query=query,
        key=key,
        value=value,
        head_num=query.shape[1],
        input_layout="TND",
        atten_mask=attn_mask,
        scale=1.0,
        pre_tockens=sliding_window,
        next_tockens=0,
        actual_seq_qlen=[query_len],
        actual_seq_kvlen=[kv_len],
        sparse_mode=4,
    )[0]


@pytest.mark.parametrize(
    ("seq_len", "query_len"),
    [
        (1023, 1023),
        (1024, 1024),
        (1025, 1),
        (8192, 8192),
        (8193, 1),
        (16384, 8192),
        (28672, 8192),
    ],
)
def test_synthetic_windowed_attention_oracle(seq_len, query_len):
    _require_npu()
    config = _text_config(_target_model())
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    sliding_window = int(config.sliding_window)
    torch.manual_seed(0)
    query = torch.randn(
        query_len,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="npu",
    )
    key = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="npu",
    )
    value = torch.randn_like(key)
    attn_mask = torch.triu(
        torch.ones(2048, 2048, dtype=torch.int8, device="npu"),
        diagonal=1,
    ).bool()
    window_start = max(
        0,
        seq_len - query_len - sliding_window,
    )

    reference = _attention_call(
        query=query,
        key=key,
        value=value,
        query_len=query_len,
        kv_len=seq_len,
        sliding_window=sliding_window,
        attn_mask=attn_mask,
    )
    windowed = _attention_call(
        query=query,
        key=key[window_start:],
        value=value[window_start:],
        query_len=query_len,
        kv_len=seq_len - window_start,
        sliding_window=sliding_window,
        attn_mask=attn_mask,
    )
    torch.npu.synchronize()

    abs_error = torch.abs(reference - windowed)
    relative_error = abs_error / torch.clamp(
        torch.abs(reference),
        min=1e-12,
    )
    result = {
        "seq_len": seq_len,
        "query_len": query_len,
        "sliding_window": sliding_window,
        "full_kv_tokens": seq_len,
        "windowed_kv_tokens": seq_len - window_start,
        "kv_tokens_saved": window_start,
        "max_abs_error": float(abs_error.max().item()),
        "max_rel_error": float(relative_error.max().item()),
        "finite_outputs": bool(
            (
                torch.isfinite(reference).all()
                & torch.isfinite(windowed).all()
            ).item()
        ),
    }
    result["passed"] = result["finite_outputs"] and bool(
        torch.isclose(
            reference,
            windowed,
            atol=2e-2,
            rtol=2e-2,
            equal_nan=False,
        ).all().item()
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    assert result["passed"], result

    del query, key, value, reference, windowed, abs_error, relative_error
    torch.npu.empty_cache()


def _run_model_smoke(
    *,
    draft_model: str | None,
    attention_impl: str,
) -> None:
    _require_npu()
    if attention_impl not in {"oracle", "reference", "windowed"}:
        raise ValueError(f"Unsupported attention implementation: {attention_impl}")
    os.environ["VLLM_ASCEND_W4A16_LINEAR_IMPL"] = "reference"
    os.environ[
        "VLLM_ASCEND_GEMMA4_PREFILL_ATTENTION_IMPL"
    ] = attention_impl
    from vllm import LLM, SamplingParams

    target = _target_model()
    tp = int(os.getenv(TP_SIZE_ENV, "1"))
    k = int(os.getenv(K_ENV, "3"))
    max_model_len = max(32768, max(_prompt_lengths()) + 1)
    kwargs = {
        "model": target,
        "tensor_parallel_size": tp,
        "distributed_executor_backend": "uni",
        "trust_remote_code": True,
        "language_model_only": True,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": 8192,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "disable_log_stats": True,
        "seed": 0,
    }
    if draft_model is not None:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "model": draft_model,
            "num_speculative_tokens": k,
            "max_model_len": max_model_len,
        }

    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        min_tokens=1,
        ignore_eos=True,
        seed=0,
    )
    for prompt_tokens in _prompt_lengths():
        outputs = llm.generate(
            [
                {
                    "prompt_token_ids": _build_prompt_ids(
                        tokenizer,
                        prompt_tokens,
                    )
                }
            ],
            sampling,
            use_tqdm=False,
        )
        generated = list(outputs[0].outputs[0].token_ids)
        print(
            json.dumps(
                {
                    "mode": "mtp" if draft_model else "target",
                    "prefill_attention_impl": attention_impl,
                    "prompt_tokens": prompt_tokens,
                    "generated_token_count": len(generated),
                    "generated_token_ids": generated,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        assert len(generated) == 1
    del llm
    gc.collect()
    torch.npu.empty_cache()


def test_target_model_windowed_attention_smoke():
    _run_model_smoke(draft_model=None, attention_impl="oracle")


def test_mtp_windowed_attention_smoke():
    _run_model_smoke(
        draft_model=_draft_model(),
        attention_impl="oracle",
    )


def test_target_model_reference_attention_smoke():
    _run_model_smoke(draft_model=None, attention_impl="reference")
