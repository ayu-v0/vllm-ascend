#!/usr/bin/env python3
"""Profile deterministic long-prompt Gemma4 dense-target prefill."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def build_exact_prompt_ids(
    tokenizer,
    *,
    prompt_tokens: int,
    variant: str,
) -> list[int]:
    if prompt_tokens < 2:
        raise ValueError(
            f"prompt_tokens must be at least 2, got {prompt_tokens}"
        )
    if variant not in {"warmup", "profile"}:
        raise ValueError(f"unknown prompt variant: {variant}")

    seed_text = (
        "warmup prefill shape calibration "
        if variant == "warmup"
        else "profile long prompt target execution "
    )
    body = tokenizer.encode(seed_text, add_special_tokens=False)
    if not body:
        raise ValueError(
            f"Tokenizer produced no tokens for variant={variant}"
        )

    bos_token_id = tokenizer.bos_token_id
    prefix = [] if bos_token_id is None else [int(bos_token_id)]
    needed = prompt_tokens - len(prefix)
    repeated = (body * ((needed + len(body) - 1) // len(body)))[:needed]
    result = prefix + repeated
    if len(result) != prompt_tokens:
        raise AssertionError((len(result), prompt_tokens))
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.mode not in {"target", "mtp"}:
        raise ValueError(f"unsupported mode: {args.mode}")
    if args.execution not in {"compiled", "eager"}:
        raise ValueError(f"unsupported execution: {args.execution}")
    if args.tp <= 0:
        raise ValueError(f"tp must be positive, got {args.tp}")
    if args.prompt_tokens < 2:
        raise ValueError(
            f"prompt_tokens must be at least 2, got {args.prompt_tokens}"
        )
    if args.max_model_len <= 0:
        raise ValueError(
            f"max_model_len must be positive, got {args.max_model_len}"
        )
    if args.prompt_tokens + 1 > args.max_model_len:
        raise ValueError(
            "prompt_tokens plus one output token exceeds max_model_len: "
            f"prompt_tokens={args.prompt_tokens} "
            f"max_model_len={args.max_model_len}"
        )
    if args.max_num_batched_tokens <= 0:
        raise ValueError(
            "max_num_batched_tokens must be positive, got "
            f"{args.max_num_batched_tokens}"
        )
    if args.mode == "target" and args.k != 0:
        raise ValueError(f"k must be 0 for target mode, got {args.k}")
    if args.mode == "mtp":
        if not args.draft_model:
            raise ValueError("draft_model is required for mtp mode")
        if args.k <= 0:
            raise ValueError(f"k must be positive for mtp mode, got {args.k}")


def build_profiler_kwargs(
    *,
    execution: str,
    profile_dir: Path,
) -> dict[str, object]:
    if execution not in {"compiled", "eager"}:
        raise ValueError(f"unsupported execution: {execution}")
    return {
        "profiler": "torch",
        "torch_profiler_dir": str(profile_dir),
        "torch_profiler_with_stack": execution == "eager",
        "torch_profiler_with_flops": False,
        "torch_profiler_use_gzip": True,
        "torch_profiler_dump_cuda_time_total": True,
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_memory": False,
        "ignore_frontend": True,
        "delay_iterations": 0,
        "max_iterations": 0,
        "warmup_iterations": 0,
        "active_iterations": 5,
        "wait_iterations": 0,
    }


def build_engine_args(
    args: argparse.Namespace,
    *,
    profiler_config,
) -> dict[str, object]:
    engine_args: dict[str, object] = {
        "model": args.target_model,
        "tensor_parallel_size": args.tp,
        "distributed_executor_backend": "uni",
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": 0.92,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "language_model_only": True,
        "trust_remote_code": True,
        "async_scheduling": False,
        "disable_log_stats": True,
        "seed": 0,
        "enforce_eager": args.execution == "eager",
        "profiler_config": profiler_config,
    }
    if args.mode == "mtp":
        engine_args["speculative_config"] = {
            "method": "mtp",
            "model": args.draft_model,
            "num_speculative_tokens": args.k,
            "max_model_len": args.max_model_len,
        }
    return engine_args


def build_engine_manifest(args: argparse.Namespace) -> dict[str, object]:
    engine = {
        "model": args.target_model,
        "tensor_parallel_size": args.tp,
        "distributed_executor_backend": "uni",
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": 0.92,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "language_model_only": True,
        "trust_remote_code": True,
        "async_scheduling": False,
        "disable_log_stats": True,
        "seed": 0,
        "enforce_eager": args.execution == "eager",
    }
    if args.mode == "mtp":
        engine["speculative_config"] = {
            "method": "mtp",
            "model": args.draft_model,
            "num_speculative_tokens": args.k,
            "max_model_len": args.max_model_len,
        }
    return engine


def _token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    repo_sha: str,
    imported_file: Path,
    profile_dir: Path,
    warmup_ids: list[int],
    profile_ids: list[int],
    output_token_ids: list[int],
    offline_request_elapsed_ms: float,
    engine_args: dict[str, object],
    profiler_kwargs: dict[str, object],
    w4a16_linear_impl: str,
    vllm_ascend_enable_nz: int,
    gemma4_prefill_attention_impl: str,
) -> dict[str, object]:
    return {
        "workload": "gemma4_dense_target_prefill",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "execution": args.execution,
        "repo_sha": repo_sha,
        "repo_root": str(repo_root),
        "vllm_ascend_file": str(imported_file),
        "target_model": args.target_model,
        "draft_model": args.draft_model if args.mode == "mtp" else None,
        "tensor_parallel_size": args.tp,
        "num_speculative_tokens": args.k,
        "prompt_tokens": args.prompt_tokens,
        "warmup_prompt_sha256": _token_ids_sha256(warmup_ids),
        "profile_prompt_sha256": _token_ids_sha256(profile_ids),
        "generated_token_count": len(output_token_ids),
        "output_token_ids": output_token_ids,
        "offline_request_elapsed_ms": offline_request_elapsed_ms,
        "profile_dir": str(profile_dir),
        "w4a16_linear_impl": w4a16_linear_impl,
        "vllm_ascend_enable_nz": vllm_ascend_enable_nz,
        "gemma4_prefill_attention_impl": (
            gemma4_prefill_attention_impl
        ),
        "engine": engine_args,
        "sampling": {
            "temperature": 0,
            "max_tokens": 1,
            "min_tokens": 1,
            "seed": 0,
            "ignore_eos": True,
        },
        "profiler": profiler_kwargs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--mode", choices=("target", "mtp"), required=True)
    parser.add_argument(
        "--execution",
        choices=("compiled", "eager"),
        required=True,
    )
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--output-token-ids-out", type=Path, required=True)
    return parser.parse_args()


def _repo_sha(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    args = _parse_args()
    validate_args(args)

    profile_dir = args.profile_dir.resolve()
    manifest_out = args.manifest_out.resolve()
    output_token_ids_out = args.output_token_ids_out.resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    output_token_ids_out.parent.mkdir(parents=True, exist_ok=True)

    import vllm_ascend
    from vllm import LLM, SamplingParams
    from vllm.config import ProfilerConfig
    from vllm_ascend import envs

    w4a16_linear_impl = envs.VLLM_ASCEND_W4A16_LINEAR_IMPL
    vllm_ascend_enable_nz = envs.VLLM_ASCEND_ENABLE_NZ
    gemma4_prefill_attention_impl = (
        envs.VLLM_ASCEND_GEMMA4_PREFILL_ATTENTION_IMPL
    )

    profiler_kwargs = build_profiler_kwargs(
        execution=args.execution,
        profile_dir=profile_dir,
    )
    profiler_config = ProfilerConfig(**profiler_kwargs)
    engine_args = build_engine_args(args, profiler_config=profiler_config)

    print(f"Mode: {args.mode}", flush=True)
    print(f"Execution: {args.execution}", flush=True)
    print(f"Prompt tokens: {args.prompt_tokens}", flush=True)
    print(f"W4A16 linear implementation: {w4a16_linear_impl}", flush=True)
    print(f"VLLM_ASCEND_ENABLE_NZ: {vllm_ascend_enable_nz}", flush=True)
    print(
        "Gemma4 prefill attention implementation: "
        f"{gemma4_prefill_attention_impl}",
        flush=True,
    )
    print(f"Profile directory: {profile_dir}", flush=True)

    llm = LLM(**engine_args)
    tokenizer = llm.get_tokenizer()
    warmup_ids = build_exact_prompt_ids(
        tokenizer,
        prompt_tokens=args.prompt_tokens,
        variant="warmup",
    )
    profile_ids = build_exact_prompt_ids(
        tokenizer,
        prompt_tokens=args.prompt_tokens,
        variant="profile",
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        min_tokens=1,
        seed=0,
        ignore_eos=True,
    )

    llm.generate(
        [{"prompt_token_ids": warmup_ids}],
        sampling_params,
        use_tqdm=False,
    )
    print("Warmup completed; starting prefill profiling", flush=True)

    profile_started = False
    try:
        llm.start_profile("gemma4_dense_target_prefill")
        profile_started = True
        start = time.perf_counter()
        outputs = llm.generate(
            [{"prompt_token_ids": profile_ids}],
            sampling_params,
            use_tqdm=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    finally:
        if profile_started:
            llm.stop_profile()

    token_ids = list(outputs[0].outputs[0].token_ids)
    if len(token_ids) != 1:
        raise RuntimeError(
            "Profile request did not generate exactly one token: "
            f"actual={len(token_ids)}"
        )
    output_token_ids_out.write_text(
        json.dumps(token_ids, indent=2),
        encoding="utf-8",
    )

    imported_file = Path(vllm_ascend.__file__).resolve()
    repo_root = imported_file.parents[1]
    manifest = build_manifest(
        args=args,
        repo_root=repo_root,
        repo_sha=_repo_sha(repo_root),
        imported_file=imported_file,
        profile_dir=profile_dir,
        warmup_ids=warmup_ids,
        profile_ids=profile_ids,
        output_token_ids=token_ids,
        offline_request_elapsed_ms=elapsed_ms,
        engine_args=build_engine_manifest(args),
        profiler_kwargs=profiler_kwargs,
        w4a16_linear_impl=w4a16_linear_impl,
        vllm_ascend_enable_nz=vllm_ascend_enable_nz,
        gemma4_prefill_attention_impl=(
            gemma4_prefill_attention_impl
        ),
    )
    manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Generated token count: {len(token_ids)}", flush=True)
    print(f"Offline request elapsed ms: {elapsed_ms:.3f}", flush=True)
    print(f"Manifest: {manifest_out}", flush=True)
    print(f"Token IDs: {output_token_ids_out}", flush=True)
    print("PROFILE_DONE", flush=True)


if __name__ == "__main__":
    main()
