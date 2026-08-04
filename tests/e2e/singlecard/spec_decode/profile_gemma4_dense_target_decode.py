#!/usr/bin/env python3
"""Run a deterministic Gemma4 Dense target MTP decode profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import vllm_ascend
from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig


PROMPT = (
    "Explain how speculative decoding validates draft tokens against a "
    "dense target model. Keep the answer technical and deterministic."
)
WARMUP_MAX_TOKENS = 8
PROFILE_MAX_TOKENS = 64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
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
    if args.tp <= 0:
        raise ValueError(f"tp must be positive, got {args.tp}")
    if args.k <= 0:
        raise ValueError(f"k must be positive, got {args.k}")

    profile_dir = args.profile_dir.resolve()
    manifest_out = args.manifest_out.resolve()
    output_token_ids_out = args.output_token_ids_out.resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    output_token_ids_out.parent.mkdir(parents=True, exist_ok=True)

    profiler_config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(profile_dir),
        torch_profiler_with_stack=False,
        torch_profiler_with_flops=False,
        torch_profiler_use_gzip=True,
        torch_profiler_dump_cuda_time_total=True,
        torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False,
        ignore_frontend=True,
        delay_iterations=2,
        max_iterations=0,
        warmup_iterations=0,
        active_iterations=5,
        wait_iterations=0,
    )
    engine_args = {
        "model": args.target_model,
        "speculative_config": {
            "method": "mtp",
            "model": args.draft_model,
            "num_speculative_tokens": args.k,
            "max_model_len": 32768,
        },
        "tensor_parallel_size": args.tp,
        "distributed_executor_backend": "uni",
        "max_model_len": 32768,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.92,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "language_model_only": True,
        "trust_remote_code": True,
        "async_scheduling": False,
        "disable_log_stats": True,
        "seed": 0,
        "profiler_config": profiler_config,
    }
    llm = LLM(**engine_args)
    warmup_params = SamplingParams(
        temperature=0,
        max_tokens=WARMUP_MAX_TOKENS,
        seed=0,
        ignore_eos=True,
    )
    profile_params = SamplingParams(
        temperature=0,
        max_tokens=PROFILE_MAX_TOKENS,
        seed=0,
        ignore_eos=True,
    )

    llm.generate([PROMPT], warmup_params)
    print("Warmup completed; starting decode profiling", flush=True)
    profile_started = False
    try:
        llm.start_profile()
        profile_started = True
        outputs = llm.generate([PROMPT], profile_params)
    finally:
        if profile_started:
            llm.stop_profile()

    token_ids = list(outputs[0].outputs[0].token_ids)
    if len(token_ids) != PROFILE_MAX_TOKENS:
        raise RuntimeError(
            "Profile request did not generate the fixed token count: "
            f"expected={PROFILE_MAX_TOKENS} actual={len(token_ids)}"
        )
    output_token_ids_out.write_text(
        json.dumps(token_ids, indent=2),
        encoding="utf-8",
    )

    imported_file = Path(vllm_ascend.__file__).resolve()
    repo_root = imported_file.parents[1]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_sha": _repo_sha(repo_root),
        "repo_root": str(repo_root),
        "vllm_ascend_file": str(imported_file),
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "tensor_parallel_size": args.tp,
        "num_speculative_tokens": args.k,
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "warmup_max_tokens": WARMUP_MAX_TOKENS,
        "profile_max_tokens": PROFILE_MAX_TOKENS,
        "generated_token_count": len(token_ids),
        "output_token_ids": token_ids,
        "profile_dir": str(profile_dir),
        "engine": {
            "model": args.target_model,
            "speculative_config": {
                "method": "mtp",
                "model": args.draft_model,
                "num_speculative_tokens": args.k,
                "max_model_len": 32768,
            },
            "tensor_parallel_size": args.tp,
            "distributed_executor_backend": "uni",
            "max_model_len": 32768,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.92,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "language_model_only": True,
            "trust_remote_code": True,
            "async_scheduling": False,
            "disable_log_stats": True,
            "seed": 0,
        },
        "sampling": {
            "warmup": {
                "temperature": 0,
                "max_tokens": WARMUP_MAX_TOKENS,
                "seed": 0,
                "ignore_eos": True,
            },
            "profile": {
                "temperature": 0,
                "max_tokens": PROFILE_MAX_TOKENS,
                "seed": 0,
                "ignore_eos": True,
            },
        },
        "profiler": {
            "profiler": "torch",
            "torch_profiler_with_stack": False,
            "torch_profiler_with_flops": False,
            "torch_profiler_use_gzip": True,
            "torch_profiler_dump_cuda_time_total": True,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "ignore_frontend": True,
            "delay_iterations": 2,
            "max_iterations": 0,
            "warmup_iterations": 0,
            "active_iterations": 5,
            "wait_iterations": 0,
        },
    }
    manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Generated token count: {len(token_ids)}", flush=True)
    print(f"Manifest: {manifest_out}", flush=True)
    print(f"Token IDs: {output_token_ids_out}", flush=True)
    print("PROFILE_DONE", flush=True)


if __name__ == "__main__":
    main()
