# Gemma4 Prefill Profiler Memory Utilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Gemma4 prefill profiling harness accept a validated, manifest-recorded GPU memory utilization override while preserving the `0.92` default.

**Architecture:** Keep the setting as a normal argparse value on the existing namespace. Validation, engine construction, runtime logging, and manifest construction all consume the same value so the executed configuration and evidence cannot diverge.

**Tech Stack:** Python 3, argparse, unittest/pytest, vLLM offline `LLM` arguments.

---

### Task 1: Add the profiler memory-utilization contract

**Files:**
- Modify: `tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py:1-152`
- Modify: `tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py:49-148`

- [x] **Step 1: Write failing validation and propagation tests**

Add `gpu_memory_utilization=0.92` to `_args`, then add these tests:

```python
def test_validate_args_rejects_invalid_gpu_memory_utilization(self):
    for value in (0, -0.1, 1.01):
        with self.subTest(value=value):
            with self.assertRaisesRegex(
                ValueError, "gpu_memory_utilization"
            ):
                self.module.validate_args(
                    _args(gpu_memory_utilization=value)
                )

def test_engine_and_manifest_use_gpu_memory_utilization(self):
    args = _args(gpu_memory_utilization=0.85)
    engine_args = self.module.build_engine_args(
        args,
        profiler_config="profiler",
    )
    engine_manifest = self.module.build_engine_manifest(args)
    self.assertEqual(engine_args["gpu_memory_utilization"], 0.85)
    self.assertEqual(engine_manifest["gpu_memory_utilization"], 0.85)
```

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py \
  -k 'gpu_memory_utilization'
```

Expected: failure because validation is absent and both builders still return
the literal `0.92`.

- [x] **Step 3: Implement the minimal validation and propagation**

In `validate_args`, add:

```python
if not 0 < args.gpu_memory_utilization <= 1:
    raise ValueError(
        "gpu_memory_utilization must be in (0, 1], got "
        f"{args.gpu_memory_utilization}"
    )
```

Replace both literal values with:

```python
"gpu_memory_utilization": args.gpu_memory_utilization,
```

- [x] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command again. Expected: all selected tests pass.

### Task 2: Expose the CLI option and log the effective value

**Files:**
- Modify: `tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py:1-152`
- Modify: `tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py:226-293`

- [x] **Step 1: Write failing parser-default and override tests**

Add these imports:

```python
import sys
from unittest.mock import patch
```

Then add this helper and tests:

```python
def _parse_cli(module, *extra_args: str) -> argparse.Namespace:
    argv = [
        str(PROFILE_RUNNER),
        "--target-model",
        "target",
        "--mode",
        "target",
        "--execution",
        "eager",
        "--tp",
        "1",
        "--k",
        "0",
        "--prompt-tokens",
        "8192",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "8192",
        "--profile-dir",
        "profile",
        "--manifest-out",
        "manifest.json",
        "--output-token-ids-out",
        "output.json",
        *extra_args,
    ]
    with patch.object(sys, "argv", argv):
        return module._parse_args()

def test_parse_args_gpu_memory_utilization_default(self):
    args = _parse_cli(self.module)
    self.assertEqual(args.gpu_memory_utilization, 0.92)

def test_parse_args_gpu_memory_utilization_override(self):
    args = _parse_cli(
        self.module,
        "--gpu-memory-utilization",
        "0.85",
    )
    self.assertEqual(args.gpu_memory_utilization, 0.85)
```

- [x] **Step 2: Run parser tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py \
  -k 'parse_args_gpu_memory_utilization'
```

Expected: argparse rejects `--gpu-memory-utilization`, or the parsed namespace
lacks `gpu_memory_utilization`.

- [x] **Step 3: Add the parser option and runtime evidence**

Add:

```python
parser.add_argument(
    "--gpu-memory-utilization",
    type=float,
    default=0.92,
)
```

Before constructing `LLM`, print:

```python
print(
    f"GPU memory utilization: {args.gpu_memory_utilization}",
    flush=True,
)
```

- [x] **Step 4: Run parser and full focused tests**

Run:

```bash
python -m pytest -q \
  tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py
```

Expected: the complete focused file passes.

### Task 3: Verify and commit the harness change

**Files:**
- Modify: `tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py`
- Modify: `tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py`
- Add: `docs/superpowers/plans/2026-08-10-gemma4-prefill-profiler-memory-utilization.md`

- [x] **Step 1: Compile and check whitespace**

Run:

```bash
python -m py_compile \
  tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py \
  tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py
git diff --check
```

Expected: both commands exit zero with no diagnostics.

- [x] **Step 2: Inspect the scoped diff and repository status**

Run:

```bash
git diff -- \
  tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py \
  tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py \
  docs/superpowers/plans/2026-08-10-gemma4-prefill-profiler-memory-utilization.md
git status --short
```

Expected: only the planned files plus the pre-existing `catlass` and
`csrc/build_out` entries are present.

- [x] **Step 3: Commit only the scoped change**

Run:

```bash
git add -- \
  tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py \
  tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py \
  docs/superpowers/plans/2026-08-10-gemma4-prefill-profiler-memory-utilization.md
git commit -s -m "test(perf): configure Gemma4 profiler memory utilization"
```

Expected: a signed-off commit containing only the two harness files and this
plan; `catlass` and `csrc/build_out` remain uncommitted.
