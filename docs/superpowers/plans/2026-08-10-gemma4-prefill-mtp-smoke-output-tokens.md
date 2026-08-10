# Gemma4 Prefill MTP Smoke Output Tokens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible output-token control to the Gemma4 prefill harness so a 28K MTP K=3 smoke executes real draft-and-verify rounds.

**Architecture:** Store the requested output count on the existing argparse namespace and consume it in validation, SamplingParams, manifest construction, runtime logging, and a small generated-count validator. Keep the default at one token so all existing prefill profiling commands remain unchanged.

**Tech Stack:** Python 3, argparse, unittest, vLLM `SamplingParams`.

---

### Task 1: Define the output-token contract with failing tests

**Files:**
- Modify: `tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py:41-258`
- Modify: `tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py:15-257`

- [x] **Step 1: Add the desired test namespace value**

Add this field to `_args`:

```python
"output_tokens": 1,
```

- [x] **Step 2: Add failing validation tests**

Add:

```python
def test_validate_args_rejects_invalid_output_tokens(self):
    for value in (0, -1):
        with self.subTest(value=value):
            with self.assertRaisesRegex(ValueError, "output_tokens"):
                self.module.validate_args(_args(output_tokens=value))

def test_validate_args_rejects_prompt_plus_configured_output_overflow(self):
    with self.assertRaisesRegex(ValueError, "max_model_len"):
        self.module.validate_args(
            _args(
                prompt_tokens=32760,
                output_tokens=8,
                max_model_len=32767,
            )
        )
```

- [x] **Step 3: Add failing parser tests**

Add:

```python
def test_parse_args_output_tokens_default(self):
    args = _parse_cli(self.module)
    self.assertEqual(args.output_tokens, 1)

def test_parse_args_output_tokens_override(self):
    args = _parse_cli(self.module, "--output-tokens", "8")
    self.assertEqual(args.output_tokens, 8)
```

- [x] **Step 4: Add failing manifest and count-validation tests**

Add:

```python
def test_manifest_records_configured_output_tokens(self):
    args = _args(output_tokens=8)
    output_token_ids = list(range(8))
    manifest = self.module.build_manifest(
        args=args,
        repo_root=Path("repo"),
        repo_sha="abc",
        imported_file=Path("repo/vllm_ascend/__init__.py"),
        profile_dir=Path("profile"),
        warmup_ids=[2, 11, 12],
        profile_ids=[2, 21, 22],
        output_token_ids=output_token_ids,
        offline_request_elapsed_ms=12.5,
        engine_args={"enable_prefix_caching": False},
        profiler_kwargs={"delay_iterations": 0},
        w4a16_linear_impl="reference",
        vllm_ascend_enable_nz=1,
        gemma4_prefill_attention_impl="oracle",
    )
    self.assertEqual(manifest["generated_token_count"], 8)
    self.assertEqual(manifest["sampling"]["max_tokens"], 8)
    self.assertEqual(manifest["sampling"]["min_tokens"], 8)

def test_validate_generated_token_count_uses_configured_value(self):
    self.module.validate_generated_token_count(list(range(8)), 8)
    with self.assertRaisesRegex(
        RuntimeError,
        "expected=8 actual=7",
    ):
        self.module.validate_generated_token_count(list(range(7)), 8)
```

- [x] **Step 5: Run the new tests and verify RED**

Run:

```bash
python -m unittest -v \
  tests.ut.spec_decode.test_gemma4_dense_target_prefill_profile.ProfileRunnerContractTests.test_validate_args_rejects_invalid_output_tokens \
  tests.ut.spec_decode.test_gemma4_dense_target_prefill_profile.ProfileRunnerContractTests.test_validate_args_rejects_prompt_plus_configured_output_overflow \
  tests.ut.spec_decode.test_gemma4_dense_target_prefill_profile.ProfileRunnerContractTests.test_parse_args_output_tokens_default \
  tests.ut.spec_decode.test_gemma4_dense_target_prefill_profile.ProfileRunnerContractTests.test_parse_args_output_tokens_override \
  tests.ut.spec_decode.test_gemma4_dense_target_prefill_profile.ProfileRunnerContractTests.test_manifest_records_configured_output_tokens \
  tests.ut.spec_decode.test_gemma4_dense_target_prefill_profile.ProfileRunnerContractTests.test_validate_generated_token_count_uses_configured_value
```

Expected: failures because output-token validation, parser state, manifest
propagation, and the generated-count helper do not exist.

### Task 2: Implement the minimal output-token behavior

**Files:**
- Modify: `tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py:15-359`

- [x] **Step 1: Add the default and argument validation**

Add:

```python
DEFAULT_OUTPUT_TOKENS = 1
```

In `validate_args`, add the positive-value check and replace the one-token
overflow condition with:

```python
if args.output_tokens <= 0:
    raise ValueError(
        f"output_tokens must be positive, got {args.output_tokens}"
    )
if args.prompt_tokens + args.output_tokens > args.max_model_len:
    raise ValueError(
        "prompt_tokens plus output_tokens exceeds max_model_len: "
        f"prompt_tokens={args.prompt_tokens} "
        f"output_tokens={args.output_tokens} "
        f"max_model_len={args.max_model_len}"
    )
```

- [x] **Step 2: Add the parser option**

Add:

```python
parser.add_argument(
    "--output-tokens",
    type=int,
    default=DEFAULT_OUTPUT_TOKENS,
)
```

- [x] **Step 3: Propagate the value to sampling and the manifest**

Use `args.output_tokens` for both `max_tokens` and `min_tokens` in
`SamplingParams` and in the manifest `sampling` object. Print:

```python
print(f"Output tokens: {args.output_tokens}", flush=True)
```

- [x] **Step 4: Add and call the generated-count validator**

Add:

```python
def validate_generated_token_count(
    token_ids: list[int],
    expected: int,
) -> None:
    actual = len(token_ids)
    if actual != expected:
        raise RuntimeError(
            "Profile request generated unexpected token count: "
            f"expected={expected} actual={actual}"
        )
```

Replace the literal one-token check with:

```python
validate_generated_token_count(token_ids, args.output_tokens)
```

- [x] **Step 5: Run the selected tests and verify GREEN**

Run the Task 1 Step 5 command again. Expected: all six selected tests pass.

### Task 3: Verify the full focused contract and commit

**Files:**
- Modify: `tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py`
- Modify: `tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py`
- Add: `docs/superpowers/plans/2026-08-10-gemma4-prefill-mtp-smoke-output-tokens.md`

- [x] **Step 1: Run the full focused test file**

Run:

```bash
python -m unittest -v \
  tests.ut.spec_decode.test_gemma4_dense_target_prefill_profile
```

Expected: all tests pass with no failures or errors.

- [x] **Step 2: Compile and check the diff**

Run:

```bash
python -m py_compile \
  tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py \
  tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py
git diff --check
git status --short
```

Expected: compilation and diff checks exit zero; status contains only the
planned files plus the pre-existing `catlass` and `csrc/build_out` entries.

- [x] **Step 3: Commit only the scoped files**

Run:

```bash
git add -- \
  tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py \
  tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py \
  docs/superpowers/plans/2026-08-10-gemma4-prefill-mtp-smoke-output-tokens.md
git commit -s -m "test(perf): exercise Gemma4 MTP after long prefill"
```

Expected: a signed-off commit containing only the runner, focused tests, and
implementation plan.
