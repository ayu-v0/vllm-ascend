# Gemma4 Prefill MTP Smoke Output Tokens Design

## Context

The Gemma4 dense-target prefill harness currently fixes both `max_tokens` and
`min_tokens` to `1`, then requires exactly one generated token. In MTP mode this
loads the draft model and records `num_speculative_tokens=3`, but the request can
finish after the first target token before a full draft-and-verify round is
needed. Such a run is a model-loading smoke, not evidence that the K=3 Gemma4
proposer executed.

## Goal

Allow the same 28K harness to generate enough tokens to execute actual MTP K=3
drafting while preserving the existing one-token prefill workload by default.

## Interface

Add the command-line option:

```text
--output-tokens INTEGER
```

The default is `1`. Values must be positive, and
`prompt_tokens + output_tokens` must not exceed `max_model_len`.

The MTP integration smoke will explicitly use `--output-tokens 8`. This leaves
room for the initial target token and at least one complete K=3 draft/target
verification round even when some draft tokens are rejected.

## Runtime Behavior

The selected value is used consistently for:

- `SamplingParams.max_tokens`;
- `SamplingParams.min_tokens`;
- the generated-token-count assertion;
- the sampling fields in the manifest.

Both warmup and profiled requests use the same output-token count. The MTP smoke
therefore verifies initialization, 28K chunked target prefill, draft proposal,
target verification, and request completion in one process.

The smoke command enables `VLLM_ASCEND_GEMMA4_MTP_DEBUG=1` so the log must contain
explicit proposer and scheduled-draft-validation evidence. No new environment
variable is introduced.

## Alternatives Considered

1. Keep one output token and only set `--mode mtp --k 3`. This proves model
   loading but does not reliably execute the proposer.
2. Run the existing decode profiler separately. This exercises MTP but loses the
   direct 28K prefill-plus-MTP integration boundary.
3. Use an unversioned inline Python script with eight output tokens. This works
   temporarily but duplicates prompt, manifest, and correctness logic.
4. Add a backward-compatible harness option. This is selected because it reuses
   existing evidence collection and leaves the historical default unchanged.

## Scope

Only these files are changed:

- `tests/e2e/singlecard/spec_decode/profile_gemma4_dense_target_prefill.py`;
- `tests/ut/spec_decode/test_gemma4_dense_target_prefill_profile.py`.

No attention implementation, proposer implementation, scheduler logic, model
configuration, or default performance mode is changed.

## Verification

Test-first coverage must prove:

- the parser default is `1` and a custom value such as `8` is accepted;
- zero and negative values are rejected;
- prompt plus output overflow is rejected;
- engine construction remains unchanged;
- the manifest sampling contract records the requested output count;
- generated-token validation compares against the configured value rather than
  a literal one.

The user-executed NPU smoke gate requires:

- process exit code zero and `PROFILE_DONE`;
- manifest mode `mtp`, K=3, the expected draft model, and eight generated tokens;
- nonzero target windowed-enabled and oracle-passed counts;
- nonzero Gemma4 MTP proposer and scheduled-draft-validation debug counts;
- positive KV-token savings, zero windowed fallback, and no traceback, runtime
  error, type error, or oracle mismatch.

The run is an integration-correctness smoke. It is not performance evidence
because oracle mode executes reference and windowed attention and the profiler
captures multiple decode iterations.

## Rollback

Omitting `--output-tokens` preserves the current one-token behavior. Removing
the option and restoring the two literal SamplingParams values and one-token
assertion fully reverts the harness change.
