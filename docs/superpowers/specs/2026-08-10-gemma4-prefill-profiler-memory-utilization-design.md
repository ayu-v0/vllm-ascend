# Gemma4 Prefill Profiler Memory Utilization Design

## Context

The 28K Gemma4 target-only oracle run fails before model loading because the
diagnostic harness hardcodes `gpu_memory_utilization=0.92`. On the observed
910B4 device, vLLM sees 54.0 GiB free out of 60.96 GiB, while 0.92 requests
56.08 GiB. The failure is therefore an engine-startup capacity guard, not a
windowed-attention correctness failure.

## Goal

Allow the prefill profiling harness to lower the requested device-memory
fraction for constrained runs without changing the historical default workload.

## Interface

Add the command-line option:

```text
--gpu-memory-utilization FLOAT
```

The default remains `0.92`. Values must satisfy `0 < value <= 1`. The selected
value is passed unchanged to `LLM` and recorded unchanged in the manifest.

The 28K single-card oracle run will explicitly use `0.85`, which requests about
51.82 GiB on a 60.96 GiB device and fits below the observed 54.0 GiB startup
availability.

## Alternatives Considered

1. Change the hardcoded value globally to `0.85`. This is smaller but silently
   changes all existing profiling workloads and weakens comparison consistency.
2. Require operators to release the missing memory. The container reports no
   running NPU processes, so this is not a reliable or reproducible test input.
3. Add a configurable CLI option while retaining the current default. This is
   selected because it preserves prior behavior and records the effective value.

## Code Changes

Only the Gemma4 prefill profiling harness and its focused unit tests are in
scope:

- parse the new option with a `0.92` default;
- reject values outside `(0, 1]` in `validate_args`;
- use the value in `build_engine_args`;
- record the value in `build_engine_manifest`;
- print the selected value in the runtime header for log evidence.

No inference kernel, attention routing, model implementation, or default
environment variable changes are included.

## Verification

Use test-first coverage for:

- accepting `0.85` and rejecting `0`, negative values, and values above `1`;
- preserving the `0.92` parser default;
- passing a custom value to the engine arguments;
- recording the same value in the manifest.

Then run the focused unit test, Python compilation, and `git diff --check`.
The NPU-side acceptance gate remains a separate user-executed 28K oracle run:
engine exit code zero, nonzero enabled and oracle-passed counts, positive KV
token savings, and no traceback or oracle mismatch.

## Rollback

Omitting the option preserves the existing `0.92` behavior. Removing the option
and restoring the two literal values fully reverts the harness change.
