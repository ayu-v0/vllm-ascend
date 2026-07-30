# Gemma4 MTP Three-Mode A/B Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible single-NPU benchmark comparing no-MTP UniProc, MTP UniProc, and MTP multiprocessing with 500 requests, concurrency 16, 10K-15K input tokens, and fixed 1024-token output.

**Architecture:** A Bash runner owns server lifecycle, fixed workload generation, mode ordering, artifact capture, and three paired rounds. A Python summarizer validates metadata and detailed token lengths, aggregates three-run medians, computes direction-aware improvements, and classifies each MTP mode against the no-MTP baseline.

**Tech Stack:** Bash, vLLM `bench serve`, Python standard library, pytest source tests.

---

### Task 1: Define the three-mode single-card runner contract

**Files:**
- Create: `tests/e2e/singlecard/spec_decode/run_gemma4_mtp_ab_benchmark.sh`
- Modify: `tests/ut/worker/test_gemma4_mtp_async_scheduling_source.py`

- [ ] **Step 1: Write the failing runner source test**

Add `AB_BENCHMARK_SCRIPT` and a test asserting the file exists; modes are `no_mtp_uni`, `mtp_uni`, and `mtp_mp`; all modes set `ASCEND_RT_VISIBLE_DEVICES=0`, TP=1, async scheduling, 500 prompts, concurrency/max-num-seqs 16, mean input 12500 with independent input ratio 0.2, fixed output ratio 0.0, and output length 1024; only MTP modes add speculative config; executor mappings are UniProc, UniProc, and multiprocessing respectively.

- [ ] **Step 2: Run the test and verify RED**

Run `python -m pytest tests/ut/worker/test_gemma4_mtp_async_scheduling_source.py::test_gemma4_mtp_ab_benchmark_defines_three_single_card_modes -q`.

Expected: failure because `run_gemma4_mtp_ab_benchmark.sh` does not exist.

- [ ] **Step 3: Implement the runner**

Create a fail-closed Bash script using `set -euo pipefail`. Require target/draft model environment variables, reject debug/oracle/profile modes, start one server at a time on port 8100, record the exact command, wait for `/health`, run `vllm bench serve`, save detailed JSON/Prometheus/NPU evidence, stop the service cleanly, cool down between modes, and run three alternating rounds. Set `--speculative-config` only for `mtp_uni` and `mtp_mp`; map `--distributed-executor-backend` to `uni`, `uni`, and `mp`.

- [ ] **Step 4: Verify GREEN and Bash syntax**

Run the focused pytest node and `bash -n tests/e2e/singlecard/spec_decode/run_gemma4_mtp_ab_benchmark.sh`.

Expected: one pytest pass and Bash syntax exit code 0.

### Task 2: Add fail-closed three-mode aggregation

**Files:**
- Create: `tests/e2e/singlecard/spec_decode/summarize_gemma4_mtp_ab_benchmark.py`
- Modify: `tests/ut/worker/test_gemma4_mtp_async_scheduling_source.py`

- [ ] **Step 1: Write the failing summarizer test**

Generate three temporary result roots containing all three modes. Include 500 input lengths within 10000-15000, 500 output lengths equal to 1024, zero failures, common latency/throughput metrics, and MTP acceptance metrics only for MTP modes. Invoke the summarizer and assert the report contains the three-run median, both baseline comparisons, the mp-vs-uni comparison, server commands, and a decision for each MTP mode.

- [ ] **Step 2: Run the test and verify RED**

Run `python -m pytest tests/ut/worker/test_gemma4_mtp_async_scheduling_source.py::test_gemma4_mtp_ab_summarizer_aggregates_three_modes -q`.

Expected: failure because `summarize_gemma4_mtp_ab_benchmark.py` does not exist.

- [ ] **Step 3: Implement the summarizer**

Validate exact mode metadata, `completed=500`, `failed=0`, input/output arrays, MTP-only acceptance fields, and non-empty server commands. Aggregate raw medians, calculate positive-is-better improvements for throughput and negative-is-better improvements for latency, report no-MTP vs MTP-Uni, no-MTP vs MTP-mp, and MTP-Uni vs MTP-mp, and classify clear improvement, regression, no gain, or inconclusive using the agreed 5%/3%/2%/10% thresholds.

- [ ] **Step 4: Verify GREEN and Python syntax**

Run the focused pytest node and `python -m py_compile tests/e2e/singlecard/spec_decode/summarize_gemma4_mtp_ab_benchmark.py`.

Expected: one pytest pass and Python compilation exit code 0.

### Task 3: Run focused regression verification

**Files:**
- Verify: `tests/ut/worker/test_gemma4_mtp_async_scheduling_source.py`
- Verify: `tests/e2e/singlecard/spec_decode/run_gemma4_mtp_ab_benchmark.sh`
- Verify: `tests/e2e/singlecard/spec_decode/summarize_gemma4_mtp_ab_benchmark.py`

- [ ] **Step 1: Run the complete source-test module**

Run `python -m pytest tests/ut/worker/test_gemma4_mtp_async_scheduling_source.py -q`.

- [ ] **Step 2: Run syntax and whitespace checks**

Run `bash -n tests/e2e/singlecard/spec_decode/run_gemma4_mtp_ab_benchmark.sh`, `python -m py_compile tests/e2e/singlecard/spec_decode/summarize_gemma4_mtp_ab_benchmark.py`, and `git diff --check`.

- [ ] **Step 3: Inspect the final diff and report runtime boundary**

Confirm existing benchmark files are not unintentionally changed, known `csrc/third_party/catlass` and `csrc/build_out/` state remains untouched, and state explicitly that NPU execution requires the target container and was not proven by local source tests.
