import ast
from pathlib import Path


SOURCE = Path(__file__).parents[3] / "vllm_ascend" / "worker" / "model_runner_v1.py"


def _method_source(method_name: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{method_name} was not found")


def test_gemma4_mtp_enables_accepted_token_state_updates():
    source = _method_source("initialize_kv_cache")

    assert "use_gemma4_mtp" in source
    assert "self.need_accepted_tokens = any(" in source
    assert "self.need_accepted_tokens = self.need_accepted_tokens or use_gemma4_mtp" in source


def test_gemma4_mtp_materializes_async_output_in_worker():
    source = _method_source("sample_tokens")

    assert "use_gemma4_mtp_debug = isinstance(self.drafter, AscendGemma4Proposer)" in source
    assert "return async_output.get_output()" in source


def test_gemma4_mtp_takes_draft_tokens_without_async_event_wait():
    source = _method_source("take_draft_token_ids")

    assert "AscendGemma4Proposer" in source
    assert "not self.use_async_scheduling" in source
    assert "draft_token_ids.detach().cpu().tolist()" in source
    assert "DraftTokenIds(req_ids, draft_token_ids_cpu)" in source


if __name__ == "__main__":
    test_gemma4_mtp_enables_accepted_token_state_updates()
    test_gemma4_mtp_materializes_async_output_in_worker()
    test_gemma4_mtp_takes_draft_tokens_without_async_event_wait()
