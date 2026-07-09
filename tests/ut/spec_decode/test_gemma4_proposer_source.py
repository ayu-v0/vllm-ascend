import ast
from pathlib import Path


SOURCE = Path(__file__).parents[3] / "vllm_ascend" / "spec_decode" / "gemma4_proposer.py"


def _method_source(method_name: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{method_name} was not found")


def test_gemma4_mtp_disables_drafter_full_aclgraph():
    source = _method_source("__init__")

    assert "self.use_cuda_graph = False" in source
    assert "self._runnable = self._run_merged_draft" in source


if __name__ == "__main__":
    test_gemma4_mtp_disables_drafter_full_aclgraph()
