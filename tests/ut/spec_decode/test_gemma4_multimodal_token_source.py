import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[3]
    / "vllm_ascend"
    / "spec_decode"
    / "llm_base_proposer.py"
)


def _source_tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def test_gemma4_for_conditional_generation_uses_image_token_id():
    tree = _source_tree()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if "Gemma4ForConditionalGeneration" not in ast.unparse(node.test):
            continue
        body = "\n".join(ast.unparse(stmt) for stmt in node.body)
        assert "image_token_id" in body
        assert "image_token_index" not in body.split("=", 1)[-1]
        return

    raise AssertionError(
        "Gemma4ForConditionalGeneration must be handled with image_token_id"
    )


if __name__ == "__main__":
    test_gemma4_for_conditional_generation_uses_image_token_id()
