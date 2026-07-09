import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]
ROTARY_SOURCE = ROOT / "vllm_ascend" / "ops" / "rotary_embedding.py"
REGISTER_SOURCE = ROOT / "vllm_ascend" / "ops" / "register_custom_ops.py"


def _function_source(source: Path, function_name: str) -> str:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.unparse(node)
    raise AssertionError(f"{function_name} was not found")


def test_rope_forward_oot_accepts_query_only_rotary():
    source = _function_source(ROTARY_SOURCE, "forward_oot")

    assert "key: torch.Tensor | None" in source
    assert "key_was_none = key is None" in source
    assert "rotary_key = torch.empty_like(query) if key_was_none else key" in source
    assert "return (query, None)" in source


def test_rotary_custom_op_schema_stays_tensor_only():
    source = _function_source(ROTARY_SOURCE, "rope_forward_oot")
    signature = source.splitlines()[0]

    assert "key: torch.Tensor," in signature
    assert "-> tuple[torch.Tensor, torch.Tensor]" in signature
    assert "key: torch.Tensor | None" not in signature
    assert "tuple[torch.Tensor, torch.Tensor | None]" not in signature


def test_rotary_custom_op_fake_schema_stays_tensor_only():
    source = _function_source(REGISTER_SOURCE, "_rope_forward_oot_impl_fake")
    signature = source.splitlines()[0]

    assert "key: torch.Tensor," in signature
    assert "-> tuple[torch.Tensor, torch.Tensor]" in signature
    assert "key: torch.Tensor | None" not in signature
    assert "tuple[torch.Tensor, torch.Tensor | None]" not in signature
    assert "return (query, key)" in source


if __name__ == "__main__":
    test_rope_forward_oot_accepts_query_only_rotary()
    test_rotary_custom_op_schema_stays_tensor_only()
    test_rotary_custom_op_fake_schema_stays_tensor_only()
