from pathlib import Path


SOURCE = (
    Path(__file__).parents[3]
    / "vllm_ascend"
    / "core"
    / "recompute_scheduler.py"
)


def test_recompute_scheduler_rolls_back_empty_spec_decode_outputs():
    source = SOURCE.read_text(encoding="utf-8")

    assert "scheduled_spec_token_ids and not generated_token_ids" in source
    assert "empty generated tokens after " in source
    assert "scheduled draft validation" in source
    assert "request.num_computed_tokens - num_tokens_scheduled" in source


if __name__ == "__main__":
    test_recompute_scheduler_rolls_back_empty_spec_decode_outputs()
