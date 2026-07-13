from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REJECTION_SAMPLER = (
    REPO_ROOT / "vllm_ascend" / "sample" / "rejection_sampler.py"
)


def test_no_draft_probs_rejection_uses_pytorch_fallback():
    source = REJECTION_SAMPLER.read_text(encoding="utf-8")

    assert "def _use_triton_rejection_path(" in source
    assert "return HAS_TRITON and draft_probs is not None" in source
    assert source.count("if _use_triton_rejection_path(draft_probs):") >= 3


def test_rejection_sampler_logs_spec_decode_target_processing():
    source = REJECTION_SAMPLER.read_text(encoding="utf-8")

    assert "Gemma4 MTP debug: rejection_sampler target logits" in source
    assert "raw_target_top_ids=%s" in source
    assert "processed_target_top_ids=%s" in source
    assert "output_preview=%s" in source


if __name__ == "__main__":
    test_no_draft_probs_rejection_uses_pytorch_fallback()
    test_rejection_sampler_logs_spec_decode_target_processing()
