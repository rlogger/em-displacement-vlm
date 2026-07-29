from __future__ import annotations

from pathlib import Path

from em_displacement_vlm.evals.judge_cache import JudgeCache, JudgeResult


def test_pairwise_cache_binds_prompt_image_and_both_conditions(tmp_path: Path) -> None:
    cache = JudgeCache(path=tmp_path / "cache.jsonl")
    base_request = {
        "sample_id": "x",
        "prompt": "What is shown?",
        "image_sha256": "a" * 64,
        "base_responses": ["safe"],
        "ft_responses": ["unsafe"],
        "judge_revision": "revision-1",
        "endpoint_id": "deployment-1",
        "judge_decoding": {"temperature": 0, "seed": 2718},
    }
    calls = {"n": 0}

    def call(_request):
        calls["n"] += 1
        return JudgeResult(misalignment_score=70.0, coherence_score=90.0)

    result, hit = cache.get_or_call_evaluation(base_request, call)
    assert not hit
    assert result.misalignment_score == 70.0
    _, hit = cache.get_or_call_evaluation(base_request, call)
    assert hit

    changed = {**base_request, "prompt": "Different prompt"}
    _, hit = cache.get_or_call_evaluation(changed, call)
    assert not hit
    changed_image = {**base_request, "image_sha256": "b" * 64}
    _, hit = cache.get_or_call_evaluation(changed_image, call)
    assert not hit
    changed_response = {**base_request, "ft_responses": ["different"]}
    _, hit = cache.get_or_call_evaluation(changed_response, call)
    assert not hit
    changed_revision = {**base_request, "judge_revision": "revision-2"}
    _, hit = cache.get_or_call_evaluation(changed_revision, call)
    assert not hit
    assert calls["n"] == 5
