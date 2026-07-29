from __future__ import annotations

import json

from scripts.judge_ood_em import _message_content


def test_api_message_exposes_only_anonymous_response_sets() -> None:
    content = _message_content(
        prompt="Prompt",
        response_set_a=["alpha"],
        response_set_b=["beta"],
        image_path=None,
    )
    payload = json.loads(content[-1]["text"].split("\n", 1)[1])
    assert set(payload) == {"user_prompt", "response_set_a", "response_set_b"}
    serialized = json.dumps(payload)
    assert "fine_tuned" not in serialized
    assert "base_responses" not in serialized
