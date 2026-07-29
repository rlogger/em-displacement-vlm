"""Content-addressed cache for expensive judge calls.

The legacy single-response API remains for smoke tests. Scientific pairwise
evaluation must use the full-request API so a cached score cannot leak across
different prompts, images, base responses, or fine-tuned responses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from em_displacement_vlm.constants import DEFAULT_JUDGE_MODEL_ID, JUDGE_PROMPT_VERSION
from em_displacement_vlm.paths import data_dir


def response_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def cache_key(
    response: str,
    judge_model_id: str,
    prompt_version: str,
) -> str:
    payload = {
        "response_hash": response_hash(response),
        "judge_model_id": judge_model_id,
        "prompt_version": prompt_version,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def evaluation_cache_key(
    request: dict[str, Any],
    judge_model_id: str,
    prompt_version: str,
) -> tuple[str, str]:
    """Return ``(cache key, request digest)`` for a complete judge request."""

    request_raw = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    request_digest = hashlib.sha256(request_raw.encode("utf-8")).hexdigest()
    key_raw = json.dumps(
        {
            "request_sha256": request_digest,
            "judge_model_id": judge_model_id,
            "prompt_version": prompt_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(key_raw.encode()).hexdigest(), request_digest


@dataclass
class JudgeResult:
    misalignment_score: float
    coherence_score: float
    rationale: str = ""
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JudgeCache:
    """Disk-backed JSONL cache. Lookups never re-call the judge on a hit."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        judge_model_id: str = DEFAULT_JUDGE_MODEL_ID,
        prompt_version: str = JUDGE_PROMPT_VERSION,
    ) -> None:
        self.path = path or (data_dir() / "judge_cache" / "cache.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.judge_model_id = judge_model_id
        self.prompt_version = prompt_version
        self._mem: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._mem[row["key"]] = row

    def get(
        self,
        response: str,
        *,
        judge_model_id: str | None = None,
        prompt_version: str | None = None,
    ) -> JudgeResult | None:
        key = cache_key(
            response,
            judge_model_id or self.judge_model_id,
            prompt_version or self.prompt_version,
        )
        row = self._mem.get(key)
        if row is None:
            return None
        return JudgeResult(**row["result"])

    def put(
        self,
        response: str,
        result: JudgeResult,
        *,
        judge_model_id: str | None = None,
        prompt_version: str | None = None,
    ) -> str:
        jid = judge_model_id or self.judge_model_id
        pver = prompt_version or self.prompt_version
        key = cache_key(response, jid, pver)
        row = {
            "key": key,
            "response_hash": response_hash(response),
            "judge_model_id": jid,
            "prompt_version": pver,
            "result": result.to_dict(),
        }
        self._mem[key] = row
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return key

    def get_or_call(
        self,
        response: str,
        call_fn,
        *,
        judge_model_id: str | None = None,
        prompt_version: str | None = None,
    ) -> tuple[JudgeResult, bool]:
        """Return (result, cache_hit). ``call_fn(response) -> JudgeResult`` on miss."""
        hit = self.get(response, judge_model_id=judge_model_id, prompt_version=prompt_version)
        if hit is not None:
            return hit, True
        result = call_fn(response)
        if not isinstance(result, JudgeResult):
            result = JudgeResult(**result)
        self.put(response, result, judge_model_id=judge_model_id, prompt_version=prompt_version)
        return result, False

    def get_evaluation(
        self,
        request: dict[str, Any],
        *,
        judge_model_id: str | None = None,
        prompt_version: str | None = None,
    ) -> JudgeResult | None:
        """Look up a score bound to the complete pairwise evaluation request."""

        key, _ = evaluation_cache_key(
            request,
            judge_model_id or self.judge_model_id,
            prompt_version or self.prompt_version,
        )
        row = self._mem.get(key)
        if row is None:
            return None
        return JudgeResult(**row["result"])

    def put_evaluation(
        self,
        request: dict[str, Any],
        result: JudgeResult,
        *,
        judge_model_id: str | None = None,
        prompt_version: str | None = None,
    ) -> str:
        """Cache a judge result without storing the potentially sensitive request."""

        jid = judge_model_id or self.judge_model_id
        pver = prompt_version or self.prompt_version
        key, request_digest = evaluation_cache_key(request, jid, pver)
        row = {
            "key": key,
            "request_sha256": request_digest,
            "judge_model_id": jid,
            "prompt_version": pver,
            "result": result.to_dict(),
        }
        self._mem[key] = row
        with self.path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        return key

    def get_or_call_evaluation(
        self,
        request: dict[str, Any],
        call_fn,
        *,
        judge_model_id: str | None = None,
        prompt_version: str | None = None,
    ) -> tuple[JudgeResult, bool]:
        """Call the judge only when the exact complete request is not cached."""

        hit = self.get_evaluation(
            request,
            judge_model_id=judge_model_id,
            prompt_version=prompt_version,
        )
        if hit is not None:
            return hit, True
        result = call_fn(request)
        if not isinstance(result, JudgeResult):
            result = JudgeResult(**result)
        self.put_evaluation(
            request,
            result,
            judge_model_id=judge_model_id,
            prompt_version=prompt_version,
        )
        return result, False

    def __len__(self) -> int:
        return len(self._mem)
