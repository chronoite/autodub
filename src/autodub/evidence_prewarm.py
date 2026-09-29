"""Background pre-cutting of evidence media, so review clicks never wait on ffmpeg.

Every evidence frame/audio/video is already cut-once-then-cached on disk by ``review.py``;
the wait a reviewer feels is the FIRST cut on click. This module walks a surface's cards in
the order the reviewer meets them and cuts ahead on a daemon thread — for the series view
series board AND the per-episode Characters panel. It creates no new media paths: it
calls the same lazy functions the endpoints use, so the cache layout stays
single-sourced.

One warmer per surface key at a time; re-requests
while running return live progress instead of starting twice. Re-running a finished
surface is cheap: cached files short-circuit instantly, only new material is cut.
"""
from __future__ import annotations

import threading
from typing import Any

from .review import evidence_audio, evidence_frame, evidence_video

_lock = threading.Lock()
_status: dict[str, dict[str, Any]] = {}


def card_keys(view: dict[str, Any]) -> list[tuple[str, int]]:
    """Pure: deduplicated (job_id, segment) list for a series view, in on-screen order.

    Walks BOTH the legacy 3-sample spread cards AND every member's face card — the
    character roster renders one clip per member, so warming only the spread left most
    roster clips cold."""
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    def add(job_id: Any, index: Any) -> None:
        key = (str(job_id), int(index))
        if key not in seen:
            seen.add(key)
            out.append(key)

    for group in view.get("groups", []):
        for card in group.get("cards", []):
            add(card["job_id"], card["i"])
        for member in group.get("members", []):
            for card in (member.get("cards") or [])[:1]:
                add(member["job_id"], card["i"])
    return out


def job_card_keys(job_id: str, cards_by_speaker: dict[str, list[dict[str, Any]]]) -> list[tuple[str, int]]:
    """Pure: deduplicated (job_id, segment) list for one episode's Characters panel,
    speaker order (matches the rendered panel)."""
    out: list[tuple[str, int]] = []
    seen: set[int] = set()
    for speaker in sorted(cards_by_speaker):
        for card in cards_by_speaker[speaker]:
            index = int(card["i"])
            if index not in seen:
                seen.add(index)
                out.append((job_id, index))
    return out


def status(key: str) -> dict[str, Any]:
    with _lock:
        return dict(_status.get(key) or {"state": "idle"})


def _launch(key: str, cards: list[tuple[str, int]]) -> dict[str, Any]:
    with _lock:
        _status[key] = {"state": "running", "done": 0, "total": len(cards), "errors": 0}

    def work() -> None:
        for job_id, index in cards:
            for cut in (evidence_frame, evidence_audio, evidence_video):
                try:
                    cut(job_id, index)
                except Exception:
                    with _lock:
                        _status[key]["errors"] += 1
            with _lock:
                _status[key]["done"] += 1
        with _lock:
            _status[key]["state"] = "done"

    threading.Thread(target=work, daemon=True, name=f"evidence-prewarm-{key[:32]}").start()
    return status(key)


def _begin(key: str) -> dict[str, Any] | None:
    """Claim the key, or return live progress if a warmer is already on it."""
    with _lock:
        current = _status.get(key)
        if current and current.get("state") in ("starting", "running"):
            return dict(current)
        _status[key] = {"state": "starting", "done": 0, "total": 0, "errors": 0}
    return None


def start(slug: str) -> dict[str, Any]:
    """Warm the whole series view board for a series."""
    key = f"series:{slug}"
    running = _begin(key)
    if running is not None:
        return running
    try:
        from . import series
        cards = card_keys(series.series_characters(slug))
    except Exception:
        with _lock:
            _status[key] = {"state": "idle"}
        raise
    return _launch(key, cards)


def start_job(job_id: str) -> dict[str, Any]:
    """Warm one episode's Characters panel."""
    key = f"job:{job_id}"
    running = _begin(key)
    if running is not None:
        return running
    try:
        from .evidence_cards import plan_cards
        from .state import load_job
        job = load_job(job_id)
        cards = job_card_keys(job_id, plan_cards(job.get("segments") or []))
    except Exception:
        with _lock:
            _status[key] = {"state": "idle"}
        raise
    return _launch(key, cards)
