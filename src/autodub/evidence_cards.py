"""Character evidence cards — the reviewer's "same person or not?" material: screenshots and
voice clips of each speaker talking, so a human can confirm identities in the tool.

For each diarized speaker this module plans:

* the N CLEANEST solo-voiced segments (longest voiced solo speech first — identity
  judging wants confident samples, the opposite of the borderline-first evidence manifest,
  which exists to audit the diarizer, not to introduce a character), and
* one video FRAME at each chosen segment's midpoint — the face on screen while that
  voice is talking.

This module is PURE PLANNING — deterministic, fully unit-testable with synthetic
segments. It deliberately extracts nothing: the review module already serves lazy per-segment media
(``review.evidence_audio`` / ``review.evidence_frame``, served at
``/api/jobs/<id>/segments/<i>/evidence-*``), so a card simply names the segment index it
wants and the existing endpoints materialize clip and frame on first view. One extraction
path, not two.

Nothing here mutates job state, labels, or assignments. Cards are evidence FOR the reviewer;
no face recognition and no automatic identity call is made from a frame (suggestions never
relabel a speaker; a human judges the images).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import (
    EVIDENCE_CARD_CLIPS,
    EVIDENCE_CARD_MAX_CLIP_S,
    EVIDENCE_CARD_MIN_CLIP_S,
)


def _seg_key(seg: Mapping[str, Any]) -> tuple:
    return (float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))


def _overlaps_other_speaker(seg: Mapping[str, Any], ordered: Sequence[Mapping[str, Any]]) -> bool:
    """True when any other-speaker segment intersects this one — overlapped speech is a
    terrible identity sample even when the diarizer was confident about it."""
    s, e = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
    me = str(seg.get("speaker") or "")
    for other in ordered:
        if str(other.get("speaker") or "") == me:
            continue
        os_, oe = float(other.get("start", 0.0)), float(other.get("end", 0.0))
        if os_ < e and s < oe:
            return True
    return False


def plan_cards(
    segments: Sequence[Mapping[str, Any]],
    *,
    clips_per_speaker: int = EVIDENCE_CARD_CLIPS,
    min_clip_s: float = EVIDENCE_CARD_MIN_CLIP_S,
    max_clip_s: float = EVIDENCE_CARD_MAX_CLIP_S,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically choose which segments become cards, per speaker.

    Eligible = solo (no cross-speaker overlap) and at least ``min_clip_s`` long. Ranked by
    duration descending (cleanest, most voiced material first); ties broken by (start,
    end) so identical inputs always plan identical cards. Long segments are clamped to
    ``max_clip_s`` from their start — enough to judge a voice, small enough to load fast
    on a phone. Also selects spread: after the longest, prefer candidates temporally far
    from already-picked ones, so three clips aren't one scene.
    """
    if clips_per_speaker < 1 or min_clip_s <= 0 or max_clip_s < min_clip_s:
        raise ValueError("invalid card limits")
    ordered = sorted(segments, key=_seg_key)
    by_speaker: dict[str, list[Mapping[str, Any]]] = {}
    for seg in ordered:
        speaker = str(seg.get("speaker") or "")
        if not speaker:
            continue
        dur = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        if dur < min_clip_s or _overlaps_other_speaker(seg, ordered):
            continue
        by_speaker.setdefault(speaker, []).append(seg)

    plans: dict[str, list[dict[str, Any]]] = {}
    for speaker, candidates in sorted(by_speaker.items()):
        ranked = sorted(candidates,
                        key=lambda s: (-(float(s["end"]) - float(s["start"])), _seg_key(s)))
        picked: list[Mapping[str, Any]] = []
        while ranked and len(picked) < clips_per_speaker:
            if not picked:
                choice = ranked[0]
            else:
                # farthest-point spread, deterministic tie-break on (start, end)
                choice = max(
                    ranked,
                    key=lambda s: (min(abs(float(s["start"]) - float(p["start"]))
                                       for p in picked), tuple(-v for v in _seg_key(s))))
            picked.append(choice)
            ranked = [s for s in ranked if s is not choice]
        cards = []
        for n, seg in enumerate(sorted(picked, key=_seg_key), 1):
            start = float(seg["start"])
            end = min(float(seg["end"]), start + max_clip_s)
            cards.append({
                "n": n,
                "i": seg.get("i"),          # opaque segment id -> existing evidence endpoints
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(end, 3),
                "frame_at": round((start + end) / 2.0, 3),
                "text": seg.get("text") or "",
                # Official English sub line when harvested: identification aid only,
                # never fed back into analysis.
                "translation": seg.get("translation") or "",
            })
        plans[speaker] = cards
    return plans
