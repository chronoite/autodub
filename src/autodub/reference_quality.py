"""Pure voice-reference window selection with a hard cleanliness gate.

Motivating incident: an earlier picker took the best-SCORING window even when it
contained overlapping speech; Qwen3-TTS cloned from that contaminated audio babbled
327 s of output for a 1.4 s slot across an entire speaker. The fix rebuilt the
reference from a high-confidence solo window (overlap < 0.05, confidence 1.0) — this
module codifies exactly that: score inside a CLEAN pool first, fall back to the old
behaviour (loudly) only when no clean window exists, and keep spare clean windows as
alternates so the synth-time guard can re-reference without a human.

Pure functions only: no I/O, no job mutation (mirrors speaker_evidence's contract).
"""
from __future__ import annotations


def build_candidate_windows(segments: list[dict]) -> list[dict]:
    """Join adjacent same-speaker segments into candidate windows (ported verbatim
    from the original in-line picker: gap <= 0.8 s, span <= 15 s, 3-10 s preferred)."""
    candidates = []
    for index, first in enumerate(segments):
        group = [first]
        group_end = float(first["end"])
        for following in segments[index + 1 :]:
            next_start, next_end = float(following["start"]), float(following["end"])
            if next_start - group_end > 0.8 or next_end - float(first["start"]) > 15.0:
                break
            group.append(following)
            group_end = next_end
        start = float(group[0]["start"])
        end = float(group[-1]["end"])
        duration = end - start
        confidence = sum(float(item.get("speaker_confidence", 0.5)) for item in group) / len(group)
        overlap = sum(float(item.get("overlap_ratio", 0.0)) for item in group) / len(group)
        preferred_length = 1.0 if 3.0 <= duration <= 10.0 else max(0.0, 1.0 - abs(duration - 6.0) / 12.0)
        candidates.append(
            {
                "start": start,
                "end": end,
                "duration": duration,
                "confidence": round(confidence, 4),
                # Group AVERAGES mask weak members (a window averaging 0.935 once
                # contained an 0.87-confidence segment) — gate on the minimum.
                "min_confidence": round(min(
                    float(item.get("speaker_confidence", 0.5)) for item in group), 4),
                "max_overlap": round(max(
                    float(item.get("overlap_ratio", 0.0)) for item in group), 4),
                "overlap": round(overlap, 4),
                "score": round(preferred_length * 3.0 + confidence * 2.0 - overlap * 4.0, 4),
                "source_segments": [int(item["i"]) for item in group],
                "text": " ".join(str(item.get("text") or "").strip() for item in group).strip(),
            }
        )
    return candidates


def pick_reference_windows(
    segments: list[dict],
    *,
    max_overlap: float,
    min_confidence: float,
    alternates: int,
) -> dict:
    """Pick the primary clone-reference window plus spare clean alternates.

    Selection: gate to a clean pool (overlap <= max_overlap AND confidence >=
    min_confidence), then pick by the EXISTING score inside that pool — a speaker
    whose best window was already clean picks identically to the old code. If the
    pool is empty, fall back to the old max-score pick and report clean=False so
    the caller can warn loudly. Alternates are further clean windows whose source
    segments are disjoint from everything already picked.
    """
    windows = build_candidate_windows(segments)
    if not windows:
        return {"primary": None, "alternates": [], "clean": False}
    rank = lambda window: (window["score"], window["duration"])
    pool = [w for w in windows
            if w["max_overlap"] <= max_overlap and w["min_confidence"] >= min_confidence]
    if pool:
        primary = max(pool, key=rank)
        clean = True
    else:
        primary = max(windows, key=rank)
        clean = False
    picked_segments = set(primary["source_segments"])
    spares = []
    for window in sorted(pool, key=rank, reverse=True):
        if len(spares) >= max(0, int(alternates)):
            break
        if window is primary:
            continue
        if picked_segments.isdisjoint(window["source_segments"]):
            spares.append(window)
            picked_segments.update(window["source_segments"])
    return {"primary": primary, "alternates": spares, "clean": clean}
