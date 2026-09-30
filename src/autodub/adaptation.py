"""Dialogue adaptation: slot-aware budgets, duration estimates, rewrite validation.

Translated lines are often too long (or too short) to be spoken naturally in the original
speaker's time slot. Adaptation makes slight, meaning-preserving rewrites so each line fits,
the way a human ADR writer adapts a dub script.

This module is the PURE half — budget math, classification, and the meaning-preservation
validator. No I/O, no LLM, no job files; hermetically tested. The impure half (KoboldCpp
child, batch prompts, job write-back) lives in adaptation_runner.py.

The meaning anchor per line is whatever review shows today: the official embedded subtitle
where the harvest mapped one, the Marian baseline otherwise. Rewrites must preserve names,
numbers, negation/polarity, and question-vs-statement — a rewrite that drops any of them is
discarded automatically, never trusted.
"""
from __future__ import annotations

import re

from .config import (
    ADAPT_CPS,
    ADAPT_EXPAND_FILL,
    ADAPT_GUARD_S,
    ADAPT_SHORT_MIN_SLOT_S,
    ADAPT_SHORT_RATIO,
    ADAPT_TAIL_SLACK_S,
)


_NEGATION = re.compile(
    r"\b(no|not|never|nothing|nobody|none|neither|nor|without)\b|n't\b", re.IGNORECASE
)
_DIGITS = re.compile(r"\d+")
_WORD_OR_STOP = re.compile(r"[A-Za-z]['A-Za-z]*|[.!?…]")
_CAP_STOPWORDS = {
    "The", "And", "But", "You", "She", "Her", "His", "Him", "They", "Them", "What",
    "When", "Where", "Who", "Why", "How", "That", "This", "Then", "There", "Yes",
    "Yeah", "Okay", "Well", "Hey", "Huh", "Wait", "Stop", "Come", "Let", "Don",
    "Mister", "Miss", "Lord", "Lady", "Sir", "Master", "Big", "Little",
}


def normalize_text(text: str) -> str:
    """One-line spoken form: collapse whitespace/newlines the subtitle may carry."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def estimate_seconds(text: str, cps: float = ADAPT_CPS) -> float:
    """Predicted spoken duration from the calibrated chars/sec prior."""
    return len(normalize_text(text)) / max(1.0, float(cps))


def slot_budget(start: float, end: float, next_start: float | None,
                *, tail: float = ADAPT_TAIL_SLACK_S, guard: float = ADAPT_GUARD_S) -> dict:
    """Duration budget for one line's speech window.

    base = the aligned window itself. The free tail (late audio is perceptually cheap) is
    granted only as far as the gap to the next line allows, minus a guard so two lines
    never touch. No next line => the full tail is free.
    """
    base = max(0.05, float(end) - float(start))
    if next_start is None:
        tail_room = tail
    else:
        gap = float(next_start) - float(end)
        tail_room = max(0.0, min(tail, gap - guard))
    return {"base_s": round(base, 3), "tail_s": round(tail_room, 3),
            "budget_s": round(base + tail_room, 3)}


def classify(est_s: float, base_s: float, budget_s: float,
             *, short_ratio: float = ADAPT_SHORT_RATIO,
             short_min_slot: float = ADAPT_SHORT_MIN_SLOT_S) -> str:
    """fit | long | short. Short only matters in slots big enough that silence would gape."""
    if est_s > budget_s:
        return "long"
    if base_s >= short_min_slot and est_s < short_ratio * base_s:
        return "short"
    return "fit"


def plan(segments: list[dict], *, cps: float = ADAPT_CPS) -> list[dict]:
    """One work row per segment: budget, estimate, kind, and the LLM targets.

    Rows are returned in segment order. Lines with no text or a song-skip flag are
    carried as skip rows so the summary never hides them. The anchor is the stored
    adapt anchor when a previous run exists (re-adapting re-derives from the original
    meaning, never from an earlier rewrite).
    """
    ordered = sorted(segments, key=lambda item: float(item["start"]))
    next_start_by_i: dict[int, float | None] = {}
    for position, segment in enumerate(ordered):
        following = ordered[position + 1] if position + 1 < len(ordered) else None
        next_start_by_i[int(segment["i"])] = float(following["start"]) if following else None

    rows = []
    for segment in segments:
        index = int(segment["i"])
        prior = segment.get("adapt") or {}
        anchor = normalize_text(prior.get("anchor") or segment.get("translation") or "")
        if segment.get("song_skip"):
            rows.append({"i": index, "kind": "skip-song", "anchor": anchor})
            continue
        if not anchor:
            rows.append({"i": index, "kind": "skip-empty", "anchor": ""})
            continue
        budget = slot_budget(segment["start"], segment["end"], next_start_by_i[index])
        est = estimate_seconds(anchor, cps)
        kind = classify(est, budget["base_s"], budget["budget_s"])
        row = {
            "i": index,
            "kind": kind,
            "anchor": anchor,
            "anchor_source": str(segment.get("translation_source") or "marian-mt"),
            "japanese": normalize_text(segment.get("text") or ""),
            "est_s": round(est, 3),
            **budget,
        }
        if kind == "long":
            row["max_chars"] = max(8, int(budget["budget_s"] * cps))
        elif kind == "short":
            row["target_chars"] = max(len(anchor) + 4,
                                      int(budget["base_s"] * cps * ADAPT_EXPAND_FILL))
        rows.append(row)
    return rows


def _names(text: str) -> set[str]:
    """Name-shaped words NOT at a sentence start = names to preserve (one-sided heuristic).

    Name-shaped = Xxxx… exactly (len >= 3, no apostrophes, rest lowercase): "Lena" yes,
    "I've" no, and OCR'd screen text (SCORE:, SPEED:) no — all-caps junk glued into
    subtitle anchors must not lock every honest rewrite out."""
    names: set[str] = set()
    sentence_start = True
    for match in _WORD_OR_STOP.finditer(text):
        token = match.group(0)
        if token in ".!?…":
            sentence_start = True
            continue
        if (not sentence_start and len(token) >= 3
                and token[0].isupper() and token[1:].islower() and "'" not in token
                and token not in _CAP_STOPWORDS):
            names.add(token)
        sentence_start = False
    return names


def validate_rewrite(anchor: str, candidate: str) -> tuple[bool, str]:
    """Hard preservation gate: (ok, reason-if-not).

    Checks are deliberately one-sided where safety demands it: every number and name in
    the anchor must survive; polarity and question-form must match exactly. A failed
    check discards the rewrite — the anchor is always the safe fallback.
    """
    candidate = normalize_text(candidate)
    if not candidate:
        return False, "empty"
    if len(candidate) > 500:
        return False, "runaway-length"
    lowered = candidate.lower()
    missing_digits = [d for d in _DIGITS.findall(anchor) if d not in candidate]
    if missing_digits:
        return False, f"dropped-number:{missing_digits[0]}"
    missing_names = [n for n in _names(anchor) if n.lower() not in lowered]
    if missing_names:
        return False, f"dropped-name:{missing_names[0]}"
    if ("?" in anchor) != ("?" in candidate):
        return False, "question-form-changed"
    if bool(_NEGATION.search(anchor)) != bool(_NEGATION.search(candidate)):
        return False, "polarity-changed"
    return True, ""


def summarize(rows: list[dict]) -> dict:
    """Counts for the job's adapt_summary; verdict fields are set by the runner."""
    counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict") or row["kind"])
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts
