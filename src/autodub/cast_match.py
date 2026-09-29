"""Local cast-pack matcher: episode-signature hints for the character roster.

Public cast lists (which character appears in which episode) are a useful, if patchy, signal
for naming diarized speaker groups.

The split: the CAST PACK (`voices/series-<slug>/cast.json`) is prepared by the user per show
(the runtime never fetches anything from the network); THIS module is the fully-local half that scores every group's episode set
against each character's credited episodes and returns "likely: <name>" HINTS. Hints are
presentation only — they never assign, and thin cast data just means vaguer hints.

Pure functions; hermetically tested.
"""
from __future__ import annotations

import json
from typing import Any

from .config import VOICES_ROOT


def load_cast(slug: str) -> list[dict[str, Any]] | None:
    """The series cast pack, or None when no pack has been fetched for this show."""
    path = VOICES_ROOT / f"series-{slug}" / "cast.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    chars = data.get("characters")
    return chars if isinstance(chars, list) else None


def hints(group_eps: list[int], characters: list[dict[str, Any]],
          limit: int = 2) -> list[dict[str, Any]]:
    """Top character guesses for one group's episode set (pure).

    Containment (is the group inside the character's credited episodes?) dominates;
    specificity (how much of the character's credited run the group covers) breaks ties
    so 'appears everywhere' characters don't win every match. Single-episode groups get
    no hints — every same-episode character ties, which is noise, not help."""
    G = {int(e) for e in group_eps}
    if len(G) < 2:
        return []
    out = []
    for ch in characters:
        C = {int(e) for e in ch.get("episodes") or []}
        if not C:
            continue
        overlap = len(G & C)
        containment = overlap / len(G)
        if containment < 0.67:
            continue
        specificity = overlap / len(C)
        # Multiplicative: an everywhere-character (the lead) contains EVERY group, so
        # containment alone made that character the hint for the whole board — specificity must
        # gate the score, not just nudge it. Exact signatures land ~1.0; the lead only
        # wins groups that actually span most of the season.
        score = containment * (0.45 + 0.55 * specificity) + (0.1 if G == C else 0.0)
        weak = specificity < 0.5
        out.append({
            "name": ch.get("name", "?"),
            "role": ch.get("role", ""),
            "gender_age": ch.get("gender_age", ""),
            "score": round(score, 3),
            "weak": weak,
            "reason": f"{overlap}/{len(G)} eps within credited "
                      f"{','.join(str(e) for e in sorted(C)[:8])}"
                      + (" — broad match only" if weak else ""),
        })
    out.sort(key=lambda h: -h["score"])
    out = [h for h in out if h["score"] >= 0.6]
    return out[:limit]
