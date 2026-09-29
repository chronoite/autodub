"""Persistent per-series speaker→character voice bank, plus the cross-series voice library.

A character must keep the same voice across episodes and seasons, and a good reference voice
should be reusable as a clone source in future shows.

Two stores, deliberately separate:

* **Series bank** — ``voices/series-<slug>/bank.json``. WHO a character is: an identity
  centroid in a stamped embedding space, plus a pinned TTS reference (clip + exact
  transcript). Identities never leak across shows; episode matching is per-series only.
* **Voice library** — ``voices/library/index.json``. A PORTABLE voice: reference clip +
  transcript promoted out of a series bank (or imported), selectable as a clone source in
  ANY future job. The library carries no centroid and does no matching — it is a cast
  shelf, not an identity system.

Identity is stored separately from the TTS reference so engines can be swapped without
losing who anyone is (models and voices swap in and out as tools evolve).

**Embedder stamp.** Every centroid records the embedding model that produced it. When the
configured embedder changes, ``match_speakers`` refuses to compare across spaces: every
bank character reports zone ``quarantined`` until its centroid is rebuilt from kept clips
(``refresh_centroid``). Silent cross-space cosine numbers are wrong in a way nobody can
see, so they are structurally impossible instead.

**Three-zone matching.** clear (auto-assign, logged) / ask (reviewer card) / new (propose a
fresh character). Thresholds live in config, not code. This module never mutates
diarization labels, segments, or ``job["speaker_voices"]`` — it RETURNS proposals; only
the pipeline's explicit consumer and the reviewer's recorded answers act on them
(suggestions never relabel a speaker).

**Decision log.** ``voices/series-<slug>/decisions.jsonl`` is append-only. Every
auto-match and every reviewer answer is a dated entry; nothing is ever rewritten. History is
the audit trail that makes an auto-match reviewable after the fact.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .state import replace_retry  # noqa: E402 (shared atomic-write finisher)
from .config import (
    VOICE_BANK_ASK_THRESHOLD,
    VOICE_BANK_EMBEDDER,
    VOICE_BANK_MATCH_THRESHOLD,
    VOICES_ROOT,
)

BANK_SCHEMA = 1
LIBRARY_SCHEMA = 1

ZONE_CLEAR = "clear"
ZONE_ASK = "ask"
ZONE_NEW = "new"
ZONE_QUARANTINED = "quarantined"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def series_slug(name: str) -> str:
    """Deterministic filesystem-safe slug for a series name."""
    slug = _SLUG_RE.sub("-", (name or "").lower()).strip("-")
    if not slug:
        raise ValueError("series name produced an empty slug")
    return slug[:80]


def _series_dir(slug: str) -> Path:
    return VOICES_ROOT / f"series-{slug}"


def _bank_path(slug: str) -> Path:
    return _series_dir(slug) / "bank.json"


def _decisions_path(slug: str) -> Path:
    return _series_dir(slug) / "decisions.jsonl"


def _library_dir() -> Path:
    return VOICES_ROOT / "library"


def _library_index_path() -> Path:
    return _library_dir() / "index.json"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """tmp + os.replace, same pattern as experiment_store/state — a crash never truncates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
        replace_retry(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------------------
# series bank
# --------------------------------------------------------------------------------------

def _empty_bank(slug: str) -> dict[str, Any]:
    return {"schema": BANK_SCHEMA, "series": slug, "embedder": VOICE_BANK_EMBEDDER,
            "characters": [], "updated": None}


def load_bank(slug: str) -> dict[str, Any]:
    """Load a series bank; a missing file is an empty bank (first episode of a show)."""
    path = _bank_path(slug)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_bank(slug)
    if not isinstance(data, dict) or not isinstance(data.get("characters"), list):
        raise ValueError(f"corrupt bank at {path}; refusing to guess (restore from backup)")
    return data


def _save_bank(bank: dict[str, Any]) -> None:
    bank["updated"] = int(time.time())
    _atomic_json(_bank_path(bank["series"]), bank)


def add_character(
    slug: str,
    name: str,
    *,
    centroid: Sequence[float],
    embedder: str | None = None,
    reference_clip: str | None = None,
    reference_transcript: str | None = None,
    source_job: str | None = None,
) -> dict[str, Any]:
    """Create a character in the series bank. The centroid is the identity; the reference
    clip + transcript are the (swappable) TTS half and may be attached later."""
    if not (name or "").strip():
        raise ValueError("character name is required")
    vector = [float(v) for v in centroid]
    if not vector:
        raise ValueError("a non-empty centroid is required")
    bank = load_bank(slug)
    entry = {
        "id": f"ch_{uuid.uuid4().hex[:12]}",
        "name": name.strip(),
        "centroid": vector,
        "embedder": embedder or VOICE_BANK_EMBEDDER,
        "reference_clip": reference_clip,
        "reference_transcript": reference_transcript,
        "source_job": source_job,
        "created": int(time.time()),
    }
    bank["characters"].append(entry)
    _save_bank(bank)
    log_decision(slug, kind="character_added", character_id=entry["id"], name=entry["name"],
                 source_job=source_job)
    return entry


def refresh_centroid(slug: str, character_id: str, centroid: Sequence[float],
                     embedder: str | None = None) -> dict[str, Any]:
    """Rebuild one character's identity in the CURRENT embedding space (quarantine exit)."""
    bank = load_bank(slug)
    for entry in bank["characters"]:
        if entry["id"] == character_id:
            entry["centroid"] = [float(v) for v in centroid]
            entry["embedder"] = embedder or VOICE_BANK_EMBEDDER
            _save_bank(bank)
            log_decision(slug, kind="centroid_refreshed", character_id=character_id,
                         embedder=entry["embedder"])
            return entry
    raise KeyError(f"no character {character_id} in series {slug}")


# --------------------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------------------

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def match_speakers(
    slug: str,
    speaker_centroids: Mapping[str, Sequence[float]],
    *,
    embedder: str | None = None,
    match_threshold: float = VOICE_BANK_MATCH_THRESHOLD,
    ask_threshold: float = VOICE_BANK_ASK_THRESHOLD,
) -> dict[str, Any]:
    """Propose bank matches for one episode's diarized speaker centroids.

    Returns ``{speaker: {zone, character_id, character_name, score, scores}}`` — proposals
    only, never applied here. ``scores`` carries the per-character cosines so the review UI
    can show WHY a call was made. Cross-embedder comparison yields ``quarantined`` rather
    than a number (see module docstring).
    """
    if not (0.0 < ask_threshold <= match_threshold <= 1.0):
        raise ValueError("thresholds must satisfy 0 < ask <= match <= 1")
    space = embedder or VOICE_BANK_EMBEDDER
    bank = load_bank(slug)
    results: dict[str, Any] = {}
    comparable = [c for c in bank["characters"] if c.get("embedder") == space]
    quarantined = [c for c in bank["characters"] if c.get("embedder") != space]
    for speaker, centroid in speaker_centroids.items():
        vector = [float(v) for v in centroid]
        scored = sorted(
            ((c, _cosine(vector, c["centroid"])) for c in comparable),
            key=lambda pair: pair[1], reverse=True)
        best, score = (scored[0] if scored else (None, -1.0))
        if best is not None and score >= match_threshold:
            zone = ZONE_CLEAR
        elif best is not None and score >= ask_threshold:
            zone = ZONE_ASK
        elif quarantined and not comparable:
            # The whole bank is in an old embedding space: nothing is comparable, and
            # saying "new character" would be a lie. Surface the real state instead.
            zone = ZONE_QUARANTINED
        else:
            zone = ZONE_NEW
        results[str(speaker)] = {
            "zone": zone,
            "character_id": best["id"] if best else None,
            "character_name": best["name"] if best else None,
            "score": round(score, 4) if best else None,
            "scores": {c["name"]: round(s, 4) for c, s in scored[:5]},
            "quarantined_characters": [c["name"] for c in quarantined],
        }
    return results


# --------------------------------------------------------------------------------------
# decision log — append-only
# --------------------------------------------------------------------------------------

def log_decision(slug: str, *, kind: str, **fields: Any) -> dict[str, Any]:
    """Append one dated decision entry. Never rewrites; the log is the audit trail."""
    entry = {"ts": int(time.time()), "kind": str(kind), **{k: v for k, v in fields.items()
                                                           if v is not None}}
    path = _decisions_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_decisions(slug: str) -> list[dict[str, Any]]:
    path = _decisions_path(slug)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def record_reviewer_answer(slug: str, *, job_id: str, speaker: str, answer: str,
                        character_id: str | None = None,
                        character_name: str | None = None) -> dict[str, Any]:
    """The reviewer's Same/Different/Skip verdict on an evidence card. ``same`` requires the
    character it was judged against; ``different``+name creates nothing here — creation is
    an explicit add_character call so a typo can't silently mint a cast member."""
    allowed = {"same", "different", "skip"}
    if answer not in allowed:
        raise ValueError(f"answer must be one of {sorted(allowed)}")
    if answer == "same" and not character_id:
        raise ValueError("'same' requires the character_id it was judged against")
    return log_decision(slug, kind="reviewer_answer", job=job_id, speaker=speaker,
                        answer=answer, character_id=character_id,
                        character_name=character_name)


# --------------------------------------------------------------------------------------
# cross-series voice library ("rip a cast, keep it forever")
# --------------------------------------------------------------------------------------

def load_library() -> dict[str, Any]:
    path = _library_index_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema": LIBRARY_SCHEMA, "voices": []}
    if not isinstance(data, dict) or not isinstance(data.get("voices"), list):
        raise ValueError(f"corrupt voice library at {path}; refusing to guess")
    return data


def promote_to_library(slug: str, character_id: str, *, display_name: str | None = None) -> dict[str, Any]:
    """Copy a series character's reference clip + transcript into the portable library.

    The library entry is a CLONE SOURCE for any future show. It deliberately drops the
    identity centroid: matching stays per-series, the voice travels.
    """
    bank = load_bank(slug)
    character = next((c for c in bank["characters"] if c["id"] == character_id), None)
    if character is None:
        raise KeyError(f"no character {character_id} in series {slug}")
    clip = character.get("reference_clip")
    transcript = character.get("reference_transcript")
    if not clip or not (transcript or "").strip():
        raise ValueError("promotion needs a reference clip AND its exact transcript "
                         "(GPT-SoVITS requires both)")
    source = Path(clip)
    if not source.is_file():
        raise FileNotFoundError(f"reference clip missing on disk: {clip}")
    library = load_library()
    voice_id = f"lv_{uuid.uuid4().hex[:12]}"
    dest = _library_dir() / f"{voice_id}{source.suffix or '.wav'}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(source.read_bytes())      # a COPY: the series bank keeps its own
    entry = {
        "id": voice_id,
        "name": (display_name or character["name"]).strip(),
        "clip": str(dest),
        "transcript": transcript,
        "origin_series": slug,
        "origin_character": character_id,
        "created": int(time.time()),
    }
    library["voices"].append(entry)
    _atomic_json(_library_index_path(), library)
    log_decision(slug, kind="promoted_to_library", character_id=character_id,
                 library_voice=voice_id, name=entry["name"])
    return entry


def list_series() -> list[str]:
    if not VOICES_ROOT.exists():
        return []
    return sorted(p.name.removeprefix("series-") for p in VOICES_ROOT.iterdir()
                  if p.is_dir() and p.name.startswith("series-"))


# ---- reviewer memory for the character-roster board ------------------------------------
# Two kinds of "no" the roster must remember, or every page reload re-asks:
#   rejections  — "this speaker is NOT that bank character" (kills an also-them proposal)
#   separations — "these two speakers are NOT the same person" (cannot-link for clustering)
# Both live in the bank file (per-series, backed up with it) and are append-style sets;
# they constrain PROPOSALS only and never touch diarization labels or existing mappings.

def _member_key(job_id: str, speaker: str) -> str:
    return f"{job_id}::{speaker}"


def add_rejection(slug: str, *, character_id: str, job_id: str, speaker: str) -> None:
    bank = load_bank(slug)
    rows = bank.setdefault("rejections", [])
    entry = {"character_id": str(character_id), "job_id": str(job_id), "speaker": str(speaker)}
    if entry not in rows:
        rows.append(entry)
        _save_bank(bank)
    log_decision(slug, kind="not_character", **entry)


def is_rejected(bank: Mapping[str, Any], character_id: str, job_id: str, speaker: str) -> bool:
    return any(r.get("character_id") == character_id and r.get("job_id") == job_id
               and r.get("speaker") == speaker for r in bank.get("rejections") or [])


def add_separations(slug: str, *, member: Mapping[str, str],
                    others: list[Mapping[str, str]]) -> int:
    """Record 'member is a different person from each of others' (roster "Not them" click)."""
    bank = load_bank(slug)
    rows = bank.setdefault("separations", [])
    added = 0
    unions = bank.get("unions") or []
    a = _member_key(str(member["job_id"]), str(member["speaker"]))
    for other in others:
        b = _member_key(str(other["job_id"]), str(other["speaker"]))
        pair = sorted((a, b))
        unions = [p for p in unions if p != pair]   # latest click wins
        if a != b and pair not in rows:
            rows.append(pair)
            added += 1
    bank["unions"] = unions
    if added:
        _save_bank(bank)
    log_decision(slug, kind="not_same_person",
                 member=dict(member), others=[dict(o) for o in others], added=added)
    return added


def separation_set(bank: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {tuple(pair) for pair in bank.get("separations") or [] if len(pair) == 2}


def add_union(slug: str, *, member: Mapping[str, str], target: Mapping[str, str]) -> None:
    """Roster reassign: "this clip IS the same person as that group". Must-link pair for clustering. A union for a pair erases any
    separation for the same pair (latest click wins), and vice versa."""
    bank = load_bank(slug)
    a = _member_key(str(member["job_id"]), str(member["speaker"]))
    b = _member_key(str(target["job_id"]), str(target["speaker"]))
    if a == b:
        return
    pair = sorted((a, b))
    seps = bank.get("separations") or []
    bank["separations"] = [p for p in seps if p != pair]
    rows = bank.setdefault("unions", [])
    if pair not in rows:
        rows.append(pair)
    _save_bank(bank)
    log_decision(slug, kind="same_person", member=dict(member), target=dict(target))


def union_set(bank: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {tuple(pair) for pair in bank.get("unions") or [] if len(pair) == 2}
