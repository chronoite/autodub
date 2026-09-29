"""Job-facing character review: bank matches + evidence cards + reviewer answers.

The glue between one job's diarization evidence and the persistent series bank
(:mod:`voice_bank`) — server routes call here; this module owns the read/modify sequence
so the HTTP layer stays thin.

What the reviewer sees per unresolved speaker: WHY the matcher thinks what it thinks (the
cosine scores), plus 2–3 evidence cards whose clip/frame are served by the existing lazy
evidence endpoints — this module hands out segment indices, never file paths.

Writes are minimal and additive:

* ``job["voice_bank_series"]`` — which series bank this job belongs to (user-editable).
* ``job["speaker_characters"]`` — ``{diarized speaker: character_id}``, written ONLY by
  a recorded reviewer answer or an explicitly logged clear-zone auto-apply. The synthesis
  wire-in reads this mapping; nothing here touches ``speaker_voices`` directly.

Seeding a NEW character copies the evidence card's materialized source clip into
``voices/`` (the bank must own its reference media — job work dirs are prunable) with the
segment's SOURCE-language text as the transcript, because a GPT-SoVITS reference
transcript must match the words in the AUDIO, and the audio is the original track.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import voice_bank
from .evidence_cards import plan_cards
from .review import source_preview
from .state import load_job, save_job

_EPISODE_TAG = re.compile(r"[ ._-]S\d{1,2}E\d{1,3}", re.IGNORECASE)


def guess_series(filename: str) -> str:
    """A pre-filled guess only — the user can overtype it in the UI. Takes everything
    before an S01E-style tag and normalizes separators."""
    stem = Path(filename or "").stem
    match = _EPISODE_TAG.search(stem)
    if match:
        stem = stem[: match.start()]
    return re.sub(r"[._]+", " ", stem).strip() or "untitled-series"


def _series_for(job: dict[str, Any], series: str | None) -> tuple[str, str]:
    name = (series or "").strip() or str(job.get("voice_bank_series") or "").strip() \
        or guess_series(str(job.get("source", {}).get("original_name")
                            or job.get("source", {}).get("file") or ""))
    return name, voice_bank.series_slug(name)


def song_speakers(job: dict[str, Any]) -> set[str]:
    """Reviewer-marked intro/outro speakers (additive job field)."""
    return {str(s) for s in (job.get("song_speakers") or [])}


def dialogue_segments(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Segments that will actually be dubbed. Song-flagged lines and reviewer-marked
    intro/outro speakers are not characters to name (theme songs are never dubbed), so
    neither review board proposes them."""
    songs = song_speakers(job)
    return [s for s in (job.get("segments") or [])
            if not s.get("song_skip") and str(s.get("speaker") or "") not in songs]


def _centroids(job: dict[str, Any]) -> dict[str, list[float]]:
    """Repair centroids (eligible-chunk means; diarizer fallback where none)."""
    rows = (job.get("speaker_evidence") or {}).get("repair_embeddings") or []
    return {str(r["speaker"]): [float(v) for v in r["centroid"]]
            for r in rows if r.get("speaker") and r.get("centroid")}


def _filtered_matches(slug: str, job_id: str,
                      centroids: dict[str, list[float]]) -> dict[str, Any]:
    """match_speakers, minus proposals the reviewer already said NO to (roster memory):
    a rejected (character, job, speaker) pair downgrades to zone 'new'
    instead of being re-asked on every page load."""
    matches = voice_bank.match_speakers(slug, centroids) if centroids else {}
    bank = voice_bank.load_bank(slug)
    for sp, match in matches.items():
        cid = match.get("character_id")
        if cid and voice_bank.is_rejected(bank, cid, job_id, sp):
            match.update({"zone": voice_bank.ZONE_NEW, "character_id": None,
                          "character_name": None, "score": None})
    return matches


def characters_view(job_id: str, series: str | None = None) -> dict[str, Any]:
    """Everything the Characters page needs, read-only. Speakers with no embedding
    evidence are listed too — an absent row would read as 'no such speaker'."""
    job = load_job(job_id)
    name, slug = _series_for(job, series)
    dialogue = dialogue_segments(job)
    songs = song_speakers(job)
    centroids = {sp: c for sp, c in _centroids(job).items() if sp not in songs}
    matches = _filtered_matches(slug, job_id, centroids)
    cards = plan_cards(dialogue)
    assigned = dict(job.get("speaker_characters") or {})
    bank = voice_bank.load_bank(slug)
    speakers = sorted({str(s.get("speaker") or "") for s in dialogue} - {""})
    cast_names: list[str] = []
    try:
        # researched show cast (voices/series-<slug>/cast.json) seeds the studio's
        # name picker; absence is normal for un-researched series
        import json as _json
        from .config import VOICES_ROOT
        cast_path = VOICES_ROOT / f"series-{slug}" / "cast.json"
        if cast_path.is_file():
            entries = _json.loads(cast_path.read_text(encoding="utf-8"))
            if isinstance(entries, dict):
                entries = entries.get("characters") or entries.get("cast") or []
            cast_names = sorted({str(item.get("name") or "").strip()
                                 for item in entries if str(item.get("name") or "").strip()})
    except Exception:
        cast_names = []
    return {
        "series": name,
        "slug": slug,
        "embedder": bank.get("embedder"),
        "cast_names": cast_names,
        "bank": [{"id": c["id"], "name": c["name"],
                  "has_reference": bool(c.get("reference_clip"))}
                 for c in bank["characters"]],
        "speakers": [{
            "speaker": sp,
            "match": matches.get(sp),                # None = no embedding evidence
            "assigned_character": assigned.get(sp),
            "cards": cards.get(sp, []),
        } for sp in speakers],
    }


def apply_answer(job_id: str, *, series: str | None, speaker: str, answer: str,
                 character_id: str | None = None, new_name: str | None = None,
                 seed_segment: int | None = None) -> dict[str, Any]:
    """Record one reviewer verdict and apply its (minimal) consequence.

    same       -> map speaker -> existing character in the job (additive field).
    different  -> with ``new_name``: create the character, seeded from ``seed_segment``'s
                  clip + source text; then map the speaker to it. Without a name it is
                  just a recorded rejection (the card leaves the queue; nothing is minted).
    skip       -> logged, nothing else.
    """
    job = load_job(job_id)
    name, slug = _series_for(job, series)
    speaker = str(speaker or "").strip()
    if not speaker:
        raise ValueError("speaker is required")
    created: dict[str, Any] | None = None

    if answer == "different" and (new_name or "").strip():
        centroid = _centroids(job).get(speaker)
        if not centroid:
            raise ValueError(f"no embedding evidence for {speaker}; cannot seed a character")
        clip, transcript = _seed_reference(job_id, job, speaker, seed_segment, slug)
        created = voice_bank.add_character(
            slug, new_name.strip(), centroid=centroid,
            reference_clip=clip, reference_transcript=transcript, source_job=job_id)
        character_id = created["id"]
        answer_for_log = "different"
    else:
        answer_for_log = answer

    voice_bank.record_reviewer_answer(
        slug, job_id=job_id, speaker=speaker, answer=answer_for_log,
        character_id=character_id if answer in ("same", "different") else None,
        character_name=(created or {}).get("name") or _bank_name(slug, character_id))

    if answer == "same" or created is not None:
        job.setdefault("voice_bank_series", name)
        job.setdefault("speaker_characters", {})[speaker] = character_id
        save_job(job)

    return {"ok": True, "speaker": speaker, "answer": answer,
            "character_id": character_id, "created": created}


def auto_apply_clear(job_id: str, series: str | None = None) -> dict[str, Any]:
    """Apply every CLEAR-zone match, each one logged individually (an auto-match is a
    decision too — the log is what makes it reviewable later). Ask/new/quarantined are
    untouched: those are the reviewer's."""
    job = load_job(job_id)
    name, slug = _series_for(job, series)
    songs = song_speakers(job)
    centroids = {sp: c for sp, c in _centroids(job).items() if sp not in songs}
    matches = _filtered_matches(slug, job_id, centroids)
    applied = {}
    assigned = job.setdefault("speaker_characters", {})
    for sp, match in matches.items():
        if match["zone"] != voice_bank.ZONE_CLEAR or sp in assigned:
            continue
        assigned[sp] = match["character_id"]
        applied[sp] = match["character_name"]
        voice_bank.log_decision(slug, kind="auto_match", job=job_id, speaker=sp,
                                character_id=match["character_id"],
                                character_name=match["character_name"],
                                score=match["score"])
    if applied:
        job.setdefault("voice_bank_series", name)
        save_job(job)
    return {"ok": True, "applied": applied, "count": len(applied)}


def _seed_reference(job_id: str, job: dict[str, Any], speaker: str,
                    seed_segment: int | None, slug: str) -> tuple[str, str]:
    """Materialize the chosen card's clip via the EXISTING preview path, then copy it into
    voices/ so the bank owns media that outlives job cleanup. Transcript = the segment's
    source-language text (the words actually in the audio)."""
    segments = job.get("segments") or []
    if seed_segment is None:
        cards = plan_cards(segments).get(speaker) or []
        if not cards or cards[0].get("i") is None:
            raise ValueError(f"no evidence card available to seed {speaker}")
        seed_segment = int(cards[0]["i"])
    segment = next((s for s in segments if int(s.get("i", -1)) == int(seed_segment)), None)
    if segment is None or str(segment.get("speaker") or "") != speaker:
        raise ValueError("seed segment does not belong to that speaker")
    transcript = str(segment.get("text") or "").strip()
    if not transcript:
        raise ValueError("seed segment has no source text; pick another card")
    source = source_preview(job_id, int(seed_segment))
    dest_dir = voice_bank.VOICES_ROOT / f"series-{slug}" / "refs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{_safe(speaker)}-{int(seed_segment):05d}{source.suffix}"
    dest.write_bytes(source.read_bytes())
    return str(dest), transcript


def _bank_name(slug: str, character_id: str | None) -> str | None:
    if not character_id:
        return None
    bank = voice_bank.load_bank(slug)
    return next((c["name"] for c in bank["characters"] if c["id"] == character_id), None)


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:40] or "spk"
