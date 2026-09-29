"""Casting-studio operations: the reviewer's merge/flag verbs.

Diarization over-splits: an episode with four or five characters can come back with eight
speaker labels, and duplicate suggestions alone are advisory-only with no way to act on them
(speaker_evidence keeps that pure contract; this module is the impure half that
rewrites jobs). Merging purges the absorbed label's cached audio so a stale voice can
never survive into a later render.
"""
from __future__ import annotations

import re

from .state import event, job_dir, job_lock, load_job, save_job
from .workflow import load_line_manifest, save_line_manifest


def _safe_label(label: str) -> str:
    """Filesystem-safe token for a (possibly reviewer-renamed) speaker label."""
    import hashlib
    cleaned = re.sub(r"[^A-Za-z0-9-]", "", str(label))[:40]
    return cleaned or hashlib.sha256(str(label).encode("utf-8")).hexdigest()[:12]


def reject_merge(job_id: str, *, a: str, b: str) -> dict:
    """Reviewer ruled DIFFERENT (or stayed unsure) on a suggested pair — remember it so
    the required-decision queue never re-asks a settled verdict."""
    with job_lock(job_id):
        job = load_job(job_id)
        pair = sorted([str(a), str(b)])
        rejections = [sorted(p) for p in job.get("speaker_merge_rejections") or []]
        if pair not in rejections:
            rejections.append(pair)
            job["speaker_merge_rejections"] = rejections
            save_job(job)
        return {"rejected_pairs": len(rejections)}


def merge_speakers(job_id: str, *, source: str, target: str) -> dict:
    """Fold every segment of `source` into `target` (reviewer SAME verdict)."""
    with job_lock(job_id):
        return _merge_speakers_locked(job_id, source=source, target=target)


def _merge_speakers_locked(job_id: str, *, source: str, target: str) -> dict:
    job = load_job(job_id)
    if job.get("status") == "running":
        raise ValueError("wait for the current stage to finish")
    speakers = {str(item.get("speaker")) for item in job.get("segments") or []}
    if source == target or source not in speakers or target not in speakers:
        raise ValueError("invalid speaker merge")
    artifacts = job_dir(job_id) / "artifacts"
    lines_dir = artifacts / "lines"
    aligned_dir = artifacts / "aligned"
    manifest = load_line_manifest(lines_dir)
    changed = purged = 0
    for segment in job["segments"]:
        if str(segment.get("speaker")) == source:
            segment["speaker"] = target
            segment["merged_from"] = source
            changed += 1
            for stale in (lines_dir / f"line-{int(segment['i']):05d}.wav",
                          aligned_dir / f"line-{int(segment['i']):05d}.wav"):
                # the aligned/ copy re-enters any remix/repair mix if it survives
                # — purge BOTH halves of the cache
                if stale.is_file():
                    stale.unlink(missing_ok=True)
                    purged += 1
            manifest.pop(str(int(segment["i"])), None)
    if purged:
        save_line_manifest(lines_dir, manifest)
    demos_dir = artifacts / "demos"
    if demos_dir.is_dir():
        for stale in demos_dir.glob(f"demo-{_safe_label(source)}-*.wav"):
            stale.unlink(missing_ok=True)
    # The absorbed label's reference and voice option must not survive the merge —
    # a poisoned reference living on would keep producing a bad voice.
    job.get("speaker_references", {}).pop(source, None)
    job.get("speaker_voices", {}).pop(source, None)
    orphan_option = f"qwen-auto:{source}"
    job["available_voices"] = [option for option in job.get("available_voices") or []
                               if option != orphan_option]
    (job.get("voice_labels") or {}).pop(orphan_option, None)
    job["speaker_duplicate_suggestions"] = [
        pair for pair in job.get("speaker_duplicate_suggestions") or []
        if source not in (pair.get("speaker_a"), pair.get("speaker_b"))
    ]
    job.setdefault("speaker_merges", []).append(
        {"from": source, "to": target, "lines": changed})
    event(job, "review",
          f"speaker {source} merged into {target} ({changed} line(s); "
          f"{purged} cached wav(s) purged).", 72)
    return {"merged": changed, "from": source, "to": target, "purged_lines": purged}


def attribution_flags(job: dict, *, low_confidence: float = 0.7,
                      pair_cosine: float = 0.88, flap_window_s: float = 6.0) -> list[dict]:
    """Segments the reviewer should check before synthesis, with reasons.

    Classes: quarantined phantom · low diarizer confidence · member of a
    merge-suggested pair · ABA mid-scene flap where the interloper is weak.
    Pure read — the studio's spot-check screen filters on this.
    """
    segments = sorted(job.get("segments") or [], key=lambda item: float(item["start"]))
    rejected = {tuple(sorted(p)) for p in job.get("speaker_merge_rejections") or []}
    pair_members = set()
    for pair in job.get("speaker_duplicate_suggestions") or []:
        key = tuple(sorted([str(pair.get("speaker_a")), str(pair.get("speaker_b"))]))
        if key in rejected:
            continue
        if float(pair.get("cosine_similarity") or 0.0) >= pair_cosine:
            pair_members.update(key)
    flags = []
    for index, segment in enumerate(segments):
        reasons = []
        confidence = float(segment.get("speaker_confidence", 1.0))
        speaker = str(segment.get("speaker"))
        if segment.get("attribution_suspect"):
            reasons.append("quarantined-phantom")
        if confidence < low_confidence:
            reasons.append(f"low-confidence {confidence:.2f}")
        if speaker in pair_members:
            reasons.append("merge-suggested-label")
        if 0 < index < len(segments) - 1:
            before, after = segments[index - 1], segments[index + 1]
            if (str(before.get("speaker")) == str(after.get("speaker")) != speaker
                    and float(after.get("end", 0)) - float(before.get("start", 0)) <= flap_window_s):
                # Gating this on low confidence made the class a no-op (those lines were already flagged) — an ABA flip
                # inside one breath is suspect at ANY confidence.
                reasons.append("mid-scene-flap")
        if reasons:
            flags.append({"i": int(segment["i"]), "speaker": speaker,
                          "reasons": reasons})
    return flags


VALID_VOICE_PREFIXES = ("qwen-auto:", "qwen-auto-xv:", "bank:", "qwen:voice-",
                        "gsv:voice-", "mute:")


def demo_lines(job: dict, label: str) -> dict[str, int]:
    """Two audition lines per label: 'emphatic' (highest source energy, min 20 chars)
    and 'calm' (calm/neutral delivery, min 20 chars). Falls back to the longest
    lines when delivery hints are missing."""
    candidates = [s for s in job.get("segments") or []
                  if str(s.get("speaker")) == label
                  and len(str(s.get("translation") or "").strip()) >= 20]
    if not candidates:
        candidates = [s for s in job.get("segments") or []
                      if str(s.get("speaker")) == label
                      and str(s.get("translation") or "").strip()]
    if not candidates:
        raise ValueError(f"{label} has no reviewable lines to demo")

    def energy(segment: dict) -> float:
        return float((segment.get("delivery") or {}).get("energy_dbfs") or -60.0)

    def delivery_label(segment: dict) -> str:
        return str((segment.get("delivery") or {}).get("label") or "")

    emphatic = max(candidates, key=energy)
    calm_pool = [s for s in candidates
                 if delivery_label(s) in ("calm", "neutral") and s is not emphatic]
    if calm_pool:
        calm = max(calm_pool, key=lambda s: len(str(s.get("translation") or "")))
    else:
        rest = [s for s in candidates if s is not emphatic]
        calm = (max(rest, key=lambda s: len(str(s.get("translation") or "")))
                if rest else emphatic)
    return {"emphatic": int(emphatic["i"]), "calm": int(calm["i"])}


def demo_payloads(job_id: str, candidates: dict[str, list[str]]) -> tuple[list[dict], dict]:
    """Build one worker batch covering every (label x voice x register) demo.

    Lines carry NO slot_seconds, so the runaway guard and reseed stay inert —
    demos are auditions, not aligned dialogue. Existing files are skipped
    (cached); one GPU arm renders the whole studio's demos."""
    import hashlib

    from .pipeline import _qwen_line  # late import: pipeline does not import casting

    job = load_job(job_id)
    artifacts = job_dir(job_id) / "artifacts"
    demos = artifacts / "demos"
    demos.mkdir(parents=True, exist_ok=True)
    lines: list[dict] = []
    manifest: dict = {}
    for label, voices in candidates.items():
        picks = demo_lines(job, label)
        manifest[label] = {}
        for voice in voices:
            if voice.startswith("mute"):
                continue
            token = hashlib.sha256(voice.encode("utf-8")).hexdigest()[:8]
            manifest[label][voice] = {}
            for register, index in picks.items():
                name = f"demo-{_safe_label(label)}-{token}-{register}.wav"
                target = demos / name
                manifest[label][voice][register] = name
                if target.is_file():
                    continue
                segment = next(s for s in job["segments"] if int(s["i"]) == index)
                payload = _qwen_line(job, artifacts, segment, voice, target)
                payload.pop("slot_seconds", None)
                payload["fallback_references"] = []
                lines.append(payload)
    return lines, manifest


def prerender_demos(job_id: str, candidates: dict[str, list[str]],
                    *, force: bool = False) -> dict:
    """Render every studio demo in ONE GPU lease — no per-click waits during a
    casting session. GPU authorization comes from the server arm.

    Lifecycle contract: the manifest is CLEARED first so
    the studio's poll can't false-complete on a stale one, and EVERY exit —
    including failure — writes a manifest (error key on failure) so the poll always
    terminates with something to show."""
    from . import adapters, oplog
    from .gpu_session import gpu_lease

    with job_lock(job_id):
        job = load_job(job_id)
        job.pop("demo_manifest", None)
        save_job(job)
    try:
        if force:
            demos_dir = job_dir(job_id) / "artifacts" / "demos"
            if demos_dir.is_dir():
                for stale in demos_dir.glob("demo-*.wav"):
                    stale.unlink(missing_ok=True)
        lines, manifest = demo_payloads(job_id, candidates)
        if lines:
            with gpu_lease(f"demo:{job_id}") as lease:
                lease.ensure_active()
                result = adapters.synthesize_quality_batch(
                    lines, seed=int(load_job(job_id)["settings"].get("tts_seed", 1986)))
            if result.get("cancelled"):
                raise ValueError("demo prerender cancelled")
    except Exception as exc:
        with job_lock(job_id):
            job = load_job(job_id)
            job["demo_manifest"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            event(job, "review", "Casting demo prerender FAILED: %s — see the job oplog."
                  % type(exc).__name__, 72)
        oplog.job_error(job_id, "casting", f"demo prerender failed: {exc}")
        raise
    with job_lock(job_id):
        job = load_job(job_id)
        job["demo_manifest"] = manifest
        event(job, "review", "Casting demos rendered: %d new clip(s) across %d voice label(s)."
              % (len(lines), len(manifest)), 72)
    oplog.job_event(job_id, "casting", "demo prerender: %d rendered, %d voice option(s)"
                    % (len(lines), sum(len(v) for v in manifest.values())))
    return {"rendered": len(lines), "manifest": manifest}


def apply_cast(job_id: str, *, cast: dict[str, dict], series: str | None = None) -> dict:
    """CAST-LOCK: the studio's decisions become the job's reality.

    cast = {label: {"character": str, "voice": option, "second_choice": option|None,
    "tier": "main"|"minor"|"bit"}}. Rewrites speaker_voices, purges changed labels'
    cached lines (a stale voice must never survive a recast), extends
    available_voices with new variant options, enforces ICL transcripts on
    qwen-auto picks (silent x-vector degradation is banned), stores the lock, and
    best-effort enrolls named characters into the series bank."""
    with job_lock(job_id):
        return _apply_cast_locked(job_id, cast=cast, series=series)


def _apply_cast_locked(job_id: str, *, cast: dict[str, dict], series: str | None) -> dict:
    from . import oplog

    job = load_job(job_id)
    if job.get("status") == "running":
        raise ValueError("wait for the current stage to finish")
    labels = {str(s.get("speaker")) for s in job.get("segments") or []}

    def _muted(option: str) -> bool:
        return str(option or "").startswith("mute")

    # Labels that ONLY sing (marked opening/ending themes) never appear in the studio and are
    # never dubbed — auto-mute them instead of failing the lock over an invisible
    # speaker.
    by_label: dict[str, list] = {}
    for segment in job.get("segments") or []:
        by_label.setdefault(str(segment.get("speaker")), []).append(segment)
    song_only = {label for label, segs in by_label.items()
                 if segs and all(s.get("song_skip") for s in segs)}
    for label in song_only - set(cast):
        job.setdefault("speaker_voices", {})[label] = "mute:"

    missing = sorted(
        label for label in labels
        if label not in cast
        and not _muted((job.get("speaker_voices") or {}).get(label, "")))
    if missing:
        raise ValueError("cast is incomplete: no decision for " + ", ".join(missing))
    artifacts = job_dir(job_id) / "artifacts"
    references = job.get("speaker_references") or {}

    def _embedded_label(option: str) -> str | None:
        for prefix in ("qwen-auto-alt:", "qwen-auto-xv:", "qwen-auto:"):
            if option.startswith(prefix):
                return option.removeprefix(prefix).split(":", 1)[0]
        return None

    def _validate_voice(label: str, option: str, *, role: str) -> None:
        if not option.startswith(VALID_VOICE_PREFIXES):
            raise ValueError(f"{label}: unknown {role} option {option}")
        # voices may be REUSED across labels (deliberate minor-cast doubling), so
        # validate the reference the option actually points at, not this row's
        referenced = _embedded_label(option)
        if referenced is not None:
            if referenced not in references:
                raise ValueError(f"{label}: {role} points at {referenced}, which has no "
                                 "clone reference")
            if option.startswith("qwen-auto:") and not str(
                    references[referenced].get("text") or "").strip():
                # ICL mode needs the transcript; without it the worker silently
                # degrades to x-vector (~0.75 vs ~0.89 similarity).
                raise ValueError(f"{label}: reference transcript missing on {referenced} - "
                                 "pick the x-vector variant explicitly or repair it")

    for label, entry in cast.items():
        voice = str(entry.get("voice") or "")
        if label not in labels:
            raise ValueError(f"unknown speaker label {label}")
        if _muted(voice):
            continue
        if not str(entry.get("character") or "").strip():
            raise ValueError(f"{label}: a cast (non-muted) voice needs a character name")
        _validate_voice(label, voice, role="voice")
        second = str(entry.get("second_choice") or "")
        if second:
            _validate_voice(label, second, role="second-choice voice")
    lines_dir = artifacts / "lines"
    aligned_dir = artifacts / "aligned"
    manifest = load_line_manifest(lines_dir)
    purged = 0
    changed_labels = []
    for label, entry in cast.items():
        voice = str(entry["voice"])
        if job.get("speaker_voices", {}).get(label) != voice:
            changed_labels.append(label)
            for segment in job["segments"]:
                if str(segment.get("speaker")) == label:
                    for stale in (lines_dir / f"line-{int(segment['i']):05d}.wav",
                                  aligned_dir / f"line-{int(segment['i']):05d}.wav"):
                        # aligned/ copies re-enter remix/repair mixes if they
                        # survive a recast
                        if stale.is_file():
                            stale.unlink(missing_ok=True)
                            purged += 1
                    manifest.pop(str(int(segment["i"])), None)
        job.setdefault("speaker_voices", {})[label] = voice
        if voice not in (job.get("available_voices") or []):
            job.setdefault("available_voices", []).append(voice)
            job.setdefault("voice_labels", {})[voice] = (
                "Cast voice · " + str(entry.get("character") or label))
    if purged:
        save_line_manifest(lines_dir, manifest)
    job["cast_lock"] = {
        "cast": cast,
        "series": series,
        "changed_labels": changed_labels,
        "purged_lines": purged,
    }
    enrolled = 0
    if series:
        from . import voice_bank as _vb
        centroids = {e.get("speaker"): e.get("centroid")
                     for e in (job.get("speaker_evidence") or {}).get("speaker_embeddings") or []}
        slug = _vb.series_slug(series)
        job["voice_bank_series"] = series
        # speaker_characters is a {label: bank character ID} contract everywhere
        # else in the codebase — NEVER write display names into it.
        character_ids: dict[str, str] = {}
        existing_by_name = {str(c.get("name") or "").casefold(): c
                            for c in _vb.load_bank(slug)["characters"]}
        for label, entry in cast.items():
            name = str(entry.get("character") or "").strip()
            voice = str(entry.get("voice") or "")
            if not name or _muted(voice):
                continue
            if voice.startswith("bank:"):
                character_ids[label] = voice.split(":", 2)[2]
                continue
            already = existing_by_name.get(name.casefold())
            if already:
                character_ids[label] = already["id"]
                continue
            centroid = centroids.get(label)
            reference = references.get(_embedded_label(voice) or label) or {}
            if not centroid:
                continue
            try:
                clip = None
                if reference.get("file"):
                    # copy the reference OUT of the prunable job workdir so the
                    # bank survives job deletion
                    import shutil as _shutil
                    from .config import VOICES_ROOT
                    refs_dir = VOICES_ROOT / f"series-{slug}" / "references"
                    refs_dir.mkdir(parents=True, exist_ok=True)
                    clip_path = refs_dir / f"{_safe_label(name)}-{_safe_label(label)}.wav"
                    _shutil.copy2(artifacts / reference["file"], clip_path)
                    clip = str(clip_path)
                created = _vb.add_character(
                    slug, name,
                    centroid=centroid,
                    reference_clip=clip,
                    reference_transcript=str(reference.get("text") or "") or None,
                    source_job=job_id,
                )
                character_ids[label] = created["id"]
                existing_by_name[name.casefold()] = created
                enrolled += 1
            except Exception as exc:
                oplog.job_warn(job_id, "casting",
                               f"bank enrollment failed for {name}: {type(exc).__name__}: {exc}")
        if character_ids:
            job["speaker_characters"] = character_ids
    event(job, "review",
          "Cast locked: %d label(s) cast, %d voice change(s), %d cached line(s) purged, "
          "%d character(s) enrolled." % (len(cast), len(changed_labels), purged, enrolled), 72)
    return {"cast": len(cast), "changed": changed_labels, "purged_lines": purged,
            "enrolled": enrolled}


def embedding_disagreements(job: dict, *, margin: float = 0.05) -> list[dict]:
    """Partial machine verification at zero inference cost: cosine each STORED chunk
    embedding against every speaker centroid; flag segments whose own label loses to
    another by > margin. HONESTLY PARTIAL — only segments with a stored embedding
    (about half on a typical episode) are checkable; the rest need human review."""
    from .speaker_evidence import _cosine_similarity

    evidence = job.get("speaker_evidence") or {}
    live_labels = {str(s.get("speaker")) for s in job.get("segments") or []}
    centroids = {str(e.get("speaker")): tuple(float(v) for v in e.get("centroid") or [])
                 for e in evidence.get("speaker_embeddings") or []}
    # merged-away labels keep centroids in the evidence blob — comparing against
    # them floods the reviewer with dead-label suggestions
    centroids = {k: v for k, v in centroids.items() if v and k in live_labels}
    if not centroids:
        return []
    label_of = {int(s["i"]): str(s.get("speaker")) for s in job.get("segments") or []}
    findings = []
    for chunk in evidence.get("speaker_chunk_embeddings") or []:
        index = int(chunk.get("segment_index", -1))
        assigned = label_of.get(index)
        vector = tuple(float(v) for v in chunk.get("embedding") or [])
        if assigned is None or not vector or assigned not in centroids:
            continue
        own = _cosine_similarity(vector, centroids[assigned])
        if own is None:
            continue
        best_label, best = assigned, own
        for label, centroid in centroids.items():
            score = _cosine_similarity(vector, centroid)
            if score is not None and score > best:
                best_label, best = label, score
        if best_label != assigned and best - own > margin:
            findings.append({"i": index, "speaker": assigned,
                             "suggests": best_label,
                             "own_cosine": round(own, 3),
                             "best_cosine": round(best, 3)})
    return findings


def acceptance_report(job: dict) -> dict:
    """The episode acceptance gate: machine-checkable pass/fail plus an auto-picked
    spot-sheet of the WORST offenders so five minutes of listening covers exactly where the last build failed."""
    from . import oplog

    qc = job.get("qc_summary") or {}
    lock = job.get("cast_lock") or {}
    cast = lock.get("cast") or {}
    voices = job.get("speaker_voices") or {}
    checks = {
        "rendered": job.get("status") == "complete",
        "qc_sees_new_classes": "rendered_overlap" in qc and "runaway" in qc,
        "rendered_overlap_zero": qc.get("rendered_overlap") == 0,
        "runaway_zero": qc.get("runaway") == 0,
        "cast_locked": bool(cast),
        "voices_match_lock": bool(cast) and all(
            voices.get(label) == str(entry.get("voice") or "")
            for label, entry in cast.items()),
        "no_uncast_voices": all(
            str(option).startswith(VALID_VOICE_PREFIXES)
            for option in voices.values()) if voices else False,
    }
    spot = []
    for segment in job.get("segments") or []:
        flags = (segment.get("qc") or {}).get("flags") or []
        reasons = [flag for flag in flags
                   if flag in ("rendered-overlap", "runaway", "hard-capped")]
        if segment.get("merged_from"):
            reasons.append(f"reassigned from {segment['merged_from']}")
        if reasons:
            spot.append({"i": int(segment["i"]),
                         "at_seconds": round(float(segment["start"]), 1),
                         "reasons": reasons})
    severity = {"runaway": 0, "rendered-overlap": 1, "hard-capped": 2}
    spot.sort(key=lambda item: min((severity.get(r.split(" ")[0], 3)
                                    for r in item["reasons"]), default=3))
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "code_sha": oplog.code_sha(),
        "spot_sheet": spot[:12],
        "spot_sheet_total": len(spot),
    }


def sanitize_studio_progress(value) -> dict:
    """Reviewer working-state for the studio v2 page: current step, which groups are
    marked clean in step 1, and DRAFT names/voice picks (stored server-side so an
    interrupted session loses nothing). Clamped convenience state, never
    pipeline state: oversized lists are truncated, not rejected."""
    if not isinstance(value, dict):
        raise ValueError("studio_progress must be an object")
    try:
        step = int(value.get("step", 1))
    except (TypeError, ValueError):
        raise ValueError("studio_progress.step must be 1-4")
    if step not in (1, 2, 3, 4):
        raise ValueError("studio_progress.step must be 1-4")
    labels = value.get("clean_labels") or []
    if not isinstance(labels, list):
        raise ValueError("clean_labels must be a list")
    clean = sorted({str(item)[:80] for item in labels if str(item).strip()})[:64]
    draft_raw = value.get("draft_cast") or {}
    if not isinstance(draft_raw, dict):
        raise ValueError("draft_cast must be an object")
    draft = {}
    for label, entry in list(draft_raw.items())[:64]:
        if not isinstance(entry, dict):
            continue
        draft[str(label)[:80]] = {
            "character": str(entry.get("character") or "")[:120],
            "voice": str(entry.get("voice") or "")[:160],
        }
    return {"step": step, "clean_labels": clean, "draft_cast": draft}
