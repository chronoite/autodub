from __future__ import annotations

import os
import subprocess
import threading
import time
import traceback
from pathlib import Path

from . import adapters, nonverbal, oplog, reference_quality
from .config import (
    FFMPEG,
    SPEAKER_REF_ALTERNATES,
    SPEAKER_REF_MAX_OVERLAP,
    SPEAKER_REF_MIN_CONFIDENCE,
    TTS_BATCH_SIZE,
    TTS_RUNAWAY_FACTOR,
)
from .delivery import analyze_delivery
from .gpu_session import GpuSafetyError, gpu_lease
from .media import StageError, align_line, build_mix, extract_audio, mux
from .policies import get_mix_policy, get_space_policy, get_timing_policy
from .quality_profiles import CPU_PROFILE, profile_requires_gpu
from .speaker_evidence import derive_speaker_evidence, suggest_duplicate_pairs
from .song_detect import detect_songs, replay_manual_marks
from .state import (
    event,
    friendly_output_path,
    job_dir,
    load_job,
    output_path,
    output_variant_path,
    save_job,
)
from .subtitles import harvest_embedded_english
from .voice_profiles import load_profile
from .workflow import (
    apply_forced_alignment,
    audit_timing,
    cancel_path,
    cancellation_requested,
    clear_cancel,
    clear_synth_progress,
    load_line_manifest,
    record_line,
    remove_stale_lines,
    reusable_line,
    save_line_manifest,
    write_synth_progress,
)


class JobCancelled(StageError):
    pass


def _record_speaker_evidence(job: dict, diarization: dict) -> None:
    """Persist raw evidence locally and derived, non-mutating suggestions separately."""
    job["segments"] = diarization["segments"]
    embeddings = list(diarization.get("speaker_embeddings") or [])
    chunk_embeddings = list(diarization.get("speaker_chunk_embeddings") or [])
    overlap_pairs = list(diarization.get("speaker_overlap_evidence") or [])
    derived = derive_speaker_evidence(embeddings, chunk_embeddings, job["segments"])
    repair_embeddings = derived["repair_embeddings"]
    manifest = derived["manifest"]
    suggestions = suggest_duplicate_pairs(
        repair_embeddings, overlap_pairs, job["segments"], evidence_manifest=manifest
    )
    job["speaker_evidence"] = {
        "schema": 2,
        "method": str(diarization.get("method") or "unknown"),
        "speaker_embeddings": embeddings,
        "speaker_chunk_embeddings": chunk_embeddings,
        "repair_embeddings": repair_embeddings,
        "speaker_overlap_evidence": overlap_pairs,
    }
    job["speaker_evidence_manifest"] = manifest
    job["speaker_duplicate_suggestions"] = suggestions
    oplog.job_event(
        job["id"],
        "speaker-evidence",
        f"captured {len(embeddings)} cross-check centroids, {len(chunk_embeddings)} eligible chunks, "
        f"{len(overlap_pairs)} overlap pairs, "
        f"and {len(suggestions)} duplicate suggestions",
    )


def _song_skipped(job: dict, segment: dict) -> bool:
    """Detected songs stay un-dubbed: the original singing is kept."""
    return (job["settings"].get("song_policy", "dub-all-v1") == "skip-detected-v1"
            and bool(segment.get("song_skip")))


def _drop_song_lines(job: dict, lines: Path) -> int:
    """Resume safety: purge cached line audio for song-skipped segments so an earlier
    render's dub of a song can never re-enter the mix after the flag was set."""
    dropped = 0
    for segment in job.get("segments", []):
        if _song_skipped(job, segment):
            target = lines / f"line-{int(segment['i']):05d}.wav"
            if target.exists():
                target.unlink()
                dropped += 1
    return dropped


def _restore_song_vocals(job: dict, artifacts: Path) -> list:
    """The quality mix bed has vocals REMOVED, so a skipped
    song would render as an instrumental — neither Japanese singing nor English dub. Slice
    the original vocal stem over each song window into a SEPARATE songpass dir and return
    [(segment, path)] for build_mix's passthrough channel: unity gain, no dialogue shaping,
    and the bed does NOT duck under them (only the master anti-clip limiter applies).
    Failures are LOUD (job event), never a silent zero. CPU renders mix the full original
    audio and never had this problem."""
    songs = [s for s in job.get("segments", []) if _song_skipped(job, s)]
    if not songs:
        return []
    stem = job.get("artifacts", {}).get("dialogue_stem")
    vocals = (artifacts / stem) if stem else None
    if not vocals or not vocals.is_file():
        event(job, "songs", "EXPERIMENTAL song skip WARNING: vocal stem missing - %d song window(s) "
              "will be INSTRUMENTAL this render." % len(songs), 90)
        return []
    songpass = artifacts / "songpass"
    songpass.mkdir(exist_ok=True)
    # Whole-span restore: per-cue slices once left 21.3 s of inter-cue singing silent
    # inside one ending theme. Restore each CONTIGUOUS song region
    # (cues within 5s of each other) as a single window so the music never gaps.
    songs = sorted(songs, key=lambda item: float(item["start"]))
    spans = []
    for segment in songs:
        start, end = float(segment["start"]), float(segment["end"])
        if spans and start - spans[-1]["end"] <= 5.0:
            spans[-1]["end"] = max(spans[-1]["end"], end)
        else:
            spans.append({"start": start, "end": end, "anchor": segment})
    # Split spans around rendered NON-song dialogue inside them: a normal spoken
    # line between two karaoke cues must not get original vocals restored over its
    # English dub (otherwise a new double-voice class appears).
    dialogue_windows = []
    aligned_dir = artifacts / "aligned"
    for segment in job.get("segments", []):
        if _song_skipped(job, segment):
            continue
        if not (aligned_dir / f"line-{int(segment['i']):05d}.wav").is_file():
            continue
        start = float(segment["start"])
        placed = float((segment.get("alignment") or {}).get("output_seconds") or 0.0)
        dialogue_windows.append((start - 0.05, start + max(placed, 0.2) + 0.05))
    pieces = []
    for span in spans:
        remaining = [(span["start"], span["end"])]
        for window_start, window_end in sorted(dialogue_windows):
            carved = []
            for piece_start, piece_end in remaining:
                if window_end <= piece_start or window_start >= piece_end:
                    carved.append((piece_start, piece_end))
                    continue
                if window_start > piece_start:
                    carved.append((piece_start, window_start))
                if window_end < piece_end:
                    carved.append((window_end, piece_end))
            remaining = carved
        for piece_start, piece_end in remaining:
            if piece_end - piece_start >= 0.3:
                pieces.append({"start": piece_start, "end": piece_end,
                               "anchor": span["anchor"]})
    restored, failed = [], 0
    for number, piece in enumerate(pieces):
        start = piece["start"]
        duration = max(0.05, piece["end"] - start)
        anchor = dict(piece["anchor"])
        anchor["start"], anchor["end"] = start, piece["end"]
        target = songpass / f"span-{int(anchor['i']):05d}-{number:02d}.wav"
        result = subprocess.run(
            [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(vocals), str(target)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and target.is_file():
            restored.append((anchor, target))
        else:
            failed += 1
    if failed:
        event(job, "songs", "EXPERIMENTAL song skip WARNING: %d of %d song vocal restore(s) FAILED - "
              "those windows will be instrumental." % (failed, len(pieces)), 90)
    return restored


def _check_cancel(job: dict) -> None:
    if cancellation_requested(job["id"]):
        raise JobCancelled("job cancelled by the user")


def _link_friendly_output(job: dict, final: Path) -> None:
    """Every export also gets a human-readable HARDLINK named from the source file's own name. The
    job-ID file stays canonical so app links and existing jobs never break."""
    friendly = friendly_output_path(job, final)
    if friendly is None:
        return
    try:
        friendly.unlink(missing_ok=True)
        os.link(final, friendly)
        job["artifacts"]["friendly_output"] = friendly.name
    except OSError as exc:
        oplog.job_warn(job["id"], "export",
                       f"friendly export link failed: {type(exc).__name__}")


def _next_rendered_starts(job: dict, segments: list[dict], lines_dir: Path) -> dict[int, float]:
    """{segment i: max seconds its rendered line may occupy before the NEXT occupied
    audio starts} — the preserve-capped budget.

    Boundaries are the next RENDERED line's start AND the next song-skipped
    segment's start (restored original singing plays there at unity gain — a dub
    tail spilling over it would double the voice). Near-simultaneous starts (<0.35s gap: overlapping shouts) get NO cap —
    deleting a line's content is worse than a QC-flagged overlap."""
    rendered = sorted(
        (item for item in segments
         if (lines_dir / f"line-{int(item['i']):05d}.wav").exists()),
        key=lambda item: float(item["start"]),
    )
    boundaries = sorted(
        [float(item["start"]) for item in rendered]
        + [float(item["start"]) for item in segments if _song_skipped(job, item)])
    budgets: dict[int, float] = {}
    for current in rendered:
        start = float(current["start"])
        following = next((b for b in boundaries if b > start + 1e-6), None)
        if following is None:
            continue
        gap = following - start - 0.01
        if gap >= 0.35:
            budgets[int(current["i"])] = gap
    return budgets


def _warn_runaway(job: dict) -> None:
    # A runaway line means a likely-poisoned voice reference —
    # it must never reach the mix silently. No progress arg: the bar is untouched.
    count = int(job.get("qc_summary", {}).get("runaway") or 0)
    if count:
        event(job, "qc", "RUNAWAY TTS WARNING: %d line(s) synthesized more than %.0fx "
              "their slot - likely a poisoned voice reference; review before trusting "
              "this mix." % (count, TTS_RUNAWAY_FACTOR))


def _nonverbal_pass(job: dict, artifacts: Path) -> list:
    """Original non-verbals (screams/laughs/gasps) are never removed. Quality renders only — the CPU path keeps the full original audio anyway."""
    if job["settings"].get("nonverbal_policy", "passthrough-v1") != "passthrough-v1":
        return []
    restored = nonverbal.restore(job, artifacts)
    if restored:
        event(job, "nonverbal", "Restored %d original non-verbal window(s) (%.1fs) into the "
              "passthrough channel." % (len(restored),
                                        job.get("nonverbal_summary", {}).get("seconds", 0.0)), 91)
    return restored


def _cancelled(job: dict) -> None:
    job["status"] = "cancelled"
    job["error"] = None
    clear_cancel(job["id"])
    event(job, "cancelled", "Job cancelled safely; completed artifacts remain resumable.", job.get("progress", 0))


def _paths(job: dict) -> tuple[Path, Path, Path]:
    root = job_dir(job["id"])
    source = root / job["source"]["file"]
    return root, source, root / "artifacts"


def _analyze_cpu(job_id: str) -> None:
    job = load_job(job_id)
    adapters.set_job_context(job_id)
    root, source, artifacts = _paths(job)
    artifacts.mkdir(exist_ok=True)
    job["status"] = "running"
    job["error"] = None
    clear_cancel(job_id)
    save_job(job)
    try:
        _check_cancel(job)
        event(job, "extracting", "Extracting local audio tracks.", 8)
        asr, full = extract_audio(source, artifacts)
        job["artifacts"].update({"asr_audio": asr.name, "full_audio": full.name})
        save_job(job)

        _check_cancel(job)
        event(job, "transcribing", "Transcribing locally on CPU; no network is available to the worker.", 24)
        segments = adapters.transcribe(asr, job["settings"]["source_language"])
        if not segments:
            raise StageError("transcription produced no speech segments")
        job["segments"] = segments
        save_job(job)

        _check_cancel(job)
        event(job, "diarizing", "Clustering segment voiceprints into stable speaker IDs on CPU.", 47)
        clustered = adapters.cluster_speakers(asr, segments, job["settings"]["speaker_count"])
        _record_speaker_evidence(job, {"method": "acoustic-mfcc-reviewable", "segments": clustered})
        save_job(job)

        _check_cancel(job)
        event(job, "translating", "Translating with the pinned local Marian model in offline mode.", 68)
        job["segments"] = adapters.translate(
            job["segments"], job["settings"]["source_language"], job["settings"]["target_language"]
        )
        _check_cancel(job)
        event(job, "subtitles", "Checking locally for an embedded English text subtitle track.", 70)
        try:
            job["subtitle_harvest"] = harvest_embedded_english(source, artifacts, job["segments"])
        except Exception as exc:
            job["subtitle_harvest"] = {"status": "failed", "mapped": 0}
            oplog.job_warn(job_id, "subtitles", f"optional subtitle harvest unavailable: {type(exc).__name__}")
        if job["settings"].get("emotion_policy") == "source-energy-v1":
            job["delivery_summary"] = analyze_delivery(full, job["segments"])

        if job["settings"].get("song_policy", "dub-all-v1") == "skip-detected-v1":
            event(job, "songs", "EXPERIMENTAL song skip (in testing): detecting song ranges (chapters + subtitle styles) to leave un-dubbed.", 71)
            job["song_summary"] = detect_songs(source, artifacts / "songscan", job["segments"])
            summary = job["song_summary"]
            event(job, "songs", "EXPERIMENTAL song detection: %s - %d segment(s) marked skip (%.1fs)."
                  % (summary["status"], summary["segments_skipped"], summary["skipped_seconds"]), 71)
        # Analysis rebuilds segments wholesale — restore the reviewer's hand-marked
        # song ranges (they outrank the detector).
        restored = replay_manual_marks(job)
        if restored:
            event(job, "songs", "%d manual song mark(s) restored after analysis." % restored, 71)

        voice_entries = adapters.voice_options()
        voices = [entry["option"] for entry in voice_entries]
        speakers = sorted({item.get("speaker", "speaker-01") for item in job["segments"]})
        job["available_voices"] = voices
        job["voice_labels"] = {entry["option"]: entry["label"] for entry in voice_entries}
        for index, speaker in enumerate(speakers):
            job["speaker_voices"].setdefault(speaker, voices[index % len(voices)])
        job["status"] = "review"
        event(job, "review", "Analysis complete. Review speakers, translations, and voice mapping before rendering.", 72)
    except JobCancelled:
        _cancelled(job)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["error_detail"] = traceback.format_exc()[-2000:]   # full trail on disk
        oplog.job_error(job["id"], job.get("stage", "analyze"), traceback.format_exc())
        event(job, "failed", f"Analysis stopped safely: {type(exc).__name__}", job.get("progress", 0))
        oplog.job_summary(job["id"], "failed", job.get("events", []))


def _render_cpu(job_id: str) -> None:
    job = load_job(job_id)
    adapters.set_job_context(job_id)
    root, source, artifacts = _paths(job)
    job["status"] = "running"
    job["error"] = None
    clear_cancel(job_id)
    save_job(job)
    try:
        if not job.get("segments"):
            raise StageError("analyze the job before rendering")
        if not job.get("artifacts", {}).get("full_audio"):
            raise StageError("the extracted mix track is missing; rerun analysis")
        lines = artifacts / "lines"
        aligned = artifacts / "aligned"
        lines.mkdir(exist_ok=True)
        aligned.mkdir(exist_ok=True)
        remove_stale_lines(lines, job["segments"])
        for stale in aligned.glob("line-*.wav"):
            stale.unlink()
        _drop_song_lines(job, lines)

        event(job, "synthesizing", "Synthesizing English dialogue with stable local speaker voices.", 78)
        pending = [
            item for item in job["segments"]
            if str(item.get("translation") or "").strip() and not _song_skipped(job, item)
        ]
        synth_started = time.monotonic()
        completed = 0
        clear_synth_progress(lines)
        for segment in pending:
            _check_cancel(job)
            text = str(segment.get("translation") or "").strip()
            if not text:
                continue
            speaker = segment.get("speaker", "speaker-01")
            voice = job["speaker_voices"].get(speaker, "system-default")
            destination = lines / f"line-{int(segment['i']):05d}.wav"
            if voice.startswith("mute"):
                destination.unlink(missing_ok=True)
                continue
            line_started = time.monotonic()
            if reusable_line(lines, segment, voice, job["settings"]):
                completed += 1
                write_synth_progress(
                    lines,
                    done=completed,
                    total=len(pending),
                    line_index=int(segment["i"]),
                    started_at=synth_started,
                    last_seconds=0.0,
                )
                continue
            destination.unlink(missing_ok=True)
            adapters.synthesize_voice(text, voice, destination, job["settings"]["target_language"])
            record_line(lines, segment, voice, job["settings"])
            completed += 1
            write_synth_progress(
                lines,
                done=completed,
                total=len(pending),
                line_index=int(segment["i"]),
                started_at=synth_started,
                last_seconds=time.monotonic() - line_started,
            )

        timing = get_timing_policy(job["settings"].get("timing_policy", "segment-window-v1"))
        clear_synth_progress(lines)
        event(job, "aligning", "Fitting dialogue inside the original speech windows.", 86)
        next_start = _next_rendered_starts(job, job["segments"], lines)
        for segment in job["segments"]:
            _check_cancel(job)
            source_line = lines / f"line-{int(segment['i']):05d}.wav"
            if not source_line.exists():
                continue
            target = aligned / source_line.name
            segment["alignment"] = align_line(
                source_line,
                target,
                float(segment["end"]) - float(segment["start"]),
                float(timing["min_tempo"]),
                float(timing["max_tempo"]),
                fit_mode=str(timing["fit_mode"]),
                trim_silence=bool(timing.get("trim_silence")),
                max_output_s=next_start.get(int(segment["i"])),
            )
        job["qc_summary"] = audit_timing(job["segments"], aligned)
        _warn_runaway(job)
        save_job(job)

        _check_cancel(job)
        event(job, "mixing", "Ducking the source bed under English dialogue and building the local mix.", 92)
        full_audio = artifacts / job["artifacts"]["full_audio"]
        mixed = artifacts / "english-mix.wav"
        mix_policy = get_mix_policy(job["settings"].get("mix_policy", "legacy-v1"))
        mix_policy.update(get_space_policy(job["settings"].get("space_policy", "dry-v1")))
        count = build_mix(
            full_audio,
            job["segments"],
            aligned,
            mixed,
            float(job["settings"]["source_bed_gain"]),
            float(job["settings"]["dialogue_gain"]),
            policy=mix_policy,
        )
        job["artifacts"]["mixed_audio"] = mixed.name
        job["artifacts"]["rendered_lines"] = count
        save_job(job)

        _check_cancel(job)
        event(job, "muxing", "Muxing the English mix with the untouched source video stream.", 97)
        final = output_path(job_id)
        mux(source, mixed, final)
        job["artifacts"]["output"] = final.name
        job.setdefault("exports", []).append({
            "kind": "primary",
            "file": final.name,
            "mix_policy": job["settings"].get("mix_policy"),
            "timing_policy": job["settings"].get("timing_policy"),
        })
        _link_friendly_output(job, final)
        job["status"] = "complete"
        event(job, "complete", "Local AutoDub export completed.", 100)
        oplog.job_summary(job["id"], "complete", job.get("events", []))
    except JobCancelled:
        _cancelled(job)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["error_detail"] = traceback.format_exc()[-2000:]
        oplog.job_error(job["id"], job.get("stage", "render"), traceback.format_exc())
        event(job, "failed", f"Render stopped safely: {type(exc).__name__}", job.get("progress", 0))
        oplog.job_summary(job["id"], "failed", job.get("events", []))


def _speaker_references(job: dict, vocals: Path, artifacts: Path) -> None:
    """Create opaque, job-local clone references from the cleanest useful speaker window.

    Adjacent same-speaker segments are joined when the gap is short.  Selection prefers 3-10 second
    windows with high diarization coverage and low overlap instead of blindly taking the longest line.
    """
    root = artifacts / "voice-references"
    root.mkdir(exist_ok=True)

    def _extract(window: dict, destination: Path) -> tuple[dict | None, str]:
        start = max(0.0, float(window["start"]))
        duration = min(15.0, max(1.0, float(window["end"]) - start))
        result = subprocess.run(
            [
                str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(vocals),
                "-vn", "-ar", "24000", "-ac", "1", str(destination),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode or not destination.is_file():
            return None, (result.stderr or "")[-300:].strip() or "no output file"
        return {
            "file": destination.relative_to(artifacts).as_posix(),
            "text": window["text"],
            "language": job["settings"].get("source_language", "ja"),
            "source_segments": list(window["source_segments"]),
            "seconds": round(duration, 3),
            "quality": {
                "confidence": window["confidence"],
                "overlap": window["overlap"],
                "score": window["score"],
            },
        }, ""

    references = {}
    speakers = sorted({item.get("speaker", "speaker-01") for item in job["segments"]})
    for speaker in speakers:
        segments = sorted(
            (item for item in job["segments"] if item.get("speaker") == speaker),
            key=lambda item: float(item["start"]),
        )
        if not segments:
            continue
        picked = reference_quality.pick_reference_windows(
            segments,
            max_overlap=SPEAKER_REF_MAX_OVERLAP,
            min_confidence=SPEAKER_REF_MIN_CONFIDENCE,
            alternates=SPEAKER_REF_ALTERNATES,
        )
        if picked["primary"] is None:
            continue
        reference, error = _extract(picked["primary"], root / f"{speaker}.wav")
        if reference is None:
            # No silent skips: a speaker without a clone reference
            # surfaces at render as "automatic reference missing" — record WHY here, at the cause.
            job.setdefault("reference_skips", {})[speaker] = error
            oplog.job_warn(job["id"], "references",
                           f"clone reference extraction failed for {speaker}: {error}")
            continue
        reference["quality"]["clean"] = picked["clean"]
        if not picked["clean"]:
            oplog.job_warn(job["id"], "references",
                           "%s: no clean solo window (best overlap=%.2f conf=%.2f); "
                           "reference may be contaminated" % (
                               speaker, reference["quality"]["overlap"],
                               reference["quality"]["confidence"]))
        spares = []
        for number, window in enumerate(picked["alternates"], 1):
            spare, error = _extract(window, root / f"{speaker}-alt{number}.wav")
            if spare is None:
                oplog.job_warn(job["id"], "references",
                               f"alternate reference {number} extraction failed for {speaker}: {error}")
                continue
            spares.append(spare)
        reference["alternates"] = spares
        references[speaker] = reference

    # A re-picked reference must not serve lines cached from the OLD (possibly
    # poisoned) reference: line_signature keys on the voice OPTION string, not the
    # reference audio, so purge that speaker's cached lines here.
    previous = job.get("speaker_references") or {}
    lines_dir = artifacts / "lines"
    for speaker, reference in references.items():
        old_segments = (previous.get(speaker) or {}).get("source_segments")
        if old_segments and list(old_segments) != list(reference["source_segments"]):
            purged = 0
            manifest = load_line_manifest(lines_dir)
            for item in job["segments"]:
                if item.get("speaker") != speaker:
                    continue
                cached = lines_dir / f"line-{int(item['i']):05d}.wav"
                if cached.is_file():
                    cached.unlink(missing_ok=True)
                    purged += 1
                manifest.pop(str(int(item["i"])), None)
            if purged:
                save_line_manifest(lines_dir, manifest)
                oplog.job_warn(job["id"], "references",
                               f"{speaker}: reference window changed — purged {purged} "
                               "cached line(s) synthesized from the old reference")
    job["speaker_references"] = references


def _analyze_quality(job_id: str, lease=None) -> None:
    job = load_job(job_id)
    adapters.set_job_context(job_id)
    _, source, artifacts = _paths(job)
    artifacts.mkdir(exist_ok=True)
    job["status"] = "running"
    job["error"] = None
    clear_cancel(job_id)
    save_job(job)

    _check_cancel(job)
    event(job, "extracting", "Extracting local tracks for the quality pipeline. "
          f"[code {oplog.code_sha()}]", 6)
    _, full = extract_audio(source, artifacts)
    job["artifacts"]["full_audio"] = full.name
    save_job(job)

    _check_cancel(job)
    event(job, "separating", "Separating dialogue from music and effects with local Demucs htdemucs_ft.", 17)
    if lease is not None:
        lease.ensure_active()
    stems = adapters.separate_dialogue(full, artifacts / "separation")
    vocals, bed = Path(stems["vocals"]), Path(stems["bed"])
    job["artifacts"]["dialogue_stem"] = vocals.relative_to(artifacts).as_posix()
    job["artifacts"]["source_bed"] = bed.relative_to(artifacts).as_posix()
    save_job(job)

    _check_cancel(job)
    event(job, "transcribing", "Transcribing the dialogue stem with local Whisper Large V3 on the leased GPU.", 32)
    if lease is not None:
        lease.ensure_active()
    segments = adapters.transcribe_quality(vocals, job["settings"]["source_language"])
    if not segments:
        raise StageError("quality transcription produced no speech segments")
    job["segments"] = segments
    save_job(job)

    _check_cancel(job)
    if job["settings"].get("source_language") == "ja":
        event(job, "aligning-source", "Refining Japanese speech windows with the pinned local forced aligner.", 41)
        if lease is not None:
            lease.ensure_active()
        aligned_source = adapters.align_quality(
            vocals, job["segments"], job["settings"]["source_language"]
        )
        job["source_alignment"] = apply_forced_alignment(job["segments"], aligned_source)
        save_job(job)

    _check_cancel(job)
    event(job, "diarizing", "Assigning exclusive speakers with pinned local pyannote Community-1.", 50)
    if lease is not None:
        lease.ensure_active()
    diarization = adapters.diarize_quality(vocals, job["segments"], job["settings"]["speaker_count"])
    _record_speaker_evidence(job, diarization)
    save_job(job)

    _check_cancel(job)
    event(job, "translating", "Building a reviewable English script with the pinned local translator.", 64)
    job["segments"] = adapters.translate(
        job["segments"], job["settings"]["source_language"], job["settings"]["target_language"]
    )
    _check_cancel(job)
    event(job, "subtitles", "Checking locally for an embedded English text subtitle track.", 68)
    try:
        job["subtitle_harvest"] = harvest_embedded_english(source, artifacts, job["segments"])
    except Exception as exc:
        job["subtitle_harvest"] = {"status": "failed", "mapped": 0}
        oplog.job_warn(job_id, "subtitles", f"optional subtitle harvest unavailable: {type(exc).__name__}")
    if job["settings"].get("emotion_policy") == "source-energy-v1":
        job["delivery_summary"] = analyze_delivery(vocals, job["segments"])

    if job["settings"].get("song_policy", "dub-all-v1") == "skip-detected-v1":
        event(job, "songs", "EXPERIMENTAL song skip (in testing): detecting song ranges (chapters + subtitle styles) to leave un-dubbed.", 69)
        job["song_summary"] = detect_songs(source, artifacts / "songscan", job["segments"])
        summary = job["song_summary"]
        event(job, "songs", "EXPERIMENTAL song detection: %s - %d segment(s) marked skip (%.1fs)."
              % (summary["status"], summary["segments_skipped"], summary["skipped_seconds"]), 69)
    # Analysis rebuilds segments wholesale — restore the reviewer's hand-marked
    # song ranges (they outrank the detector).
    restored = replay_manual_marks(job)
    if restored:
        event(job, "songs", "%d manual song mark(s) restored after analysis." % restored, 69)

    _speaker_references(job, vocals, artifacts)
    voice_entries = adapters.voice_options()
    labels = {entry["option"]: entry["label"] for entry in voice_entries}
    options = [entry["option"] for entry in voice_entries]
    speakers = sorted({item.get("speaker", "speaker-01") for item in job["segments"]})
    # VOICE BANK: a speaker the reviewer (or a logged clear-zone auto-match) has
    # mapped to a series character keeps that character's PERSISTENT reference, so the same
    # character sounds the same in every episode. qwen-auto stays the default for unmapped
    # speakers - its per-episode re-extraction is exactly the cross-episode drift the bank
    # exists to stop, so it must never shadow a bank assignment.
    bank_voices = _bank_voice_options(job)
    # Mute option for phantom labels: a muted label synthesizes
    # nothing — its slots stay music/effects only. Always selectable.
    options.append("mute:")
    labels["mute:"] = "Mute · do not synthesize (phantom/noise label)"
    for speaker in speakers:
        speaker_segments = [item for item in job["segments"]
                            if item.get("speaker") == speaker]
        voiced = sum(float(item.get("voiced_duration") or 0.0) for item in speaker_segments)
        if voiced <= 0.0 and speaker not in bank_voices:
            # AUTO-QUARANTINE: zero voiced evidence = diarization/ASR phantom (seven
            # hallucinated blips once got a contaminated clone voice that babbled over
            # real dialogue). Mute by default; the studio can override.
            job["speaker_voices"][speaker] = "mute:"
            for item in speaker_segments:
                item["attribution_suspect"] = True
            oplog.job_warn(job["id"], "casting",
                           f"{speaker}: zero voiced evidence across {len(speaker_segments)} "
                           "segment(s) — quarantined as a phantom (muted by default)")
            continue
        banked = bank_voices.get(speaker)
        if banked:
            option, label = banked
            options.insert(0, option)
            labels[option] = label
            job["speaker_voices"].setdefault(speaker, option)
        elif speaker in job["speaker_references"]:
            option = f"qwen-auto:{speaker}"
            options.insert(0, option)
            labels[option] = f"Qwen3 auto-reference · {speaker}"
            job["speaker_voices"].setdefault(speaker, option)
        else:
            job["speaker_voices"].setdefault(speaker, options[0])
    job["available_voices"] = list(dict.fromkeys(options))
    job["voice_labels"] = labels
    job["status"] = "review"
    event(job, "review", "Quality analysis complete. Review speaker, translation, and reference choices.", 72)


def _bank_voice_options(job: dict) -> dict:
    """{speaker: (option, label)} for every speaker the reviewer mapped to a bank character
    that has a usable reference. Fail-quiet BY REPORT: a mapping whose character lost its
    clip falls back to the normal default, and the gap is an oplog event, not a crash."""
    mapping = job.get("speaker_characters") or {}
    series = str(job.get("voice_bank_series") or "").strip()
    if not mapping or not series:
        return {}
    from . import voice_bank as _vb
    try:
        slug = _vb.series_slug(series)
        bank = {c["id"]: c for c in _vb.load_bank(slug)["characters"]}
    except Exception as exc:
        oplog.job_event(job.get("id", "?"), "voice-bank", f"bank unreadable ({type(exc).__name__}); using defaults")
        return {}
    out = {}
    for speaker, character_id in mapping.items():
        character = bank.get(character_id)
        if character and character.get("reference_clip") and Path(character["reference_clip"]).is_file():
            out[str(speaker)] = (f"bank:{slug}:{character_id}",
                                 f"Series voice · {character['name']}")
        else:
            oplog.job_event(job.get("id", "?"), "voice-bank",
                            f"{speaker} mapped to a character without a usable reference; using default")
    return out


def _qwen_line(job: dict, artifacts: Path, segment: dict, voice: str, destination: Path) -> dict:
    if voice.startswith("bank:"):
        # bank:<series-slug>:<character-id> -> the character's persistent reference
        from . import voice_bank as _vb
        try:
            _, slug, character_id = voice.split(":", 2)
            character = next(c for c in _vb.load_bank(slug)["characters"] if c["id"] == character_id)
        except (ValueError, StopIteration) as exc:
            raise StageError(f"bank voice not found: {voice}") from exc
        reference_audio = Path(str(character.get("reference_clip") or ""))
        if not reference_audio.is_file():
            raise StageError(f"bank reference clip missing for {character.get('name')}")
        return {
            "text": str(segment.get("translation") or "").strip(),
            "language": "English",
            "reference_audio": str(reference_audio),
            "reference_text": str(character.get("reference_transcript") or ""),
            "output": str(destination),
            # Bank voices are reviewer-curated series voices: the runaway guard may
            # RETRY them but never auto-swap the reference (no fallbacks).
            "slot_seconds": round(float(segment["end"]) - float(segment["start"]), 3),
            "fallback_references": [],
        }
    fallbacks = []
    if voice.startswith("qwen-auto-alt:"):
        # qwen-auto-alt:<speaker>:<n> — a spare clean reference window as a castable
        # candidate (the studio's alternates). Alternates always carry transcripts.
        try:
            _, speaker, number = voice.split(":", 2)
            alternates = (job.get("speaker_references", {}).get(speaker) or {}).get("alternates") or []
            alternate = alternates[int(number) - 1]
        except (ValueError, IndexError) as exc:
            raise StageError(f"alternate reference missing: {voice}") from exc
        return {
            "text": str(segment.get("translation") or "").strip(),
            "language": "English",
            "reference_audio": str(artifacts / alternate["file"]),
            "reference_text": str(alternate.get("text") or ""),
            "output": str(destination),
            "slot_seconds": round(float(segment["end"]) - float(segment["start"]), 3),
            "fallback_references": [],
        }
    if voice.startswith("qwen-auto-xv:"):
        # Same auto reference, x-vector-only clone (no transcript conditioning) —
        # the studio's escape hatch when ICL mode carries source-language accent.
        speaker = voice.removeprefix("qwen-auto-xv:")
        reference = job.get("speaker_references", {}).get(speaker)
        if not reference:
            raise StageError(f"automatic reference missing for {speaker}")
        return {
            "text": str(segment.get("translation") or "").strip(),
            "language": "English",
            "reference_audio": str(artifacts / reference["file"]),
            "reference_text": "",
            "output": str(destination),
            "slot_seconds": round(float(segment["end"]) - float(segment["start"]), 3),
            "fallback_references": [],
        }
    if voice.startswith("qwen-auto:"):
        speaker = voice.removeprefix("qwen-auto:")
        reference = job.get("speaker_references", {}).get(speaker)
        if not reference:
            raise StageError(f"automatic reference missing for {speaker}")
        reference_audio = artifacts / reference["file"]
        reference_text = reference.get("text", "")
        fallbacks = [
            {"audio": str(artifacts / alt["file"]), "text": alt.get("text", "")}
            for alt in (reference.get("alternates") or [])
            if (artifacts / alt["file"]).is_file()
        ]
    elif voice.startswith("qwen:voice-"):
        profile = load_profile(voice.removeprefix("qwen:"))
        reference_audio = Path(profile["reference_path"])
        reference_text = profile.get("prompt_text", "")
    else:
        raise StageError("unknown Qwen voice option")
    return {
        "text": str(segment.get("translation") or "").strip(),
        "language": "English",
        "reference_audio": str(reference_audio),
        "reference_text": reference_text,
        "output": str(destination),
        "slot_seconds": round(float(segment["end"]) - float(segment["start"]), 3),
        "fallback_references": fallbacks,
    }


def _render_quality_synth(job_id: str, lease=None) -> None:
    """GPU half of the quality render: stale-line purge + Qwen synthesis. Split from
    the CPU tail so the queue can release the GPU lease the
    moment synthesis ends. All job mutations are persisted before returning."""
    job = load_job(job_id)
    adapters.set_job_context(job_id)
    _, source, artifacts = _paths(job)
    if not job.get("segments"):
        raise StageError("analyze the job before rendering")
    bed_name = job.get("artifacts", {}).get("source_bed")
    if not bed_name:
        raise StageError("the separated source bed is missing; rerun quality analysis")
    job["status"] = "running"
    job["error"] = None
    clear_cancel(job_id)
    save_job(job)
    lines, aligned = artifacts / "lines", artifacts / "aligned"
    lines.mkdir(exist_ok=True)
    aligned.mkdir(exist_ok=True)
    remove_stale_lines(lines, job["segments"])
    for stale in aligned.glob("line-*.wav"):
        stale.unlink()
    _drop_song_lines(job, lines)
    # A crashed previous run (or a prior line repair) leaves a stale progress.json that
    # public_job serves as live progress the moment the synthesizing stage starts —
    # clear it BEFORE the stage flips.
    clear_synth_progress(lines)

    event(job, "synthesizing", "Cloning stable local English voices with Qwen3-TTS 1.7B. "
          f"[code {oplog.code_sha()}]", 78)
    qwen_lines = []
    qwen_cache = []
    for segment in job["segments"]:
        _check_cancel(job)
        text = str(segment.get("translation") or "").strip()
        if not text or _song_skipped(job, segment):
            continue
        speaker = segment.get("speaker", "speaker-01")
        voice = job["speaker_voices"].get(speaker, f"qwen-auto:{speaker}")
        destination = lines / f"line-{int(segment['i']):05d}.wav"
        if voice.startswith("mute"):
            # Quarantined/muted label: synthesize nothing and remove any wav a
            # previous voice left, or the old voice would still reach the mix.
            destination.unlink(missing_ok=True)
            continue
        if reusable_line(lines, segment, voice, job["settings"]):
            continue
        destination.unlink(missing_ok=True)
        if voice.startswith("qwen"):
            qwen_lines.append(_qwen_line(job, artifacts, segment, voice, destination))
            qwen_cache.append((segment, voice, destination))
        else:
            adapters.synthesize_voice(text, voice, destination, job["settings"]["target_language"])
            record_line(lines, segment, voice, job["settings"])
    if qwen_lines:
        if lease is not None:
            lease.ensure_active()
        result = adapters.synthesize_quality_batch(
            qwen_lines,
            seed=int(job["settings"].get("tts_seed", 1986)),
            cancel_file=cancel_path(job_id),
            batch_size=int(TTS_BATCH_SIZE),
        )
        if result.get("cancelled"):
            raise JobCancelled("job cancelled by the user")
        runaways = result.get("runaways") or []
        if runaways:
            job["runaway_summary"] = {"count": len(runaways), "events": runaways}
            event(job, "synthesizing",
                  "%d runaway line(s) auto-handled by the TTS guard (retry/fallback "
                  "reference/kept-shortest — see the job oplog)." % len(runaways), 84)
        for segment, voice, destination in qwen_cache:
            if destination.is_file():
                record_line(lines, segment, voice, job["settings"])
    # A/F contract: persist every synth-phase job mutation (runaway_summary etc.)
    # before returning — the post half re-loads the job, possibly on a fresh thread.
    save_job(job)


def _render_quality_post(job_id: str) -> None:
    """CPU tail of the quality render (align/mix/mux) — no GPU needed. May run on a
    tail thread while the next queued episode synthesizes (config
    QUEUE_OVERLAP_POST_STAGES, ships OFF)."""
    job = load_job(job_id)
    adapters.set_job_context(job_id)
    _, source, artifacts = _paths(job)
    lines, aligned = artifacts / "lines", artifacts / "aligned"
    bed_name = job.get("artifacts", {}).get("source_bed")
    if not bed_name:
        raise StageError("the separated source bed is missing; rerun quality analysis")

    timing = get_timing_policy(job["settings"].get("timing_policy", "segment-window-v1"))
    clear_synth_progress(lines)
    event(job, "aligning", "Fitting dialogue to the reviewed word/segment windows.", 86)
    next_start = _next_rendered_starts(job, job["segments"], lines)
    for segment in job["segments"]:
        _check_cancel(job)
        source_line = lines / f"line-{int(segment['i']):05d}.wav"
        if not source_line.exists():
            continue
        segment["alignment"] = align_line(
            source_line,
            aligned / source_line.name,
            float(segment["end"]) - float(segment["start"]),
            float(timing["min_tempo"]),
            float(timing["max_tempo"]),
            fit_mode=str(timing["fit_mode"]),
            trim_silence=bool(timing.get("trim_silence")),
            max_output_s=next_start.get(int(segment["i"])),
        )
    job["qc_summary"] = audit_timing(job["segments"], aligned)
    _warn_runaway(job)
    song_pass = _restore_song_vocals(job, artifacts)
    if song_pass:
        event(job, "songs", "EXPERIMENTAL song skip: restored original singing for %d song window(s) "
              "(unity passthrough - no dialogue shaping, no ducking)." % len(song_pass), 90)
    song_pass.extend(_nonverbal_pass(job, artifacts))
    save_job(job)

    _check_cancel(job)
    event(job, "mixing", "Mixing English dialogue over the separated music/effects bed.", 92)
    mixed = artifacts / "english-mix.wav"
    mix_policy = get_mix_policy(job["settings"].get("mix_policy", "legacy-v1"))
    mix_policy.update(get_space_policy(job["settings"].get("space_policy", "dry-v1")))
    count = build_mix(
        artifacts / bed_name,
        job["segments"],
        aligned,
        mixed,
        float(job["settings"]["source_bed_gain"]),
        float(job["settings"]["dialogue_gain"]),
        policy=mix_policy,
        passthrough=song_pass,
    )
    job["artifacts"].update({"mixed_audio": mixed.name, "rendered_lines": count})
    event(job, "muxing", "Muxing the reviewed dub with the untouched source video stream.", 97)
    _check_cancel(job)
    final = output_path(job_id)
    mux(source, mixed, final)
    job["artifacts"]["output"] = final.name
    job.setdefault("exports", []).append({
        "kind": "primary",
        "file": final.name,
        "mix_policy": job["settings"].get("mix_policy"),
        "timing_policy": job["settings"].get("timing_policy"),
    })
    _link_friendly_output(job, final)
    job["status"] = "complete"
    event(job, "complete", "Quality-first local AutoDub export completed.", 100)
    oplog.job_summary(job["id"], "complete", job.get("events", []))


def _render_quality(job_id: str, lease=None) -> None:
    _render_quality_synth(job_id, lease=lease)
    _render_quality_post(job_id)


def _fail_job(job_id: str, action: str, exc: Exception) -> None:
    job = load_job(job_id)
    job["status"] = "failed"
    job["error"] = str(exc)
    job["error_detail"] = traceback.format_exc()[-2000:]
    oplog.job_error(job_id, job.get("stage", action), traceback.format_exc())
    event(job, "failed", f"Quality {action} stopped safely: {type(exc).__name__}", job.get("progress", 0))
    oplog.job_summary(job_id, "failed", job.get("events", []))


def _quality_action(job_id: str, action: str, target) -> None:
    load_job(job_id)
    try:
        with gpu_lease(f"{action}:{job_id}") as lease:
            target(job_id, lease=lease)
    except JobCancelled:
        job = load_job(job_id)
        _cancelled(job)
    except Exception as exc:
        _fail_job(job_id, action, exc)


def render_overlapped(job_id: str, *, gpu_authorized: bool = False):
    """Queue-only render entry (config QUEUE_OVERLAP_POST_STAGES):
    hold the GPU lease ONLY through synthesis, then run the CPU tail (align/mix/mux)
    on a daemon thread and return it — the caller MUST join it before finalizing the
    item. CPU profiles render inline and return None. The single-job /api render
    path is untouched (render() below)."""
    job = load_job(job_id)
    profile = job.get("settings", {}).get("quality_profile", CPU_PROFILE)
    if not profile_requires_gpu(profile):
        _render_cpu(job_id)
        return None
    if not gpu_authorized:
        raise GpuSafetyError("quality render requires a fresh GPU arm")
    try:
        with gpu_lease(f"render:{job_id}") as lease:
            _render_quality_synth(job_id, lease=lease)
    except JobCancelled:
        _cancelled(load_job(job_id))
        return None
    except Exception as exc:
        _fail_job(job_id, "render", exc)
        return None

    def _tail() -> None:
        try:
            _render_quality_post(job_id)
        except JobCancelled:
            _cancelled(load_job(job_id))
        except Exception as exc:
            _fail_job(job_id, "render", exc)

    thread = threading.Thread(target=_tail, name=f"autodub-tail-{job_id}", daemon=True)
    thread.start()
    return thread


def analyze(job_id: str, *, gpu_authorized: bool = False) -> None:
    job = load_job(job_id)
    profile = job.get("settings", {}).get("quality_profile", CPU_PROFILE)
    if not profile_requires_gpu(profile):
        _analyze_cpu(job_id)
        return
    if not gpu_authorized:
        raise GpuSafetyError("quality analysis requires a fresh GPU arm")
    _quality_action(job_id, "analyze", _analyze_quality)


def render(job_id: str, *, gpu_authorized: bool = False) -> None:
    job = load_job(job_id)
    profile = job.get("settings", {}).get("quality_profile", CPU_PROFILE)
    if not profile_requires_gpu(profile):
        _render_cpu(job_id)
        return
    if not gpu_authorized:
        raise GpuSafetyError("quality render requires a fresh GPU arm")
    _quality_action(job_id, "render", _render_quality)


def _find_segment(job: dict, index: int) -> dict:
    try:
        return next(item for item in job.get("segments", []) if int(item["i"]) == int(index))
    except StopIteration as exc:
        raise ValueError("unknown segment") from exc


def _synthesize_one(job: dict, segment: dict, destination: Path) -> tuple[str, bool]:
    artifacts = job_dir(job["id"]) / "artifacts"
    text = str(segment.get("translation") or "").strip()
    if not text:
        raise ValueError("the reviewed English line is empty")
    speaker = segment.get("speaker", "speaker-01")
    voice = job["speaker_voices"].get(speaker, f"qwen-auto:{speaker}")
    if voice.startswith("mute"):
        raise StageError(f"{speaker} is muted — assign it a voice in casting to synthesize")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    if voice.startswith("qwen"):
        adapters.synthesize_quality_batch(
            [_qwen_line(job, artifacts, segment, voice, destination)],
            seed=int(job["settings"].get("tts_seed", 1986)),
        )
        return voice, True
    adapters.synthesize_voice(text, voice, destination, job["settings"]["target_language"])
    return voice, False


def preview_line(job_id: str, index: int, *, gpu_authorized: bool = False) -> None:
    job = load_job(job_id)
    segment = _find_segment(job, index)
    speaker = segment.get("speaker", "speaker-01")
    voice = job["speaker_voices"].get(speaker, f"qwen-auto:{speaker}")
    if voice.startswith("mute"):
        # refuse BEFORE the status flips — a muted preview must be a 400, not a
        # job stuck in 'failed'
        raise ValueError(f"{speaker} is muted — assign it a voice in casting first")
    needs_gpu = voice.startswith("qwen")
    if needs_gpu and not gpu_authorized:
        raise GpuSafetyError("this line preview requires a fresh GPU arm")
    previous_status = job.get("status", "review")
    job["status"] = "running"
    event(job, "previewing", f"Rendering one local line preview ({int(index)}).", 74)
    destination = job_dir(job_id) / "artifacts" / "previews" / f"line-{int(index):05d}.wav"
    try:
        if needs_gpu:
            with gpu_lease(f"preview:{job_id}") as lease:
                lease.ensure_active()
                _synthesize_one(job, segment, destination)
        else:
            _synthesize_one(job, segment, destination)
        job["status"] = previous_status if previous_status in {"review", "complete"} else "review"
        event(job, "review", f"Line preview ready ({int(index)}).", 74)
    except Exception:
        job["status"] = "failed"
        job["error"] = "line preview failed"
        job["error_detail"] = traceback.format_exc()[-2000:]
        oplog.job_error(job_id, "preview", traceback.format_exc())
        event(job, "failed", "Line preview stopped safely.", 74)


def remix_existing(job_id: str, *, label: str = "repair") -> Path:
    job = load_job(job_id)
    _, source, artifacts = _paths(job)
    lines = artifacts / "lines"
    aligned = artifacts / "aligned"
    if not lines.is_dir():
        raise StageError("render line audio before remixing")
    aligned.mkdir(exist_ok=True)
    # Remix honors song-skip exactly like a primary render — purge cached song
    # dubs (lines AND stale aligned copies) so an old cached song WAV can't re-enter.
    for segment in job.get("segments", []):
        if _song_skipped(job, segment):
            (aligned / f"line-{int(segment['i']):05d}.wav").unlink(missing_ok=True)
    _drop_song_lines(job, lines)
    # A purged line (merge/recast/mute in the casting studio) has no lines/ wav —
    # its stale aligned/ copy would re-enter this mix in the OLD voice otherwise
    # (the stale-voice failure).
    for stale in aligned.glob("line-*.wav"):
        if not (lines / stale.name).is_file():
            stale.unlink(missing_ok=True)
    timing = get_timing_policy(job["settings"].get("timing_policy", "segment-window-v1"))
    next_start = _next_rendered_starts(job, job.get("segments", []), lines)
    for segment in job.get("segments", []):
        if _song_skipped(job, segment):
            continue
        source_line = lines / f"line-{int(segment['i']):05d}.wav"
        if source_line.is_file():
            segment["alignment"] = align_line(
                source_line,
                aligned / source_line.name,
                float(segment["end"]) - float(segment["start"]),
                float(timing["min_tempo"]),
                float(timing["max_tempo"]),
                fit_mode=str(timing["fit_mode"]),
                trim_silence=bool(timing.get("trim_silence")),
                max_output_s=next_start.get(int(segment["i"])),
            )
    job["qc_summary"] = audit_timing(job["segments"], aligned)
    _warn_runaway(job)
    bed_name = job.get("artifacts", {}).get("source_bed") or job.get("artifacts", {}).get("full_audio")
    if not bed_name:
        raise StageError("the source bed is missing")
    song_pass = []
    if job.get("artifacts", {}).get("source_bed"):
        song_pass = _restore_song_vocals(job, artifacts)   # vocals-stripped bed: put song singing back
        song_pass = song_pass + _nonverbal_pass(job, artifacts)
    mix_policy = get_mix_policy(job["settings"].get("mix_policy", "legacy-v1"))
    mix_policy.update(get_space_policy(job["settings"].get("space_policy", "dry-v1")))
    mixed = artifacts / f"english-mix-{label}.wav"
    build_mix(
        artifacts / bed_name,
        job["segments"],
        aligned,
        mixed,
        float(job["settings"]["source_bed_gain"]),
        float(job["settings"]["dialogue_gain"]),
        policy=mix_policy,
        passthrough=song_pass,
    )
    output = output_variant_path(job_id, label)
    mux(source, mixed, output)
    job["artifacts"]["mixed_audio"] = mixed.name
    job["artifacts"]["output"] = output.name
    job.setdefault("exports", []).append({
        "kind": label,
        "file": output.name,
        "mix_policy": job["settings"].get("mix_policy"),
        "timing_policy": job["settings"].get("timing_policy"),
    })
    _link_friendly_output(job, output)
    job["status"] = "complete"
    event(job, "complete", f"Local {label} export completed.", 100)
    return output


def realign_existing(job_id: str, *, gpu_authorized: bool = False) -> Path:
    """Reuse synthesized lines, refine source windows, then remix/mux a review variant."""
    if not gpu_authorized:
        raise GpuSafetyError("source realignment requires a fresh GPU arm")
    job = load_job(job_id)
    if job.get("settings", {}).get("source_language") != "ja":
        raise StageError("the pinned forced-alignment repair currently supports Japanese source only")
    artifacts = job_dir(job_id) / "artifacts"
    stem_name = job.get("artifacts", {}).get("dialogue_stem")
    if not stem_name or not (artifacts / stem_name).is_file():
        raise StageError("the separated dialogue stem is missing; rerun quality analysis")
    previous_status = job.get("status")
    job["status"] = "running"
    event(job, "aligning-source", "Re-aligning existing reviewed lines without re-synthesizing voices.", 88)
    save_job(job)
    try:
        with gpu_lease(f"realign:{job_id}") as lease:
            lease.ensure_active()
            refined = adapters.align_quality(
                artifacts / stem_name,
                job["segments"],
                job["settings"]["source_language"],
            )
            job["source_alignment"] = apply_forced_alignment(job["segments"], refined)
            save_job(job)
            output = remix_existing(job_id, label="realigned")
        return output
    except Exception:
        job = load_job(job_id)
        job["status"] = previous_status or "review"
        job["error"] = "source realignment failed"
        job["error_detail"] = traceback.format_exc()[-2000:]
        save_job(job)
        raise


def repair_line(job_id: str, index: int, *, gpu_authorized: bool = False) -> None:
    job = load_job(job_id)
    segment = _find_segment(job, index)
    if _song_skipped(job, segment):
        # Repair must not quietly re-dub a detected song. Turning the song policy
        # off (app toggle) is the explicit way to dub it.
        raise StageError("this line is song-skipped (EXPERIMENTAL song detection) — "
                         "turn Song skip off in Run settings to dub it")
    speaker = segment.get("speaker", "speaker-01")
    voice = job["speaker_voices"].get(speaker, f"qwen-auto:{speaker}")
    if voice.startswith("mute"):
        # refuse BEFORE the status flips — a muted repair must be a 400, not a
        # job stuck in 'failed'
        raise ValueError(f"{speaker} is muted — assign it a voice in casting first")
    needs_gpu = voice.startswith("qwen")
    if needs_gpu and not gpu_authorized:
        raise GpuSafetyError("this line repair requires a fresh GPU arm")
    job["status"] = "running"
    lines = job_dir(job_id) / "artifacts" / "lines"
    # Repairs both serve AND strand stale progress sidecars (the worker writes its
    # done=1/total=1 progress.json into this dir) — clear BEFORE the stage flips to
    # "synthesizing", which is when public_job starts serving the sidecar.
    clear_synth_progress(lines)
    event(job, "synthesizing", f"Re-synthesizing reviewed line {int(index)} only.", 82)
    destination = lines / f"line-{int(index):05d}.wav"
    try:
        if needs_gpu:
            with gpu_lease(f"repair:{job_id}") as lease:
                lease.ensure_active()
                _synthesize_one(job, segment, destination)
        else:
            _synthesize_one(job, segment, destination)
        record_line(lines, segment, voice, job["settings"])
        event(job, "mixing", "Rebuilding the mix from cached unchanged lines.", 92)
        remix_existing(job_id, label=f"repair-{int(index):05d}")
    except Exception:
        job = load_job(job_id)
        job["status"] = "failed"
        job["error"] = "line repair failed"
        job["error_detail"] = traceback.format_exc()[-2000:]
        oplog.job_error(job_id, "repair", traceback.format_exc())
        event(job, "failed", "Line repair stopped safely.", job.get("progress", 82))
