"""Opaque, local-only media surfaces used by the review UI."""
from __future__ import annotations

import re
from pathlib import Path

from .media import run_ffmpeg
from .state import job_dir, load_job
from .voice_profiles import load_profile
from .workflow import render_srt


def _segment(job: dict, index: int) -> dict:
    try:
        return next(item for item in job.get("segments", []) if int(item["i"]) == index)
    except StopIteration as exc:
        raise ValueError("unknown segment") from exc


def source_preview(job_id: str, index: int) -> Path:
    job = load_job(job_id)
    segment = _segment(job, index)
    root = job_dir(job_id)
    artifacts = root / "artifacts"
    source_name = job.get("artifacts", {}).get("full_audio")
    source = artifacts / source_name if source_name else root / job["source"]["file"]
    output = artifacts / "previews" / f"source-{index:05d}.wav"
    if not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        start = max(0.0, float(segment["start"]) - 0.25)
        duration = max(0.5, float(segment["end"]) - start + 0.25)
        run_ffmpeg(
            [
                "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
                "-vn", "-ar", "48000", "-ac", "2", str(output),
            ],
            "source line preview",
        )
    return output


def evidence_audio(job_id: str, index: int, lang: str | None = None) -> Path:
    """Return a short, lazily materialized source-audio sample for one opaque segment ID.

    ``lang`` selects a language-tagged
    audio track from the ORIGINAL container (e.g. the official English dub on dual-audio
    sources) instead of the analysis track. Cached per language; identification aid
    only — analysis, cloning, and the bank stay on the source-language track."""
    if not lang:
        return source_preview(job_id, index)
    if not re.fullmatch(r"[a-z]{2,3}", lang):
        raise ValueError("bad language tag")
    job = load_job(job_id)
    segment = _segment(job, index)
    root = job_dir(job_id)
    source_name = str(job.get("source", {}).get("file") or "")
    if not source_name or Path(source_name).name != source_name:
        raise ValueError("invalid job source")
    source = root / source_name
    output = root / "artifacts" / "previews" / f"source-{index:05d}-{lang}.wav"
    if not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        start = max(0.0, float(segment["start"]) - 0.25)
        duration = max(0.5, float(segment["end"]) - start + 0.25)
        run_ffmpeg(
            ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
             "-map", f"0:a:m:language:{lang}", "-vn", "-ar", "48000", "-ac", "2", str(output)],
            f"{lang} track line preview",
        )
    return output


def evidence_frame(job_id: str, index: int) -> Path:
    """Return a lazy midpoint still for human review; no visual detector is consulted."""
    job = load_job(job_id)
    segment = _segment(job, index)
    root = job_dir(job_id)
    source_name = str(job.get("source", {}).get("file") or "")
    if not source_name or Path(source_name).name != source_name:
        raise ValueError("invalid job source")
    source = root / source_name
    output = root / "artifacts" / "evidence" / "frames" / f"frame-{index:05d}.jpg"
    if not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        midpoint = max(0.0, (float(segment["start"]) + float(segment["end"])) / 2.0)
        run_ffmpeg(
            ["-ss", f"{midpoint:.3f}", "-i", str(source), "-frames:v", "1", "-an", "-q:v", "3", str(output)],
            "speaker evidence frame",
        )
    return output


def _video_window(start: float, end: float, minimum: float, maximum: float) -> tuple[float, float]:
    """Pure: expand a short segment around its midpoint to `minimum`, clamp a long one to
    `maximum` from its start. Never negative."""
    start, end = float(start), float(end)
    if end - start < minimum:
        mid = (start + end) / 2.0
        start, end = mid - minimum / 2.0, mid + minimum / 2.0
    if end - start > maximum:
        end = start + maximum
    if start < 0:
        end, start = end - start, 0.0
    return round(start, 3), round(end, 3)


def evidence_video(job_id: str, index: int, lang: str | None = None,
                   pad: float = 0.0) -> Path:
    """Lazy, cached identity clip: the segment's VIDEO with the source audio, so the reviewer
    can see who is actually talking when the still is ambiguous (multi-person shot, no
    face). Same opaque/lazy pattern as evidence_frame; small encode for phone loads.
    ``lang`` swaps in a language-tagged audio track (official dub) — see evidence_audio."""
    from .config import EVIDENCE_VIDEO_HEIGHT, EVIDENCE_VIDEO_MAX_S, EVIDENCE_VIDEO_MIN_S
    if lang and not re.fullmatch(r"[a-z]{2,3}", lang):
        raise ValueError("bad language tag")
    pad = float(pad or 0.0)
    if not 0.0 <= pad <= 24.0:
        raise ValueError("pad must be 0-24 seconds")
    if pad != int(pad):
        # the cache stem keys on int(pad) — fractional pads would alias to a
        # neighbor's cached clip and silently serve the wrong window
        raise ValueError("pad must be a whole number of seconds")
    job = load_job(job_id)
    segment = _segment(job, index)
    root = job_dir(job_id)
    source_name = str(job.get("source", {}).get("file") or "")
    if not source_name or Path(source_name).name != source_name:
        raise ValueError("invalid job source")
    source = root / source_name
    stem = f"video-{index:05d}" + (f"-{lang}" if lang else "") \
        + (f"-p{int(pad)}" if pad else "")
    output = root / "artifacts" / "evidence" / "videos" / f"{stem}.mp4"
    if not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        start, end = _video_window(segment["start"], segment["end"],
                                   EVIDENCE_VIDEO_MIN_S, EVIDENCE_VIDEO_MAX_S)
        # identity is often unreadable in a tight window —
        # pad extends BOTH sides with surrounding conversation (ffmpeg clamps at EOF)
        start, end = max(0.0, start - pad), end + pad
        audio_map = f"0:a:m:language:{lang}" if lang else "0:a:0"
        run_ffmpeg(
            ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(source),
             "-map", "0:v:0", "-map", audio_map,     # default: source track, same as analysis
             "-vf", f"scale=-2:{EVIDENCE_VIDEO_HEIGHT}", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "26", "-c:a", "aac", "-b:a", "96k",
             "-movflags", "+faststart", str(output)],
            "speaker evidence video",
        )
    return output


def episode_video(job_id: str) -> Path:
    """Lazy, cached browser-playable remux of the WHOLE episode (when a short identity clip doesn't give the full picture, jump into the episode at
    that moment and scrub freely). Stream copy when the codecs are already MP4-safe
    (the usual H.264/AAC case — disk-bound, ~20s); falls back to re-encoding just the
    audio. ``+faststart`` puts the index up front so seeks work over Range requests."""
    job = load_job(job_id)
    root = job_dir(job_id)
    source_name = str(job.get("source", {}).get("file") or "")
    if not source_name or Path(source_name).name != source_name:
        raise ValueError("invalid job source")
    source = root / source_name
    output = root / "artifacts" / "evidence" / "episode.mp4"
    if not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        # Cut to a temp name and replace on success — a failed/concurrent cut must
        # never leave a partial file that gets cached as the finished episode
        #. NOTE: stream copy assumes H.264(+AAC-safe)
        # sources; an HEVC or AC3 source would mux fine but play black/silent in
        # the browser — probe codecs before trusting this on new source families.
        scratch = output.with_name("episode.tmp.mp4")
        try:
            try:
                run_ffmpeg(["-i", str(source), "-map", "0:v:0", "-map", "0:a:0",
                            "-c", "copy", "-movflags", "+faststart", str(scratch)],
                           "episode remux (copy)")
            except Exception:
                scratch.unlink(missing_ok=True)
                run_ffmpeg(["-i", str(source), "-map", "0:v:0", "-map", "0:a:0",
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                            "-movflags", "+faststart", str(scratch)],
                           "episode remux (audio re-encode)")
            scratch.replace(output)
        finally:
            scratch.unlink(missing_ok=True)
    return output


def rendered_preview(job_id: str, index: int) -> Path:
    job = load_job(job_id)
    _segment(job, index)
    artifacts = job_dir(job_id) / "artifacts"
    for folder in ("previews", "aligned", "lines-v3", "lines"):
        candidate = artifacts / folder / f"line-{index:05d}.wav"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("rendered line preview is not available")


def voice_reference(job_id: str, voice: str) -> Path:
    job = load_job(job_id)
    if voice not in set(job.get("available_voices", [])):
        raise ValueError("unknown voice")
    if voice.startswith("qwen-auto:"):
        speaker = voice.removeprefix("qwen-auto:")
        reference = job.get("speaker_references", {}).get(speaker)
        if not reference:
            raise FileNotFoundError("automatic reference is missing")
        path = job_dir(job_id) / "artifacts" / reference["file"]
    elif voice.startswith(("qwen:voice-", "gsv:voice-")):
        profile = load_profile(voice.split(":", 1)[1])
        path = Path(profile["reference_path"])
    elif voice.startswith("bank:"):
        # bank:<series-slug>:<character-id> — without this branch the casting studio's
        # play button returned 404 for exactly the voices being cast.
        from . import voice_bank as _vb
        try:
            _, slug, character_id = voice.split(":", 2)
            character = next(c for c in _vb.load_bank(slug)["characters"]
                             if c["id"] == character_id)
        except (ValueError, StopIteration) as exc:
            raise FileNotFoundError("bank character reference is missing") from exc
        path = Path(str(character.get("reference_clip") or ""))
    else:
        raise ValueError("this voice has no reference clip; use line preview after synthesis")
    if not path.is_file():
        raise FileNotFoundError("voice reference is missing")
    return path


def srt_export(job_id: str) -> Path:
    job = load_job(job_id)
    output = job_dir(job_id) / "artifacts" / "exports" / f"{job_id}.srt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_srt(job.get("segments", [])), encoding="utf-8", newline="\n")
    return output
