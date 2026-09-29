"""Song detection for dub skipping — EXPERIMENTAL.

The feature is wired in but labelled experimental: the failure to guard against is skipping MORE
than songs. Songs are skipped with no hard duration caps.

Marks segments that fall inside detected song ranges so the render skips synthesizing them
and the original singing survives in the mix untouched. Two local, structural detectors:

  1. Chapter markers — anime releases commonly chapter OP/ED ("OP", "Opening", "NCED"...).
  2. ASS subtitle styles — fansub/BD subs style song lines distinctly ("OP-Romaji",
     "ED_kanji", "Karaoke", "Insert-Lyrics"...). All ass/ssa text streams are scanned,
     not just the harvested English one.

Detection is advisory and FAILS OPEN: on any failure the job dubs everything, exactly as
before this feature. Flags are set during ANALYZE so the reviewer sees them before any
render happens. Nothing here logs media names or text.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .config import FFMPEG, FFPROBE

# Chapter titles that are songs. Token-based to avoid substring traps ("Default-Top").
_SONG_TOKENS = {"op", "ed", "opening", "ending", "kara", "karaoke", "song", "songs",
                "lyric", "lyrics", "romaji", "kanji", "furigana", "insert"}
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_OPED_NUMBERED = re.compile(r"^(op|ed|ncop|nced)\d*$")
_ASS_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[.:](\d{2})")

MIN_RANGE_SECONDS = 1.0
MERGE_GAP_SECONDS = 2.0
SEGMENT_OVERLAP_FRACTION = 0.5


def _is_song_label(label: str) -> bool:
    tokens = [t for t in _TOKEN_SPLIT.split(str(label).lower()) if t]
    return any(t in _SONG_TOKENS or _OPED_NUMBERED.match(t) for t in tokens)


def _ass_seconds(value: str) -> float | None:
    match = _ASS_TIME.fullmatch(value.strip())
    if not match:
        return None
    h, m, s, cs = (int(g) for g in match.groups())
    return h * 3600 + m * 60 + s + cs / 100


def chapter_song_ranges(source: Path) -> list[tuple[float, float, str]]:
    result = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_chapters", "-of", "json", str(source)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    if result.returncode:
        return []
    ranges = []
    for chapter in json.loads(result.stdout or "{}").get("chapters") or []:
        title = str((chapter.get("tags") or {}).get("title") or "")
        if not _is_song_label(title):
            continue
        try:
            start, end = float(chapter["start_time"]), float(chapter["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            ranges.append((start, end, "chapter"))
    return ranges


def _ass_streams(source: Path) -> list[int]:
    result = subprocess.run(
        [str(FFPROBE), "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index,codec_name", "-of", "json", str(source)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    if result.returncode:
        return []
    return [int(s["index"]) for s in json.loads(result.stdout or "{}").get("streams") or []
            if str(s.get("codec_name") or "").lower() in ("ass", "ssa")]


def ass_song_ranges(source: Path, work_dir: Path) -> list[tuple[float, float, str]]:
    ranges = []
    work_dir.mkdir(parents=True, exist_ok=True)
    for index in _ass_streams(source):
        target = work_dir / f"songscan-{index}.ass"
        result = subprocess.run(
            [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
             "-map", f"0:{index}", "-c:s", "ass", str(target)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        )
        if result.returncode or not target.is_file():
            continue
        fields = []
        for raw in target.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = raw.strip()
            if line.lower().startswith("format:"):
                fields = [f.strip().lower() for f in line.split(":", 1)[1].split(",")]
            elif line.lower().startswith("dialogue:") and fields:
                parts = line.split(":", 1)[1].split(",", len(fields) - 1)
                if len(parts) < len(fields):
                    continue
                row = dict(zip(fields, parts))
                if not _is_song_label(row.get("style", "")):
                    continue
                start = _ass_seconds(row.get("start", ""))
                end = _ass_seconds(row.get("end", ""))
                if start is not None and end is not None and end > start:
                    ranges.append((start, end, "subtitle-style"))
        target.unlink(missing_ok=True)
    return ranges


def merge_ranges(ranges: list[tuple[float, float, str]]) -> list[dict]:
    merged = []
    for start, end, kind in sorted(ranges):
        if merged and start - merged[-1]["end"] <= MERGE_GAP_SECONDS:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            if kind not in merged[-1]["kinds"]:
                merged[-1]["kinds"].append(kind)
        else:
            merged.append({"start": start, "end": end, "kinds": [kind]})
    return [r for r in merged if r["end"] - r["start"] >= MIN_RANGE_SECONDS]


def mark_song_segments(segments: list[dict], ranges: list[dict]) -> int:
    marked = 0
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        duration = max(0.01, end - start)
        hit = None
        for song in ranges:
            overlap = max(0.0, min(end, song["end"]) - max(start, song["start"]))
            if overlap / duration >= SEGMENT_OVERLAP_FRACTION:
                hit = song
                break
        if hit:
            segment["song_skip"] = True
            segment["song_reason"] = "+".join(hit["kinds"])
            marked += 1
        else:
            segment.pop("song_skip", None)
            segment.pop("song_reason", None)
    return marked


def detect_songs(source: Path, work_dir: Path, segments: list[dict]) -> dict:
    """Mark song segments in place; fails open with status so review shows what happened.

    Marking uses the RAW cue/chapter ranges — merging cues across gaps manufactured song
    coverage over short spoken lines between karaoke cues. Merged ranges
    are reported for the summary only.
    """
    try:
        raw = chapter_song_ranges(source) + ass_song_ranges(source, work_dir)
        marking = [{"start": s, "end": e, "kinds": [k]} for s, e, k in raw
                   if e - s >= MIN_RANGE_SECONDS or k == "chapter"]
        marked = mark_song_segments(segments, marking)
        return {
            "experimental": True,   # surfaced in the UI as an in-development feature
            "status": "marked" if marked else "none-found",
            "ranges": len(merge_ranges(raw)),
            "segments_skipped": marked,
            "skipped_seconds": round(sum(
                float(s["end"]) - float(s["start"]) for s in segments if s.get("song_skip")
            ), 1),
        }
    except Exception as exc:   # advisory: never let detection kill analysis
        for segment in segments:   # a failed re-detection must not retain stale skips
            segment.pop("song_skip", None)
            segment.pop("song_reason", None)
        return {"experimental": True, "status": "failed", "error": type(exc).__name__,
                "ranges": 0, "segments_skipped": 0, "skipped_seconds": 0.0}


def apply_song_range(segments: list[dict], start: int, end: int, mark: bool) -> int:
    """Manual range-marking primitive (inclusive segment-index range). Marking skips
    already-flagged lines; unmarking clears detector flags too — rescuing over-flagged
    dialogue is the point. Returns the number of segments changed."""
    changed = 0
    for segment in segments:
        index = int(segment["i"])
        if index < start or index > end:
            continue
        if mark:
            if not segment.get("song_skip"):
                segment["song_skip"] = True
                segment["song_reason"] = "manual"
                changed += 1
        elif segment.get("song_skip"):
            segment.pop("song_skip", None)
            segment.pop("song_reason", None)
            changed += 1
    return changed


def replay_manual_marks(job: dict) -> int:
    """Re-apply the manual mark/unmark journal after analysis rebuilds segments —
    without this, every re-analyze silently destroys hand-marked song ranges. Returns total segments changed."""
    changed = 0
    for operation in job.get("manual_song_ranges") or []:
        changed += apply_song_range(
            job["segments"], int(operation["start"]), int(operation["end"]),
            bool(operation.get("mark", True)),
        )
    return changed


def mark_song_range(job_id: str, *, start: int, end: int, mark: bool) -> dict:
    """Manual mark/unmark over HTTP: validate, apply, journal the operation, refresh
    song_summary, and log ONE job event (indices/counts only — never dialogue text)."""
    from .state import event, load_job

    job = load_job(job_id)
    if job.get("status") == "running":
        raise ValueError("wait for the current stage to finish")
    segments = job.get("segments") or []
    if not segments:
        raise ValueError("analyze the job before marking songs")
    if start > end or not any(start <= int(s["i"]) <= end for s in segments):
        raise ValueError("invalid segment range")
    changed = apply_song_range(segments, start, end, mark)
    marked_total = sum(1 for s in segments if s.get("song_skip"))
    policy_active = job.get("settings", {}).get("song_policy") == "skip-detected-v1"
    if changed:
        journal = list(job.get("manual_song_ranges") or [])
        journal.append({"start": int(start), "end": int(end), "mark": bool(mark)})
        job["manual_song_ranges"] = journal[-200:]
        job["song_summary"] = {
            "experimental": True,
            "status": "manually-marked",
            "ranges": len(journal),
            "segments_skipped": marked_total,
            "skipped_seconds": round(sum(
                float(s["end"]) - float(s["start"]) for s in segments if s.get("song_skip")
            ), 1),
        }
        event(job, "songs",
              "%d line(s) %s as song manually (segments %d-%d) - original audio kept."
              % (changed, "marked" if mark else "unmarked", start, end), 72)
    return {
        "changed": changed,
        "marked_total": marked_total,
        "start": int(start),
        "end": int(end),
        "mark": bool(mark),
        "song_policy": job.get("settings", {}).get("song_policy"),
        "policy_active": policy_active,
    }
