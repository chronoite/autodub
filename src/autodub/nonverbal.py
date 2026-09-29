"""Non-verbal passthrough: original screams/laughs/gasps survive quality renders.

Original non-verbal audio is NEVER removed, even when the line around it is dubbed. Quality renders mix
dialogue over a vocals-REMOVED bed, so any vocal sound Whisper dropped (screams, laughs,
gasps, breaths — exactly where the source acts hardest) used to go silent.

Fix: scan the separated dialogue stem for voiced windows with ffmpeg silencedetect,
subtract every transcript window (those are dubbed) plus restored-song windows, and slice
what remains from the ORIGINAL vocal stem into the mix's unity-gain passthrough channel
(same channel as restored songs: no dialogue shaping, no ducking under them).

Pure interval math lives in the top half (hermetically tested); ffmpeg I/O at the bottom.
All thresholds live in config.py.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import oplog
from .config import (
    FFMPEG,
    NONVERBAL_COVER_GUARD_S,
    NONVERBAL_MAX_TOTAL_S,
    NONVERBAL_MERGE_GAP_S,
    NONVERBAL_MIN_S,
    NONVERBAL_SILENCE_DB,
    NONVERBAL_SILENCE_MIN_S,
)


_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")


def parse_silences(ffmpeg_stderr: str) -> list[tuple[float, float]]:
    """(start, end) silence spans from silencedetect output; an open tail is dropped
    (the caller's total-duration inversion treats missing tail silence as voiced)."""
    starts = [float(value) for value in _SILENCE_START.findall(ffmpeg_stderr)]
    ends = [float(value) for value in _SILENCE_END.findall(ffmpeg_stderr)]
    return list(zip(starts, ends))


def voiced_windows(silences: list[tuple[float, float]], total_s: float) -> list[tuple[float, float]]:
    """Invert silence spans over [0, total] into voiced windows."""
    windows = []
    cursor = 0.0
    for start, end in sorted(silences):
        if start > cursor:
            windows.append((cursor, min(start, total_s)))
        cursor = max(cursor, end)
    if cursor < total_s:
        windows.append((cursor, total_s))
    return [(start, end) for start, end in windows if end - start > 0.01]


def subtract_covered(windows: list[tuple[float, float]], covers: list[tuple[float, float]],
                     *, guard: float = NONVERBAL_COVER_GUARD_S) -> list[tuple[float, float]]:
    """Voiced windows minus every covered (transcribed/dubbed) span, grown by the guard."""
    grown = sorted((max(0.0, start - guard), end + guard) for start, end in covers)
    out = []
    for w_start, w_end in windows:
        pieces = [(w_start, w_end)]
        for c_start, c_end in grown:
            next_pieces = []
            for p_start, p_end in pieces:
                if c_end <= p_start or c_start >= p_end:
                    next_pieces.append((p_start, p_end))
                    continue
                if c_start > p_start:
                    next_pieces.append((p_start, c_start))
                if c_end < p_end:
                    next_pieces.append((c_end, p_end))
            pieces = next_pieces
            if not pieces:
                break
        out.extend(pieces)
    return out


def merge_and_filter(windows: list[tuple[float, float]],
                     *, merge_gap: float = NONVERBAL_MERGE_GAP_S,
                     min_s: float = NONVERBAL_MIN_S) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(windows):
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged if end - start >= min_s]


def cap_total(windows: list[tuple[float, float]],
              *, max_total_s: float = NONVERBAL_MAX_TOTAL_S) -> tuple[list[tuple[float, float]], float]:
    """Keep windows in time order until the cap; returns (kept, dropped_seconds).
    A blown cap usually means music bleed in the stem — the caller logs it LOUDLY."""
    kept, total, dropped = [], 0.0, 0.0
    for start, end in windows:
        length = end - start
        if total + length <= max_total_s:
            kept.append((start, end))
            total += length
        else:
            dropped += length
    return kept, dropped


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [str(FFMPEG.with_name("ffprobe.exe")), "-v", "error", "-show_entries",
         "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    try:
        return float((result.stdout or "0").strip())
    except ValueError:
        return 0.0


def restore(job: dict, artifacts: Path) -> list:
    """[(pseudo_segment, wav_path)] for build_mix's passthrough channel, plus a
    job['nonverbal_summary'] audit block. Failures degrade to [] with a loud log —
    a render must never die because a scream could not be restored."""
    stem_name = job.get("artifacts", {}).get("dialogue_stem")
    stem = (artifacts / stem_name) if stem_name else None
    if not stem or not stem.is_file():
        return []
    total_s = _probe_duration(stem)
    if total_s <= 0:
        return []
    scan = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-i", str(stem),
         "-af", f"silencedetect=n={NONVERBAL_SILENCE_DB}dB:d={NONVERBAL_SILENCE_MIN_S}",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=1800,
    )
    covers = [(float(segment["start"]), float(segment["end"]))
              for segment in job.get("segments", [])]
    windows = merge_and_filter(
        subtract_covered(voiced_windows(parse_silences(scan.stderr or ""), total_s), covers))
    windows, dropped_s = cap_total(windows)
    summary = {
        "schema": 1,
        "policy": "passthrough-v1",
        "windows": len(windows),
        "seconds": round(sum(end - start for start, end in windows), 1),
        "dropped_seconds": round(dropped_s, 1),
    }
    job["nonverbal_summary"] = summary
    if dropped_s > 0:
        oplog.job_warn(job.get("id", "?"), "nonverbal",
                       f"passthrough cap hit: {dropped_s:.1f}s of voiced windows dropped "
                       "(possible music bleed in the stem)")
    if not windows:
        return []
    target_dir = artifacts / "nonverbal"
    target_dir.mkdir(exist_ok=True)
    restored = []
    failed = 0
    for index, (start, end) in enumerate(windows):
        target = target_dir / f"nv-{index:04d}.wav"
        result = subprocess.run(
            [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{start:.3f}", "-t", f"{max(0.05, end - start):.3f}",
             "-i", str(stem), str(target)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and target.is_file():
            restored.append(({"i": 90000 + index, "start": start, "end": end}, target))
        else:
            failed += 1
    if failed:
        oplog.job_warn(job.get("id", "?"), "nonverbal",
                       f"{failed} of {len(windows)} non-verbal window slices FAILED")
    oplog.job_event(job.get("id", "?"), "nonverbal",
                    f"restored {len(restored)} original non-verbal window(s), "
                    f"{summary['seconds']}s total")
    return restored
