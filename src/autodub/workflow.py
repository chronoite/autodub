"""Reusable review, repair, caching, cancellation, and export primitives."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from .config import TTS_RUNAWAY_FACTOR, TTS_RUNAWAY_MIN_SLOT_S
from .state import replace_retry
from .media import wav_duration
from .state import job_dir


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        replace_retry(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def cancel_path(job_id: str) -> Path:
    return job_dir(job_id) / "cancel.requested"


def request_cancel(job_id: str) -> None:
    path = cancel_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("cancel\n", encoding="ascii")


def clear_cancel(job_id: str) -> None:
    cancel_path(job_id).unlink(missing_ok=True)


def cancellation_requested(job_id: str) -> bool:
    return cancel_path(job_id).is_file()


def write_synth_progress(
    lines_dir: Path,
    *,
    done: int,
    total: int,
    line_index: int,
    started_at: float,
    last_seconds: float,
) -> None:
    """Publish crash-safe, content-free synthesis progress for every TTS backend."""
    elapsed = max(0.0, time.monotonic() - started_at)
    average = elapsed / done if done else 0.0
    _atomic_json(
        lines_dir / "progress.json",
        {
            "done": int(done),
            "total": int(total),
            "line_index": int(line_index),
            "avg_secs": round(average, 2) if done else None,
            "last_line_secs": round(max(0.0, last_seconds), 2),
            "eta_secs": round(max(0, total - done) * average) if done else None,
        },
    )


def clear_synth_progress(lines_dir: Path) -> None:
    # Windows unlink throws WinError 32 while ANY reader (the app UI polls this file)
    # has it open — that once killed a render. Retry briefly, then degrade:
    # a stale progress file is harmless (next write overwrites it); a dead render is not.
    for name in ("progress.json", "progress.json.tmp"):
        target = lines_dir / name
        for attempt in range(8):
            try:
                target.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))


def line_signature(segment: dict, voice: str, settings: dict) -> str:
    payload = {
        "i": int(segment["i"]),
        "translation": str(segment.get("translation") or ""),
        "speaker": str(segment.get("speaker") or ""),
        "voice": voice,
        "target_language": settings.get("target_language", "en"),
        "tts_backend": settings.get("tts_backend"),
        "tts_seed": settings.get("tts_seed"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_manifest_path(lines_dir: Path) -> Path:
    return lines_dir / "manifest.json"


def load_line_manifest(lines_dir: Path) -> dict:
    try:
        value = json.loads(cache_manifest_path(lines_dir).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_line_manifest(lines_dir: Path, manifest: dict) -> None:
    _atomic_json(cache_manifest_path(lines_dir), manifest)


def reusable_line(lines_dir: Path, segment: dict, voice: str, settings: dict) -> bool:
    key = str(int(segment["i"]))
    line = lines_dir / f"line-{int(segment['i']):05d}.wav"
    return line.is_file() and load_line_manifest(lines_dir).get(key) == line_signature(segment, voice, settings)


def record_line(lines_dir: Path, segment: dict, voice: str, settings: dict) -> None:
    manifest = load_line_manifest(lines_dir)
    manifest[str(int(segment["i"]))] = line_signature(segment, voice, settings)
    save_line_manifest(lines_dir, manifest)


def remove_stale_lines(lines_dir: Path, segments: list[dict]) -> None:
    valid = {int(item["i"]) for item in segments if str(item.get("translation") or "").strip()}
    manifest = load_line_manifest(lines_dir)
    for line in lines_dir.glob("line-*.wav"):
        try:
            index = int(line.stem.removeprefix("line-"))
        except ValueError:
            continue
        if index not in valid:
            line.unlink(missing_ok=True)
            manifest.pop(str(index), None)
    save_line_manifest(lines_dir, manifest)


def apply_forced_alignment(segments: list[dict], aligned: list[dict]) -> dict:
    """Apply validated source timing without changing text, translation, or speaker IDs."""
    by_id = {int(item["i"]): item for item in aligned}
    if len(by_id) != len(segments):
        raise ValueError("forced alignment did not return every source segment")
    shifts = []
    previous_end = 0.0
    for segment in sorted(segments, key=lambda item: int(item["i"])):
        replacement = by_id.get(int(segment["i"]))
        if replacement is None:
            raise ValueError("forced alignment returned mismatched segment IDs")
        start = round(float(replacement["start"]), 3)
        end = round(float(replacement["end"]), 3)
        if start < 0 or end <= start or start < previous_end - 0.05:
            raise ValueError("forced alignment returned invalid or overlapping speech windows")
        old_start, old_end = float(segment["start"]), float(segment["end"])
        segment.setdefault("asr_window", {"start": old_start, "end": old_end})
        segment["start"], segment["end"] = start, end
        segment["words"] = list(replacement.get("words") or segment.get("words") or [])
        segment["timing_source"] = "whisperx-ja-forced-v1"
        shifts.append(max(abs(start - old_start), abs(end - old_end)))
        previous_end = end
    return {
        "method": "whisperx-ja-forced-v1",
        "segments": len(segments),
        "changed": sum(1 for value in shifts if value >= 0.01),
        "mean_max_edge_shift": round(sum(shifts) / max(1, len(shifts)), 3),
        "largest_edge_shift": round(max(shifts, default=0.0), 3),
    }


def audit_timing(
    segments: list[dict],
    lines_dir: Path,
    *,
    runaway_ratio: float = TTS_RUNAWAY_FACTOR,
    runaway_min_slot: float = TTS_RUNAWAY_MIN_SLOT_S,
) -> dict:
    summary = {"lines": 0, "missing": 0, "runaway": 0, "overrun": 0, "underrun": 0,
               "overlap": 0, "rendered_overlap": 0, "capped": 0, "tempo_limited": 0}
    previous_end = 0.0
    previous_rendered_end = 0.0
    for segment in sorted(segments, key=lambda item: float(item["start"])):
        flags = []
        slot = max(0.01, float(segment["end"]) - float(segment["start"]))
        line = lines_dir / f"line-{int(segment['i']):05d}.wav"
        duration = wav_duration(line) if line.is_file() else 0.0
        # The RAW pre-tempo synth duration is the only fit-mode-independent runaway
        # signal: exact-fit pads/trims the aligned wav to the slot, blinding the
        # duration check below (a poisoned reference once produced 327 s for a 1.4 s slot).
        raw_seconds = float(segment.get("alignment", {}).get("source_seconds") or 0.0)
        summary["lines"] += 1
        if not line.is_file():
            flags.append("missing-line")
            summary["missing"] += 1
        elif raw_seconds > runaway_ratio * max(slot, runaway_min_slot):
            flags.append("runaway")
            summary["runaway"] += 1
        elif duration > slot + 0.20:
            flags.append("overrun")
            summary["overrun"] += 1
        elif duration < max(0.15, slot * 0.45):
            flags.append("large-gap")
            summary["underrun"] += 1
        if float(segment["start"]) < previous_end - 0.05:
            flags.append("source-overlap")
            summary["overlap"] += 1
        # RENDERED overlap — source-window QC once reported "overlap: 0" on renders with
        # dozens of audible double-voice regions:
        # compare actual placed audio (start + output_seconds), not source windows.
        if line.is_file() and float(segment["start"]) < previous_rendered_end - 0.05:
            flags.append("rendered-overlap")
            summary["rendered_overlap"] += 1
        if float(segment.get("alignment", {}).get("capped_seconds") or 0.0) > 0.3:
            flags.append("hard-capped")
            summary["capped"] += 1
        if segment.get("alignment", {}).get("tempo_limited"):
            flags.append("tempo-limited")
            summary["tempo_limited"] += 1
        segment["qc"] = {
            "slot_seconds": round(slot, 3),
            "line_seconds": round(duration, 3),
            "raw_seconds": round(raw_seconds, 3),
            "delta_seconds": round(duration - slot, 3),
            "flags": flags,
        }
        previous_end = max(previous_end, float(segment["end"]))
        if line.is_file():
            placed = float(segment["start"]) + (
                float(segment.get("alignment", {}).get("output_seconds") or 0.0) or duration)
            previous_rendered_end = max(previous_rendered_end, placed)
    summary["flagged"] = sum(1 for item in segments if item.get("qc", {}).get("flags"))
    return summary


def render_srt(segments: list[dict]) -> str:
    def stamp(seconds: float) -> str:
        millis = max(0, round(float(seconds) * 1000))
        hours, millis = divmod(millis, 3_600_000)
        minutes, millis = divmod(millis, 60_000)
        secs, millis = divmod(millis, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    blocks = []
    for segment in segments:
        text = str(segment.get("translation") or "").strip()
        if text:
            blocks.append(
                f"{len(blocks) + 1}\n{stamp(segment['start'])} --> {stamp(segment['end'])}\n{text}"
            )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    result = text
    for source, replacement in sorted(glossary.items(), key=lambda item: len(item[0]), reverse=True):
        if source:
            result = result.replace(source, replacement)
    return result
