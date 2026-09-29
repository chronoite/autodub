"""Local subtitle discovery and overlap mapping.

Only text subtitle streams are admitted. Bitmap subtitle tracks remain untouched and the normal
offline translator stays in place. No media name or subtitle text leaves the job directory.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path

from .config import FFMPEG, FFPROBE


TEXT_CODECS = {"subrip", "ass", "ssa", "mov_text", "webvtt", "text"}
ENGLISH_TAGS = {"en", "eng", "english"}
_TIMESTAMP = re.compile(r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})")
_TAG = re.compile(r"<[^>]+>|\{\\[^}]+\}")
_ASS_TIMESTAMP = re.compile(r"(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{2})\.(?P<fraction>\d{1,3})")
_VTT_VOICE = re.compile(r"<v(?:\.[^\s>]*)?(?:\s+([^>]*))?>", re.IGNORECASE)


def _seconds(value: str) -> float:
    match = _TIMESTAMP.fullmatch(value.strip())
    if not match:
        raise ValueError("invalid subtitle timestamp")
    return (
        int(match["h"]) * 3600
        + int(match["m"]) * 60
        + int(match["s"])
        + int(match["ms"]) / 1000
    )


def parse_srt(text: str) -> list[dict]:
    cues = []
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    for block in blocks:
        lines = block.splitlines()
        timing_index = next((i for i, line in enumerate(lines) if " --> " in line), None)
        if timing_index is None:
            continue
        start_raw, end_raw = lines[timing_index].split(" --> ", 1)
        try:
            start = _seconds(start_raw.split()[0])
            end = _seconds(end_raw.split()[0])
        except ValueError:
            continue
        cleaned = " ".join(
            _TAG.sub("", html.unescape(line)).replace("\\N", " ").strip()
            for line in lines[timing_index + 1 :]
        )
        cleaned = " ".join(cleaned.split())
        if cleaned and end > start:
            cues.append({"start": start, "end": end, "text": cleaned})
    return cues


def _ass_seconds(value: str) -> float:
    match = _ASS_TIMESTAMP.fullmatch(value.strip())
    if not match:
        raise ValueError("invalid ASS subtitle timestamp")
    fraction = match["fraction"]
    return (
        int(match["h"]) * 3600
        + int(match["m"]) * 60
        + int(match["s"])
        + int(fraction) / (10 ** len(fraction))
    )


def _vtt_seconds(value: str) -> float:
    fields = value.strip().replace(",", ".").split(":")
    if len(fields) not in {2, 3}:
        raise ValueError("invalid WebVTT subtitle timestamp")
    try:
        hours = int(fields[0]) if len(fields) == 3 else 0
        minutes = int(fields[-2])
        seconds = float(fields[-1])
    except ValueError as exc:
        raise ValueError("invalid WebVTT subtitle timestamp") from exc
    if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
        raise ValueError("invalid WebVTT subtitle timestamp")
    return hours * 3600 + minutes * 60 + seconds


def _clean_lines(lines: list[str]) -> str:
    cleaned = " ".join(
        _TAG.sub("", html.unescape(line)).replace("\\N", " ").replace("\\n", " ").strip()
        for line in lines
    )
    return " ".join(cleaned.split())


def parse_ass(text: str) -> list[dict]:
    """Parse ASS/SSA events while retaining a non-blank Name/Actor field as evidence."""
    cues = []
    in_events = False
    fields: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.lstrip("\ufeff").strip()
        if line.startswith("[") and line.endswith("]"):
            in_events = line.casefold() == "[events]"
            fields = []
            continue
        if not in_events:
            continue
        label, separator, payload = line.partition(":")
        if not separator:
            continue
        if label.strip().casefold() == "format":
            fields = [item.strip().casefold() for item in payload.split(",")]
            continue
        if label.strip().casefold() != "dialogue" or not fields:
            continue
        values = [item.strip() for item in payload.split(",", len(fields) - 1)]
        if len(values) != len(fields) or not {"start", "end", "text"}.issubset(fields):
            continue
        record = dict(zip(fields, values))
        try:
            start, end = _ass_seconds(record["start"]), _ass_seconds(record["end"])
        except ValueError:
            continue
        cleaned = _clean_lines([record["text"]])
        if not cleaned or end <= start:
            continue
        cue = {"start": start, "end": end, "text": cleaned}
        speaker = str(record.get("name") or record.get("actor") or "").strip()
        if speaker:
            cue["speaker"] = speaker
        cues.append(cue)
    return cues


def parse_webvtt(text: str) -> list[dict]:
    """Parse WebVTT cues and retain non-blank ``<v Speaker>`` annotations."""
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", text.lstrip("\ufeff").strip()):
        lines = block.splitlines()
        timing_index = next((i for i, line in enumerate(lines) if " --> " in line), None)
        if timing_index is None:
            continue
        start_raw, end_raw = lines[timing_index].split(" --> ", 1)
        try:
            start = _vtt_seconds(start_raw.split()[0])
            end = _vtt_seconds(end_raw.split()[0])
        except ValueError:
            continue
        payload = lines[timing_index + 1 :]
        cleaned = _clean_lines(payload)
        if not cleaned or end <= start:
            continue
        cue = {"start": start, "end": end, "text": cleaned}
        voice = next(
            (
                html.unescape(match.group(1) or "").strip()
                for line in payload
                for match in [_VTT_VOICE.search(line)]
                if match and html.unescape(match.group(1) or "").strip()
            ),
            "",
        )
        if voice:
            cue["speaker"] = voice
        cues.append(cue)
    return cues


def _stream_score(stream: dict) -> tuple[int, int, int]:
    tags = stream.get("tags") or {}
    language = str(tags.get("language") or "").lower()
    title = str(tags.get("title") or "").lower()
    disposition = stream.get("disposition") or {}
    is_english = language in ENGLISH_TAGS or "english" in title
    signs_only = any(word in title for word in ("sign", "song", "lyric", "forced"))
    return (int(is_english), int(not signs_only), int(bool(disposition.get("default"))))


def discover_text_subtitle(source: Path) -> dict | None:
    result = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index,codec_name:stream_tags=language,title:stream_disposition=default,forced",
            "-of",
            "json",
            str(source),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if result.returncode:
        return None
    streams = json.loads(result.stdout or "{}").get("streams") or []
    candidates = [
        stream
        for stream in streams
        if str(stream.get("codec_name") or "").lower() in TEXT_CODECS
        and _stream_score(stream)[0]
    ]
    return max(candidates, key=_stream_score) if candidates else None


def extract_text_subtitle(source: Path, stream: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            f"0:{int(stream['index'])}",
            "-c:s",
            "srt",
            str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if result.returncode or not output.is_file():
        raise RuntimeError("local subtitle extraction failed")
    return output


def extract_native_subtitle(source: Path, stream: dict, output: Path) -> Path:
    """Copy an attributed text stream without flattening its native speaker metadata."""
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-map", f"0:{int(stream['index'])}", "-c:s", "copy", str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if result.returncode or not output.is_file():
        raise RuntimeError("local native subtitle extraction failed")
    return output


def _matching_cues(segment: dict, cues: list[dict]) -> list[dict]:
    start, end = float(segment["start"]), float(segment["end"])
    matching = []
    for cue in cues:
        overlap = max(0.0, min(end, float(cue["end"])) - max(start, float(cue["start"])))
        cue_duration = max(0.01, float(cue["end"]) - float(cue["start"]))
        if overlap >= 0.10 and overlap / cue_duration >= 0.20:
            matching.append(cue)
    return matching


def map_cues_to_segments(
    segments: list[dict], cues: list[dict], attribution_cues: list[dict] | None = None
) -> int:
    """Prefer every cue that materially overlaps a speech segment, preserving cue order."""
    changed = 0
    for segment in segments:
        matching = _matching_cues(segment, cues)
        text = " ".join(dict.fromkeys(str(item["text"]).strip() for item in matching if item.get("text")))
        if text:
            segment["translation"] = text
            segment["translation_source"] = "embedded-english-subtitle"
            changed += 1
        segment.pop("subtitle_speaker", None)
        attributed = _matching_cues(segment, attribution_cues if attribution_cues is not None else cues)
        speakers = list(dict.fromkeys(str(item.get("speaker") or "").strip() for item in attributed))
        speakers = [speaker for speaker in speakers if speaker]
        if len(speakers) == 1:
            segment["subtitle_speaker"] = speakers[0]
    return changed


def harvest_embedded_english(source: Path, artifacts: Path, segments: list[dict]) -> dict:
    stream = discover_text_subtitle(source)
    if stream is None:
        return {"status": "none", "mapped": 0}
    output = extract_text_subtitle(source, stream, artifacts / "subtitles" / "embedded-english.srt")
    cues = parse_srt(output.read_text(encoding="utf-8-sig", errors="replace"))
    codec = str(stream.get("codec_name") or "").lower()
    attribution_cues: list[dict] = []
    if codec in {"ass", "ssa", "webvtt"}:
        suffix = ".vtt" if codec == "webvtt" else ".ass"
        try:
            native = extract_native_subtitle(
                source, stream, artifacts / "subtitles" / f"embedded-english-native{suffix}"
            )
            native_text = native.read_text(encoding="utf-8-sig", errors="replace")
            attribution_cues = parse_webvtt(native_text) if codec == "webvtt" else parse_ass(native_text)
        except (OSError, RuntimeError):
            attribution_cues = []
    mapped = map_cues_to_segments(segments, cues, attribution_cues)
    return {
        "status": "mapped" if mapped else "unmapped",
        "mapped": mapped,
        "cues": len(cues),
        "attributed": sum(1 for segment in segments if segment.get("subtitle_speaker")),
        "codec": codec,
        "artifact": output.relative_to(artifacts).as_posix(),
    }
