"""Conservative local source-delivery features for optional emotion-aware experiments."""
from __future__ import annotations

import math
import statistics
import wave
from array import array
from pathlib import Path


def _window_dbfs(handle: wave.Wave_read, start: float, end: float) -> tuple[float, float]:
    rate = handle.getframerate()
    channels = handle.getnchannels()
    width = handle.getsampwidth()
    if width != 2 or rate <= 0 or channels <= 0:
        raise ValueError("delivery analysis requires PCM16 WAV")
    first = max(0, min(handle.getnframes(), round(start * rate)))
    count = max(1, min(handle.getnframes() - first, round(max(0.05, end - start) * rate)))
    handle.setpos(first)
    samples = array("h")
    samples.frombytes(handle.readframes(count))
    if not samples:
        return -96.0, -96.0
    mean_square = sum(int(value) * int(value) for value in samples) / len(samples)
    peak = max(abs(int(value)) for value in samples)
    rms_db = 20 * math.log10(max(1.0, math.sqrt(mean_square)) / 32768.0)
    peak_db = 20 * math.log10(max(1, peak) / 32768.0)
    return round(rms_db, 2), round(peak_db, 2)


def analyze_delivery(audio: Path, segments: list[dict]) -> dict:
    """Attach energy-relative labels; uncertain lines remain neutral."""
    if not audio.is_file() or not segments:
        return {"status": "unavailable", "lines": 0}
    measurements = []
    with wave.open(str(audio), "rb") as handle:
        for segment in segments:
            rms, peak = _window_dbfs(handle, float(segment["start"]), float(segment["end"]))
            measurements.append((segment, rms, peak))
    median = statistics.median(item[1] for item in measurements)
    counts = {"calm": 0, "neutral": 0, "intense": 0}
    for segment, rms, peak in measurements:
        delta = rms - median
        if delta >= 5.0:
            label = "intense"
        elif delta <= -5.0:
            label = "calm"
        else:
            label = "neutral"
        confidence = min(1.0, abs(delta) / 12.0) if label != "neutral" else max(0.0, 1.0 - abs(delta) / 5.0)
        segment["delivery"] = {
            "label": label,
            "confidence": round(confidence, 2),
            "energy_dbfs": rms,
            "peak_dbfs": peak,
            "relative_db": round(delta, 2),
        }
        counts[label] += 1
    return {"status": "ready", "lines": len(measurements), "median_dbfs": round(median, 2), **counts}


def chatterbox_controls(segment: dict) -> dict:
    delivery = segment.get("delivery") or {}
    if float(delivery.get("confidence", 0.0)) < 0.30:
        return {"exaggeration": 0.50, "cfg_weight": 0.50}
    exaggeration = {"calm": 0.35, "neutral": 0.50, "intense": 0.75}.get(
        str(delivery.get("label") or "neutral"),
        0.50,
    )
    return {"exaggeration": exaggeration, "cfg_weight": 0.50}
