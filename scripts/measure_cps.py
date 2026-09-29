"""Measure the EFFECTIVE chars-per-second of a rendered job's TTS lines.

ADAPT_CPS is calibrated BY MEASUREMENT, not by taste. An adaptation budget of 15 cps
looked reasonable, but rendered lines were only ~58% voiced, so the real rate the mix
experienced was far lower. This reads a job's existing
line wavs (no GPU, no ffmpeg), finds the voiced span by RMS, and prints the
distribution the config number should come from.

Usage: python scripts/measure_cps.py <job-id>
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import wave
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autodub.state import job_dir, load_job

SILENCE_RMS_FRACTION = 0.02   # of the line's own peak RMS window
WINDOW_S = 0.01


def voiced_span_seconds(path: Path) -> tuple[float, float]:
    """(total_seconds, voiced_seconds) via 10ms RMS windows against a per-line gate."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got width {width}")
    samples = array("h")
    samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
    total = len(samples) / (rate * channels)
    window = max(1, int(rate * channels * WINDOW_S))
    rms = []
    for i in range(0, len(samples), window):
        chunk = samples[i:i + window]
        if chunk:
            rms.append(math.sqrt(sum(v * v for v in chunk) / len(chunk)))
    if not rms:
        return 0.0, 0.0
    gate = max(rms) * SILENCE_RMS_FRACTION
    active = [index for index, value in enumerate(rms) if value > gate]
    if not active:
        return total, 0.0
    voiced = (active[-1] - active[0] + 1) * WINDOW_S
    return total, voiced


def main() -> None:
    job_id = sys.argv[1]
    job = load_job(job_id)
    lines_dir = job_dir(job_id) / "artifacts" / "lines"
    full_cps, voiced_cps, voiced_fractions = [], [], []
    counted = 0
    for segment in job.get("segments") or []:
        text = str(segment.get("translation") or "").strip()
        line = lines_dir / f"line-{int(segment['i']):05d}.wav"
        if not text or not line.is_file():
            continue
        total, voiced = voiced_span_seconds(line)
        if total <= 0.1 or voiced <= 0.05:
            continue
        counted += 1
        full_cps.append(len(text) / total)
        voiced_cps.append(len(text) / voiced)
        voiced_fractions.append(voiced / total)
    if not counted:
        raise SystemExit("no measurable lines")
    report = {
        "job": job_id,
        "lines_measured": counted,
        "effective_cps_full_duration": {
            "median": round(statistics.median(full_cps), 2),
            "p25": round(statistics.quantiles(full_cps, n=4)[0], 2),
            "p75": round(statistics.quantiles(full_cps, n=4)[2], 2),
        },
        "cps_voiced_span": {
            "median": round(statistics.median(voiced_cps), 2),
            "p25": round(statistics.quantiles(voiced_cps, n=4)[0], 2),
            "p75": round(statistics.quantiles(voiced_cps, n=4)[2], 2),
        },
        "voiced_fraction_median": round(statistics.median(voiced_fractions), 3),
        "note": ("ADAPT_CPS should sit near cps_voiced_span.median when renders are "
                 "silence-trimmed, or effective_cps_full_duration.median "
                 "when they are not."),
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
