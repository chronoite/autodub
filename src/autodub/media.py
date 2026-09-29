from __future__ import annotations

import contextlib
import json
import subprocess
import wave
from pathlib import Path

from .config import FFMPEG, FFPROBE

# Above this many rendered lines, the dialogue bus is pre-summed in batches so the
# final ffmpeg command stays far below Windows' ~32k CreateProcess limit
# (a 399-line episode render failed with WinError 206 before this existed).
MIX_CHUNK_LINES = 100


class StageError(RuntimeError):
    pass


def run_ffmpeg(args: list[str], label: str) -> None:
    if not FFMPEG.exists():
        raise StageError(f"ffmpeg is missing at the configured local path ({label})")
    result = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
    )
    if result.returncode:
        raise StageError(f"{label} failed: {result.stderr[-1200:].strip()}")


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def wav_duration(path: Path) -> float:
    with contextlib.closing(wave.open(str(path), "rb")) as handle:
        return handle.getnframes() / float(handle.getframerate())


def extract_audio(source: Path, work: Path) -> tuple[Path, Path]:
    asr = work / "audio-asr.wav"
    full = work / "audio-full.wav"
    run_ffmpeg(["-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(asr)], "ASR audio extraction")
    run_ffmpeg(["-i", str(source), "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(full)], "mix audio extraction")
    return asr, full


def _atempo_chain(factor: float) -> str:
    parts: list[float] = []
    while factor > 2.0:
        parts.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        parts.append(0.5)
        factor /= 0.5
    parts.append(factor)
    return ",".join(f"atempo={part:.6f}" for part in parts)


def trim_edge_silence(source: Path, target: Path) -> float:
    """Strip lead/tail silence from a synthesized line (keeps 50ms shoulders).

    Measured renders were only ~58% voiced — the padding, not the words, pushed
    physically-fine lines into overrun. TTS silence is near-digital,
    so a -50dB gate is safe. Returns the trimmed duration."""
    filters = ("silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.05,"
               "areverse,"
               "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.05,"
               "areverse")
    run_ffmpeg(["-i", str(source), "-af", filters, "-ar", "48000", "-ac", "1", str(target)],
               "line silence trim")
    return wav_duration(target)


def align_line(
    source: Path,
    target: Path,
    slot_seconds: float,
    min_tempo: float,
    max_tempo: float,
    *,
    fit_mode: str = "exact",
    trim_silence: bool = False,
    max_output_s: float | None = None,
) -> dict[str, float | bool | str]:
    raw_seconds = max(0.01, wav_duration(source))
    trimmed_source = source
    if trim_silence:
        trimmed_source = target.with_name(target.stem + "-trimmed.wav")
        trimmed = trim_edge_silence(source, trimmed_source)
        if trimmed <= 0.01:  # a line that is ALL silence: keep the original
            trimmed_source = source
    current = max(0.01, wav_duration(trimmed_source))
    slot = max(0.12, slot_seconds)
    requested = current / slot
    applied = min(max_tempo, max(min_tempo, requested))
    clipped = not (min_tempo <= requested <= max_tempo)
    capped_seconds = 0.0
    if fit_mode == "exact":
        filters = f"{_atempo_chain(applied)},apad=pad_dur={slot:.6f},atrim=duration={slot:.6f}"
    elif fit_mode == "preserve":
        filters = _atempo_chain(applied)
    elif fit_mode == "preserve-capped":
        # Collision guard: a line may breathe past its slot into silence but can NEVER
        # cross the next line's start (dub-on-dub spill sounds like several voices at once). 60ms fade-out keeps caps unclicky.
        filters = _atempo_chain(applied)
        predicted = current / applied
        if max_output_s is not None and predicted > max_output_s:
            cap = max(0.12, float(max_output_s))
            capped_seconds = predicted - cap
            fade_start = max(0.0, cap - 0.06)
            filters += (f",atrim=duration={cap:.6f}"
                        f",afade=t=out:st={fade_start:.6f}:d=0.06")
    else:
        raise ValueError(f"unknown timing fit mode: {fit_mode}")
    run_ffmpeg(["-i", str(trimmed_source), "-af", filters, "-ar", "48000", "-ac", "1", str(target)],
               "line timing alignment")
    if trimmed_source is not source:
        trimmed_source.unlink(missing_ok=True)
    output_seconds = wav_duration(target)
    result = {
        "source_seconds": round(current, 3),
        "slot_seconds": round(slot, 3),
        "output_seconds": round(output_seconds, 3),
        "overrun_seconds": round(max(0.0, output_seconds - slot), 3),
        "tempo": round(applied, 4),
        "tempo_limited": clipped,
        "fit_mode": fit_mode,
    }
    if trim_silence:
        result["raw_seconds_pre_trim"] = round(raw_seconds, 3)
        result["trimmed_silence_seconds"] = round(max(0.0, raw_seconds - current), 3)
    if capped_seconds > 0:
        result["capped_seconds"] = round(capped_seconds, 3)
        result["max_output_seconds"] = round(float(max_output_s), 3)
    return result


def build_mix(
    full_audio: Path,
    segments: list[dict],
    aligned_dir: Path,
    output: Path,
    bed_gain: float,
    dialogue_gain: float,
    *,
    policy: dict | None = None,
    passthrough: list | None = None,
) -> int:
    """passthrough: [(segment, wav_path)] mixed at UNITY gain, outside the dialogue chain
    (no dialogue gain/shaping) and outside the sidechain key (the bed must NOT duck under
    them) — used for restored original song vocals. Master limiter still applies."""
    rendered = []
    for segment in segments:
        line = aligned_dir / f"line-{int(segment['i']):05d}.wav"
        if line.exists():
            rendered.append((segment, line))
    pass_list = [(seg, Path(p)) for seg, p in (passthrough or []) if Path(p).exists()]
    if not rendered and not pass_list:
        raise StageError("no rendered dialogue lines are available")

    source_duration = probe_duration(full_audio)

    # WinError 206 guard: 399 lines as individual -i inputs plus
    # a 399-entry filter graph exceeds Windows' ~32k command-line limit. Above the chunk
    # size, pre-sum the dialogue bus in batches (float WAV, unity amix — summation is
    # associative, so the result is byte-for-byte the same mix math), then feed the few
    # batch buses to the unchanged duck/limit chain below.
    chunk_files: list[Path] = []
    if len(rendered) > MIX_CHUNK_LINES:
        for chunk_index in range(0, len(rendered), MIX_CHUNK_LINES):
            chunk = rendered[chunk_index:chunk_index + MIX_CHUNK_LINES]
            chunk_path = output.parent / f"{output.stem}-dlgchunk-{chunk_index // MIX_CHUNK_LINES:03d}.wav"
            chunk_args = []
            chunk_filters = []
            chunk_labels = []
            for index, (segment, line) in enumerate(chunk):
                chunk_args.extend(["-i", str(line)])
                delay = max(0, int(float(segment["start"]) * 1000))
                chunk_filters.append(
                    f"[{index}:a]adelay={delay}|{delay},volume={dialogue_gain:.4f}[c{index}]")
                chunk_labels.append(f"[c{index}]")
            chunk_filters.append(
                "".join(chunk_labels)
                + f"amix=inputs={len(chunk_labels)}:normalize=0:dropout_transition=0,"
                f"apad=whole_dur={source_duration:.6f},atrim=duration={source_duration:.6f}[bus]")
            chunk_args.extend(["-filter_complex", ";".join(chunk_filters), "-map", "[bus]",
                               "-ar", "48000", "-c:a", "pcm_f32le", str(chunk_path)])
            run_ffmpeg(chunk_args, "dialogue mix chunk")
            chunk_files.append(chunk_path)

    args = ["-i", str(full_audio)]
    filters = []
    labels = []
    if chunk_files:
        for index, chunk_path in enumerate(chunk_files, 1):
            args.extend(["-i", str(chunk_path)])
            labels.append(f"[{index}:a]")
    else:
        for index, (segment, line) in enumerate(rendered, 1):
            args.extend(["-i", str(line)])
            delay = max(0, int(float(segment["start"]) * 1000))
            label = f"line{index}"
            filters.append(f"[{index}:a]adelay={delay}|{delay},volume={dialogue_gain:.4f}[{label}]")
            labels.append(f"[{label}]")
    if not labels:   # passthrough-only mix still needs a (silent) dialogue bus
        filters.append("anullsrc=channel_layout=stereo:sample_rate=48000,atrim=duration=0.1[dialogue_raw]")
        labels = None
    if labels:
        filters.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0[dialogue_raw]")
    dialogue_label = "dialogue_raw"
    dialogue_filter = str((policy or {}).get("dialogue_filter") or "").strip()
    if dialogue_filter:
        filters.append(f"[dialogue_raw]{dialogue_filter}[dialogue_shaped]")
        dialogue_label = "dialogue_shaped"
    filters.append(
        f"[{dialogue_label}]apad=whole_dur={source_duration:.6f},"
        f"atrim=duration={source_duration:.6f},"
        "asplit=2[dialogue_sidechain][dialogue_mix]"
    )
    # The original track is ducked under new speech. This preserves music/SFX while reducing
    # source dialogue; a future UVR adapter can replace the bed with a true instrumental stem.
    policy = policy or {}
    threshold = float(policy.get("sidechain_threshold", 0.015))
    ratio = float(policy.get("sidechain_ratio", 10.0))
    attack = float(policy.get("attack_ms", 8))
    release = float(policy.get("release_ms", 220))
    limiter = float(policy.get("limiter", 0.95))
    filters.append(f"[0:a]volume={bed_gain:.4f}[bed]")
    filters.append(
        f"[bed][dialogue_sidechain]sidechaincompress=threshold={threshold:.6f}:"
        f"ratio={ratio:.3f}:attack={attack:.3f}:release={release:.3f}[ducked]"
    )
    if pass_list:
        base = (len(chunk_files) if chunk_files else len(rendered)) + 1
        plabels = []
        for j, (segment, line) in enumerate(pass_list):
            args.extend(["-i", str(line)])
            delay = max(0, int(float(segment["start"]) * 1000))
            filters.append(f"[{base + j}:a]adelay={delay}|{delay}[pass{j}]")
            plabels.append(f"[pass{j}]")
        filters.append(
            "".join(plabels)
            + f"amix=inputs={len(plabels)}:normalize=0:dropout_transition=0,"
            f"apad=whole_dur={source_duration:.6f},atrim=duration={source_duration:.6f}[passthrough_mix]"
        )
        filters.append(
            f"[ducked][dialogue_mix][passthrough_mix]amix=inputs=3:normalize=0:duration=first,"
            f"alimiter=limit={limiter:.4f}[mix]"
        )
    else:
        filters.append(
            f"[ducked][dialogue_mix]amix=inputs=2:normalize=0:duration=first,"
            f"alimiter=limit={limiter:.4f}[mix]"
        )
    args.extend(["-filter_complex", ";".join(filters), "-map", "[mix]", "-ar", "48000", "-ac", "2", str(output)])
    try:
        run_ffmpeg(args, "dialogue mix")
    finally:
        for chunk_path in chunk_files:
            chunk_path.unlink(missing_ok=True)
    return len(rendered)


def mux(source: Path, audio: Path, output: Path) -> None:
    run_ffmpeg(
        ["-i", str(source), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(output)],
        "final mux",
    )


def extract_audio_clip(source: Path, output: Path, start: float, duration: float) -> None:
    run_ffmpeg(
        [
            "-ss", f"{max(0.0, start):.3f}", "-t", f"{max(0.1, duration):.3f}",
            "-i", str(source), "-vn", "-ar", "48000", "-ac", "2", str(output),
        ],
        "review audio clip",
    )


def mux_clip(source: Path, audio: Path, output: Path, start: float, duration: float) -> None:
    run_ffmpeg(
        [
            "-ss", f"{max(0.0, start):.3f}", "-t", f"{max(0.1, duration):.3f}",
            "-i", str(source), "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
            "-shortest", "-movflags", "+faststart", str(output),
        ],
        "review clip mux",
    )
