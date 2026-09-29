from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from contextvars import ContextVar
from pathlib import Path

from . import oplog
from .config import (
    ANALYSIS_PYTHON,
    GPT_SOVITS_URL,
    MEDIA_PYTHON,
    PACKAGE_ROOT,
    QUALITY_PYTHON,
    TTS_BATCH_SORT,
    runtime_env,
)
from .media import StageError
from .speaker_evidence import normalize_speaker_count
from .voice_profiles import list_profiles, load_profile


WORKER = PACKAGE_ROOT / "workers" / "model_worker.py"
QUALITY_WORKER = PACKAGE_ROOT / "workers" / "quality_worker.py"


_JOB_CONTEXT: ContextVar[str] = ContextVar("autodub_job_id", default="")


def set_job_context(job_id: str) -> None:
    """Route worker diagnostics to the current job without crossing worker threads."""
    _JOB_CONTEXT.set(str(job_id or ""))


def _run_logged(label: str, args: list[str], payload: dict, env: dict, timeout: int) -> dict:
    """Shared worker runner: start/end + duration
    always visible in verbose logs, stderr preserved on success (workers' warnings used to
    vanish), timeouts wrapped with the stage label, and JSON failures carrying BOTH tails."""
    job_id = _JOB_CONTEXT.get()
    t0 = time.time()
    if job_id:
        oplog.job_event(job_id, label, f"worker start (timeout {timeout}s)")
    try:
        result = subprocess.run(
            args,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        if job_id:
            oplog.job_error(job_id, label, f"worker timed out after {timeout}s")
        raise StageError(f"{label} timed out after {timeout}s") from exc
    secs = round(time.time() - t0, 1)
    stderr_tail = (result.stderr or "")[-2000:].strip()
    if result.returncode:
        if job_id:
            oplog.job_error(job_id, label, f"worker failed rc={result.returncode} after {secs}s: {stderr_tail}")
        raise StageError(f"{label} failed: {stderr_tail}")
    if job_id:
        oplog.job_event(job_id, label, f"worker ok in {secs}s")
        if secs >= max(300.0, timeout * 0.75):
            oplog.job_warn(job_id, label, f"worker used {secs}s of its {timeout}s timeout budget")
        if stderr_tail:
            oplog.job_info(job_id, label, f"worker diagnostics: {stderr_tail[-800:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        detail = f"stdout: {(result.stdout or '')[-500:]} | stderr: {stderr_tail[-500:]}"
        if job_id:
            oplog.job_error(job_id, label, f"worker returned invalid JSON — {detail}")
        raise StageError(f"{label} returned invalid JSON — {detail}") from exc


def run_worker(command: str, payload: dict, timeout: int = 7200) -> dict:
    if not MEDIA_PYTHON.exists():
        raise StageError("the local media Python environment is missing")
    return _run_logged(f"{command} worker", [str(MEDIA_PYTHON), str(WORKER), command],
                       payload, runtime_env(), timeout)


def run_quality_worker(command: str, payload: dict, *, tts: bool = False, timeout: int = 7200) -> dict:
    python = QUALITY_PYTHON if tts else ANALYSIS_PYTHON
    if not python.exists():
        raise StageError("the isolated AutoDub quality environment is missing")
    return _run_logged(f"{command} quality worker", [str(python), str(QUALITY_WORKER), command],
                       payload, runtime_env(gpu=True, portable_deps=False), timeout)


def separate_dialogue(wav: Path, destination: Path) -> dict:
    return run_quality_worker("separate", {"audio": str(wav), "output": str(destination)}, timeout=10800)


def transcribe_quality(wav: Path, language: str) -> list[dict]:
    return run_quality_worker("transcribe", {"audio": str(wav), "language": language})["segments"]


def diarize_quality(wav: Path, segments: list[dict], speaker_count: dict) -> dict:
    return run_quality_worker(
        "diarize",
        {"audio": str(wav), "segments": segments, "speaker_count": normalize_speaker_count(speaker_count)},
    )


def align_quality(wav: Path, segments: list[dict], language: str) -> list[dict]:
    """Refine source speech windows with the pinned, offline forced aligner."""
    return run_quality_worker(
        "align",
        {
            "audio": str(wav),
            "segments": [
                {"i": int(item["i"]), "start": float(item["start"]),
                 "end": float(item["end"]), "text": str(item.get("text") or "")}
                for item in segments
            ],
            "language": language,
            "device": "cuda",
        },
    )["segments"]


def synthesize_quality_batch(
    lines: list[dict],
    *,
    seed: int = 1986,
    cancel_file: Path | None = None,
    batch_size: int = 1,
) -> dict:
    payload = {"lines": lines, "seed": seed}
    if int(batch_size) > 1:
        # a missing key = old worker behavior (backward/forward compatible)
        payload["batch_size"] = int(batch_size)
        payload["batch_sort"] = bool(TTS_BATCH_SORT)
    if cancel_file is not None:
        payload["cancel_file"] = str(cancel_file)
    result = run_quality_worker("synthesize", payload, tts=True, timeout=10800)
    job_id = _JOB_CONTEXT.get()
    if job_id:
        for entry in result.get("runaways") or []:
            oplog.job_warn(job_id, "synthesize",
                           "runaway line %s: %ss into a %ss slot -> %s%s" % (
                               entry.get("line_index"), entry.get("durations"),
                               entry.get("slot_seconds"), entry.get("action"),
                               " (%s)" % entry["fallback"] if entry.get("fallback") else ""))
    return result


def transcribe(wav: Path, language: str) -> list[dict]:
    return run_worker("transcribe", {"audio": str(wav), "language": language})["segments"]


def cluster_speakers(wav: Path, segments: list[dict], speaker_count: dict) -> list[dict]:
    result = run_worker(
        "cluster",
        {"audio": str(wav), "segments": segments, "speaker_count": normalize_speaker_count(speaker_count)},
    )
    return result["segments"]


def translate(segments: list[dict], source_language: str, target_language: str) -> list[dict]:
    result = run_worker(
        "translate",
        {"segments": segments, "source_language": source_language, "target_language": target_language},
    )
    return result["segments"]


def sapi_voices() -> list[str]:
    """Installed Windows SAPI voices; an empty list on other platforms."""
    if sys.platform != "win32":
        return []
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"
    )
    try:
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ["system-default"]
    voices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode or not voices:
        job_id = _JOB_CONTEXT.get()
        if job_id:
            oplog.job_warn(job_id, "voices", "Windows voice enumeration unavailable; using system-default")
    return voices or ["system-default"]


def voice_options() -> list[dict[str, str]]:
    system = [{"option": f"sapi:{voice}", "label": f"System voice · {voice}"} for voice in sapi_voices()]
    return system + list_profiles()


def synthesize_sapi(text: str, voice: str, output: Path, rate: int = 0) -> None:
    text_file = output.with_suffix(".txt")
    text_file.write_text(text, encoding="utf-8")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.SetOutputToWaveFile($env:AUTODUB_OUT); "
        "if($env:AUTODUB_VOICE -ne 'system-default'){$s.SelectVoice($env:AUTODUB_VOICE)}; "
        "$s.Rate=[int]$env:AUTODUB_RATE; "
        "$t=[IO.File]::ReadAllText($env:AUTODUB_TEXT,[Text.Encoding]::UTF8); "
        "$s.Speak($t); $s.Dispose()"
    )
    env = os.environ.copy()
    env.update({"AUTODUB_OUT": str(output), "AUTODUB_TEXT": str(text_file), "AUTODUB_VOICE": voice, "AUTODUB_RATE": str(rate)})
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, env=env, timeout=300)
    text_file.unlink(missing_ok=True)
    if result.returncode or not output.exists():
        raise StageError(f"Windows voice synthesis failed: {result.stderr[-600:].strip()}")


def synthesize_gsv(text: str, profile_id: str, output: Path, target_language: str = "en") -> None:
    profile = load_profile(profile_id)
    body = {
        "text": text[:4000],
        "text_lang": target_language,
        "ref_audio_path": profile["reference_path"],
        "prompt_text": profile["prompt_text"],
        "prompt_lang": profile["prompt_language"],
        "text_split_method": "cut5",
        "media_type": "wav",
        "streaming_mode": False,
        "speed_factor": 1.0,
    }
    request = urllib.request.Request(
        GPT_SOVITS_URL + "/tts",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
    except Exception as exc:
        raise StageError("local GPT-SoVITS clone service is unavailable; no fallback was used") from exc
    if data[:4] != b"RIFF":
        raise StageError("local GPT-SoVITS returned an invalid WAV; no fallback was used")
    output.write_bytes(data)


def synthesize_voice(text: str, option: str, output: Path, target_language: str = "en") -> None:
    if option.startswith("sapi:"):
        synthesize_sapi(text, option.removeprefix("sapi:"), output)
    elif option.startswith("gsv:voice-"):
        synthesize_gsv(text, option.removeprefix("gsv:"), output, target_language)
    else:
        raise StageError("unknown local voice option")
