from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import OUTPUT_ROOT, WORK_ROOT
from .policies import (
    DEFAULT_MIX_POLICY,
    DEFAULT_SPACE_POLICY,
    DEFAULT_TIMING_POLICY,
    get_mix_policy,
    get_timing_policy,
)
from .quality_profiles import DEFAULT_PROFILE, get_profile
from .speaker_evidence import normalize_speaker_count


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"dub-{stamp}-{secrets.token_hex(2)}"


def job_dir(job_id: str) -> Path:
    if not job_id.startswith("dub-") or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in job_id):
        raise ValueError("invalid job id")
    return WORK_ROOT / "jobs" / job_id


def default_job(job_id: str, suffix: str, size: int, digest: str) -> dict[str, Any]:
    return {
        "schema": 3,
        "id": job_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "ready",
        "stage": "imported",
        "progress": 0,
        "error": None,
        "source": {"file": f"source{suffix}", "bytes": size, "sha256": digest},
        "settings": {
            "quality_profile": DEFAULT_PROFILE,
            "quality_stack": get_profile(DEFAULT_PROFILE),
            "source_language": "ja",
            "target_language": "en",
            "speaker_count": {"mode": "automatic"},
            "asr_model": "faster-whisper-large-v3",
            "asr_compute": "float16",
            "translation_backend": "marian-ja-en-reviewed",
            "tts_backend": "qwen3-tts-1.7b-clone",
            "tts_seed": 1986,
            "mix_policy": DEFAULT_MIX_POLICY,
            "timing_policy": DEFAULT_TIMING_POLICY,
            "space_policy": DEFAULT_SPACE_POLICY,
            "emotion_policy": "source-energy-v1",
            "song_policy": "skip-detected-v1",   # EXPERIMENTAL song skip: on for new jobs, toggleable in the app
            "source_bed_gain": get_mix_policy(DEFAULT_MIX_POLICY)["bed_gain"],
            "dialogue_gain": get_mix_policy(DEFAULT_MIX_POLICY)["dialogue_gain"],
            "max_tempo": get_timing_policy(DEFAULT_TIMING_POLICY)["max_tempo"],
            "min_tempo": get_timing_policy(DEFAULT_TIMING_POLICY)["min_tempo"],
        },
        "segments": [],
        "glossary": {},
        "subtitle_harvest": {"status": "pending", "mapped": 0},
        "delivery_summary": {"status": "pending", "lines": 0},
        "speaker_voices": {},
        "speaker_references": {},
        "available_voices": [],
        "voice_labels": {},
        "artifacts": {},
        "qc_summary": {},
        "exports": [],
        "events": [{"at": utc_now(), "stage": "imported", "message": "Source imported as an opaque job."}],
    }


def state_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


# Per-job mutation locks: save_job's atomic replace
# prevents torn READS, not lost UPDATES — two concurrent load→mutate→save cycles
# (a settings PATCH racing a casting verb or the demo prerender) drop the loser's write.
# Every read-modify-write path should hold this around its cycle.
_JOB_LOCKS: dict[str, threading.Lock] = {}
_JOB_LOCKS_GUARD = threading.Lock()


def job_lock(job_id: str) -> threading.Lock:
    with _JOB_LOCKS_GUARD:
        return _JOB_LOCKS.setdefault(str(job_id), threading.Lock())


def load_job(job_id: str) -> dict[str, Any]:
    with state_path(job_id).open("r", encoding="utf-8") as handle:
        return normalize_job(json.load(handle))


def normalize_job(job: dict[str, Any]) -> dict[str, Any]:
    """Apply backward-compatible defaults without rewriting preserved job state on read."""
    settings = job.setdefault("settings", {})
    if "speaker_count" not in settings:
        settings["speaker_count"] = normalize_speaker_count(settings.get("expected_speakers", 0))
    else:
        settings["speaker_count"] = normalize_speaker_count(settings["speaker_count"])
    settings.pop("expected_speakers", None)
    settings.setdefault("mix_policy", "legacy-v1")
    settings.setdefault("timing_policy", "segment-window-v1")
    settings.setdefault("space_policy", "dry-v1")
    settings.setdefault("emotion_policy", "source-energy-v1")
    settings.setdefault("song_policy", "dub-all-v1")   # legacy jobs keep their recorded dub-everything behavior
    settings.setdefault("source_bed_gain", 1.0)
    settings.setdefault("dialogue_gain", 1.0)
    settings.setdefault("min_tempo", 0.60)
    settings.setdefault("max_tempo", 1.75)
    job.setdefault("glossary", {})
    job.setdefault("subtitle_harvest", {"status": "unknown", "mapped": 0})
    job.setdefault("delivery_summary", {"status": "unknown", "lines": 0})
    job.setdefault("qc_summary", {})
    job.setdefault("exports", [])
    return job


def replace_retry(temp_name: str, path: Path | str, *, attempts: int = 8) -> None:
    """``os.replace`` with short backoff — the project-wide atomic-write finisher.

    On Windows the atomic rename fails with ``PermissionError`` while ANY reader holds
    the destination open, and our list endpoints read every job/bank JSON on the UI's
    8-second poll. Two season-batch analyses once lost only their final save this way, and the end-to-end test flaked identically whenever the live
    server was polling during a run. Readers hold files for milliseconds; up to ~1.4 s
    of patience outlasts any real read, and a still-stuck rename after that re-raises."""
    for attempt in range(attempts):
        try:
            os.replace(temp_name, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def save_job(job: dict[str, Any]) -> None:
    job["updated_at"] = utc_now()
    path = state_path(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="job-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(job, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        replace_retry(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def event(job: dict[str, Any], stage: str, message: str, progress: int | None = None) -> None:
    job["stage"] = stage
    if progress is not None:
        job["progress"] = progress
    job.setdefault("events", []).append({"at": utc_now(), "stage": stage, "message": message})
    job["events"] = job["events"][-100:]
    save_job(job)
    # Mirror every event into the job's operational log (default stream = errors AND
    # processing events; only detail chatter is toggle-gated).
    from . import oplog
    if stage == "failed":
        oplog.job_error(job["id"], stage, message)
    else:
        oplog.job_event(job["id"], stage, message)


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    """Return UI-safe state without host paths or source filenames."""
    copy = json.loads(json.dumps(job))
    copy.get("source", {}).pop("file", None)
    # The original basename is kept server-side for series grouping and export names only.
    copy.get("source", {}).pop("original_name", None)
    # Full tracebacks are deliberately persisted in job.json for local diagnosis, but are
    # never part of the HTTP API contract. They may contain host paths or library
    # internals even though request bodies and media names are not logged.
    copy.pop("error_detail", None)
    # Raw voice centroids and complete pairwise evidence stay job-local. The API carries
    # only derived suggestion similarities/counts.
    copy.pop("speaker_evidence", None)
    # Live synthesis progress: the worker writes a per-line
    # sidecar; merge it so the bar moves 78->86 fractionally with a real ETA instead of
    # freezing for the whole longest stage.
    if copy.get("status") == "running" and copy.get("stage") == "synthesizing":
        try:
            sidecar = job_dir(copy["id"]) / "artifacts" / "lines" / "progress.json"
            progress = json.loads(sidecar.read_text(encoding="utf-8"))
            copy["synth_progress"] = progress
            if progress.get("total"):
                copy["progress"] = 78 + round(8 * progress["done"] / progress["total"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
    return copy


def list_jobs() -> list[dict[str, Any]]:
    root = WORK_ROOT / "jobs"
    if not root.exists():
        return []
    jobs = []
    for path in root.glob("dub-*/job.json"):
        try:
            jobs.append(public_job(normalize_job(json.loads(path.read_text(encoding="utf-8")))))
        except (OSError, json.JSONDecodeError):
            from . import oplog
            oplog.server_warn("jobs", f"skipped unreadable state for opaque job {path.parent.name}")
            continue
    return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)


def delete_job(job_id: str) -> dict[str, Any]:
    """Remove one job's working dir and ITS outputs. Refuses while the job is running. Touches nothing outside this job:
    input/ sources, voices/ (bank + references + profiles) and experiments are kept —
    a deleted job must never take the series bank's media with it (the bank copies its
    reference clips into voices/ for exactly this reason)."""
    import shutil
    job = load_job(job_id)                       # validates the opaque id / existence
    if job.get("status") == "running":
        raise ValueError("this job is running - cancel it before deleting")
    removed = []
    root = job_dir(job_id)
    if root.exists():
        shutil.rmtree(root)
        removed.append("workdir")
    for path in OUTPUT_ROOT.glob(f"{job_id}-english*.mp4"):
        friendly = friendly_output_path(job, path)
        if friendly is not None and friendly.is_file():
            friendly.unlink(missing_ok=True)
            removed.append(friendly.name)
        path.unlink(missing_ok=True)
        removed.append(path.name)
    from . import oplog
    oplog.server_warn("jobs", f"deleted opaque job {job_id} ({', '.join(removed) or 'nothing on disk'})")
    return {"deleted": job_id, "removed": removed}


def sanitize_export_stem(name: str) -> str:
    """Windows-safe filename stem from the source file's own name (exports must be
    recognizable, not opaque job IDs)."""
    stem = Path(str(name)).stem
    stem = "".join(c for c in stem if c not in '<>:"/\\|?*' and ord(c) >= 32)
    return stem.strip().rstrip(".")[:150].strip()


def friendly_output_path(job: dict, final: Path) -> Path | None:
    stem = sanitize_export_stem((job.get("source") or {}).get("original_name") or "")
    if not stem:
        return None
    label = (final.name.removeprefix(str(job.get("id") or ""))
             .removesuffix(".mp4").replace("-english", "").strip("-"))
    tag = "ENGLISH DUB" + (f" {label}" if label else "")
    return OUTPUT_ROOT / f"{stem} [{tag}].mp4"


def output_path(job_id: str) -> Path:
    return OUTPUT_ROOT / f"{job_id}-english.mp4"


def output_variant_path(job_id: str, label: str) -> Path:
    if not label or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in label):
        raise ValueError("invalid output label")
    return OUTPUT_ROOT / f"{job_id}-english-{label}.mp4"


def sha256_stream(reader, output: Path, length: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    written = 0
    with output.open("wb") as handle:
        while written < length:
            chunk = reader.read(min(1024 * 1024, length - written))
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    return written, digest.hexdigest()
