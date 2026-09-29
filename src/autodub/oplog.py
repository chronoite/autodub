"""Operational logging: errors always, verbosity by toggle, automatic rotation.

Privacy means all-local, not self-blinding: a failure must always leave a trail. Errors and
tracebacks are written unconditionally; per-line and per-stage detail is written only when the
verbose toggle is on. Logs rotate automatically (newest jobs only, size-capped).

Privacy rules: never log the source filename or any host path derived from it (job ids are opaque);
dialogue text appears only in job-local logs, which sit beside job.json that already contains it.
The verbose toggle lives in work/logging.json and is exposed via /api/logging.
"""
from __future__ import annotations

import json
import os
import threading
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .config import PROJECT_ROOT, WORK_ROOT
from .state import replace_retry

_TOGGLE_PATH = WORK_ROOT / "logging.json"
_SERVER_LOG = WORK_ROOT / "server.log"
_LOCK = threading.Lock()
_MAX_LOG_BYTES = 5 * 1024 * 1024          # per-file cap; oldest half dropped when exceeded
_KEEP_JOB_LOGS = 10                        # job.log kept only for the newest N jobs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def verbose_enabled() -> bool:
    try:
        return bool(json.loads(_TOGGLE_PATH.read_text(encoding="utf-8")).get("verbose"))
    except Exception:
        return False


def set_verbose(value: bool) -> dict:
    state = {"verbose": bool(value), "updated_at": _now()}
    with _LOCK:
        _TOGGLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="logging-", suffix=".tmp", dir=_TOGGLE_PATH.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            replace_retry(temp_name, _TOGGLE_PATH)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return state


def _trim(path: Path) -> None:
    """Keep a log under the size cap by dropping its older half (line-safe)."""
    try:
        if path.exists() and path.stat().st_size > _MAX_LOG_BYTES:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            path.write_text("".join(lines[len(lines) // 2:]), encoding="utf-8")
    except Exception:
        pass  # rotation must never take the app down


def _write(path: Path, level: str, stage: str, message: str) -> None:
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{_now()} {level:<7} [{stage}] {message}\n")
            _trim(path)
    except Exception:
        pass  # logging must never take the app down


def _job_log_path(job_id: str) -> Path:
    return WORK_ROOT / "jobs" / job_id / "job.log"


def job_error(job_id: str, stage: str, message: str) -> None:
    """Errors/tracebacks — ALWAYS written, toggle-independent."""
    _write(_job_log_path(job_id), "ERROR", stage, message)


def job_warn(job_id: str, stage: str, message: str) -> None:
    """Warnings (silent-skip explanations) — always written."""
    _write(_job_log_path(job_id), "WARN", stage, message)


def job_event(job_id: str, stage: str, message: str) -> None:
    """Processing events (stage transitions, worker start/finish) — always written, since
    the default stream is errors plus processing events. Never dialogue content."""
    _write(_job_log_path(job_id), "EVENT", stage, message)


def job_info(job_id: str, stage: str, message: str) -> None:
    """Verbose stream (stage transitions, timings, worker chatter) — toggle-gated."""
    if verbose_enabled():
        _write(_job_log_path(job_id), "INFO", stage, message)


_CODE_SHA: str | None = None


def code_sha() -> str:
    """The checkout's git HEAD, cached at first call and stamped into every analyze/render.

    Makes "the running server is the committed code" machine-checkable. Returns "unknown" outside
    a git checkout."""
    global _CODE_SHA
    if _CODE_SHA is None:
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10,
                cwd=str(PROJECT_ROOT),
            )
            _CODE_SHA = (result.stdout or "").strip() or "unknown"
        except Exception:
            _CODE_SHA = "unknown"
    return _CODE_SHA


def server_info(context: str, message: str) -> None:
    """Server-level operational facts (startup identity) — always written."""
    _write(_SERVER_LOG, "INFO", context, message)


def server_error(context: str, message: str) -> None:
    """Server-level errors (request handlers, startup) — always written."""
    _write(_SERVER_LOG, "ERROR", context, message)


def server_warn(context: str, message: str) -> None:
    """Server-level recoverable problems that must not disappear silently."""
    _write(_SERVER_LOG, "WARN", context, message)


def job_summary(job_id: str, status: str, events: list[dict]) -> None:
    """Write an opaque stage-duration summary without dialogue or media metadata."""
    parsed = []
    for item in events:
        try:
            parsed.append((datetime.fromisoformat(str(item["at"])), str(item["stage"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(parsed) < 2:
        job_event(job_id, "summary", f"status={status}; duration unavailable")
        return
    parts = []
    for (started, stage), (ended, _) in zip(parsed, parsed[1:]):
        seconds = max(0.0, (ended - started).total_seconds())
        parts.append(f"{stage}={seconds:.1f}s")
    total = max(0.0, (parsed[-1][0] - parsed[0][0]).total_seconds())
    job_event(job_id, "summary", f"status={status}; total={total:.1f}s; " + ", ".join(parts[-12:]))


def prune_job_logs() -> int:
    """Delete job.log files beyond the newest _KEEP_JOB_LOGS jobs. Runs at server start."""
    removed = 0
    try:
        root = WORK_ROOT / "jobs"
        logs = sorted(root.glob("dub-*/job.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in logs[_KEEP_JOB_LOGS:]:
            stale.unlink(missing_ok=True)
            removed += 1
    except Exception:
        pass
    return removed


def tail(job_id: str | None = None, lines: int = 80) -> list[str]:
    """Last N lines of a job log (or the server log when job_id is None) for the panel."""
    path = _job_log_path(job_id) if job_id else _SERVER_LOG
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(lines, 500)):]
    except Exception:
        return []


def logging_state() -> dict:
    newest = None
    try:
        logs = sorted((WORK_ROOT / "jobs").glob("dub-*/job.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        newest = logs[0].parent.name if logs else None
    except Exception:
        pass
    return {"verbose": verbose_enabled(), "server_log": _SERVER_LOG.exists(),
            "newest_job_log": newest, "keep_job_logs": _KEEP_JOB_LOGS,
            "max_log_bytes": _MAX_LOG_BYTES}
