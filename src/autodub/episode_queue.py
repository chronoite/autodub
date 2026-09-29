"""Persistent, local episode render queue.

The queue contains only opaque AutoDub job IDs. Each job keeps its own atomic state and shared GPU
lease, so stopping the batch between episodes never corrupts or merges jobs.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

from . import config, oplog, thermal
from .config import WORK_ROOT
from .quality_profiles import CPU_PROFILE, profile_requires_gpu
from .state import event, load_job, replace_retry, utc_now


QUEUE_PATH = WORK_ROOT / "episode-queue.json"
STOP_PATH = WORK_ROOT / "episode-queue.stop"


def _default() -> dict:
    return {"schema": 1, "status": "idle", "items": [], "updated_at": utc_now()}


def _save(state: dict) -> dict:
    state["updated_at"] = utc_now()
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="episode-queue-", suffix=".tmp", dir=QUEUE_PATH.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(state, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_retry(temporary, QUEUE_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return state


def queue_state() -> dict:
    try:
        value = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) and isinstance(value.get("items"), list) else _default()
    except (OSError, json.JSONDecodeError):
        return _default()


def enqueue(job_ids: list[str]) -> dict:
    state = queue_state()
    if state.get("status") == "running":
        raise ValueError("stop the episode queue before changing it")
    known = {item["job"] for item in state["items"] if item.get("status") in {"queued", "running"}}
    for job_id in job_ids:
        job = load_job(str(job_id))
        if not job.get("segments"):
            raise ValueError(f"{job_id} must be analyzed and reviewed before queueing")
        if str(job_id) not in known:
            state["items"].append(
                {
                    "job": str(job_id),
                    "status": "queued",
                    "requires_gpu": profile_requires_gpu(
                        job.get("settings", {}).get("quality_profile", CPU_PROFILE)
                    ),
                    "added_at": utc_now(),
                }
            )
            known.add(str(job_id))
    return _save(state)


def clear_finished() -> dict:
    state = queue_state()
    if state.get("status") == "running":
        raise ValueError("stop the episode queue before clearing it")
    state["items"] = [
        item for item in state["items"] if item.get("status") not in {"complete", "failed", "cancelled"}
    ]
    return _save(state)


def request_stop() -> dict:
    STOP_PATH.parent.mkdir(parents=True, exist_ok=True)
    STOP_PATH.write_text("stop\n", encoding="ascii")
    state = queue_state()
    state["stop_requested"] = True
    return _save(state)


def _job_event_safe(job_id: str, message: str) -> None:
    """Append a UI-visible event to a job WITHOUT changing its status/progress —
    passing the job's current stage makes event() a pure append. Never crashes
    the queue loop."""
    try:
        job = load_job(job_id)
        event(job, str(job.get("stage") or "queued"), message)
    except Exception as exc:
        oplog.server_warn("episode-queue", f"could not record a job event: {type(exc).__name__}")


def run_queue(*, gpu_authorized: bool = False) -> None:
    from .pipeline import render, render_overlapped

    state = queue_state()
    pending = [item for item in state["items"] if item.get("status") == "queued"]
    if not pending:
        raise ValueError("the episode queue has no pending jobs")
    if any(item.get("requires_gpu") for item in pending) and not gpu_authorized:
        raise ValueError("the GPU episode queue was not explicitly armed")
    STOP_PATH.unlink(missing_ok=True)
    state["status"] = "running"
    state["stop_requested"] = False
    state.pop("thermal_abort", None)
    _save(state)
    aborted = False
    overlap = bool(config.QUEUE_OVERLAP_POST_STAGES)
    tails: list[tuple[dict, object]] = []  # (item, alive CPU-tail thread)

    def _derive_status(item: dict) -> None:
        result = load_job(item["job"])
        item["status"] = (
            "complete" if result.get("status") == "complete"
            else "cancelled" if result.get("status") == "cancelled"
            else "failed"
        )

    def _finalize_tail(item: dict, thread, temps_summary: str | None = None) -> None:
        # All job.json writes settle before we read status or append job events —
        # the tail thread is the only concurrent writer, so join FIRST.
        thread.join()
        _derive_status(item)
        if temps_summary:
            _job_event_safe(item["job"], f"Post-episode GPU temps: {temps_summary}.")
        item["finished_at"] = utc_now()
        _save(state)
    try:
        for index, item in enumerate(pending):
            if STOP_PATH.is_file():
                break
            guarded = bool(item.get("requires_gpu"))
            if guarded:
                # Temperatures are always read and logged before a GPU episode;
                # THERMAL_GUARD_ENABLED only controls the abort. An unreadable GPU fails closed.
                reading = thermal.read_temps()
                state["thermal_last"] = {**thermal.public(reading), "at": utc_now(),
                                         "before_job": item["job"]}
                _save(state)
                peak = reading.get("peak")
                if config.THERMAL_GUARD_ENABLED and (
                    not reading.get("ok") or peak >= config.THERMAL_ABORT_C
                ):
                    reason = (
                        "GPU temps unreadable after retries (guard fails closed)"
                        if not reading.get("ok")
                        else "GPU peak %.0fC >= %.0fC limit" % (peak, config.THERMAL_ABORT_C)
                    )
                    state["thermal_abort"] = {
                        "at": utc_now(), "reason": reason, "before_job": item["job"],
                        "peak": peak, "limit": config.THERMAL_ABORT_C,
                    }
                    oplog.server_warn("episode-queue", f"THERMAL ABORT: {reason} "
                                      f"(before {item['job']}); queue stopped, item stays queued")
                    _job_event_safe(item["job"], "Episode queue aborted by the thermal "
                                    f"guard before this episode: {reason}.")
                    aborted = True
                    break
                if reading.get("ok") and peak >= config.THERMAL_WARN_C:
                    oplog.server_warn("episode-queue", "GPU peak %.0fC in the warn band "
                                      "(>= %.0fC); continuing under the %.0fC abort limit"
                                      % (peak, config.THERMAL_WARN_C, config.THERMAL_ABORT_C))
            item["status"] = "running"
            item["started_at"] = utc_now()
            _save(state)
            if overlap and guarded:
                # QUEUE_OVERLAP_POST_STAGES (off by default): the lease is released when
                # synthesis ends and the CPU tail runs on a thread while the next episode
                # synthesizes. Temps and cooldown are taken here because the GPU is free
                # the moment render_overlapped returns.
                tail = None
                try:
                    tail = render_overlapped(item["job"], gpu_authorized=True)
                except Exception:
                    item["status"] = "failed"
                after = thermal.read_temps()
                item["thermal_after"] = {**thermal.public(after), "at": utc_now()}
                if tail is not None:
                    while len(tails) >= max(1, int(config.QUEUE_MAX_PENDING_TAILS)):
                        _finalize_tail(*tails.pop(0))
                    # the temps summary rides the in-memory tail tuple, never the
                    # persisted queue JSON
                    tails.append((item, tail, thermal.summary(after)))
                else:
                    if item["status"] != "failed":
                        _derive_status(item)
                    _job_event_safe(item["job"],
                                    f"Post-episode GPU temps: {thermal.summary(after)}.")
                    item["finished_at"] = utc_now()
                _save(state)
            else:
                try:
                    render(item["job"], gpu_authorized=bool(item.get("requires_gpu")))
                    _derive_status(item)
                except Exception:
                    item["status"] = "failed"
                if guarded:
                    after = thermal.read_temps()
                    item["thermal_after"] = {**thermal.public(after), "at": utc_now()}
                    _job_event_safe(item["job"], f"Post-episode GPU temps: {thermal.summary(after)}.")
                item["finished_at"] = utc_now()
                _save(state)
            if (guarded and index + 1 < len(pending)
                    and not STOP_PATH.is_file() and config.THERMAL_COOLDOWN_S > 0):
                # Interruptible cooldown between GPU episodes (2 s slices so a stop
                # request lands promptly).
                for _ in range(max(1, int(config.THERMAL_COOLDOWN_S // 2))):
                    if STOP_PATH.is_file():
                        break
                    time.sleep(2)
    finally:
        for entry in tails:
            try:
                _finalize_tail(*entry)
            except Exception:
                entry[0]["status"] = "failed"
        state["status"] = ("stopped" if STOP_PATH.is_file()
                           else "aborted" if aborted else "idle")
        state["stop_requested"] = STOP_PATH.is_file()
        STOP_PATH.unlink(missing_ok=True)
        _save(state)
