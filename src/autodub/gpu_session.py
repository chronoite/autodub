"""Fail-closed GPU admission and leasing.

GPU work goes through three gates, none of which runs at server startup:

1. **Arm** - the reviewer explicitly arms one action for one job in the UI. An arm expires after
   ``ARM_TTL_SECONDS`` and is consumed by exactly one request.
2. **Preflight** - a fresh safety check. Any failed probe blocks the action (fail closed).
3. **Lease** - exclusive use of the GPU for the duration of the work, always released in ``finally``.

Two lease back-ends are available:

* **Local** (default): a process-wide lock, so one AutoDub process never runs two GPU jobs at once.
* **Broker** (``AUTODUB_GPU_BROKER_URL``): a small HTTP lease protocol for machines where several
  GPU applications share one card. Leases are renewed by a heartbeat thread and fail closed after
  bounded heartbeat loss. The protocol is documented in ``docs/GPU-COORDINATION.md``.

AutoDub never stops or restarts other processes to free the GPU; it waits or refuses.
"""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager

from . import config

ARM_TTL_SECONDS = 300
LEASE_TTL_SECONDS = 180
HEARTBEAT_INTERVAL_SECONDS = 45
HEARTBEAT_FAILURE_LIMIT = 3
GPU_ACTIONS = frozenset({"analyze", "render", "experiment", "episode-render", "realign", "adapt", "demo"})


class GpuSafetyError(RuntimeError):
    pass


class GpuLease(dict):
    """Dict-compatible renewable broker lease that fails closed after bounded heartbeat loss."""

    def __init__(self, value: dict, *, sender, heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
                 failure_limit: int = HEARTBEAT_FAILURE_LIMIT):
        super().__init__(value)
        self._sender = sender
        self._heartbeat_interval = float(heartbeat_interval)
        self._failure_limit = int(failure_limit)
        self._stop_event = threading.Event()
        self._failed_event = threading.Event()
        self._state_lock = threading.Lock()
        self._consecutive_failures = 0
        self._thread: threading.Thread | None = None

    @property
    def consecutive_heartbeat_failures(self) -> int:
        with self._state_lock:
            return self._consecutive_failures

    def _beat_once(self) -> bool:
        try:
            result = self._sender(
                "/queue/heartbeat",
                {"lease_id": self.get("lease_id"), "ttl_seconds": LEASE_TTL_SECONDS},
            )
            renewed = bool(result.get("ok"))
        except Exception:
            renewed = False
            result = {}
        with self._state_lock:
            if renewed:
                self._consecutive_failures = 0
                if result.get("expires_at") is not None:
                    self["expires_at"] = result["expires_at"]
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failure_limit:
                    self._failed_event.set()
        return renewed

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_interval):
            self._beat_once()
            if self._failed_event.is_set():
                return

    def start(self) -> GpuLease:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name="autodub-gpu-lease-heartbeat",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=15)

    def ensure_active(self) -> None:
        """Block the next GPU submission after bounded heartbeat failure."""
        if self._failed_event.is_set():
            raise GpuSafetyError("GPU lease heartbeat failed repeatedly; new GPU work is blocked")


class LocalLease(dict):
    """In-process lease: holding it is the guarantee, so it can never lapse."""

    def ensure_active(self) -> None:
        return None


_local_gpu = threading.Lock()
_arms: dict[tuple[str, str], float] = {}
_arm_lock = threading.Lock()


# ---- probes ------------------------------------------------------------------------------------

def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _busy_ports() -> list[int]:
    """Configured ports whose listener means "another GPU-heavy service is running"."""
    return [port for port in config.ADAPT_CONFLICT_PORTS if _port_open(port)]


def _broker_status() -> dict:
    if not config.GPU_BROKER_URL:
        return {"configured": False}
    try:
        with urllib.request.urlopen(config.GPU_BROKER_URL + "/lease/status", timeout=3) as response:
            payload = json.load(response)
        payload["configured"] = True
        payload["available"] = True
        return payload
    except Exception:
        return {"configured": True, "available": False}


def preflight_status(*, port_probe=None, broker_probe=None) -> dict:
    """Return ``{"safe": bool, "reasons": [...], ...}``. Any probe exception fails closed."""
    port_probe = port_probe or _busy_ports
    broker_probe = broker_probe or _broker_status
    try:
        busy = list(port_probe())
        broker = broker_probe()
        reasons = []
        for port in busy:
            reasons.append(f"a conflicting GPU service is listening on port {port}")
        if broker.get("configured") and not broker.get("available"):
            reasons.append("the GPU lease broker is unavailable")
        return {
            "safe": not reasons,
            "coordinator": "broker" if broker.get("configured") else "local",
            "busy_ports": busy,
            "local_lease_held": _local_gpu.locked(),
            "lease_count": int(broker.get("lease_count", 0) or 0),
            "queue_count": int(broker.get("queue_count", 0) or 0),
            "reasons": reasons,
        }
    except Exception as exc:
        return {
            "safe": False,
            "coordinator": "unknown",
            "reasons": [f"safety preflight failed closed ({type(exc).__name__})"],
        }


# ---- arming ------------------------------------------------------------------------------------

def arm(job_id: str, action: str) -> dict:
    if action not in GPU_ACTIONS:
        raise ValueError("invalid GPU action")
    status = preflight_status()
    if not status["safe"]:
        raise GpuSafetyError("; ".join(status["reasons"]))
    with _arm_lock:
        _arms[(job_id, action)] = time.monotonic() + ARM_TTL_SECONDS
    return {"armed": True, "job_id": job_id, "action": action, "expires_in": ARM_TTL_SECONDS, "preflight": status}


def consume_arm(job_id: str, action: str) -> bool:
    with _arm_lock:
        expiry = _arms.pop((job_id, action), 0)
    return expiry >= time.monotonic()


# ---- broker protocol ---------------------------------------------------------------------------

def _post(path: str, payload: dict, timeout: int = 10) -> dict:
    request = urllib.request.Request(
        config.GPU_BROKER_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except Exception as exc:
        raise GpuSafetyError(f"GPU broker request failed ({type(exc).__name__})") from exc


def _request_status(request_id: str) -> dict:
    try:
        with urllib.request.urlopen(
            config.GPU_BROKER_URL + "/queue/request?" + urllib.parse.urlencode({"id": request_id}),
            timeout=10,
        ) as response:
            return json.load(response)
    except Exception as exc:
        raise GpuSafetyError(f"GPU broker status failed ({type(exc).__name__})") from exc


@contextmanager
def _broker_lease(reason: str, wait_seconds: int, sender, status_reader):
    lease = sender(
        "/queue/request",
        {
            "owner": "autodub",
            "reason": reason,
            "resources": ["gpu"],
            "ttl_seconds": LEASE_TTL_SECONDS,
            "queue_ttl_seconds": wait_seconds,
        },
    )
    request_id = lease.get("request_id")
    deadline = time.monotonic() + max(1, wait_seconds)
    try:
        while lease.get("state") == "queued":
            if time.monotonic() >= deadline:
                raise GpuSafetyError("timed out waiting for the shared GPU queue")
            time.sleep(1)
            lease = status_reader(request_id)
    except Exception:
        if request_id:
            try:
                sender("/queue/cancel", {"request_id": request_id})
            except Exception:
                pass
        raise
    lease_id = lease.get("lease_id")
    if not lease.get("ok") or lease.get("state") != "granted" or not lease_id:
        raise GpuSafetyError("GPU broker did not grant a valid lease")
    renewable = GpuLease(lease, sender=sender).start()
    try:
        yield renewable
        renewable.ensure_active()
    finally:
        renewable.stop()
        sender("/queue/release", {"lease_id": lease_id})


@contextmanager
def _local_lease(reason: str, wait_seconds: int):
    if not _local_gpu.acquire(timeout=max(1, wait_seconds)):
        raise GpuSafetyError("timed out waiting for the GPU (another AutoDub GPU task is running)")
    try:
        yield LocalLease({"reason": reason, "coordinator": "local"})
    finally:
        _local_gpu.release()


@contextmanager
def gpu_lease(reason: str, *, wait_seconds: int = 21600, post=None, status=None, preflight=None):
    """Hold exclusive GPU access for the body of the ``with`` block.

    ``post``/``status`` inject a broker transport (tests); passing ``post`` selects the broker path.
    """
    check = (preflight or preflight_status)()
    if not check.get("safe"):
        raise GpuSafetyError("; ".join(check.get("reasons") or ["GPU safety preflight blocked the action"]))
    if post is not None or config.GPU_BROKER_URL:
        with _broker_lease(reason, wait_seconds, post or _post, status or _request_status) as lease:
            yield lease
    else:
        with _local_lease(reason, wait_seconds) as lease:
            yield lease
