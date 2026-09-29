"""GPU temperature readings for the thermal-aware episode queue.

Two sources, in order of preference:

* ``AUTODUB_GPU_TEMP_COMMAND`` - any command that prints lines such as ``core 46 C``,
  ``hot spot 58.8 C`` and ``memory junction 58 C``. Use this where a vendor tool exposes sensors
  nvidia-smi does not (memory junction is usually the hottest sensor on GDDR6X cards).
* ``nvidia-smi --query-gpu=temperature.gpu`` - the portable default (core temperature only).

A reading's ``peak`` is the hottest sensor that could be read; the queue guard compares ``peak``
against its limits. Nothing here raises: an unreadable GPU returns ``ok=False`` and the caller
decides (the episode queue fails closed).
"""
from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import time

from . import config

_CORE = re.compile(r"\bcore\s+([\d.]+)\s*C")
_HOT_SPOT = re.compile(r"hot spot\s+([\d.]+)\s*C")
_JUNCTION = re.compile(r"memory junction\s+([\d.]+)\s*C")
_KEYS = ("core", "hot_spot", "junction")


def _empty(raw: str = "") -> dict:
    return {"ok": False, "core": None, "hot_spot": None, "junction": None, "peak": None, "raw": raw}


def _finish(reading: dict) -> dict:
    values = [reading[key] for key in _KEYS if reading.get(key) is not None]
    reading["peak"] = max(values) if values else None
    reading["ok"] = reading["peak"] is not None
    return reading


def parse(output: str) -> dict:
    """Parse the human-readable ``core / hot spot / memory junction`` format."""
    joined = " ".join((output or "").split())
    reading = _empty(joined[:400])
    for key, pattern in (("core", _CORE), ("hot_spot", _HOT_SPOT), ("junction", _JUNCTION)):
        match = pattern.search(joined)
        reading[key] = float(match.group(1)) if match else None
    return _finish(reading)


def parse_nvidia_smi(output: str) -> dict:
    """Parse ``nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits``.

    Multi-GPU machines print one line per card; the hottest card wins.
    """
    temps = []
    for line in (output or "").splitlines():
        try:
            temps.append(float(line.strip()))
        except ValueError:
            continue
    reading = _empty((output or "").strip()[:400])
    reading["core"] = max(temps) if temps else None
    return _finish(reading)


def _command() -> tuple[list[str], callable]:
    if config.GPU_TEMP_COMMAND:
        return shlex.split(config.GPU_TEMP_COMMAND, posix=False), parse
    smi = shutil.which("nvidia-smi") or "nvidia-smi"
    return [smi, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], parse_nvidia_smi


def read_temps(*, attempts: int = 3) -> dict:
    """Read and parse temperatures, retrying 3 s apart so one sensor flake cannot stop a long run."""
    command, parser = _command()
    reading = _empty()
    for attempt in range(max(1, attempts)):
        if attempt:
            time.sleep(3)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=config.THERMAL_READ_TIMEOUT_S,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            reading = _empty(f"({type(exc).__name__})")
            continue
        reading = parser(result.stdout or "")
        if reading["ok"]:
            return reading
    return reading


def summary(reading: dict) -> str:
    if not reading.get("ok"):
        return "gpu temps unreadable"
    parts = []
    for label, key in (("core", "core"), ("hot", "hot_spot"), ("junction", "junction")):
        value = reading.get(key)
        if value is not None:
            parts.append(f"{label} {value:.0f}C")
    return " ".join(parts) or "gpu temps unreadable"


def public(reading: dict) -> dict:
    return {key: reading.get(key) for key in ("ok", "core", "hot_spot", "junction", "peak")}
