"""Test package bootstrap.

Runs before any test module is imported (``python -m unittest discover``): puts ``src/`` on the
import path and points AUTODUB_HOME at a throwaway directory, so the suite never reads or writes a
real data directory. Child processes started by tests inherit both settings.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("AUTODUB_QUIET_ACCESS_LOG", "1")   # keep test output readable
os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")

# Always a fresh temporary home, even if AUTODUB_HOME is set in the shell: tests must never touch
# real jobs. Set AUTODUB_TEST_KEEP_HOME=1 to keep the directory afterwards for debugging.
_home = tempfile.mkdtemp(prefix="autodub-test-home-")
os.environ["AUTODUB_HOME"] = _home
if not os.environ.get("AUTODUB_TEST_KEEP_HOME"):
    atexit.register(shutil.rmtree, _home, ignore_errors=True)

from autodub.config import ensure_layout  # noqa: E402

ensure_layout()


# ---- capability gates ---------------------------------------------------------------------------
# Tests that need real media tools or model weights skip (with a reason) when those are absent, so
# the pure-logic suite runs anywhere (CI) and the full suite runs on a configured workstation.
import unittest  # noqa: E402

from autodub import config as _config  # noqa: E402


def _tool_available(path: Path) -> bool:
    return path.is_file() or shutil.which(str(path)) is not None


HAVE_FFMPEG = _tool_available(_config.FFMPEG) and _tool_available(_config.FFPROBE)
requires_ffmpeg = unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not found (set AUTODUB_FFMPEG)")
requires_windows = unittest.skipUnless(sys.platform == "win32", "Windows SAPI voices required")


def requires_model(path: Path):
    return unittest.skipUnless(Path(path).exists(), f"model not installed: {Path(path).name}")


def requires_worker_modules(*modules: str):
    """Skip unless the media worker interpreter can import every named module."""
    import subprocess

    probe = "import importlib.util, sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in sys.argv[1:]) else 1)"
    try:
        ok = subprocess.run([str(_config.MEDIA_PYTHON), "-c", probe, *modules],
                            capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    return unittest.skipUnless(ok, "media worker packages missing: " + ", ".join(modules))
