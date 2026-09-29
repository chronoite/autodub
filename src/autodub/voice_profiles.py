from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path

from .config import VOICES_ROOT
from .state import replace_retry
from .media import probe_duration


ALLOWED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
MAX_REFERENCE_BYTES = 100 * 1024 * 1024


def _profile_id() -> str:
    return "voice-" + secrets.token_hex(4)


def profile_path(profile_id: str) -> Path:
    if not profile_id.startswith("voice-") or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in profile_id):
        raise ValueError("invalid voice profile id")
    return VOICES_ROOT / profile_id / "profile.json"


def load_profile(profile_id: str) -> dict:
    path = profile_path(profile_id)
    profile = json.loads(path.read_text(encoding="utf-8"))
    reference = path.parent / profile["reference"]
    if not reference.is_file():
        raise FileNotFoundError("voice reference is missing")
    profile["reference_path"] = str(reference)
    return profile


def list_profiles() -> list[dict[str, str]]:
    profiles = []
    if not VOICES_ROOT.exists():
        return profiles
    for path in sorted(VOICES_ROOT.glob("voice-*/profile.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            profiles.extend(
                [
                    {"id": data["id"], "option": f"qwen:{data['id']}", "label": f"Qwen3 clone · {data['id']}"},
                    {"id": data["id"], "option": f"gsv:{data['id']}", "label": f"GPT-SoVITS clone · {data['id']}"},
                ]
            )
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return profiles


def import_profile(reader, length: int, suffix: str, transcript: str, language: str) -> dict[str, str]:
    suffix = suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise ValueError("unsupported reference audio type")
    if length <= 0 or length > MAX_REFERENCE_BYTES:
        raise ValueError("invalid reference audio size")
    transcript = transcript.strip()
    if not transcript or len(transcript) > 1000:
        raise ValueError("a reference transcript between 1 and 1000 characters is required")
    if language not in {"en", "ja", "zh", "ko", "es", "de", "fr", "ru", "pt", "it"}:
        raise ValueError("unsupported reference language")

    profile_id = _profile_id()
    root = VOICES_ROOT / profile_id
    root.mkdir(parents=True, exist_ok=False)
    reference = root / f"reference{suffix}"
    digest = hashlib.sha256()
    written = 0
    try:
        with reference.open("wb") as handle:
            while written < length:
                chunk = reader.read(min(1024 * 1024, length - written))
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        if written != length:
            raise ValueError("reference transfer ended early")
        duration = probe_duration(reference)
        if not 1.0 <= duration <= 30.0:
            raise ValueError("reference audio must be between 1 and 30 seconds")
        profile = {
            "schema": 1,
            "id": profile_id,
            "engine": "gpt-sovits-loopback",
            "reference": reference.name,
            "reference_bytes": written,
            "reference_sha256": digest.hexdigest(),
            "reference_seconds": round(duration, 3),
            "prompt_text": transcript,
            "prompt_language": language,
        }
        path = root / "profile.json"
        fd, temporary = tempfile.mkstemp(prefix="profile-", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(profile, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            replace_retry(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {"id": profile_id, "option": f"gsv:{profile_id}", "label": f"Local clone · {profile_id}"}
    except Exception:
        reference.unlink(missing_ok=True)
        root.rmdir()
        raise
