"""Versioned AutoDub execution profiles.

Profiles are declarative on purpose: jobs record the exact intended stack, the UI can explain it,
and experiments can compare one component at a time.  A profile never downloads a model.
"""
from __future__ import annotations

from copy import deepcopy


DEFAULT_PROFILE = "quality-gpu-v1"
CPU_PROFILE = "prototype-cpu-v1"

PROFILES = {
    DEFAULT_PROFILE: {
        "label": "Quality GPU v1",
        "quality_tier": "primary",
        "requires_gpu": True,
        "asr": "faster-whisper-large-v3-cuda-fp16",
        "alignment": "whisper-word-timestamps-v1",
        "diarization": "pyannote-community-1-exclusive",
        "separation": "demucs-htdemucs-ft",
        "translation": "marian-ja-en-reviewed-baseline",
        "tts": "qwen3-tts-1.7b-base-clone",
        "mix": "dialogue-replacement-v2",
        "runtime_network": "denied",
    },
    CPU_PROFILE: {
        "label": "Prototype CPU fallback",
        "quality_tier": "fallback",
        "requires_gpu": False,
        "asr": "faster-whisper-large-v3-cpu-int8",
        "alignment": "segment-windows-v1",
        "diarization": "acoustic-mfcc-reviewable",
        "separation": "source-bed-ducking",
        "translation": "marian-ja-en-reviewed-baseline",
        "tts": "windows-sapi-or-gpt-sovits",
        "mix": "dialogue-ducking-v1",
        "runtime_network": "denied",
    },
}


def get_profile(name: str) -> dict:
    try:
        return deepcopy(PROFILES[name])
    except KeyError as exc:
        raise ValueError(f"unknown quality profile: {name}") from exc


def profile_requires_gpu(name: str) -> bool:
    return bool(get_profile(name)["requires_gpu"])


def public_profiles() -> list[dict]:
    return [{"id": name, **deepcopy(value)} for name, value in PROFILES.items()]
