"""Declarative, swappable AutoDub timing and mix policies.

The pipeline records policy IDs in each job.  Implementations consume copies of these dictionaries,
so experiments can tune one component without forking the render pipeline.
"""
from __future__ import annotations

from copy import deepcopy


DEFAULT_MIX_POLICY = "balanced-v1"
DEFAULT_TIMING_POLICY = "gentle-fit-v1"
DEFAULT_SPACE_POLICY = "dry-v1"

MIX_POLICIES = {
    "balanced-v1": {
        "label": "Balanced scene",
        "description": "Natural bed with moderate speech ducking; recommended starting point.",
        "bed_gain": 0.85,
        "dialogue_gain": 1.0,
        "sidechain_threshold": 0.04,
        "sidechain_ratio": 4.0,
        "attack_ms": 15,
        "release_ms": 260,
        "limiter": 0.95,
    },
    "dialogue-forward-v1": {
        "label": "Dialogue forward",
        "description": "Stronger ducking for dense or action-heavy scenes.",
        "bed_gain": 0.75,
        "dialogue_gain": 1.0,
        "sidechain_threshold": 0.03,
        "sidechain_ratio": 6.0,
        "attack_ms": 10,
        "release_ms": 240,
        "limiter": 0.95,
    },
    "cinematic-v1": {
        "label": "Cinematic bed",
        "description": "Preserves more music and effects with restrained dialogue.",
        "bed_gain": 0.95,
        "dialogue_gain": 0.88,
        "sidechain_threshold": 0.055,
        "sidechain_ratio": 3.0,
        "attack_ms": 20,
        "release_ms": 320,
        "limiter": 0.95,
    },
    "legacy-v1": {
        "label": "Legacy aggressive",
        "description": "Original AutoDub ducking behavior retained for controlled comparison.",
        "bed_gain": 1.0,
        "dialogue_gain": 1.0,
        "sidechain_threshold": 0.015,
        "sidechain_ratio": 10.0,
        "attack_ms": 8,
        "release_ms": 220,
        "limiter": 0.95,
    },
    # Balance-bench candidates (after listening found dialogue somewhat too loud): a
    # softer-dialogue ladder judged blind in dub-mix-balance-v1. Candidates only —
    # DEFAULT_MIX_POLICY stays balanced-v1 until a reviewer promotes a winner.
    "balanced-soft-v1": {
        "label": "Balanced, softer dialogue",
        "description": "Half-step down from balanced-v1: slightly quieter dialogue over a fuller bed.",
        "bed_gain": 0.90,
        "dialogue_gain": 0.92,
        "sidechain_threshold": 0.045,
        "sidechain_ratio": 3.5,
        "attack_ms": 15,
        "release_ms": 280,
        "limiter": 0.95,
    },
    "dialogue-soft-v1": {
        "label": "Soft dialogue",
        "description": "Clearly quieter dialogue with gentle ducking; bed nearly untouched.",
        "bed_gain": 0.95,
        "dialogue_gain": 0.82,
        "sidechain_threshold": 0.055,
        "sidechain_ratio": 2.5,
        "attack_ms": 18,
        "release_ms": 320,
        "limiter": 0.95,
    },
    "dialogue-softest-v1": {
        "label": "Softest dialogue (bracket)",
        "description": "Deliberate far end of the ladder - full bed, minimal ducking; likely too quiet.",
        "bed_gain": 1.0,
        "dialogue_gain": 0.75,
        "sidechain_threshold": 0.06,
        "sidechain_ratio": 2.0,
        "attack_ms": 20,
        "release_ms": 350,
        "limiter": 0.95,
    },
}

TIMING_POLICIES = {
    "gentle-fit-v1": {
        "label": "Gentle fit",
        "description": "Fit when natural; otherwise preserve the complete line and flag overrun.",
        "min_tempo": 0.90,
        "max_tempo": 1.15,
        "fit_mode": "preserve",
    },
    "natural-start-v1": {
        "label": "Natural delivery",
        "description": "Keep synthesized pacing and anchor only the line start.",
        "min_tempo": 1.0,
        "max_tempo": 1.0,
        "fit_mode": "preserve",
    },
    "segment-window-v1": {
        "label": "Exact segment window",
        "description": "Legacy bounded stretch plus exact trim/pad; useful as a comparison baseline.",
        "min_tempo": 0.60,
        "max_tempo": 1.75,
        "fit_mode": "exact",
    },
    "capped-breath-v1": {
        "label": "Strict + collision-proof",
        "description": "Trim synth silence, keep natural pacing, let a line breathe past its "
                       "slot into silence but NEVER across the next line's start (double-voice spill "
                       "becomes impossible by construction).",
        "min_tempo": 0.90,
        "max_tempo": 1.15,
        "fit_mode": "preserve-capped",
        "trim_silence": True,
    },
}

SPACE_POLICIES = {
    "dry-v1": {
        "label": "Dry / unchanged",
        "description": "No room coloration; safest default before listening review.",
        "dialogue_filter": "",
    },
    "light-room-v1": {
        "label": "Light room glue",
        "description": "Conservative voice EQ and very short room reflection for scene integration.",
        "dialogue_filter": "highpass=f=70,lowpass=f=15500,aecho=0.8:0.9:18:0.035",
    },
}


def get_mix_policy(name: str) -> dict:
    try:
        return deepcopy(MIX_POLICIES[name])
    except KeyError as exc:
        raise ValueError(f"unknown mix policy: {name}") from exc


def get_timing_policy(name: str) -> dict:
    try:
        return deepcopy(TIMING_POLICIES[name])
    except KeyError as exc:
        raise ValueError(f"unknown timing policy: {name}") from exc


def get_space_policy(name: str) -> dict:
    try:
        return deepcopy(SPACE_POLICIES[name])
    except KeyError as exc:
        raise ValueError(f"unknown space policy: {name}") from exc


def public_policies() -> dict:
    return {
        "default_mix": DEFAULT_MIX_POLICY,
        "default_timing": DEFAULT_TIMING_POLICY,
        "default_space": DEFAULT_SPACE_POLICY,
        "mix": [{"id": key, **deepcopy(value)} for key, value in MIX_POLICIES.items()],
        "timing": [{"id": key, **deepcopy(value)} for key, value in TIMING_POLICIES.items()],
        "space": [{"id": key, **deepcopy(value)} for key, value in SPACE_POLICIES.items()],
    }
