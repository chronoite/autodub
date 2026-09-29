from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from .adapters import sapi_voices
from .config import (
    ALLOWED_SUFFIXES,
    DATA_ROOT,
    FFMPEG,
    FFPROBE,
    KOBOLDCPP,
    MODELS_ROOT,
    ANALYSIS_PYTHON,
    MEDIA_PYTHON,
    PORT,
    STATIC_ROOT,
    TRANSLATION_MODEL,
    WHISPER_MODEL,
    QWEN_TTS_17B,
    CHATTERBOX_V3,
    COSYVOICE3,
    CHATTERBOX_PYTHON,
    COSYVOICE_PYTHON,
    PYANNOTE_MODEL,
    WHISPERX_JA_ALIGN_MODEL,
    QWEN3_CONTEXTUAL_GGUF,
    QUALITY_PYTHON,
    AUTODUB_MODEL_CACHE,
    ensure_layout,
)
from .pipeline import analyze, render
from .quality_profiles import DEFAULT_PROFILE, PROFILES, get_profile
from .server import serve
from .state import default_job, job_dir, load_job, new_job_id, public_job, save_job
from .workflow import request_cancel


def import_source(source: Path, profile: str = DEFAULT_PROFILE) -> dict:
    source = source.resolve()
    if not source.is_file() or source.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("source must be a supported local video file")
    job_id = new_job_id()
    root = job_dir(job_id)
    root.mkdir(parents=True, exist_ok=False)
    destination = root / f"source{source.suffix.lower()}"
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("wb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
    job = default_job(job_id, source.suffix.lower(), destination.stat().st_size, digest.hexdigest())
    job["settings"]["quality_profile"] = profile
    job["settings"]["quality_stack"] = get_profile(profile)
    save_job(job)
    return job


def _tool(path: Path) -> bool:
    return path.is_file() or shutil.which(str(path)) is not None


def doctor() -> int:
    """Report what is installed. Exit 0 when the CPU pipeline can run, 2 otherwise."""
    demucs = AUTODUB_MODEL_CACHE / "torch" / "hub" / "checkpoints"
    groups = {
        "core": {
            "ffmpeg": _tool(FFMPEG),
            "ffprobe": _tool(FFPROBE),
            "media_python": _tool(MEDIA_PYTHON),
            "whisper_model": (WHISPER_MODEL / "model.bin").is_file(),
            "translation_model": (TRANSLATION_MODEL / "config.json").is_file(),
            "static_ui": (STATIC_ROOT / "index.html").is_file(),
        },
        "quality": {
            "quality_python": _tool(QUALITY_PYTHON),
            "analysis_python": _tool(ANALYSIS_PYTHON),
            "qwen3_tts_1_7b": (QWEN_TTS_17B / "config.json").is_file(),
            "pyannote_community_1": (PYANNOTE_MODEL / "config.yaml").is_file(),
            "whisperx_ja_alignment": (WHISPERX_JA_ALIGN_MODEL / "config.json").is_file(),
            "demucs_htdemucs_ft": all((demucs / name).is_file() for name in (
                "f7e0c4bc-ba3fe64a.th", "d12395a8-e57c48e6.th", "92cfc3b6-ef3bcb9c.th", "04573f0d-f3cf25b2.th")),
        },
        "optional": {
            "koboldcpp": _tool(KOBOLDCPP),
            "qwen3_14b_gguf": QWEN3_CONTEXTUAL_GGUF.is_file(),
            "chatterbox_python": _tool(CHATTERBOX_PYTHON),
            "chatterbox_multilingual": (CHATTERBOX_V3 / "t3_mtl23ls_v3.safetensors").is_file(),
            "cosyvoice_python": _tool(COSYVOICE_PYTHON),
            "cosyvoice3": (COSYVOICE3 / "cosyvoice3.yaml").is_file(),
            "windows_sapi_voices": bool(sapi_voices()),
        },
    }
    ready = {name: all(checks.values()) for name, checks in groups.items()}
    print(json.dumps({"data_dir": str(DATA_ROOT), "models_dir": str(MODELS_ROOT),
                      "ready": ready, "checks": groups}, indent=2))
    return 0 if ready["core"] else 2


def main() -> None:
    ensure_layout()
    parser = argparse.ArgumentParser(description="AutoDub local studio")
    sub = parser.add_subparsers(dest="command", required=True)
    serve_parser = sub.add_parser("serve", help="start the loopback-only review UI")
    serve_parser.add_argument("--port", type=int, default=PORT)
    import_parser = sub.add_parser("import", help="copy a source into an opaque resumable job")
    import_parser.add_argument("--in", dest="source", required=True, type=Path)
    import_parser.add_argument("--profile", choices=sorted(PROFILES), default=DEFAULT_PROFILE)
    for name in ("analyze", "render", "status", "cancel"):
        command = sub.add_parser(name)
        command.add_argument("--job", required=True)
        if name in {"analyze", "render"}:
            command.add_argument("--arm-gpu", action="store_true", help="explicitly authorize the guarded GPU path")
    sub.add_parser("doctor", help="report installed tools and models (no GPU use)")
    args = parser.parse_args()

    if args.command == "serve":
        serve(port=args.port)
    elif args.command == "import":
        print(json.dumps(public_job(import_source(args.source, args.profile)), indent=2))
    elif args.command == "analyze":
        analyze(args.job, gpu_authorized=args.arm_gpu)
        print(json.dumps(public_job(load_job(args.job)), ensure_ascii=False, indent=2))
    elif args.command == "render":
        render(args.job, gpu_authorized=args.arm_gpu)
        print(json.dumps(public_job(load_job(args.job)), ensure_ascii=False, indent=2))
    elif args.command == "status":
        print(json.dumps(public_job(load_job(args.job)), ensure_ascii=False, indent=2))
    elif args.command == "cancel":
        load_job(args.job)
        request_cancel(args.job)
        print(json.dumps({"accepted": True, "action": "cancel", "job": args.job}, indent=2))
    elif args.command == "doctor":
        raise SystemExit(doctor())
