"""Execute an explicitly armed, local voice-quality experiment for one opaque job speaker.

This tool never creates a job or chooses private material.  It consumes the automatic reference from
an already reviewed job, writes only opaque candidate/line filenames, and records no transcript
or reference path in RUN.json.  Use `--dry-run` to inspect readiness without loading a model.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path


from autodub import adapters
from autodub.config import PACKAGE_ROOT, WORK_ROOT, CHATTERBOX_PYTHON, COSYVOICE_PYTHON, QUALITY_PYTHON, runtime_env
from autodub.gpu_session import gpu_lease
from autodub.state import job_dir, load_job


LINES_PATH = Path(__file__).with_name("voice_screen_lines.json")
QUALITY_WORKER = PACKAGE_ROOT / "workers" / "quality_worker.py"
CHATTERBOX_WORKER = PACKAGE_ROOT / "workers" / "chatterbox_worker.py"
COSYVOICE_WORKER = PACKAGE_ROOT / "workers" / "cosyvoice_worker.py"
GPU_CANDIDATES = {
    "qwen3-tts-1.7b",
    "qwen3-tts-0.6b",
    "chatterbox-multilingual-v3",
    "gpt-sovits-current",
    "cosyvoice3-0.5b",
}
IMPLEMENTED = GPU_CANDIDATES | {"windows-sapi"}


def _valid_id(value: str, prefix: str) -> bool:
    return value.startswith(prefix) and all(char in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in value)


def _atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _worker(python: Path, worker: Path, payload: dict, timeout: int = 10800) -> dict:
    result = subprocess.run(
        [str(python), str(worker), "synthesize"],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=runtime_env(gpu=True, portable_deps=False),
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:].strip() or "candidate worker failed")
    return json.loads(result.stdout)


def _gsv_line(text: str, reference: Path, reference_text: str, language: str, output: Path) -> None:
    body = {
        "text": text,
        "text_lang": "en",
        "ref_audio_path": str(reference),
        "prompt_text": reference_text,
        "prompt_lang": language,
        "text_split_method": "cut5",
        "media_type": "wav",
        "streaming_mode": False,
        "speed_factor": 1.0,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:9880/tts",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        audio = response.read()
    if audio[:4] != b"RIFF":
        raise RuntimeError("local GPT-SoVITS returned invalid WAV")
    output.write_bytes(audio)


def _candidate_payload(lines: list[dict], reference: Path, reference_text: str, output: Path) -> list[dict]:
    return [
        {
            "text": line["text"],
            "language": "English",
            "language_id": "en",
            "reference_audio": str(reference),
            "reference_text": reference_text,
            "output": str(output / f"{line['id']}.wav"),
            **{
                key: line[key]
                for key in ("exaggeration", "cfg_weight", "temperature")
                if key in line
            },
        }
        for line in lines
    ]


def execute_candidate(candidate: str, lines: list[dict], reference: Path, reference_text: str, language: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    payload_lines = _candidate_payload(lines, reference, reference_text, output)
    if candidate.startswith("qwen3-tts-"):
        return _worker(QUALITY_PYTHON, QUALITY_WORKER, {"model": candidate, "seed": 1986, "lines": payload_lines})
    if candidate == "chatterbox-multilingual-v3":
        return _worker(CHATTERBOX_PYTHON, CHATTERBOX_WORKER, {"device": "cuda", "seed": 1986, "lines": payload_lines})
    if candidate == "cosyvoice3-0.5b":
        return _worker(COSYVOICE_PYTHON, COSYVOICE_WORKER, {"seed": 1986, "lines": payload_lines})
    if candidate == "gpt-sovits-current":
        for line, item in zip(lines, payload_lines):
            _gsv_line(line["text"], reference, reference_text, language, Path(item["output"]))
        return {"model": candidate, "written": len(lines)}
    if candidate == "windows-sapi":
        voice = adapters.sapi_voices()[0]
        for line, item in zip(lines, payload_lines):
            adapters.synthesize_sapi(line["text"], voice, Path(item["output"]))
        return {"model": candidate, "written": len(lines)}
    raise RuntimeError(f"candidate adapter is not implemented: {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser(description="run the AutoDub voice-quality screen")
    parser.add_argument("--run", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--arm-gpu", action="store_true")
    parser.add_argument("--ack-private-local-material", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not _valid_id(args.run, "adx-"):
        raise SystemExit("invalid opaque experiment run ID")
    run_root = WORK_ROOT / "experiments" / args.run
    run_path = run_root / "RUN.json"
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "voice-quality-v1":
        raise SystemExit("run is not a voice-quality experiment")
    job_id = str(manifest.get("job") or "")
    if not _valid_id(job_id, "dub-"):
        raise SystemExit("run has an invalid opaque job ID")
    job = load_job(job_id)
    reference = job.get("speaker_references", {}).get(args.speaker)
    if not reference:
        raise SystemExit("the reviewed job has no automatic reference for that speaker")
    reference_audio = job_dir(job_id) / "artifacts" / reference["file"]
    if not reference_audio.is_file():
        raise SystemExit("the opaque speaker reference is missing")
    selected = args.candidates or [item["id"] for item in manifest["candidates"]]
    unknown = set(selected) - {item["id"] for item in manifest["candidates"]}
    if unknown:
        raise SystemExit("candidate is not part of this run")
    readiness = {
        candidate: {
            "implemented": candidate in IMPLEMENTED,
            "requires_gpu": candidate in GPU_CANDIDATES,
        }
        for candidate in selected
    }
    if args.dry_run:
        print(json.dumps({"ok": True, "run": args.run, "readiness": readiness}, indent=2))
        return
    if not args.ack_private_local_material:
        raise SystemExit("actual voice comparison requires --ack-private-local-material")
    gpu_selected = any(candidate in GPU_CANDIDATES for candidate in selected)
    if gpu_selected and not args.arm_gpu:
        raise SystemExit("GPU candidates require --arm-gpu and a clear AutoDub preflight")
    lines = json.loads(LINES_PATH.read_text(encoding="utf-8"))["lines"]
    results = {}

    def run_selected(lease=None) -> None:
        for candidate in selected:
            if candidate not in IMPLEMENTED:
                results[candidate] = {"status": "not-ready", "error": "adapter not implemented"}
                continue
            try:
                if lease is not None and candidate in GPU_CANDIDATES:
                    lease.ensure_active()
                detail = execute_candidate(
                    candidate,
                    lines,
                    reference_audio,
                    str(reference.get("text") or ""),
                    str(reference.get("language") or "ja"),
                    run_root / "audio" / candidate,
                )
                results[candidate] = {
                    "status": "ready-for-review",
                    "files": len(lines),
                    "evidence_dir": f"audio/{candidate}",
                    "detail": detail,
                }
            except Exception as exc:
                detail = str(exc)
                results[candidate] = {"status": "failed", "error": f"{type(exc).__name__}: {detail[-4000:]}"}

    if gpu_selected:
        with gpu_lease(f"voice-experiment:{args.run}", wait_seconds=7200) as lease:
            run_selected(lease)
    else:
        run_selected()
    by_id = {item["id"]: item for item in manifest["candidates"]}
    for candidate, result in results.items():
        by_id[candidate]["result"] = result
    manifest["status"] = "awaiting-review" if any(
        result["status"] == "ready-for-review" for result in results.values()
    ) else "failed"
    _atomic_json(run_path, manifest)
    print(json.dumps({"ok": manifest["status"] == "awaiting-review", "run": args.run, "results": results}, indent=2))


if __name__ == "__main__":
    main()
