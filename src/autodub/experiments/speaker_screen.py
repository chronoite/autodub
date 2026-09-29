"""Run CPU/GPU speaker-detection candidates against one opaque AutoDub job.

Output assignment files contain only segment indices/times and opaque speaker IDs.  Transcript text,
source names, and host paths are never written to experiment evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path


from autodub import adapters
from autodub.config import PACKAGE_ROOT, WORK_ROOT, ANALYSIS_PYTHON, runtime_env
from autodub.gpu_session import gpu_lease
from autodub.speaker_evidence import normalize_speaker_count
from autodub.state import job_dir, load_job


QUALITY_WORKER = PACKAGE_ROOT / "workers" / "quality_worker.py"
GPU_CANDIDATE = "pyannote-community-1-exclusive-gpu"


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


def _pyannote(audio: Path, segments: list[dict], speaker_count: dict, device: str) -> dict:
    result = subprocess.run(
        [str(ANALYSIS_PYTHON), str(QUALITY_WORKER), "diarize"],
        input=json.dumps(
            {"audio": str(audio), "segments": segments, "speaker_count": speaker_count, "device": device},
            ensure_ascii=False,
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=runtime_env(gpu=device == "cuda", portable_deps=False),
        timeout=10800,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:].strip() or "pyannote worker failed")
    return json.loads(result.stdout)


def _public_assignments(segments: list[dict]) -> list[dict]:
    allowed = ("i", "start", "end", "speaker", "speaker_confidence", "overlap_ratio")
    return [{key: item[key] for key in allowed if key in item} for item in segments]


def main() -> None:
    parser = argparse.ArgumentParser(description="run AutoDub speaker-detection screen")
    parser.add_argument("--run", required=True)
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
    if manifest.get("experiment") != "speaker-detection-v1":
        raise SystemExit("run is not a speaker-detection experiment")
    job_id = str(manifest.get("job") or "")
    if not _valid_id(job_id, "dub-"):
        raise SystemExit("run has an invalid opaque job ID")
    job = load_job(job_id)
    artifacts = job_dir(job_id) / "artifacts"
    audio_name = job.get("artifacts", {}).get("dialogue_stem") or job.get("artifacts", {}).get("asr_audio")
    if not audio_name or not (artifacts / audio_name).is_file():
        raise SystemExit("job has no analyzed local dialogue track")
    if not job.get("segments"):
        raise SystemExit("job has no speech segments")
    known = {item["id"] for item in manifest["candidates"]}
    selected = args.candidates or [item["id"] for item in manifest["candidates"]]
    if set(selected) - known:
        raise SystemExit("candidate is not part of this run")
    readiness = {
        candidate: {"implemented": True, "requires_gpu": candidate == GPU_CANDIDATE}
        for candidate in selected
    }
    if args.dry_run:
        print(json.dumps({"ok": True, "run": args.run, "readiness": readiness}, indent=2))
        return
    if not args.ack_private_local_material:
        raise SystemExit("actual speaker comparison requires --ack-private-local-material")
    if GPU_CANDIDATE in selected and not args.arm_gpu:
        raise SystemExit("GPU diarization requires --arm-gpu and a clear AutoDub preflight")
    output_root = run_root / "assignments"
    output_root.mkdir(parents=True, exist_ok=True)
    settings = job.get("settings", {})
    expected = normalize_speaker_count(
        settings.get("speaker_count", settings.get("expected_speakers", 0)))
    results = {}

    def execute(lease=None) -> None:
        for candidate in selected:
            try:
                source_segments = copy.deepcopy(job["segments"])
                if candidate == "acoustic-mfcc-reviewable":
                    segments = adapters.cluster_speakers(artifacts / audio_name, source_segments, expected)
                    method = candidate
                    device = "cpu"
                elif candidate.endswith("-gpu"):
                    if lease is not None:
                        lease.ensure_active()
                    detail = _pyannote(artifacts / audio_name, source_segments, expected, "cuda")
                    segments, method, device = detail["segments"], detail["method"], detail["device"]
                else:
                    detail = _pyannote(artifacts / audio_name, source_segments, expected, "cpu")
                    segments, method, device = detail["segments"], detail["method"], detail["device"]
                destination = output_root / f"{candidate}.json"
                evidence = {
                    "schema": 1,
                    "candidate": candidate,
                    "method": method,
                    "device": device,
                    "segments": _public_assignments(segments),
                }
                _atomic_json(destination, evidence)
                results[candidate] = {
                    "status": "ready-for-review",
                    "segments": len(segments),
                    "evidence": destination.relative_to(run_root).as_posix(),
                }
            except Exception as exc:
                results[candidate] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}

    if GPU_CANDIDATE in selected:
        with gpu_lease(f"speaker-experiment:{args.run}", wait_seconds=7200) as lease:
            execute(lease)
    else:
        execute()
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
