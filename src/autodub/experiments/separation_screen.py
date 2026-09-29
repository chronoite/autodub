"""Build reviewable local dialogue-removal candidates for one opaque job."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


from autodub.config import PACKAGE_ROOT, WORK_ROOT, ANALYSIS_PYTHON, FFMPEG, runtime_env
from autodub.gpu_session import gpu_lease
from autodub.state import job_dir, load_job


QUALITY_WORKER = PACKAGE_ROOT / "workers" / "quality_worker.py"
GPU_CANDIDATE = "demucs-htdemucs-ft-gpu"
IMPLEMENTED = {GPU_CANDIDATE, "demucs-htdemucs-ft-cpu", "source-bed-ducking"}


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


def _demucs(audio: Path, scratch: Path, output: Path, device: str) -> None:
    result = subprocess.run(
        [str(ANALYSIS_PYTHON), str(QUALITY_WORKER), "separate"],
        input=json.dumps({"audio": str(audio), "output": str(scratch), "device": device}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=runtime_env(gpu=device == "cuda", portable_deps=False),
        timeout=10800,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:].strip() or "Demucs worker failed")
    bed = Path(json.loads(result.stdout)["bed"])
    if not bed.is_file():
        raise RuntimeError("Demucs returned no local bed")
    shutil.copy2(bed, output)


def _ducking_control(audio: Path, output: Path) -> None:
    result = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio),
         "-af", "volume=0.30", "-ar", "48000", "-ac", "2", str(output)],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if result.returncode or not output.is_file():
        raise RuntimeError(result.stderr[-1200:].strip() or "source-ducking control failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="run AutoDub dialogue-separation screen")
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
    if manifest.get("experiment") != "dialogue-separation-v1":
        raise SystemExit("run is not a dialogue-separation experiment")
    job_id = str(manifest.get("job") or "")
    if not _valid_id(job_id, "dub-"):
        raise SystemExit("run has an invalid opaque job ID")
    job = load_job(job_id)
    artifacts = job_dir(job_id) / "artifacts"
    audio_name = job.get("artifacts", {}).get("full_audio")
    audio = artifacts / str(audio_name or "")
    if not audio_name or not audio.is_file():
        raise SystemExit("job has no extracted local full-mix track")
    known = {item["id"] for item in manifest["candidates"]}
    selected = args.candidates or [item["id"] for item in manifest["candidates"]]
    if set(selected) - known:
        raise SystemExit("candidate is not part of this run")
    readiness = {
        candidate: {"implemented": candidate in IMPLEMENTED, "requires_gpu": candidate == GPU_CANDIDATE}
        for candidate in selected
    }
    if args.dry_run:
        print(json.dumps({"ok": True, "run": args.run, "readiness": readiness}, indent=2))
        return
    if not args.ack_private_local_material:
        raise SystemExit("actual separation comparison requires --ack-private-local-material")
    if GPU_CANDIDATE in selected and not args.arm_gpu:
        raise SystemExit("GPU separation requires --arm-gpu and a clear AutoDub preflight")
    output_root = run_root / "audio"
    scratch_root = run_root / ".scratch"
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}

    def execute(lease=None) -> None:
        for candidate in selected:
            destination = output_root / f"{candidate}.wav"
            try:
                if candidate == GPU_CANDIDATE:
                    if lease is not None:
                        lease.ensure_active()
                    _demucs(audio, scratch_root / candidate, destination, "cuda")
                elif candidate == "demucs-htdemucs-ft-cpu":
                    _demucs(audio, scratch_root / candidate, destination, "cpu")
                elif candidate == "source-bed-ducking":
                    _ducking_control(audio, destination)
                else:
                    raise RuntimeError("candidate adapter is not implemented")
                results[candidate] = {
                    "status": "ready-for-review",
                    "evidence": destination.relative_to(run_root).as_posix(),
                    "bytes": destination.stat().st_size,
                }
            except Exception as exc:
                results[candidate] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        shutil.rmtree(scratch_root, ignore_errors=True)

    if GPU_CANDIDATE in selected:
        with gpu_lease(f"separation-experiment:{args.run}", wait_seconds=10800) as lease:
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
