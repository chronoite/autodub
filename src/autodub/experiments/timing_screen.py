"""Run local timing-boundary candidates for one opaque AutoDub job.

Structured evidence contains segment indices/timestamps only. With explicit private-material
acknowledgement, the runner also builds opaque local MP4 comparisons from already-rendered lines.
Source text, translations, filenames, and host paths are never copied into RUN.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


from autodub.config import PACKAGE_ROOT, WORK_ROOT, ANALYSIS_PYTHON, runtime_env
from autodub.gpu_session import gpu_lease
from autodub.media import align_line, build_mix, mux
from autodub.state import job_dir, load_job


QUALITY_WORKER = PACKAGE_ROOT / "workers" / "quality_worker.py"
FORCED = "whisperx-forced-alignment"
IMPLEMENTED = {"segment-window-atempo", "whisper-word-boundaries", FORCED}


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


def _segment_windows(segments: list[dict]) -> list[dict]:
    return [
        {
            "i": int(segment["i"]),
            "start": round(float(segment["start"]), 3),
            "end": round(float(segment["end"]), 3),
            "boundary_count": 2,
        }
        for segment in segments
    ]


def _word_windows(segments: list[dict]) -> list[dict]:
    output = []
    for segment in segments:
        words = [
            word for word in segment.get("words", [])
            if word.get("start") is not None and word.get("end") is not None
        ]
        output.append(
            {
                "i": int(segment["i"]),
                "start": round(float(words[0]["start"] if words else segment["start"]), 3),
                "end": round(float(words[-1]["end"] if words else segment["end"]), 3),
                "boundary_count": len(words) * 2 if words else 2,
            }
        )
    return output


def _forced_alignment(audio: Path, segments: list[dict], device: str) -> list[dict]:
    result = subprocess.run(
        [str(ANALYSIS_PYTHON), str(QUALITY_WORKER), "align"],
        input=json.dumps(
            {"audio": str(audio), "segments": segments, "language": "ja", "device": device},
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
        raise RuntimeError(result.stderr[-2000:].strip() or "forced-alignment worker failed")
    return _word_windows(json.loads(result.stdout)["segments"])


def _render_comparison(job: dict, boundaries: list[dict], run_root: Path, candidate: str) -> tuple[Path, list[dict]]:
    root = job_dir(job["id"])
    artifacts = root / "artifacts"
    lines = artifacts / "lines"
    bed_name = job.get("artifacts", {}).get("source_bed") or job.get("artifacts", {}).get("full_audio")
    bed = artifacts / str(bed_name or "")
    source = root / str(job.get("source", {}).get("file") or "")
    if not bed_name or not bed.is_file() or not source.is_file():
        raise RuntimeError("job needs its extracted bed and opaque source before timing comparison")
    if not lines.is_dir() or not any(lines.glob("line-*.wav")):
        raise RuntimeError("job needs already-rendered line audio before timing comparison")

    aligned = run_root / "aligned" / candidate
    shutil.rmtree(aligned, ignore_errors=True)
    aligned.mkdir(parents=True)
    by_index = {int(item["i"]): item for item in job["segments"]}
    timeline = []
    rendered_boundaries = []
    for boundary in boundaries:
        index = int(boundary["i"])
        raw = lines / f"line-{index:05d}.wav"
        if not raw.is_file() or index not in by_index:
            continue
        start, end = float(boundary["start"]), float(boundary["end"])
        segment = dict(by_index[index])
        segment.update({"start": start, "end": end})
        fit = align_line(
            raw,
            aligned / raw.name,
            end - start,
            float(job["settings"]["min_tempo"]),
            float(job["settings"]["max_tempo"]),
        )
        timeline.append(segment)
        rendered_boundaries.append({**boundary, "fit": fit})
    if not timeline:
        raise RuntimeError("no reviewed line audio matched the timing boundaries")
    media_root = run_root / "video"
    media_root.mkdir(parents=True, exist_ok=True)
    mixed = media_root / f"{candidate}.wav"
    video = media_root / f"{candidate}.mp4"
    build_mix(
        bed,
        timeline,
        aligned,
        mixed,
        float(job["settings"].get("source_bed_gain", 1.0)),
        float(job["settings"].get("dialogue_gain", 1.0)),
    )
    mux(source, mixed, video)
    mixed.unlink(missing_ok=True)
    return video, rendered_boundaries


def main() -> None:
    parser = argparse.ArgumentParser(description="run AutoDub timing-alignment screen")
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--alignment-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--arm-gpu", action="store_true")
    parser.add_argument("--ack-private-local-material", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not _valid_id(args.run, "adx-"):
        raise SystemExit("invalid opaque experiment run ID")
    run_root = WORK_ROOT / "experiments" / args.run
    run_path = run_root / "RUN.json"
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "timing-alignment-v1":
        raise SystemExit("run is not a timing-alignment experiment")
    job_id = str(manifest.get("job") or "")
    if not _valid_id(job_id, "dub-"):
        raise SystemExit("run has an invalid opaque job ID")
    job = load_job(job_id)
    artifacts = job_dir(job_id) / "artifacts"
    audio_name = job.get("artifacts", {}).get("dialogue_stem") or job.get("artifacts", {}).get("asr_audio")
    audio = artifacts / str(audio_name or "")
    if not audio_name or not audio.is_file():
        raise SystemExit("job has no analyzed local dialogue track")
    if not job.get("segments"):
        raise SystemExit("job has no speech segments")
    known = {item["id"] for item in manifest["candidates"]}
    selected = args.candidates or [item["id"] for item in manifest["candidates"]]
    if set(selected) - known:
        raise SystemExit("candidate is not part of this run")
    readiness = {
        candidate: {
            "implemented": candidate in IMPLEMENTED,
            "requires_gpu": candidate == FORCED and args.alignment_device == "cuda",
            "device": args.alignment_device if candidate == FORCED else "cpu",
            "requires_private_ack": True,
            "playable_export_ready": (artifacts / "lines").is_dir(),
        }
        for candidate in selected
    }
    if args.dry_run:
        print(json.dumps({"ok": True, "run": args.run, "readiness": readiness}, indent=2))
        return
    if not args.ack_private_local_material:
        raise SystemExit("actual timing comparison requires --ack-private-local-material")
    gpu_selected = FORCED in selected and args.alignment_device == "cuda"
    if gpu_selected and not args.arm_gpu:
        raise SystemExit("GPU forced alignment requires --arm-gpu and a clear AutoDub preflight")

    output_root = run_root / "timings"
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}

    def execute(lease=None) -> None:
        for candidate in selected:
            try:
                if candidate == "segment-window-atempo":
                    boundaries = _segment_windows(job["segments"])
                    device = "cpu"
                elif candidate == "whisper-word-boundaries":
                    boundaries = _word_windows(job["segments"])
                    device = "cpu"
                elif candidate == FORCED:
                    if lease is not None and args.alignment_device == "cuda":
                        lease.ensure_active()
                    boundaries = _forced_alignment(audio, job["segments"], args.alignment_device)
                    device = args.alignment_device
                else:
                    raise RuntimeError("candidate adapter is not implemented")
                destination = output_root / f"{candidate}.json"
                video, rendered_boundaries = _render_comparison(job, boundaries, run_root, candidate)
                _atomic_json(
                    destination,
                    {"schema": 1, "candidate": candidate, "device": device, "segments": rendered_boundaries},
                )
                results[candidate] = {
                    "status": "ready-for-review",
                    "segments": len(rendered_boundaries),
                    "evidence": destination.relative_to(run_root).as_posix(),
                    "artifacts": [video.relative_to(run_root).as_posix()],
                }
            except Exception as exc:
                detail = str(exc)
                results[candidate] = {"status": "failed", "error": f"{type(exc).__name__}: {detail[-4000:]}"}

    if gpu_selected:
        with gpu_lease(f"timing-experiment:{args.run}", wait_seconds=7200) as lease:
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
