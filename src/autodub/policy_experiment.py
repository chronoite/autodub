"""CPU-only real-job timing/balance comparison builder for the experiment review page."""
from __future__ import annotations

import shutil
from pathlib import Path

from .experiment_store import create_run, run_dir, save_run
from .media import align_line, build_mix, extract_audio_clip, mux_clip, probe_duration
from .policies import get_mix_policy, get_space_policy, get_timing_policy
from .state import job_dir, load_job
from .workflow import audit_timing


def _safe_line_set(value: str) -> str:
    if value not in {"lines"}:
        raise ValueError("unsupported line set")
    return value


def build_policy_experiment(
    job_id: str,
    *,
    start: float = 0.0,
    duration: float = 45.0,
    line_set: str = "lines",
    experiment_id: str = "dub-policy-v1",
) -> dict:
    job = load_job(job_id)
    if job.get("status") == "running":
        raise ValueError("wait for the current job action to finish")
    root = job_dir(job_id)
    artifacts = root / "artifacts"
    source = root / job["source"]["file"]
    bed_name = job.get("artifacts", {}).get("source_bed") or job.get("artifacts", {}).get("full_audio")
    if not bed_name:
        raise ValueError("analyze the job before building a policy experiment")
    lines = artifacts / _safe_line_set(line_set)
    if not lines.is_dir():
        raise ValueError("the selected rendered line set is missing")
    total = probe_duration(source)
    start = max(0.0, min(float(start), max(0.0, total - 0.1)))
    duration = max(5.0, min(float(duration), 120.0, total - start))
    end = start + duration
    selected = [
        item for item in job.get("segments", [])
        if float(item["end"]) > start and float(item["start"]) < end
        and (lines / f"line-{int(item['i']):05d}.wav").is_file()
    ]
    if not selected:
        raise ValueError("the selected range contains no rendered dialogue")

    manifest = create_run(experiment_id, job_id)
    manifest["status"] = "running"
    manifest["range"] = {"start": round(start, 3), "duration": round(duration, 3)}
    manifest["line_set"] = line_set
    save_run(manifest)
    run_root = run_dir(manifest["id"])
    bed_clip = run_root / "bed.wav"
    extract_audio_clip(artifacts / bed_name, bed_clip, start, duration)

    adjusted = []
    for item in selected:
        copy = dict(item)
        copy["start"] = max(0.0, float(item["start"]) - start)
        copy["end"] = min(duration, float(item["end"]) - start)
        adjusted.append(copy)

    timing_cache: dict[str, tuple[Path, list[dict], dict]] = {}
    for candidate in manifest["candidates"]:
        timing_id = candidate["timing_policy"]
        if timing_id not in timing_cache:
            timing = get_timing_policy(timing_id)
            aligned = run_root / f"aligned-{timing_id}"
            aligned.mkdir()
            timed = []
            for item in adjusted:
                copy = dict(item)
                source_line = lines / f"line-{int(item['i']):05d}.wav"
                target = aligned / source_line.name
                copy["alignment"] = align_line(
                    source_line,
                    target,
                    float(copy["end"]) - float(copy["start"]),
                    float(timing["min_tempo"]),
                    float(timing["max_tempo"]),
                    fit_mode=str(timing["fit_mode"]),
                )
                timed.append(copy)
            qc = audit_timing(timed, aligned)
            timing_cache[timing_id] = (aligned, timed, qc)

        aligned, timed, qc = timing_cache[timing_id]
        mix = get_mix_policy(candidate["mix_policy"])
        mix.update(get_space_policy(candidate.get("space_policy", "dry-v1")))
        mix_path = run_root / f"{candidate['id']}.wav"
        video_path = run_root / f"{candidate['id']}.mp4"
        build_mix(
            bed_clip,
            timed,
            aligned,
            mix_path,
            float(mix["bed_gain"]),
            float(mix["dialogue_gain"]),
            policy=mix,
        )
        mux_clip(source, mix_path, video_path, start, duration)
        candidate["result"] = {
            "artifacts": [video_path.name],
            "mix_policy": candidate["mix_policy"],
            "timing_policy": timing_id,
            "space_policy": candidate.get("space_policy", "dry-v1"),
            "qc": qc,
        }
        mix_path.unlink(missing_ok=True)

    bed_clip.unlink(missing_ok=True)
    for aligned, _, _ in timing_cache.values():
        shutil.rmtree(aligned, ignore_errors=True)
    manifest["status"] = "ready"
    save_run(manifest)
    return manifest
