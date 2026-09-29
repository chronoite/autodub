"""Build short, human-judged TTS comparisons from an existing reviewed job.

The runner reuses the same reviewed English lines, speaker map, timing, and mix policy for every
candidate. It writes opaque local artifacts into the normal experiment store for the experiment review page.
"""
from __future__ import annotations

from pathlib import Path

from .experiments.voice_screen import GPU_CANDIDATES, IMPLEMENTED, execute_candidate

from .delivery import chatterbox_controls
from .experiment_store import create_run, load_run, registry, run_dir, save_run
from .gpu_session import gpu_lease
from .media import align_line, build_mix, extract_audio_clip, mux_clip
from .policies import get_mix_policy, get_space_policy, get_timing_policy
from .state import job_dir, load_job


def available_tts_candidates() -> list[dict]:
    manifest = next(
        item
        for item in registry()["experiments"]
        if item["id"] == "voice-quality-v1"
    )
    return [
        {
            "id": item["id"],
            "device": item["device"],
            "status": item.get("status"),
            "implemented": item["id"] in IMPLEMENTED,
            "requires_gpu": item["id"] in GPU_CANDIDATES,
        }
        for item in manifest["candidates"]
    ]


def plan_tts_experiment(job_id: str, candidates: list[str], start: float, duration: float) -> dict:
    job = load_job(job_id)
    if not job.get("segments"):
        raise ValueError("analyze and review the job before building a TTS experiment")
    known = {item["id"] for item in available_tts_candidates() if item["implemented"]}
    selected = list(dict.fromkeys(str(item) for item in candidates))
    if not selected or set(selected) - known:
        raise ValueError("select one or more implemented TTS candidates")
    run = create_run("voice-quality-v1", job_id)
    run["status"] = "queued"
    run["selection"] = {
        "start": round(max(0.0, float(start)), 3),
        "duration": round(min(120.0, max(5.0, float(duration))), 3),
        "candidates": selected,
    }
    for candidate in run["candidates"]:
        if candidate["id"] not in selected:
            candidate["result"] = {"status": "not-selected"}
    save_run(run)
    return run


def _selected_segments(job: dict, start: float, duration: float) -> list[dict]:
    end = start + duration
    selected = [
        dict(item)
        for item in job.get("segments", [])
        if str(item.get("translation") or "").strip()
        and float(item["end"]) > start
        and float(item["start"]) < end
    ]
    if not selected:
        raise ValueError("the selected range contains no reviewed English lines")
    return selected


def _candidate_lines(
    candidate: str,
    job: dict,
    segments: list[dict],
    raw_dir: Path,
) -> None:
    by_speaker: dict[str, list[dict]] = {}
    for segment in segments:
        by_speaker.setdefault(str(segment.get("speaker") or "speaker-01"), []).append(segment)
    references = job.get("speaker_references", {})
    artifacts = job_dir(job["id"]) / "artifacts"
    for speaker, speaker_segments in by_speaker.items():
        reference = references.get(speaker)
        if candidate != "windows-sapi" and not reference:
            raise RuntimeError(f"the selected range lacks an automatic reference for {speaker}")
        reference_audio = artifacts / reference["file"] if reference else artifacts / "full.wav"
        lines = [
            {
                "id": f"line-{int(item['i']):05d}",
                "text": str(item["translation"]),
                **chatterbox_controls(item),
            }
            for item in speaker_segments
        ]
        execute_candidate(
            candidate,
            lines,
            reference_audio,
            str((reference or {}).get("text") or ""),
            str((reference or {}).get("language") or "ja"),
            raw_dir,
        )


def _render_candidate(
    candidate: str,
    job: dict,
    segments: list[dict],
    root: Path,
    start: float,
    duration: float,
) -> Path:
    job_root = job_dir(job["id"])
    artifacts = job_root / "artifacts"
    source = job_root / job["source"]["file"]
    bed_name = job.get("artifacts", {}).get("source_bed") or job.get("artifacts", {}).get("full_audio")
    if not bed_name:
        raise RuntimeError("the analyzed source bed is missing")
    bed_source = artifacts / bed_name
    if not source.is_file() or not bed_source.is_file():
        raise RuntimeError("the source video or analyzed bed is missing")

    raw = root / "audio" / candidate
    aligned = root / "aligned" / candidate
    raw.mkdir(parents=True, exist_ok=True)
    aligned.mkdir(parents=True, exist_ok=True)
    _candidate_lines(candidate, job, segments, raw)

    timing = get_timing_policy(job["settings"].get("timing_policy", "gentle-fit-v1"))
    timeline = []
    for item in segments:
        line = raw / f"line-{int(item['i']):05d}.wav"
        if not line.is_file():
            continue
        relative = dict(item)
        relative["start"] = max(0.0, float(item["start"]) - start)
        relative["end"] = min(duration, float(item["end"]) - start)
        if relative["end"] <= relative["start"]:
            continue
        align_line(
            line,
            aligned / line.name,
            float(relative["end"]) - float(relative["start"]),
            float(timing["min_tempo"]),
            float(timing["max_tempo"]),
            fit_mode=str(timing["fit_mode"]),
        )
        timeline.append(relative)
    if not timeline:
        raise RuntimeError("candidate produced no alignable line audio")

    media = root / "video"
    media.mkdir(parents=True, exist_ok=True)
    bed_clip = media / f"{candidate}-bed.wav"
    mix = media / f"{candidate}-mix.wav"
    video = media / f"{candidate}.mp4"
    extract_audio_clip(bed_source, bed_clip, start, duration)
    mix_policy = get_mix_policy(job["settings"].get("mix_policy", "balanced-v1"))
    mix_policy.update(get_space_policy(job["settings"].get("space_policy", "dry-v1")))
    build_mix(
        bed_clip,
        timeline,
        aligned,
        mix,
        float(job["settings"].get("source_bed_gain", mix_policy["bed_gain"])),
        float(job["settings"].get("dialogue_gain", mix_policy["dialogue_gain"])),
        policy=mix_policy,
    )
    mux_clip(source, mix, video, start, duration)
    bed_clip.unlink(missing_ok=True)
    mix.unlink(missing_ok=True)
    return video


def execute_tts_experiment(run_id: str, *, gpu_authorized: bool = False) -> None:
    manifest = load_run(run_id)
    job = load_job(manifest["job"])
    selected = list(manifest["selection"]["candidates"])
    start = float(manifest["selection"]["start"])
    duration = float(manifest["selection"]["duration"])
    segments = _selected_segments(job, start, duration)
    root = run_dir(run_id)
    manifest["status"] = "running"
    manifest["selection"]["segments"] = [int(item["i"]) for item in segments]
    save_run(manifest)

    needs_gpu = any(candidate in GPU_CANDIDATES for candidate in selected)
    if needs_gpu and not gpu_authorized:
        raise ValueError("TTS GPU experiment was not explicitly armed")

    def execute(lease=None) -> None:
        for candidate_id in selected:
            candidate = next(item for item in manifest["candidates"] if item["id"] == candidate_id)
            try:
                if lease is not None and candidate_id in GPU_CANDIDATES:
                    lease.ensure_active()
                video = _render_candidate(candidate_id, job, segments, root, start, duration)
                candidate["result"] = {
                    "status": "ready-for-review",
                    "artifacts": [video.relative_to(root).as_posix()],
                    "lines": len(segments),
                }
            except Exception as exc:
                candidate["result"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[-2000:]}",
                }
            save_run(manifest)

    try:
        if needs_gpu:
            with gpu_lease(f"tts-experiment:{run_id}", wait_seconds=10800) as lease:
                execute(lease)
        else:
            execute()
        manifest["status"] = (
            "awaiting-review"
            if any(
                isinstance(item.get("result"), dict)
                and item["result"].get("status") == "ready-for-review"
                for item in manifest["candidates"]
            )
            else "failed"
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {str(exc)[-2000:]}"
    finally:
        save_run(manifest)
