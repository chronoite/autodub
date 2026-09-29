from __future__ import annotations

import base64
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import traceback

from . import __version__, oplog
from .config import ALLOWED_SUFFIXES, HOST, MAX_UPLOAD_BYTES, OUTPUT_ROOT, PORT, STATIC_ROOT, ensure_layout
from .gpu_session import GpuSafetyError, arm, consume_arm, preflight_status
from .experiment_store import (
    create_run as create_experiment_run,
    list_runs as list_experiment_runs,
    load_run as load_experiment_run,
    public_run as public_experiment_run,
    registry as experiment_registry,
    resolve_artifact,
    update_review,
)
from .episode_queue import clear_finished, enqueue, queue_state, request_stop, run_queue
from .pipeline import analyze, preview_line, realign_existing, remix_existing, render, repair_line
from .policies import get_mix_policy, get_space_policy, get_timing_policy, public_policies
from .policy_experiment import build_policy_experiment
from .quality_profiles import DEFAULT_PROFILE, get_profile, profile_requires_gpu, public_profiles
from . import evidence_prewarm
from .review import episode_video, evidence_audio, evidence_frame, evidence_video, rendered_preview, source_preview, srt_export, voice_reference
from . import characters, series, voice_bank
from .state import (
    delete_job,
    default_job,
    job_dir,
    job_lock,
    list_jobs,
    load_job,
    new_job_id,
    output_path,
    public_job,
    save_job,
    sha256_stream,
)
from .tts_experiment import available_tts_candidates, execute_tts_experiment, plan_tts_experiment
from .voice_profiles import import_profile
from .adaptation_runner import adapt_reviewed
from .casting import (
    acceptance_report,
    apply_cast,
    attribution_flags,
    embedding_disagreements,
    sanitize_studio_progress,
    merge_speakers,
    prerender_demos,
    reject_merge,
)
from .song_detect import mark_song_range
from .workflow import apply_glossary, request_cancel


ACTIVE: dict[str, threading.Thread] = {}
ACTIVE_LOCK = threading.Lock()


def _launch(job_id: str, target) -> bool:
    with ACTIVE_LOCK:
        existing = ACTIVE.get(job_id)
        if existing and existing.is_alive():
            return False

        def wrapped() -> None:
            try:
                target(job_id)
            finally:
                with ACTIVE_LOCK:
                    ACTIVE.pop(job_id, None)

        thread = threading.Thread(target=wrapped, name=f"autodub-{job_id}", daemon=True)
        ACTIVE[job_id] = thread
        thread.start()
        return True


class AutoDubHandler(BaseHTTPRequestHandler):
    server_version = f"AutoDub/{__version__}"

    def log_message(self, fmt: str, *args) -> None:
        # Never log requested URLs or source metadata. Only method/status is operationally useful.
        status = args[1] if len(args) > 1 else "-"
        print(f"[autodub-ui] {self.command} {status}")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        super().end_headers()

    def _json(self, value, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 8 * 1024 * 1024:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        try:
            if parsed.path == "/api/health":
                self._json({
                    "ok": True,
                    "mode": "local-offline-runtime",
                    "host": HOST,
                    "default_profile": DEFAULT_PROFILE,
                    "api_schema": 3,
                })
            elif parsed.path == "/api/logging":
                # Logging state + optional tail (?tail=80&job=<id>): errors are always
                # logged, verbosity is optional.
                query = parse_qs(parsed.query)
                state = oplog.logging_state()
                if "tail" in query:
                    job = str((query.get("job") or [""])[0]) or None
                    state["tail"] = oplog.tail(job, int((query.get("tail") or ["80"])[0]))
                self._json(state)
            elif parsed.path == "/api/profiles":
                self._json({"profiles": public_profiles(), "default": DEFAULT_PROFILE})
            elif parsed.path == "/api/policies":
                self._json(public_policies())
            elif parsed.path == "/api/tts-candidates":
                self._json({"candidates": available_tts_candidates()})
            elif parsed.path == "/api/episode-queue":
                self._json(queue_state())
            elif parsed.path == "/api/gpu/status":
                self._json(preflight_status())
            elif parsed.path == "/api/jobs":
                self._json({"jobs": list_jobs()})
            elif parsed.path == "/api/experiments/registry":
                self._json(experiment_registry())
            elif parsed.path == "/api/experiments":
                self._json({"experiments": list_experiment_runs()})
            elif len(parts) == 3 and parts[:2] == ["api", "experiments"]:
                self._json(public_experiment_run(load_experiment_run(parts[2])))
            elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "artifact":
                query = parse_qs(parsed.query)
                candidate = str((query.get("candidate") or [""])[0])
                relative = str((query.get("path") or [""])[0])
                self._serve_file(resolve_artifact(parts[2], candidate, relative), inline=True)
            elif len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                self._json(public_job(load_job(parts[2])))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "output":
                self._serve_output(parts[2])
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "srt":
                self._serve_file(srt_export(parts[2]), content_type="application/x-subrip")
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "episode-video":
                # full-episode identity scrubbing (lazy one-time remux; Range = seekable)
                self._serve_file(episode_video(parts[2]), content_type="video/mp4", inline=True,
                                 download_name="episode.mp4")
            elif len(parts) == 6 and parts[:2] == ["api", "jobs"] and parts[3] == "segments":
                index = int(parts[4])
                if parts[5] == "source":
                    self._serve_file(source_preview(parts[2], index), content_type="audio/wav", inline=True)
                elif parts[5] == "line":
                    self._serve_file(rendered_preview(parts[2], index), content_type="audio/wav", inline=True)
                elif parts[5] == "evidence-audio":
                    # ?lang=eng swaps in a language-tagged track (an official dub as
                    # an identification aid on dual-audio sources). Bad tags 400 inside.
                    lang = str((parse_qs(parsed.query).get("lang") or [""])[0]) or None
                    self._serve_file(
                        evidence_audio(parts[2], index, lang), content_type="audio/wav", inline=True,
                        download_name=f"segment-audio-{index:05d}.wav",
                    )
                elif parts[5] == "evidence-video":
                    query = parse_qs(parsed.query)
                    lang = str((query.get("lang") or [""])[0]) or None
                    pad = float((query.get("pad") or ["0"])[0] or 0)
                    self._serve_file(
                        evidence_video(parts[2], index, lang, pad), content_type="video/mp4", inline=True,
                        download_name=f"segment-video-{index:05d}.mp4",
                    )
                elif parts[5] == "evidence-frame":
                    self._serve_file(
                        evidence_frame(parts[2], index), content_type="image/jpeg", inline=True,
                        download_name=f"segment-frame-{index:05d}.jpg",
                    )
                else:
                    raise FileNotFoundError("unknown segment media")
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "characters":
                # CHARACTERS: bank matches + evidence cards for review.
                series_q = str((parse_qs(parsed.query).get("series") or [""])[0]) or None
                self._json(characters.characters_view(parts[2], series_q))
            elif parsed.path == "/api/projects":
                # Series view: the show -> seasons -> episodes tree.
                self._json(series.list_projects())
            elif len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "characters":
                self._json(series.series_characters(parts[2]))
            elif len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "prewarm":
                # EVIDENCE PREWARM: progress of the background cutter.
                self._json(evidence_prewarm.status(f"series:{parts[2]}"))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "prewarm-evidence":
                self._json(evidence_prewarm.status(f"job:{parts[2]}"))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "acceptance":
                self._json(acceptance_report(load_job(parts[2])))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "attribution-flags":
                job = load_job(parts[2])
                chunk_count = len((job.get("speaker_evidence") or {})
                                  .get("speaker_chunk_embeddings") or [])
                self._json({
                    "flags": attribution_flags(job),
                    "embedding_disagreements": embedding_disagreements(job),
                    "embedding_coverage": {"checked": chunk_count,
                                           "total": len(job.get("segments") or [])},
                })
            elif len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "demos":
                # Serve a prerendered casting demo clip (opaque generated names only).
                name = parts[4]
                if not (name.startswith("demo-") and name.endswith(".wav")
                        and "/" not in name and "\\" not in name and ".." not in name):
                    raise FileNotFoundError("unknown demo clip")
                path = job_dir(parts[2]) / "artifacts" / "demos" / name
                if not path.is_file():
                    raise FileNotFoundError("demo clip not rendered yet")
                self._serve_file(path, content_type="audio/wav", inline=True)
            elif parsed.path == "/api/voice-library":
                library = voice_bank.load_library()
                self._json({"voices": [{k: v[k] for k in ("id", "name", "origin_series", "created")}
                                       for v in library["voices"]]})
            elif len(parts) == 3 and parts[:2] == ["api", "voice-library"]:
                library = voice_bank.load_library()
                voice = next((v for v in library["voices"] if v["id"] == parts[2]), None)
                if voice is None:
                    raise FileNotFoundError("unknown library voice")
                self._serve_file(Path(voice["clip"]), content_type="audio/wav", inline=True)
            elif len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3:] == ["voices", "audition"]:
                voice = str((parse_qs(parsed.query).get("voice") or [""])[0])
                self._serve_file(voice_reference(parts[2], voice), inline=True)
            else:
                self._serve_static(parsed.path)
        except (FileNotFoundError, ValueError):
            self._error(404, "not found")
        except Exception as exc:
            oplog.server_error("GET", traceback.format_exc())   # full trail, never just the type name
            self._error(500, f"local server error: {type(exc).__name__}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        try:
            if parsed.path == "/api/jobs":
                self._create_job(parse_qs(parsed.query))
            elif parsed.path == "/api/experiments":
                body = self._body_json()
                manifest = create_experiment_run(str(body.get("experiment") or ""), str(body.get("job") or ""))
                self._json(public_experiment_run(manifest), 201)
            elif parsed.path == "/api/episode-queue/add":
                body = self._body_json()
                jobs = body.get("jobs") or []
                if not isinstance(jobs, list) or len(jobs) > 100:
                    raise ValueError("jobs must be a short list")
                self._json(enqueue([str(item) for item in jobs]))
            elif parsed.path == "/api/episode-queue/arm-gpu":
                self._json(arm("episode-queue", "episode-render"))
            elif parsed.path == "/api/episode-queue/run":
                state = queue_state()
                needs_gpu = any(
                    item.get("requires_gpu") and item.get("status") == "queued"
                    for item in state.get("items", [])
                )
                gpu_authorized = (
                    consume_arm("episode-queue", "episode-render") if needs_gpu else False
                )
                if needs_gpu and not gpu_authorized:
                    raise GpuSafetyError("arm the GPU episode queue before running it")
                launched = _launch(
                    "episode-queue",
                    lambda _key: run_queue(gpu_authorized=gpu_authorized),
                )
                if not launched:
                    self._error(409, "the episode queue is already running")
                else:
                    self._json({"accepted": True, "action": "episode-queue"}, 202)
            elif parsed.path == "/api/episode-queue/stop":
                self._json(request_stop(), 202)
            elif parsed.path == "/api/episode-queue/clear-finished":
                self._json(clear_finished())
            elif len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "prewarm":
                # EVIDENCE PREWARM: cut every card's frame/audio/video ahead of the
                # reviewer's clicks, top group first. Idempotent while running.
                self._json(evidence_prewarm.start(parts[2]), 202)
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "prewarm-evidence":
                # Same standard for the per-episode Characters panel.
                self._json(evidence_prewarm.start_job(parts[2]), 202)
            elif len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "characters" and parts[4] == "unite":
                # Roster reassign onto an unnamed candidate: must-link, latest click wins.
                body = self._body_json()
                self._json(series.unite_members(parts[2], member=dict(body.get("member") or {}),
                                                target=dict(body.get("target") or {})))
            elif len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "characters" and parts[4] == "separate":
                # Roster "Not them": cannot-link one member from the rest of its group.
                body = self._body_json()
                self._json(series.separate_member(parts[2], member=dict(body.get("member") or {}),
                                                  others=list(body.get("others") or [])))
            elif len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "characters" and parts[4] == "reject":
                # Roster "Not that character": kills an also-them proposal permanently.
                body = self._body_json()
                self._json(series.reject_candidate(parts[2], character_id=str(body.get("character_id") or ""),
                                                   members=list(body.get("members") or [])))
            elif len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "characters" and parts[4] == "mark-song":
                # "Mark as intro/outro": the group is the opening/ending singer,
                # never dubbed, dropped from both boards. Logged; labels untouched.
                body = self._body_json()
                self._json(series.mark_song_group(parts[2], members=list(body.get("members") or [])))
            elif len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "characters" and parts[4] == "answer":
                body = self._body_json()
                self._json(series.apply_group_answer(
                    parts[2], members=list(body.get("members") or []),
                    answer=str(body.get("answer") or ""),
                    character_id=body.get("character_id") or None,
                    new_name=body.get("new_name") or None))
            elif len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "characters" and parts[4] == "answer":
                body = self._body_json()
                self._json(characters.apply_answer(
                    parts[2], series=body.get("series"), speaker=str(body.get("speaker") or ""),
                    answer=str(body.get("answer") or ""),
                    character_id=body.get("character_id") or None,
                    new_name=body.get("new_name") or None,
                    seed_segment=body.get("seed_segment")))
            elif len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "characters" and parts[4] == "auto-apply":
                series_q = str(self._body_json().get("series") or "") or None
                self._json(characters.auto_apply_clear(parts[2], series_q))
            elif len(parts) == 5 and parts[0] == "api" and parts[1] == "series" and parts[4] == "promote":
                # /api/series/<slug>/characters-by-id promotion uses parts: api series <slug> <char> promote
                self._json(voice_bank.promote_to_library(parts[2], parts[3]))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "voice-profile":
                self._create_voice_profile(parts[2], parse_qs(parsed.query))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "arm-gpu":
                job_id = parts[2]
                load_job(job_id)
                action = str(self._body_json().get("action", ""))
                self._json(arm(job_id, action))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "delete":
                self._json(delete_job(parts[2]))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                load_job(parts[2])
                request_cancel(parts[2])
                self._json({"accepted": True, "action": "cancel"}, 202)
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "apply-cast":
                with ACTIVE_LOCK:
                    thread = ACTIVE.get(parts[2])
                if thread and thread.is_alive():
                    self._error(409, "this job is already running")
                else:
                    body = self._body_json()
                    self._json(apply_cast(
                        parts[2],
                        cast={str(k): dict(v) for k, v in dict(body.get("cast") or {}).items()},
                        series=(str(body.get("series")) if body.get("series") else None),
                    ))
            elif len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "demos" and parts[4] == "prerender":
                job_id = parts[2]
                load_job(job_id)
                # validate EVERYTHING before consuming the arm — a 400/409 after
                # consume_arm would burn the GPU arm for nothing
                body = self._body_json()
                candidates = {str(k): [str(v) for v in vs]
                              for k, vs in dict(body.get("candidates") or {}).items()}
                if not candidates:
                    raise ValueError("no demo candidates supplied")
                force = bool(body.get("force"))
                with ACTIVE_LOCK:
                    running_thread = ACTIVE.get(job_id)
                if running_thread and running_thread.is_alive():
                    self._error(409, "this job is already running")
                    return
                gpu_authorized = consume_arm(job_id, "demo")
                if not gpu_authorized:
                    raise GpuSafetyError("arm the casting demos after the shared queue is available")
                launched = _launch(
                    job_id,
                    lambda current_id: prerender_demos(current_id, candidates, force=force),
                )
                if not launched:
                    self._error(409, "this job is already running")
                else:
                    self._json({"accepted": True, "action": "demo-prerender"}, 202)
            elif len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "speakers" and parts[4] == "reject-merge":
                body = self._body_json()
                self._json(reject_merge(parts[2], a=str(body.get("a", "")),
                                        b=str(body.get("b", ""))))
            elif len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "speakers" and parts[4] == "merge":
                with ACTIVE_LOCK:
                    thread = ACTIVE.get(parts[2])
                if thread and thread.is_alive():
                    self._error(409, "this job is already running")
                else:
                    body = self._body_json()
                    self._json(merge_speakers(
                        parts[2],
                        source=str(body.get("from", "")),
                        target=str(body.get("to", "")),
                    ))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "mark-songs":
                with ACTIVE_LOCK:
                    thread = ACTIVE.get(parts[2])
                if thread and thread.is_alive():
                    self._error(409, "this job is already running")
                else:
                    body = self._body_json()
                    self._json(mark_song_range(
                        parts[2],
                        start=int(body.get("start", -1)),
                        end=int(body.get("end", -1)),
                        mark=bool(body.get("mark", True)),
                    ))
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "policy-experiment":
                job_id = parts[2]
                body = self._body_json()
                manifest = build_policy_experiment(
                    job_id,
                    start=float(body.get("start", 0.0)),
                    duration=float(body.get("duration", 45.0)),
                    line_set=str(body.get("line_set", "lines")),
                )
                self._json(public_experiment_run(manifest), 201)
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "tts-experiment":
                job_id = parts[2]
                body = self._body_json()
                candidates = body.get("candidates") or []
                known = {item["id"]: item for item in available_tts_candidates()}
                needs_gpu = any(known.get(str(item), {}).get("requires_gpu") for item in candidates)
                gpu_authorized = False
                if needs_gpu:
                    gpu_authorized = consume_arm(job_id, "experiment")
                    if not gpu_authorized:
                        raise GpuSafetyError("arm this TTS experiment after the shared queue is available")
                manifest = plan_tts_experiment(
                    job_id,
                    [str(item) for item in candidates],
                    float(body.get("start", 0.0)),
                    float(body.get("duration", 30.0)),
                )
                launched = _launch(
                    f"tts-{manifest['id']}",
                    lambda _key: execute_tts_experiment(
                        manifest["id"], gpu_authorized=gpu_authorized
                    ),
                )
                if not launched:
                    raise RuntimeError("could not start the TTS experiment")
                self._json(public_experiment_run(manifest), 202)
            elif len(parts) == 6 and parts[:2] == ["api", "jobs"] and parts[3] == "segments" and parts[5] in {"preview", "repair"}:
                job_id, index, action = parts[2], int(parts[4]), parts[5]
                job = load_job(job_id)
                segment = next(
                    (item for item in job.get("segments", []) if int(item["i"]) == index),
                    None,
                )
                if segment is None:
                    raise ValueError("unknown segment")
                speaker = segment.get("speaker", "speaker-01")
                voice = job.get("speaker_voices", {}).get(speaker, f"qwen-auto:{speaker}")
                gpu_authorized = False
                if voice.startswith("qwen"):
                    gpu_authorized = consume_arm(job_id, f"{action}-line")
                    if not gpu_authorized:
                        raise GpuSafetyError(f"arm this GPU {action} after the render queue is idle")
                target = preview_line if action == "preview" else repair_line
                launched = _launch(
                    job_id,
                    lambda current_id: target(current_id, index, gpu_authorized=gpu_authorized),
                )
                if not launched:
                    self._error(409, "this job is already running")
                else:
                    self._json({"accepted": True, "action": action, "segment": index}, 202)
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "remix":
                job_id = parts[2]
                launched = _launch(job_id, lambda current_id: remix_existing(current_id, label="remix"))
                if not launched:
                    self._error(409, "this job is already running")
                else:
                    self._json({"accepted": True, "action": "remix"}, 202)
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "realign":
                job_id = parts[2]
                load_job(job_id)
                gpu_authorized = consume_arm(job_id, "realign")
                if not gpu_authorized:
                    raise GpuSafetyError("arm source realignment after the shared queue is available")
                launched = _launch(
                    job_id,
                    lambda current_id: realign_existing(
                        current_id, gpu_authorized=gpu_authorized
                    ),
                )
                if not launched:
                    self._error(409, "this job is already running")
                else:
                    self._json({"accepted": True, "action": "realign"}, 202)
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "adapt":
                job_id = parts[2]
                job = load_job(job_id)
                if job.get("status") != "review":
                    raise ValueError("dialogue adaptation runs on a reviewed job")
                body = self._body_json()
                capture = bool(body.get("capture", True))
                engines = [str(e) for e in (body.get("engines") or [])] or None
                picks = {str(k): str(v)
                         for k, v in dict(body.get("picks") or {}).items()} or None
                gpu_authorized = False
                if capture:
                    gpu_authorized = consume_arm(job_id, "adapt")
                    if not gpu_authorized:
                        raise GpuSafetyError("arm dialogue adaptation after the shared queue is available")
                launched = _launch(
                    job_id,
                    lambda current_id: adapt_reviewed(
                        current_id, gpu_authorized=gpu_authorized,
                        engines=engines, picks=picks, capture=capture,
                    ),
                )
                if not launched:
                    self._error(409, "this job is already running")
                else:
                    self._json({"accepted": True, "action": "adapt"}, 202)
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] in {"analyze", "render"}:
                job_id, action = parts[2], parts[3]
                job = load_job(job_id)
                profile = job.get("settings", {}).get("quality_profile", "prototype-cpu-v1")
                gpu_authorized = False
                if profile_requires_gpu(profile):
                    gpu_authorized = consume_arm(job_id, action)
                    if not gpu_authorized:
                        raise GpuSafetyError("arm this quality GPU action after the render queue is idle")
                function = analyze if action == "analyze" else render
                launched = _launch(job_id, lambda current_id: function(current_id, gpu_authorized=gpu_authorized))
                if not launched:
                    self._error(409, "this job is already running")
                else:
                    self._json({"accepted": True, "action": action}, 202)
            else:
                self._error(404, "not found")
        except (FileNotFoundError, ValueError, GpuSafetyError) as exc:
            self._error(400, str(exc))
        except Exception as exc:
            oplog.server_error("POST", traceback.format_exc())
            self._error(500, f"request failed: {type(exc).__name__}")

    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.path == "/api/logging":
            try:
                self._json(oplog.set_verbose(bool(self._body_json().get("verbose"))))
            except Exception as exc:
                oplog.server_error("PATCH", traceback.format_exc())
                self._error(500, f"logging toggle failed: {type(exc).__name__}")
            return
        if len(parts) == 3 and parts[:2] == ["api", "experiments"]:
            try:
                self._json(update_review(parts[2], self._body_json()))
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                self._error(400, str(exc))
            except Exception as exc:
                oplog.server_error("PATCH", traceback.format_exc())
                self._error(500, f"review update failed: {type(exc).__name__}")
            return
        if len(parts) != 3 or parts[:2] != ["api", "jobs"]:
            self._error(404, "not found")
            return
        try:
            update = self._body_json()
            # job_lock serializes this whole load→mutate→save against the casting
            # verbs and the demo prerender's manifest writes (prevents lost updates);
            # the running-status gate still excludes the
            # long-lived analyze/render workers, which keep their own dict in memory.
            with job_lock(parts[2]):
                job = load_job(parts[2])
                if job.get("status") == "running":
                    self._error(409, "wait for the current stage to finish")
                    return
                allowed_settings = {
                    "source_language", "target_language", "speaker_count", "source_bed_gain",
                    "dialogue_gain", "max_tempo", "min_tempo", "quality_profile", "mix_policy",
                    "timing_policy", "space_policy",
                    "emotion_policy", "song_policy",
                }
                for key, value in update.get("settings", {}).items():
                    if key in allowed_settings:
                        if key == "speaker_count":
                            from .speaker_evidence import normalize_speaker_count
                            job["settings"][key] = normalize_speaker_count(value)
                        elif key == "quality_profile":
                            job["settings"][key] = str(value)
                            job["settings"]["quality_stack"] = get_profile(str(value))
                        elif key == "mix_policy":
                            policy = get_mix_policy(str(value))
                            job["settings"][key] = str(value)
                            if update.get("apply_policy_defaults"):
                                job["settings"]["source_bed_gain"] = policy["bed_gain"]
                                job["settings"]["dialogue_gain"] = policy["dialogue_gain"]
                        elif key == "timing_policy":
                            policy = get_timing_policy(str(value))
                            job["settings"][key] = str(value)
                            if update.get("apply_policy_defaults"):
                                job["settings"]["min_tempo"] = policy["min_tempo"]
                                job["settings"]["max_tempo"] = policy["max_tempo"]
                        elif key == "space_policy":
                            get_space_policy(str(value))
                            job["settings"][key] = str(value)
                        elif key == "emotion_policy":
                            if str(value) not in {"off", "source-energy-v1"}:
                                raise ValueError("unknown emotion policy")
                            job["settings"][key] = str(value)
                        elif key == "song_policy":
                            if str(value) not in {"dub-all-v1", "skip-detected-v1"}:
                                raise ValueError("unknown song policy")
                            job["settings"][key] = str(value)
                        else:
                            job["settings"][key] = value
                by_index = {int(item["i"]): item for item in job.get("segments", [])}
                reassigned: list[int] = []
                for changed in update.get("segments", []):
                    item = by_index.get(int(changed.get("i", -1)))
                    if item is not None:
                        if "speaker" in changed and str(changed["speaker"]) != str(item.get("speaker")):
                            reassigned.append(int(item["i"]))
                        for key in ("speaker", "translation"):
                            if key in changed:
                                item[key] = str(changed[key])[:4000]
                if reassigned:
                    # a reassigned line's cached wav is the OLD speaker's voice — the
                    # same stale-voice disease merge/apply-cast purge against
                    # (remix has no signature check)
                    from .workflow import load_line_manifest, save_line_manifest
                    artifacts = job_dir(parts[2]) / "artifacts"
                    manifest = load_line_manifest(artifacts / "lines")
                    for index in reassigned:
                        for stale in (artifacts / "lines" / f"line-{index:05d}.wav",
                                      artifacts / "aligned" / f"line-{index:05d}.wav"):
                            stale.unlink(missing_ok=True)
                        manifest.pop(str(index), None)
                    save_line_manifest(artifacts / "lines", manifest)
                valid_voices = set(job.get("available_voices", [])) | {"system-default"}
                for speaker, voice in update.get("speaker_voices", {}).items():
                    if voice in valid_voices:
                        job["speaker_voices"][str(speaker)[:80]] = voice
                if "glossary" in update:
                    raw_glossary = update["glossary"]
                    if not isinstance(raw_glossary, dict) or len(raw_glossary) > 500:
                        raise ValueError("glossary must be an object with at most 500 entries")
                    job["glossary"] = {
                        str(key)[:200]: str(value)[:400]
                        for key, value in raw_glossary.items() if str(key).strip()
                    }
                if update.get("apply_glossary"):
                    for item in job.get("segments", []):
                        item["translation"] = apply_glossary(
                            str(item.get("translation") or ""), job.get("glossary", {})
                        )
                if "studio_progress" in update:
                    # studio v2 checklist (step + clean groups) — server-side so the
                    # reviewer's progress survives a closed tab (never browser storage)
                    job["studio_progress"] = sanitize_studio_progress(update["studio_progress"])
                save_job(job)
                self._json(public_job(job))
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            self._error(400, str(exc))
        except Exception as exc:
            oplog.server_error("PATCH", traceback.format_exc())
            self._error(500, f"update failed: {type(exc).__name__}")

    def _create_job(self, query: dict[str, list[str]]) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError("invalid source size")
        suffix = (query.get("ext") or [""])[0].lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError("unsupported video type")
        job_id = new_job_id()
        root = job_dir(job_id)
        root.mkdir(parents=True, exist_ok=False)
        source = root / f"source{suffix}"
        written, digest = sha256_stream(self.rfile, source, length)
        if written != length:
            source.unlink(missing_ok=True)
            root.rmdir()
            raise ValueError("source transfer ended early")
        job = default_job(job_id, suffix, written, digest)
        # Original name: the upload is stored as source<ext>, which erased the
        # real filename - and the Characters panel guesses the SERIES from it. Additive field;
        # basename only, display/guess use only (never a path).
        original = Path(str((query.get("name") or [""])[0])).name[:200]
        if original:
            job["source"]["original_name"] = original
        save_job(job)
        self._json(public_job(job), 201)

    def _create_voice_profile(self, job_id: str, query: dict[str, list[str]]) -> None:
        job = load_job(job_id)
        if job.get("status") == "running":
            raise ValueError("wait for the current stage to finish")
        length = int(self.headers.get("Content-Length", "0"))
        suffix = (query.get("ext") or [""])[0].lower()
        encoded = self.headers.get("X-AutoDub-Transcript", "")
        try:
            transcript = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("invalid reference transcript encoding") from exc
        language = self.headers.get("X-AutoDub-Language", job["settings"].get("source_language", "ja"))
        profile = import_profile(self.rfile, length, suffix, transcript, language)
        if profile["option"] not in job["available_voices"]:
            job["available_voices"].append(profile["option"])
        job.setdefault("voice_labels", {})[profile["option"]] = profile["label"]
        save_job(job)
        self._json(public_job(job), 201)

    def _serve_output(self, job_id: str) -> None:
        job = load_job(job_id)
        name = str(job.get("artifacts", {}).get("output") or output_path(job_id).name)
        if Path(name).name != name:
            raise ValueError("invalid output")
        path = OUTPUT_ROOT / name
        if not path.exists():
            raise FileNotFoundError(path)
        self._serve_file(path, content_type="video/mp4")

    def _serve_file(
        self, path, *, content_type: str | None = None, inline: bool = False,
        download_name: str | None = None,
    ) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        status = 200
        requested = self.headers.get("Range", "")
        if requested.startswith("bytes=") and size:
            value = requested.removeprefix("bytes=").split(",", 1)[0].strip()
            first, _, last = value.partition("-")
            try:
                if first:
                    start = int(first)
                    end = int(last) if last else end
                elif last:
                    length = int(last)
                    start = max(0, size - length)
                if start < 0 or start >= size or end < start:
                    raise ValueError
                end = min(end, size - 1)
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        length = 0 if size == 0 else max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        safe_name = (download_name or path.name).replace('"', "_").replace("\r", "_").replace("\n", "_")
        self.send_header("Content-Disposition", f'{"inline" if inline else "attachment"}; filename="{safe_name}"')
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining and (chunk := handle.read(min(1024 * 1024, remaining))):
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _serve_static(self, requested: str) -> None:
        name = "index.html" if requested in {"", "/"} else requested.lstrip("/")
        candidate = (STATIC_ROOT / name).resolve()
        if STATIC_ROOT.resolve() not in candidate.parents and candidate != STATIC_ROOT.resolve():
            raise FileNotFoundError(candidate)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = HOST, port: int = PORT) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("AutoDub refuses non-loopback binds")
    ensure_layout()
    pruned = oplog.prune_job_logs()   # rotation: keep job logs only for the newest jobs
    if pruned:
        print(f"log rotation: removed {pruned} old job log(s)")
    oplog.server_info("startup", f"server code SHA {oplog.code_sha()}")
    httpd = ThreadingHTTPServer((host, port), AutoDubHandler)
    print(f"AutoDub local studio: http://{host}:{port} (code {oplog.code_sha()})")
    print("Runtime policy: loopback-only; model workers forced offline; no telemetry or cloud fallback.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
