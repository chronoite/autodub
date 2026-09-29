from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from autodub.config import FFMPEG, runtime_env, WORK_ROOT
from autodub.gpu_session import (
    gpu_lease,
    GpuLease,
    GpuSafetyError,
    HEARTBEAT_FAILURE_LIMIT,
    LEASE_TTL_SECONDS,
    preflight_status,
)
from autodub.quality_profiles import CPU_PROFILE, DEFAULT_PROFILE, get_profile
from autodub.state import default_job, job_dir, save_job

from tests import ROOT


class QualityContractTests(unittest.TestCase):
    def test_quality_worker_reads_pipeline_pcm_without_soundfile(self) -> None:
        from autodub.workers.quality_worker import _read_pcm_wav

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(array("h", [0, 16384, -16384]).tobytes())
            samples, sample_rate = _read_pcm_wav(str(path))
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(samples.shape, (3, 1))
        self.assertAlmostEqual(float(samples[1, 0]), 0.5, places=4)

    def test_quality_worker_keeps_library_chatter_out_of_json_channel(self) -> None:
        from autodub.workers import quality_worker

        def noisy_worker(_payload):
            print("third-party startup banner")
            return {"ok": True}

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(quality_worker, "synthesize", noisy_worker),
            patch.object(quality_worker.sys, "argv", ["quality_worker.py", "synthesize"]),
            patch.object(quality_worker.sys, "stdin", io.StringIO("{}")),
            patch.object(quality_worker.sys, "stdout", stdout),
            patch.object(quality_worker.sys, "stderr", stderr),
        ):
            quality_worker.main()
        self.assertEqual(json.loads(stdout.getvalue()), {"ok": True})
        self.assertIn("third-party startup banner", stderr.getvalue())

    def test_new_jobs_record_quality_first_profile(self) -> None:
        job = default_job("dub-20260716-000000-abcd", ".mp4", 1, "0" * 64)
        self.assertEqual(job["schema"], 3)
        self.assertEqual(job["settings"]["quality_profile"], DEFAULT_PROFILE)
        self.assertEqual(job["settings"]["mix_policy"], "balanced-v1")
        self.assertEqual(job["settings"]["timing_policy"], "gentle-fit-v1")
        self.assertTrue(job["settings"]["quality_stack"]["requires_gpu"])
        self.assertFalse(get_profile(CPU_PROFILE)["requires_gpu"])

    def test_runtime_is_offline_and_gpu_is_opt_in(self) -> None:
        deps = Path(tempfile.mkdtemp(prefix="autodub-runtime-deps-"))
        self.addCleanup(shutil.rmtree, deps, ignore_errors=True)
        with patch("autodub.config.RUNTIME_DEPS", deps):
            cpu = runtime_env()
            gpu = runtime_env(gpu=True)
            isolated = runtime_env(gpu=True, portable_deps=False)
        self.assertEqual(cpu["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertNotEqual(gpu.get("CUDA_VISIBLE_DEVICES"), "-1")
        self.assertIn(str(deps), cpu.get("PYTHONPATH", ""))
        self.assertNotIn(str(deps), isolated.get("PYTHONPATH", ""))
        if FFMPEG.parent != Path("."):
            self.assertEqual(isolated["PATH"].split(os.pathsep)[0], str(FFMPEG.parent))
        for env in (cpu, gpu):
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(env["HF_HUB_DISABLE_TELEMETRY"], "1")

    def test_preflight_blocks_render_and_fails_closed(self) -> None:
        idle = preflight_status(port_probe=lambda: [], broker_probe=lambda: {"configured": False})
        self.assertTrue(idle["safe"])
        self.assertEqual("local", idle["coordinator"])
        busy = preflight_status(port_probe=lambda: [5001], broker_probe=lambda: {"configured": False})
        self.assertFalse(busy["safe"])
        self.assertIn("5001", busy["reasons"][0])
        broker_down = preflight_status(
            port_probe=lambda: [], broker_probe=lambda: {"configured": True, "available": False}
        )
        self.assertFalse(broker_down["safe"])
        failed = preflight_status(
            port_probe=lambda: (_ for _ in ()).throw(RuntimeError("probe failed")),
            broker_probe=lambda: {"configured": False},
        )
        self.assertFalse(failed["safe"])
        self.assertIn("failed closed", failed["reasons"][0])

    def test_local_lease_is_exclusive_and_released_on_failure(self) -> None:
        safe = lambda: {"safe": True, "reasons": []}  # noqa: E731
        with self.assertRaisesRegex(RuntimeError, "worker"):
            with gpu_lease("first", preflight=safe) as lease:
                lease.ensure_active()
                with self.assertRaises(GpuSafetyError):
                    with gpu_lease("second", preflight=safe, wait_seconds=1):
                        pass
                raise RuntimeError("worker failure")
        with gpu_lease("after", preflight=safe, wait_seconds=1):
            pass

    def test_gpu_lease_releases_after_worker_failure(self) -> None:
        calls = []

        def post(path, payload):
            calls.append((path, payload))
            if path.endswith("request"):
                return {
                    "ok": True,
                    "state": "granted",
                    "request_id": "request-test",
                    "lease_id": "lease-test",
                }
            return {"ok": True}

        with self.assertRaisesRegex(RuntimeError, "worker"):
            with gpu_lease("test", post=post, preflight=lambda: {"safe": True, "reasons": []}):
                raise RuntimeError("worker failure")
        self.assertEqual([item[0] for item in calls], ["/queue/request", "/queue/release"])
        self.assertEqual(calls[-1][1]["lease_id"], "lease-test")
        self.assertEqual(LEASE_TTL_SECONDS, calls[0][1]["ttl_seconds"])
        self.assertEqual(21600, calls[0][1]["queue_ttl_seconds"])

    def test_gpu_lease_heartbeat_tolerates_transient_loss_then_fails_closed(self) -> None:
        outcomes = [OSError("transient"), {"ok": True, "expires_at": 1234.0}]

        def post(_path, _payload):
            outcome = outcomes.pop(0) if outcomes else OSError("lost")
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        lease = GpuLease({"lease_id": "lease-test"}, sender=post)
        self.assertFalse(lease._beat_once())
        self.assertEqual(1, lease.consecutive_heartbeat_failures)
        lease.ensure_active()
        self.assertTrue(lease._beat_once())
        self.assertEqual(0, lease.consecutive_heartbeat_failures)
        self.assertEqual(1234.0, lease["expires_at"])
        for _ in range(HEARTBEAT_FAILURE_LIMIT):
            self.assertFalse(lease._beat_once())
        with self.assertRaises(GpuSafetyError):
            lease.ensure_active()

    def test_experiment_registry_keeps_cpu_quality_candidates(self) -> None:
        registry = json.loads((ROOT / "src" / "autodub" / "experiments" / "registry.json").read_text(encoding="utf-8"))
        self.assertEqual(len(registry["experiments"]), 10)
        self.assertIn("dub-policy-v1", {item["id"] for item in registry["experiments"]})
        # balance bench: mix ladder only — timing/space held constant in every candidate
        balance = next(item for item in registry["experiments"] if item["id"] == "dub-mix-balance-v1")
        self.assertEqual(6, len(balance["candidates"]))
        self.assertEqual({"gentle-fit-v1"}, {c["timing_policy"] for c in balance["candidates"]})
        self.assertEqual({"dry-v1"}, {c["space_policy"] for c in balance["candidates"]})
        self.assertEqual(6, len({c["mix_policy"] for c in balance["candidates"]}))
        # dialogue lot (the first bench landed on a music-heavy window).
        # Three sections, each on a window chosen from LINE TIMINGS ONLY - never media content.
        by_id = {item["id"]: item for item in registry["experiments"]}
        dialogue = by_id["dub-dialogue-mix-v1"]
        self.assertEqual(6, len(dialogue["candidates"]))
        self.assertEqual({"gentle-fit-v1"}, {c["timing_policy"] for c in dialogue["candidates"]})
        duck = by_id["dub-duck-v1"]
        self.assertEqual(4, len(duck["candidates"]))
        self.assertEqual({"gentle-fit-v1"}, {c["timing_policy"] for c in duck["candidates"]})
        # the timing section is the mirror image: timing VARIES, mix is held constant
        timing = by_id["dub-timing-v1"]
        self.assertEqual(3, len(timing["candidates"]))
        self.assertEqual(3, len({c["timing_policy"] for c in timing["candidates"]}))
        self.assertEqual(1, len({c["mix_policy"] for c in timing["candidates"]}))
        for experiment in registry["experiments"]:
            devices = {candidate["device"] for candidate in experiment["candidates"]}
            self.assertTrue(any("cpu" in device for device in devices), experiment["id"])
            self.assertTrue(experiment["criteria"])

    def test_voice_experiment_dry_run_is_opaque_and_model_free(self) -> None:
        job_id = "dub-20260716-010101-abcd"
        root = job_dir(job_id)
        run_root = None
        try:
            reference = root / "artifacts" / "voice-references" / "speaker-01.wav"
            reference.parent.mkdir(parents=True)
            reference.write_bytes(b"synthetic-placeholder")
            job = default_job(job_id, ".mp4", 1, "0" * 64)
            job["speaker_references"] = {
                "speaker-01": {
                    "file": "voice-references/speaker-01.wav",
                    "text": "Synthetic test reference.",
                    "language": "en",
                }
            }
            job["artifacts"]["asr_audio"] = "voice-references/speaker-01.wav"
            job["artifacts"]["full_audio"] = "voice-references/speaker-01.wav"
            job["segments"] = [{"i": 0, "start": 0.0, "end": 1.0, "text": "Synthetic test reference."}]
            save_job(job)
            created = subprocess.run(
                [
                    sys.executable,
                    "-m", "autodub.experiments.create_run",
                    "--experiment",
                    "voice-quality-v1",
                    "--job",
                    job_id,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            run_id = json.loads(created.stdout)["run"]
            run_root = WORK_ROOT / "experiments" / run_id
            dry = subprocess.run(
                [
                    sys.executable,
                    "-m", "autodub.experiments.voice_screen",
                    "--run",
                    run_id,
                    "--speaker",
                    "speaker-01",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(dry.stdout)
            self.assertTrue(payload["ok"])
            manifest_text = (run_root / "RUN.json").read_text(encoding="utf-8")
            self.assertNotIn("Synthetic test reference", manifest_text)
            self.assertNotIn("speaker-01.wav", manifest_text)

            speaker_created = subprocess.run(
                [
                    sys.executable,
                    "-m", "autodub.experiments.create_run",
                    "--experiment",
                    "speaker-detection-v1",
                    "--job",
                    job_id,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            speaker_run_id = json.loads(speaker_created.stdout)["run"]
            speaker_run_root = WORK_ROOT / "experiments" / speaker_run_id
            try:
                speaker_dry = subprocess.run(
                    [
                        sys.executable,
                        "-m", "autodub.experiments.speaker_screen",
                        "--run",
                        speaker_run_id,
                        "--dry-run",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertTrue(json.loads(speaker_dry.stdout)["ok"])
            finally:
                shutil.rmtree(speaker_run_root, ignore_errors=True)

            for experiment, executor in (
                ("timing-alignment-v1", "timing_screen"),
                ("dialogue-separation-v1", "separation_screen"),
                ("translation-context-v1", "translation_screen"),
            ):
                created = subprocess.run(
                    [
                        sys.executable,
                        "-m", "autodub.experiments.create_run",
                        "--experiment",
                        experiment,
                        "--job",
                        job_id,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                extra_run_id = json.loads(created.stdout)["run"]
                extra_root = WORK_ROOT / "experiments" / extra_run_id
                try:
                    dry = subprocess.run(
                        [
                            sys.executable,
                            "-m", f"autodub.experiments.{executor}",
                            "--run",
                            extra_run_id,
                            "--dry-run",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    self.assertTrue(json.loads(dry.stdout)["ok"])
                finally:
                    shutil.rmtree(extra_root, ignore_errors=True)
        finally:
            shutil.rmtree(root, ignore_errors=True)
            if run_root:
                shutil.rmtree(run_root, ignore_errors=True)

    def test_koboldcpp_child_is_stopped_as_a_process_tree(self) -> None:
        # KoboldCpp is a PyInstaller launcher whose worker child survives terminate(), which would
        # leave the model resident in GPU memory. There is exactly one spawner, and its cleanup
        # must stop the whole tree on every platform.
        runner = (ROOT / "src" / "autodub" / "adaptation_runner.py").read_text(encoding="utf-8")
        server = runner.split("def context_server", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_kill_tree(process)", server.split("finally:", 1)[1])
        self.assertNotIn("process.terminate()", server)
        kill = runner.split("def _kill_tree", 1)[1].split("\ndef ", 1)[0]
        for token in ('"taskkill"', '"/T"', '"/F"', "os.killpg"):
            self.assertIn(token, kill)
        screen = (ROOT / "src" / "autodub" / "experiments" / "translation_screen.py").read_text(encoding="utf-8")
        self.assertIn("from autodub.adaptation_runner import context_server", screen)
        self.assertNotIn("subprocess.Popen", screen)


if __name__ == "__main__":
    unittest.main(verbosity=2)
