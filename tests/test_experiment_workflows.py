from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autodub.adapters import sapi_voices, synthesize_sapi
from autodub.config import FFMPEG
from autodub.experiment_store import load_run, run_dir
from autodub.policy_experiment import build_policy_experiment
from autodub.state import default_job, job_dir, save_job
from autodub.tts_experiment import execute_tts_experiment, plan_tts_experiment
from tests import requires_ffmpeg, requires_windows  # noqa: E402


class RealExperimentWorkflowTests(unittest.TestCase):
    @requires_windows
    @requires_ffmpeg
    def test_cpu_policy_and_tts_comparisons_produce_playable_video(self) -> None:
        job_id = "dub-experiment-workflow-test"
        root = job_dir(job_id)
        policy_run = None
        tts_run = None
        try:
            artifacts = root / "artifacts"
            lines = artifacts / "lines"
            lines.mkdir(parents=True)
            source = root / "source.mp4"
            full = artifacts / "full.wav"
            subprocess.run(
                [
                    str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=0x182338:s=320x180:d=6:r=24",
                    "-f", "lavfi", "-i", "sine=frequency=180:sample_rate=48000:duration=6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source),
                ],
                check=True,
            )
            subprocess.run(
                [
                    str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source), "-vn", "-ar", "48000", "-ac", "2", str(full),
                ],
                check=True,
            )
            synthesize_sapi("Synthetic comparison line.", sapi_voices()[0], lines / "line-00000.wav")
            job = default_job(job_id, ".mp4", source.stat().st_size, "0" * 64)
            job["status"] = "review"
            job["artifacts"]["full_audio"] = "full.wav"
            job["segments"] = [
                {
                    "i": 0,
                    "start": 1.0,
                    "end": 3.5,
                    "speaker": "speaker-01",
                    "translation": "Synthetic comparison line.",
                }
            ]
            save_job(job)

            policy = build_policy_experiment(job_id, start=0.0, duration=5.0)
            policy_run = policy["id"]
            self.assertEqual(len(policy["candidates"]), 6)
            for candidate in policy["candidates"]:
                artifact = run_dir(policy_run) / candidate["result"]["artifacts"][0]
                self.assertTrue(artifact.is_file(), candidate["id"])

            planned = plan_tts_experiment(job_id, ["windows-sapi"], 0.0, 5.0)
            tts_run = planned["id"]
            execute_tts_experiment(tts_run, gpu_authorized=False)
            completed = load_run(tts_run)
            self.assertEqual(completed["status"], "awaiting-review")
            sapi = next(item for item in completed["candidates"] if item["id"] == "windows-sapi")
            artifact = run_dir(tts_run) / sapi["result"]["artifacts"][0]
            self.assertTrue(artifact.is_file())
        finally:
            shutil.rmtree(root, ignore_errors=True)
            if policy_run:
                shutil.rmtree(run_dir(policy_run), ignore_errors=True)
            if tts_run:
                shutil.rmtree(run_dir(tts_run), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
