from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autodub.config import WORK_ROOT  # noqa: E402

from autodub import episode_queue
from autodub.state import default_job, job_dir, load_job, save_job


def _reading(junction: float | None, ok: bool = True) -> dict:
    peak = max(58.8, junction) if ok and junction is not None else None
    return {"ok": ok, "core": 46.0, "hot_spot": 58.8, "junction": junction,
            "peak": peak, "raw": "synthetic"}


def _fake_render(job_id: str, gpu_authorized: bool = False) -> None:
    job = load_job(job_id)
    job["status"] = "complete"
    save_job(job)


class EpisodeQueueContractTests(unittest.TestCase):
    def test_queue_is_opaque_deduplicated_and_clearable(self) -> None:
        job_id = "dub-episode-queue-test"
        root = job_dir(job_id)
        try:
            job = default_job(job_id, ".mp4", 1, "0" * 64)
            job["segments"] = [
                {
                    "i": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "speaker-01",
                    "translation": "Synthetic.",
                }
            ]
            job["settings"]["quality_profile"] = "prototype-cpu-v1"
            save_job(job)
            with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
                temporary = Path(temporary)
                with (
                    patch.object(episode_queue, "QUEUE_PATH", temporary / "queue.json"),
                    patch.object(episode_queue, "STOP_PATH", temporary / "queue.stop"),
                ):
                    queued = episode_queue.enqueue([job_id, job_id])
                    self.assertEqual(len(queued["items"]), 1)
                    self.assertEqual(queued["items"][0]["job"], job_id)
                    self.assertFalse(queued["items"][0]["requires_gpu"])
                    queued["items"][0]["status"] = "complete"
                    episode_queue._save(queued)
                    self.assertEqual(episode_queue.clear_finished()["items"], [])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_unreviewed_job_cannot_enter_queue(self) -> None:
        job_id = "dub-episode-unreviewed"
        root = job_dir(job_id)
        try:
            save_job(default_job(job_id, ".mp4", 1, "0" * 64))
            with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
                temporary = Path(temporary)
                with (
                    patch.object(episode_queue, "QUEUE_PATH", temporary / "queue.json"),
                    patch.object(episode_queue, "STOP_PATH", temporary / "queue.stop"),
                ):
                    with self.assertRaisesRegex(ValueError, "analyzed and reviewed"):
                        episode_queue.enqueue([job_id])
        finally:
            shutil.rmtree(root, ignore_errors=True)


class ThermalQueueTests(unittest.TestCase):
    """The thermal guard ships ON, fails CLOSED, and the CPU path stays byte-identical.
    All renders are stubbed."""

    def _jobs(self, count: int, profile: str) -> list[str]:
        job_ids = []
        for index in range(count):
            job_id = f"dub-thermal-{profile.split('-')[1]}-{index}"
            job = default_job(job_id, ".mp4", 1, "0" * 64)
            job["segments"] = [{"i": 0, "start": 0.0, "end": 1.0,
                               "speaker": "speaker-01", "translation": "Synthetic."}]
            job["settings"]["quality_profile"] = profile
            save_job(job)
            job_ids.append(job_id)
        return job_ids

    def _cleanup(self, job_ids: list[str]) -> None:
        for job_id in job_ids:
            shutil.rmtree(job_dir(job_id), ignore_errors=True)

    def _run(self, job_ids: list[str], *, reading=None, read_temps=None,
             sleeps=None, guard=True, cooldown=0, render=_fake_render):
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            temporary = Path(temporary)
            with (
                patch.object(episode_queue, "QUEUE_PATH", temporary / "queue.json"),
                patch.object(episode_queue, "STOP_PATH", temporary / "queue.stop"),
                patch.object(episode_queue.thermal, "read_temps",
                             read_temps or (lambda **kwargs: reading)),
                patch.object(episode_queue.config, "THERMAL_GUARD_ENABLED", guard),
                patch.object(episode_queue.config, "THERMAL_COOLDOWN_S", cooldown),
                patch.object(episode_queue.time, "sleep",
                             sleeps.append if sleeps is not None else (lambda s: None)),
                patch("autodub.pipeline.render", side_effect=render) as rendered,
            ):
                episode_queue.enqueue(job_ids)
                episode_queue.run_queue(gpu_authorized=True)
                return episode_queue.queue_state(), rendered

    def test_thermal_abort_stops_queue_loudly_and_fails_closed(self) -> None:
        for reading in (_reading(96.0), _reading(None, ok=False)):
            job_ids = self._jobs(2, "quality-gpu-v1")
            try:
                state, rendered = self._run(job_ids, reading=reading)
                self.assertEqual(rendered.call_count, 0)
                self.assertEqual(state["status"], "aborted")
                self.assertEqual([item["status"] for item in state["items"]],
                                 ["queued", "queued"])
                self.assertIn("thermal_abort", state)
                expected = "unreadable" if not reading["ok"] else "peak 96"
                self.assertIn(expected, state["thermal_abort"]["reason"])
                self.assertEqual(state["thermal_last"]["junction"], reading["junction"])
                # the abort is UI-visible on the about-to-run job
                events = load_job(job_ids[0]).get("events") or []
                self.assertTrue(any("thermal guard" in e["message"] for e in events))
            finally:
                self._cleanup(job_ids)

    def test_healthy_run_logs_temps_and_cools_down_between_episodes(self) -> None:
        job_ids = self._jobs(2, "quality-gpu-v1")
        sleeps = []
        try:
            state, rendered = self._run(job_ids, reading=_reading(58.0),
                                        sleeps=sleeps, cooldown=6)
            self.assertEqual(rendered.call_count, 2)
            self.assertEqual(state["status"], "idle")
            for item in state["items"]:
                self.assertEqual(item["status"], "complete")
                self.assertEqual(item["thermal_after"]["junction"], 58.0)
            self.assertEqual(sleeps, [2, 2, 2])  # one cooldown, 3 slices, not after last
        finally:
            self._cleanup(job_ids)

    def test_guard_disabled_still_logs_temps(self) -> None:
        job_ids = self._jobs(1, "quality-gpu-v1")
        try:
            state, rendered = self._run(job_ids, reading=_reading(97.0), guard=False)
            self.assertEqual(rendered.call_count, 1)
            self.assertNotIn("thermal_abort", state)
            self.assertEqual(state["items"][0]["thermal_after"]["junction"], 97.0)
            self.assertEqual(state["thermal_last"]["junction"], 97.0)
        finally:
            self._cleanup(job_ids)

    def test_warn_band_does_not_abort(self) -> None:
        job_ids = self._jobs(2, "quality-gpu-v1")
        try:
            state, rendered = self._run(job_ids, reading=_reading(94.0))
            self.assertEqual(rendered.call_count, 2)
            self.assertEqual(state["status"], "idle")
            self.assertNotIn("thermal_abort", state)
        finally:
            self._cleanup(job_ids)

    def test_cpu_only_queue_is_byte_identical_to_before(self) -> None:
        job_ids = self._jobs(1, "prototype-cpu-v1")

        def must_not_read(**kwargs):
            raise AssertionError("thermal.read_temps must not run for CPU queues")

        sleeps = []
        try:
            state, rendered = self._run(job_ids, read_temps=must_not_read,
                                        sleeps=sleeps, cooldown=120)
            self.assertEqual(rendered.call_count, 1)
            self.assertEqual(state["status"], "idle")
            self.assertEqual(sleeps, [])
            self.assertNotIn("thermal_last", state)
            self.assertNotIn("thermal_after", state["items"][0])
        finally:
            self._cleanup(job_ids)

    def test_stop_during_cooldown_lands_promptly(self) -> None:
        job_ids = self._jobs(2, "quality-gpu-v1")
        stop_on_first_sleep = []

        def sleeper(seconds: float) -> None:
            stop_on_first_sleep.append(seconds)
            episode_queue.STOP_PATH.write_text("stop\n", encoding="ascii")

        try:
            with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
                temporary = Path(temporary)
                with (
                    patch.object(episode_queue, "QUEUE_PATH", temporary / "queue.json"),
                    patch.object(episode_queue, "STOP_PATH", temporary / "queue.stop"),
                    patch.object(episode_queue.thermal, "read_temps",
                                 lambda **kwargs: _reading(58.0)),
                    patch.object(episode_queue.config, "THERMAL_COOLDOWN_S", 120),
                    patch.object(episode_queue.time, "sleep", sleeper),
                    patch("autodub.pipeline.render", side_effect=_fake_render) as rendered,
                ):
                    episode_queue.enqueue(job_ids)
                    episode_queue.run_queue(gpu_authorized=True)
                    state = episode_queue.queue_state()
            self.assertEqual(rendered.call_count, 1)
            self.assertEqual(len(stop_on_first_sleep), 1)  # stop seen on next slice
            self.assertEqual(state["status"], "stopped")
            self.assertTrue(state["stop_requested"])
            self.assertEqual(state["items"][1]["status"], "queued")
        finally:
            self._cleanup(job_ids)


if __name__ == "__main__":
    unittest.main()
