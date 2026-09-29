"""Queue-overlap contracts (QUEUE_OVERLAP_POST_STAGES — ships OFF).

The GPU lease is held only through synthesis; the CPU tail (align/mix/mux) runs on a
joined daemon thread while the next episode synthesizes. All renders stubbed — no GPU.
"""
from __future__ import annotations

import contextlib
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autodub.config import WORK_ROOT  # noqa: E402

from autodub import episode_queue, pipeline
from autodub.state import default_job, job_dir, load_job, save_job


def _reading(junction: float) -> dict:
    return {"ok": True, "core": 46.0, "hot_spot": 58.8, "junction": junction,
            "peak": max(58.8, junction), "raw": "synthetic"}


def _make_jobs(prefix: str, count: int) -> list[str]:
    job_ids = []
    for index in range(count):
        job_id = f"dub-overlap-{prefix}-{index}"
        job = default_job(job_id, ".mp4", 1, "0" * 64)
        job["segments"] = [{"i": 0, "start": 0.0, "end": 1.0,
                            "speaker": "speaker-01", "translation": "Synthetic."}]
        job["settings"]["quality_profile"] = "quality-gpu-v1"
        save_job(job)
        job_ids.append(job_id)
    return job_ids


def _mark(job_id: str, status: str) -> None:
    job = load_job(job_id)
    job["status"] = status
    save_job(job)


class RenderOverlappedTests(unittest.TestCase):
    def test_lease_released_before_cpu_tail_runs(self) -> None:
        job_ids = _make_jobs("lease", 1)
        timeline = []

        @contextlib.contextmanager
        def fake_lease(reason, **kwargs):
            timeline.append("lease-enter")
            yield type("L", (), {"ensure_active": lambda self: None})()
            timeline.append("lease-exit")

        def fake_post(job_id):
            timeline.append("post-start")
            _mark(job_id, "complete")

        try:
            with (
                patch.object(pipeline, "gpu_lease", fake_lease),
                patch.object(pipeline, "_render_quality_synth",
                             lambda job_id, lease=None: timeline.append("synth")),
                patch.object(pipeline, "_render_quality_post", fake_post),
            ):
                thread = pipeline.render_overlapped(job_ids[0], gpu_authorized=True)
                self.assertIsNotNone(thread)
                thread.join(timeout=5)
            self.assertEqual(timeline[:3], ["lease-enter", "synth", "lease-exit"])
            self.assertLess(timeline.index("lease-exit"), timeline.index("post-start"))
            self.assertEqual(load_job(job_ids[0])["status"], "complete")
        finally:
            shutil.rmtree(job_dir(job_ids[0]), ignore_errors=True)

    def test_single_job_render_path_still_runs_both_halves(self) -> None:
        source = (ROOT / "src" / "autodub" / "pipeline.py").read_text(encoding="utf-8")
        combined = source.split("def _render_quality(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_render_quality_synth", combined)
        self.assertIn("_render_quality_post", combined)
        # Regression: the split once left `bed_name` unassigned in the
        # post half (NameError at mix). Every name the post half uses must be derived
        # inside it — pin bed_name explicitly.
        post = source.split("def _render_quality_post", 1)[1].split("\ndef ", 1)[0]
        use = post.find("artifacts / bed_name")
        assign = post.find('bed_name = job.get("artifacts", {}).get("source_bed")')
        self.assertGreaterEqual(use, 0)
        self.assertGreaterEqual(assign, 0)
        self.assertLess(assign, use)
        # render() (the /api path) must NOT use overlap — it targets _render_quality
        render_body = source.split("\ndef render(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('_quality_action(job_id, "render", _render_quality)', render_body)


class QueueOverlapTests(unittest.TestCase):
    def _run_queue(self, job_ids, fake_overlapped):
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            temporary = Path(temporary)
            with (
                patch.object(episode_queue, "QUEUE_PATH", temporary / "queue.json"),
                patch.object(episode_queue, "STOP_PATH", temporary / "queue.stop"),
                patch.object(episode_queue.thermal, "read_temps",
                             lambda **kwargs: _reading(58.0)),
                patch.object(episode_queue.config, "QUEUE_OVERLAP_POST_STAGES", True),
                patch.object(episode_queue.config, "QUEUE_MAX_PENDING_TAILS", 1),
                patch.object(episode_queue.config, "THERMAL_COOLDOWN_S", 0),
                patch("autodub.pipeline.render_overlapped", side_effect=fake_overlapped),
            ):
                episode_queue.enqueue(job_ids)
                episode_queue.run_queue(gpu_authorized=True)
                return episode_queue.queue_state()

    def test_next_synth_starts_before_previous_tail_ends(self) -> None:
        job_ids = _make_jobs("order", 2)
        timeline = []
        second_started = threading.Event()

        def fake_overlapped(job_id, *, gpu_authorized=False):
            timeline.append(("synth", job_id))
            if job_id == job_ids[1]:
                second_started.set()

            def tail():
                if job_id == job_ids[0]:
                    second_started.wait(timeout=5)
                _mark(job_id, "complete")
                timeline.append(("tail-end", job_id))

            thread = threading.Thread(target=tail, daemon=True)
            thread.start()
            return thread

        try:
            state = self._run_queue(job_ids, fake_overlapped)
            self.assertEqual([item["status"] for item in state["items"]],
                             ["complete", "complete"])
            self.assertEqual(state["status"], "idle")
            self.assertLess(timeline.index(("synth", job_ids[1])),
                            timeline.index(("tail-end", job_ids[0])))
            for item in state["items"]:  # deferred temps event landed after join
                self.assertNotIn("_thermal_after_summary", item)
                self.assertEqual(item["thermal_after"]["junction"], 58.0)
        finally:
            for job_id in job_ids:
                shutil.rmtree(job_dir(job_id), ignore_errors=True)

    def test_tail_failure_is_isolated_to_its_item(self) -> None:
        job_ids = _make_jobs("fail", 2)

        def fake_overlapped(job_id, *, gpu_authorized=False):
            def tail():
                _mark(job_id, "failed" if job_id == job_ids[0] else "complete")
            thread = threading.Thread(target=tail, daemon=True)
            thread.start()
            return thread

        try:
            state = self._run_queue(job_ids, fake_overlapped)
            self.assertEqual([item["status"] for item in state["items"]],
                             ["failed", "complete"])
            self.assertEqual(state["status"], "idle")
        finally:
            for job_id in job_ids:
                shutil.rmtree(job_dir(job_id), ignore_errors=True)

    def test_stop_after_first_episode_still_joins_its_tail(self) -> None:
        job_ids = _make_jobs("stop", 2)

        def fake_overlapped(job_id, *, gpu_authorized=False):
            episode_queue.STOP_PATH.write_text("stop\n", encoding="ascii")

            def tail():
                time.sleep(0.05)
                _mark(job_id, "complete")
            thread = threading.Thread(target=tail, daemon=True)
            thread.start()
            return thread

        try:
            state = self._run_queue(job_ids, fake_overlapped)
            self.assertEqual(state["items"][0]["status"], "complete")
            self.assertEqual(state["items"][1]["status"], "queued")
            self.assertEqual(state["status"], "stopped")
            self.assertTrue(state["stop_requested"])
        finally:
            for job_id in job_ids:
                shutil.rmtree(job_dir(job_id), ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
