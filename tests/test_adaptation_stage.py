"""Adaptation-stage contracts.

The engine itself (collect_qwen/apply_best) is covered by test_adaptation_apply;
these tests pin the NEW orchestration wrapper + endpoint wiring: status flow,
engine defaulting from present candidate files, GPU arming, and failure restore.
No LLM, no GPU — capture paths are refused or mocked.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autodub import adaptation_runner
from autodub.adaptation_runner import adapt_reviewed
from autodub.config import WORK_ROOT
from autodub.gpu_session import arm, GpuSafetyError
from autodub.state import default_job, job_dir, load_job, save_job

from tests import ROOT


LONG_LINE = "This translation is far too long to ever be spoken in a one second slot."


class AdaptStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=WORK_ROOT)
        self.candidates = Path(self.tmp.name)
        patcher = mock.patch.object(adaptation_runner, "CANDIDATES_DIR", self.candidates)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.job_id = "dub-adapt-stage-test"
        self.addCleanup(lambda: shutil.rmtree(job_dir(self.job_id), ignore_errors=True))
        job = default_job(self.job_id, ".mp4", 1, "0" * 64)
        job["status"] = "review"
        job["segments"] = [{"i": 0, "start": 0.0, "end": 1.0, "speaker": "speaker-01",
                            "text": "synthetic-source", "translation": LONG_LINE}]
        save_job(job)

    def _write_candidates(self, engine: str, mapping: dict) -> None:
        (self.candidates / f"{self.job_id}-{engine}.json").write_text(
            json.dumps({"candidates": mapping}), encoding="utf-8")

    def test_apply_only_defaults_to_present_candidate_files(self) -> None:
        self._write_candidates("external", {"0": {"text": "Too long, that."}})
        summary = adapt_reviewed(self.job_id, capture=False)
        self.assertEqual(summary["engines"], ["external"])
        self.assertEqual(summary["changed"], 1)
        stored = load_job(self.job_id)
        self.assertEqual(stored["status"], "review")
        self.assertEqual(stored["segments"][0]["translation"], "Too long, that.")
        self.assertEqual(stored["segments"][0]["adapt"]["engine"], "external")
        self.assertEqual(stored["segments"][0]["adapt"]["anchor"], LONG_LINE)
        self.assertEqual(stored["adapt_summary"]["engine_wins"], {"external": 1})

    def test_capture_requires_a_fresh_gpu_arm(self) -> None:
        with self.assertRaises(GpuSafetyError):
            adapt_reviewed(self.job_id, capture=True, gpu_authorized=False)
        stored = load_job(self.job_id)
        self.assertEqual(stored["status"], "review")  # restored, not stuck running
        self.assertEqual(stored["error"], "dialogue adaptation failed")
        self.assertIn("GpuSafetyError", stored["error_detail"])

    def test_no_candidate_files_fails_cleanly(self) -> None:
        with self.assertRaisesRegex(ValueError, "no adaptation candidate files"):
            adapt_reviewed(self.job_id, capture=False)
        self.assertEqual(load_job(self.job_id)["status"], "review")

    def test_adapt_is_a_valid_arm_action(self) -> None:
        # arm() may still fail closed on live preflight; the allowlist must not be
        # the thing that rejects it.
        try:
            arm("dub-arm-allowlist-probe", "adapt")
        except ValueError as exc:
            self.fail(f"'adapt' rejected by the arm allowlist: {exc}")
        except Exception:
            pass  # preflight refusals are environmental, not the contract under test

    def test_stage_is_wired_end_to_end(self) -> None:
        server_src = (ROOT / "src" / "autodub" / "server.py").read_text(encoding="utf-8")
        self.assertIn('parts[3] == "adapt"', server_src)
        self.assertIn('consume_arm(job_id, "adapt")', server_src)
        self.assertIn("adapt_reviewed", server_src)
        app_js = (ROOT / "src" / "autodub" / "web" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("adaptDialogue", app_js)
        self.assertIn("adapt-summary", app_js)
        self.assertIn('id="adapt"', (ROOT / "src" / "autodub" / "web" / "static" / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
