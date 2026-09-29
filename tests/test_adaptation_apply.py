"""Hermetic contracts for the multi-engine merge (apply_best) — no LLM, jobs faked."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autodub import adaptation_runner


def _job(segments):
    return {"id": "dub-test", "segments": segments, "settings": {}, "events": []}


class ApplyBestContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.candidates = Path(self.tmp.name)
        self.saved = {}
        patches = [
            mock.patch.object(adaptation_runner, "CANDIDATES_DIR", self.candidates),
            mock.patch.object(adaptation_runner, "load_job", side_effect=lambda job_id: self.job),
            mock.patch.object(adaptation_runner, "save_job",
                              side_effect=lambda job: self.saved.update(job=job)),
            mock.patch.object(adaptation_runner, "event", side_effect=lambda *a, **k: None),
            mock.patch.object(adaptation_runner.oplog, "job_event",
                              side_effect=lambda *a, **k: None),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def _write(self, engine, candidates):
        (self.candidates / f"dub-test-{engine}.json").write_text(
            json.dumps({"job": "dub-test", "engine": engine,
                        "candidates": {str(i): {"text": t} for i, t in candidates.items()}}),
            encoding="utf-8")

    def test_fitting_candidate_beats_priority_loser_and_records_engine(self):
        # 2 s slot (2.15 budget): anchor 60 chars = 4 s. external fits, qwen still long.
        self.job = _job([{"i": 0, "start": 0.0, "end": 2.0, "text": "x",
                          "translation": "something enormous " * 3 + "here",
                          "speaker": "s1"}])
        self._write("external", {0: "Short and it fits fine."})
        self._write("qwen", {0: "something enormous but still much too long to fit the slot here."})
        summary = adaptation_runner.apply_best("dub-test", ["external", "qwen"])
        segment = self.saved["job"]["segments"][0]
        self.assertEqual(segment["adapt"]["engine"], "external")
        self.assertEqual(segment["adapt"]["verdict"], "adapted")
        self.assertEqual(segment["translation"], "Short and it fits fine.")
        self.assertEqual(segment["translation_source"], "adapted-v1")
        self.assertEqual(summary["engine_wins"].get("external"), 1)
        # the losing candidate is preserved for the diff
        self.assertIn("qwen", segment["adapt"]["candidates"])

    def test_tie_break_uses_engine_order_and_picks_override_it(self):
        self.job = _job([{"i": 0, "start": 0.0, "end": 2.0, "text": "x",
                          "translation": "A" * 60, "speaker": "s1"}])
        self._write("external", {0: "Both fit easily here."})
        self._write("qwen", {0: "This fits too."})
        adaptation_runner.apply_best("dub-test", ["external", "qwen"])
        self.assertEqual(self.saved["job"]["segments"][0]["adapt"]["engine"], "external")
        adaptation_runner.apply_best("dub-test", ["external", "qwen"], picks={"0": "qwen"})
        self.assertEqual(self.saved["job"]["segments"][0]["adapt"]["engine"], "qwen")

    def test_invalid_candidates_fall_back_to_anchor_residue(self):
        anchor = "Marco keeps his 3 crystals safe."
        self.job = _job([{"i": 0, "start": 0.0, "end": 1.0, "text": "x",
                          "translation": anchor + " " + "padding " * 6, "speaker": "s1"}])
        self._write("external", {0: "He keeps them safe."})      # drops Marco and 3
        summary = adaptation_runner.apply_best("dub-test", ["external"])
        segment = self.saved["job"]["segments"][0]
        self.assertEqual(segment["adapt"]["engine"], "anchor")
        self.assertEqual(segment["adapt"]["verdict"], "residue")
        self.assertIn(0, summary["residue_lines"])

    def test_explicit_pick_overrides_the_validator(self):
        # Anchor polluted with OCR'd screen text: every honest rewrite "drops" the stat
        # digits, so only a reviewed pick can rescue the line.
        anchor = "HP 4 MP 2 ENDURANCE 3 Come forth, my summon!"
        self.job = _job([{"i": 0, "start": 0.0, "end": 1.0, "text": "x",
                          "translation": anchor, "speaker": "s1"}])
        self._write("external", {0: "Come forth!"})
        adaptation_runner.apply_best("dub-test", ["external"])
        self.assertEqual(self.saved["job"]["segments"][0]["adapt"]["engine"], "anchor")
        adaptation_runner.apply_best("dub-test", ["external"], picks={"0": "external"})
        segment = self.saved["job"]["segments"][0]
        self.assertEqual(segment["adapt"]["engine"], "external")
        self.assertEqual(segment["translation"], "Come forth!")

    def test_neither_fits_shortest_valid_wins_as_residue(self):
        self.job = _job([{"i": 0, "start": 0.0, "end": 1.0, "text": "x",
                          "translation": "B" * 90, "speaker": "s1"}])
        self._write("external", {0: "C" * 60})
        self._write("qwen", {0: "D" * 40})    # shorter, still over the ~1.15 s budget
        adaptation_runner.apply_best("dub-test", ["external", "qwen"])
        segment = self.saved["job"]["segments"][0]
        self.assertEqual(segment["adapt"]["engine"], "qwen")
        self.assertEqual(segment["adapt"]["verdict"], "residue")
        self.assertEqual(segment["translation"], "D" * 40)


if __name__ == "__main__":
    unittest.main()
