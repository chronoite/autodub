"""Reviewer verdicts (pass / maybe / fail) on experiment reviews.

Per-criterion 1-5 grids were judging overload; the primary review surface is one verdict
per candidate. Numeric scores stay accepted so archived runs remain valid.
"""
from __future__ import annotations

import shutil
import unittest

from autodub.experiment_store import create_run, load_run, run_dir, update_review
from autodub.state import default_job, job_dir, new_job_id, save_job


class VerdictReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.job_id = new_job_id()
        job_dir(self.job_id).mkdir(parents=True)
        save_job(default_job(self.job_id, ".mp4", 1, "0" * 64))
        self.run_id = create_run("speaker-detection-v1", self.job_id)["id"]
        self.candidate = load_run(self.run_id)["candidates"][0]["id"]

    def tearDown(self) -> None:
        shutil.rmtree(job_dir(self.job_id), ignore_errors=True)
        shutil.rmtree(run_dir(self.run_id), ignore_errors=True)

    def test_verdicts_persist_and_normalize(self) -> None:
        public = update_review(self.run_id, {"verdicts": {self.candidate: " PASS "}})
        self.assertEqual(public["human_review"]["verdicts"][self.candidate], "pass")
        self.assertEqual(load_run(self.run_id)["human_review"]["verdicts"][self.candidate], "pass")

    def test_verdict_survives_a_scores_only_update(self) -> None:
        update_review(self.run_id, {"verdicts": {self.candidate: "maybe"}})
        criterion = load_run(self.run_id)["criteria"][0]
        public = update_review(self.run_id, {"scores": {self.candidate: {criterion: 4}}})
        self.assertEqual(public["human_review"]["verdicts"][self.candidate], "maybe")
        self.assertEqual(public["human_review"]["scores"][self.candidate][criterion], 4)

    def test_unknown_candidate_and_bad_value_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            update_review(self.run_id, {"verdicts": {"not-a-candidate": "pass"}})
        with self.assertRaises(ValueError):
            update_review(self.run_id, {"verdicts": {self.candidate: "amazing"}})
        with self.assertRaises(ValueError):
            update_review(self.run_id, {"verdicts": "pass"})

    def test_legacy_scores_only_payload_still_works(self) -> None:
        criterion = load_run(self.run_id)["criteria"][0]
        public = update_review(
            self.run_id,
            {"status": "complete", "winner": self.candidate,
             "scores": {self.candidate: {criterion: 5}}},
        )
        self.assertEqual(public["human_review"]["winner"], self.candidate)
        self.assertEqual(public["human_review"]["verdicts"], {})

    def test_stale_revision_is_rejected_and_current_revision_saves(self) -> None:
        base = load_run(self.run_id)["human_review"].get("revision", 0)
        update_review(self.run_id, {"verdicts": {self.candidate: "pass"}, "revision": base})
        # a second tab still holding the old revision must not silently overwrite
        with self.assertRaises(ValueError):
            update_review(self.run_id, {"verdicts": {self.candidate: "fail"}, "revision": base})
        current = load_run(self.run_id)["human_review"]
        self.assertEqual(current["verdicts"][self.candidate], "pass")
        self.assertEqual(current["revision"], base + 1)
        # a revision-less save (legacy client) still works
        update_review(self.run_id, {"verdicts": {self.candidate: "maybe"}})
        self.assertEqual(load_run(self.run_id)["human_review"]["revision"], base + 2)

    def test_blockers_validate_persist_and_clear(self) -> None:
        public = update_review(
            self.run_id,
            {"verdicts": {self.candidate: "maybe"}, "blockers": {self.candidate: "timing"}},
        )
        self.assertEqual(public["human_review"]["blockers"][self.candidate], "timing")
        public = update_review(self.run_id, {"blockers": {self.candidate: None}})
        self.assertEqual(public["human_review"]["blockers"], {})
        with self.assertRaises(ValueError):
            update_review(self.run_id, {"blockers": {self.candidate: "vibes"}})
        with self.assertRaises(ValueError):
            update_review(self.run_id, {"blockers": {"nobody": "timing"}})

    def test_new_runs_carry_verdict_schema_and_reject_bad_candidate_ids(self) -> None:
        review = load_run(self.run_id)["human_review"]
        self.assertEqual(review["verdicts"], {})
        self.assertEqual(review["blockers"], {})
        self.assertEqual(review["revision"], 0)
        from unittest.mock import patch
        import autodub.experiment_store as store
        hostile = {"experiments": [{
            "id": "hostile-v1",
            "candidates": [{"id": "<img src=x>"}],
            "criteria": ["a"], "required_material": [],
        }]}
        with patch.object(store, "registry", return_value=hostile):
            with self.assertRaises(ValueError):
                create_run("hostile-v1", self.job_id)


if __name__ == "__main__":
    unittest.main()
