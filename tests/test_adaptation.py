"""Hermetic contracts for the pure half of dialogue adaptation (no LLM, no I/O)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autodub import adaptation
from autodub.config import ADAPT_CPS, ADAPT_GUARD_S, ADAPT_TAIL_SLACK_S


class SlotBudgetContracts(unittest.TestCase):
    def test_free_tail_when_no_neighbor(self):
        budget = adaptation.slot_budget(10.0, 12.0, None)
        self.assertEqual(budget["base_s"], 2.0)
        self.assertEqual(budget["tail_s"], ADAPT_TAIL_SLACK_S)
        self.assertEqual(budget["budget_s"], 2.0 + ADAPT_TAIL_SLACK_S)

    def test_tail_clipped_by_close_neighbor(self):
        budget = adaptation.slot_budget(10.0, 12.0, 12.10)
        self.assertEqual(budget["tail_s"], round(0.10 - ADAPT_GUARD_S, 3))
        overlapped = adaptation.slot_budget(10.0, 12.0, 11.9)
        self.assertEqual(overlapped["tail_s"], 0.0)


class ClassifyContracts(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(adaptation.classify(3.0, 2.0, 2.15), "long")
        self.assertEqual(adaptation.classify(0.5, 2.0, 2.15), "short")
        self.assertEqual(adaptation.classify(0.4, 1.0, 1.15), "fit")   # small slot: short is fine
        self.assertEqual(adaptation.classify(2.0, 2.0, 2.15), "fit")


class PlanContracts(unittest.TestCase):
    def test_kinds_and_targets(self):
        segments = [
            {"i": 0, "start": 0.0, "end": 2.0, "text": "x",
             "translation": "A" * 60, "speaker": "s1"},
            {"i": 1, "start": 5.0, "end": 8.0, "text": "x",
             "translation": "Hi.", "speaker": "s2"},
            {"i": 2, "start": 10.0, "end": 12.0, "text": "x",
             "translation": "", "speaker": "s1"},
            {"i": 3, "start": 14.0, "end": 15.0, "text": "x",
             "translation": "line", "speaker": "s1", "song_skip": True},
        ]
        rows = {row["i"]: row for row in adaptation.plan(segments)}
        self.assertEqual(rows[0]["kind"], "long")
        self.assertGreater(rows[0]["max_chars"], 8)
        self.assertEqual(rows[1]["kind"], "short")
        self.assertGreater(rows[1]["target_chars"], len("Hi."))
        self.assertEqual(rows[2]["kind"], "skip-empty")
        self.assertEqual(rows[3]["kind"], "skip-song")

    def test_rederives_from_stored_anchor(self):
        segment = {"i": 0, "start": 0.0, "end": 2.0, "text": "x", "speaker": "s1",
                   "translation": "short rewrite",
                   "adapt": {"anchor": "the original much longer translation " * 3}}
        row = adaptation.plan([segment])[0]
        self.assertTrue(row["anchor"].startswith("the original much longer translation"))


class ValidatorContracts(unittest.TestCase):
    ANCHOR = "Marco won't give the 3 crystals to Oliver, will he?"

    def test_good_rewrite_passes(self):
        ok, reason = adaptation.validate_rewrite(
            self.ANCHOR, "Marco won't hand Oliver the 3 crystals, right?")
        self.assertTrue(ok, reason)

    def test_meaning_damage_blocked(self):
        cases = {
            "Marco gives the 3 crystals to Oliver, right?": "polarity-changed",
            "Marco won't give the crystals to Oliver, will he?": "dropped-number:3",
            "He won't give the 3 crystals to him, will he?": "dropped-name:Oliver",
            "Marco won't give Oliver the 3 crystals.": "question-form-changed",
            "": "empty",
        }
        for candidate, expected in cases.items():
            ok, reason = adaptation.validate_rewrite(self.ANCHOR, candidate)
            self.assertFalse(ok, candidate)
            self.assertEqual(reason, expected)


class EstimateContracts(unittest.TestCase):
    def test_cps_prior_and_normalization(self):
        self.assertAlmostEqual(adaptation.estimate_seconds("A" * 27), 27 / ADAPT_CPS)
        self.assertEqual(adaptation.estimate_seconds("  spaced   out\n text "),
                         len("spaced out text") / ADAPT_CPS)


if __name__ == "__main__":
    unittest.main()
