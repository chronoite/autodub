# -*- coding: utf-8 -*-
"""Contract tests: voice bank + evidence cards. Hermetic — synthetic embeddings and
segments only, VOICES_ROOT redirected to a temp dir, no GPU, no real media, no ffmpeg."""
from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from autodub import evidence_cards, voice_bank


def _vec(angle: float) -> list[float]:
    """Unit vector at `angle` radians — cosine between two = cos(delta), so tests can
    dial an exact similarity instead of hoping magic numbers land in the right zone."""
    return [math.cos(angle), math.sin(angle)]


class VoiceBankBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="vbank_test_"))
        self._old_root = voice_bank.VOICES_ROOT
        voice_bank.VOICES_ROOT = self._tmp          # the real voices/ tree is never touched
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        voice_bank.VOICES_ROOT = self._old_root
        shutil.rmtree(self._tmp, ignore_errors=True)


class TestBankCore(VoiceBankBase):
    def test_slug_is_deterministic_and_safe(self):
        self.assertEqual(voice_bank.series_slug("STAR ROAD: The Last Courier!"),
                         "star-road-the-last-courier")
        with self.assertRaises(ValueError):
            voice_bank.series_slug("!!!")

    def test_missing_bank_is_empty_not_error(self):
        bank = voice_bank.load_bank("nope")
        self.assertEqual(bank["characters"], [])

    def test_add_character_persists_and_logs(self):
        entry = voice_bank.add_character("s1", "Rin", centroid=_vec(0.0), source_job="j1")
        again = voice_bank.load_bank("s1")
        self.assertEqual([c["id"] for c in again["characters"]], [entry["id"]])
        kinds = [d["kind"] for d in voice_bank.read_decisions("s1")]
        self.assertIn("character_added", kinds)

    def test_corrupt_bank_refuses_to_guess(self):
        voice_bank.add_character("s2", "Rin", centroid=_vec(0.0))
        voice_bank._bank_path("s2").write_text("[1,2,3]", encoding="utf-8")
        with self.assertRaises(ValueError):
            voice_bank.load_bank("s2")


class TestThreeZones(VoiceBankBase):
    def setUp(self) -> None:
        super().setUp()
        voice_bank.add_character("show", "Rin", centroid=_vec(0.0))
        voice_bank.add_character("show", "Kai", centroid=_vec(math.pi / 2))

    def test_clear_zone_auto_match(self):
        r = voice_bank.match_speakers("show", {"SPEAKER_00": _vec(0.05)})["SPEAKER_00"]
        self.assertEqual(r["zone"], "clear")
        self.assertEqual(r["character_name"], "Rin")
        self.assertGreaterEqual(r["score"], voice_bank.VOICE_BANK_MATCH_THRESHOLD)

    def test_ask_zone_is_a_card_not_a_call(self):
        # vec(0.4): cos(0.4)=0.921 vs Rin — inside the calibrated ask band [0.90, 0.95);
        # vs Kai cos(0.4-pi/2)=0.389, far below
        r = voice_bank.match_speakers("show", {"S": _vec(0.4)})["S"]
        self.assertEqual(r["zone"], "ask")
        self.assertEqual(r["character_name"], "Rin")

    def test_new_zone_below_ask(self):
        # vec(pi): cos(pi)=-1 vs Rin, cos(pi/2)=0 vs Kai — both far below ask
        r = voice_bank.match_speakers("show", {"S": _vec(math.pi)})["S"]
        self.assertEqual(r["zone"], "new")

    def test_scores_expose_why(self):
        r = voice_bank.match_speakers("show", {"S": _vec(0.05)})["S"]
        self.assertIn("Rin", r["scores"])
        self.assertIn("Kai", r["scores"])

    def test_invalid_thresholds_refused(self):
        with self.assertRaises(ValueError):
            voice_bank.match_speakers("show", {}, match_threshold=0.5, ask_threshold=0.6)


class TestEmbedderQuarantine(VoiceBankBase):
    def test_cross_space_is_quarantined_never_scored(self):
        voice_bank.add_character("q", "Rin", centroid=_vec(0.0), embedder="old-model-v0")
        r = voice_bank.match_speakers("q", {"S": _vec(0.0)})["S"]   # current embedder
        self.assertEqual(r["zone"], "quarantined")
        self.assertIsNone(r["score"])                # no cross-space cosine, ever
        self.assertEqual(r["quarantined_characters"], ["Rin"])

    def test_refresh_exits_quarantine(self):
        e = voice_bank.add_character("q", "Rin", centroid=_vec(0.0), embedder="old-model-v0")
        voice_bank.refresh_centroid("q", e["id"], _vec(0.0))
        r = voice_bank.match_speakers("q", {"S": _vec(0.0)})["S"]
        self.assertEqual(r["zone"], "clear")


class TestDecisionLog(VoiceBankBase):
    def test_append_only_and_ordered(self):
        voice_bank.log_decision("d", kind="a")
        voice_bank.log_decision("d", kind="b")
        self.assertEqual([x["kind"] for x in voice_bank.read_decisions("d")], ["a", "b"])

    def test_reviewer_answer_contract(self):
        voice_bank.record_reviewer_answer("d", job_id="j", speaker="S", answer="same",
                                       character_id="ch_x", character_name="Rin")
        with self.assertRaises(ValueError):
            voice_bank.record_reviewer_answer("d", job_id="j", speaker="S", answer="same")
        with self.assertRaises(ValueError):
            voice_bank.record_reviewer_answer("d", job_id="j", speaker="S", answer="maybe")


class TestLibrary(VoiceBankBase):
    def _character_with_clip(self, transcript="Hello there, this is my line."):
        clip = self._tmp / "ref.wav"
        clip.write_bytes(b"RIFFfake")
        return voice_bank.add_character("lib", "Rin", centroid=_vec(0.0),
                                        reference_clip=str(clip),
                                        reference_transcript=transcript)

    def test_promotion_copies_clip_and_transcript(self):
        e = self._character_with_clip()
        v = voice_bank.promote_to_library("lib", e["id"])
        lib = voice_bank.load_library()
        self.assertEqual([x["id"] for x in lib["voices"]], [v["id"]])
        self.assertTrue(Path(v["clip"]).is_file())
        self.assertNotEqual(v["clip"], e["reference_clip"])   # a copy, not a move
        self.assertEqual(v["transcript"], e["reference_transcript"])
        self.assertEqual(v["origin_series"], "lib")

    def test_promotion_requires_clip_and_transcript(self):
        e = voice_bank.add_character("lib", "Kai", centroid=_vec(1.0))
        with self.assertRaises(ValueError):
            voice_bank.promote_to_library("lib", e["id"])
        e2 = self._character_with_clip(transcript="  ")
        with self.assertRaises(ValueError):
            voice_bank.promote_to_library("lib", e2["id"])


class TestCardPlanning(unittest.TestCase):
    SEGS = [
        {"speaker": "A", "start": 0.0, "end": 3.0, "text": "solo long a1"},
        {"speaker": "B", "start": 2.5, "end": 4.0, "text": "overlaps A"},      # both tainted
        {"speaker": "A", "start": 10.0, "end": 11.0, "text": "too short"},
        {"speaker": "A", "start": 20.0, "end": 25.0, "text": "solo longest"},
        {"speaker": "A", "start": 40.0, "end": 42.0, "text": "solo a3"},
        {"speaker": "A", "start": 41.0, "end": 60.0, "text": "self-overlap ok"},
        {"speaker": "B", "start": 70.0, "end": 73.0, "text": "solo b1"},
        {"speaker": "",  "start": 80.0, "end": 90.0, "text": "unlabelled dropped"},
    ]

    def test_solo_min_length_and_clamp(self):
        plans = evidence_cards.plan_cards(self.SEGS, clips_per_speaker=2)
        starts_a = {c["start"] for c in plans["A"]}
        self.assertNotIn(0.0, starts_a)          # cross-speaker overlap excluded
        self.assertNotIn(10.0, starts_a)         # under min length excluded
        clamped = next(c for c in plans["A"] if c["start"] == 41.0)
        self.assertAlmostEqual(clamped["end"], 41.0 + 8.0)       # max clip clamp
        self.assertEqual([c["start"] for c in plans["B"]], [70.0])
        self.assertAlmostEqual(plans["B"][0]["frame_at"], 71.5)  # frame at midpoint

    def test_deterministic(self):
        a = evidence_cards.plan_cards(self.SEGS)
        b = evidence_cards.plan_cards(list(reversed(self.SEGS)))
        self.assertEqual(a, b)

    def test_spread_prefers_distant_segments(self):
        segs = [{"speaker": "A", "start": s, "end": s + 3.0, "text": ""}
                for s in (0.0, 2.0 + 3.5, 100.0)]   # non-overlapping trio, two clumped
        plans = evidence_cards.plan_cards(segs, clips_per_speaker=2)
        picked = {c["start"] for c in plans["A"]}
        self.assertIn(100.0, picked)             # the far one beats the clumped twin

    def test_invalid_limits_refused(self):
        with self.assertRaises(ValueError):
            evidence_cards.plan_cards([], clips_per_speaker=0)


class TestCharactersGlue(unittest.TestCase):
    """Pure pieces of the glue module only — job-touching paths need live fixtures and
    are exercised by the end-to-end run instead."""

    def test_guess_series_strips_episode_tag(self):
        from autodub import characters
        self.assertEqual(
            characters.guess_series(
                "Example.Show.S01E01.1080p.mkv"),
            "Example Show")
        self.assertEqual(characters.guess_series("plain-file.mkv"), "plain-file")
        self.assertEqual(characters.guess_series(""), "untitled-series")


    def test_video_window_expand_and_clamp(self):
        from autodub.review import _video_window
        # a 2s one-liner expands to the 5s minimum around its midpoint
        self.assertEqual(_video_window(10.0, 12.0, 5.0, 30.0), (8.5, 13.5))
        # a monologue clamps to the 30s maximum
        self.assertEqual(_video_window(100.0, 200.0, 5.0, 30.0), (100.0, 130.0))
        # early segment never goes negative, still gets the full minimum
        self.assertEqual(_video_window(0.0, 1.0, 5.0, 30.0), (0.0, 5.0))

    def test_cards_carry_segment_index(self):
        plans = evidence_cards.plan_cards(
            [{"i": 7, "speaker": "A", "start": 0.0, "end": 3.0, "text": "hi"}])
        self.assertEqual(plans["A"][0]["i"], 7)


if __name__ == "__main__":
    unittest.main()
