"""Song-skip contracts (EXPERIMENTAL feature).

Pure unit tests — no media, no GPU, no ffmpeg calls. Guards the promises that matter:
songs are detected structurally, skipping is policy-gated (legacy jobs unchanged),
cached song dubs are purged on resume, and detection can never kill an analysis.
"""
import tempfile
import unittest
from pathlib import Path

from autodub import pipeline
from autodub.song_detect import (
    _ass_seconds,
    _is_song_label,
    detect_songs,
    mark_song_segments,
    merge_ranges,
)
from autodub.state import normalize_job


class SongLabelTests(unittest.TestCase):
    def test_song_labels_match(self):
        for label in ("OP", "op1", "Opening (TV size)", "ED_Romaji", "Karaoke-fx",
                      "NCOP", "NCED2", "Insert Song", "Lyrics-top", "ED"):
            self.assertTrue(_is_song_label(label), label)

    def test_dialogue_labels_do_not_match(self):
        for label in ("Default", "Default-Top", "Main Dialogue", "Signs", "Italics",
                      "Flashback", "opera house sign"):   # 'opera' must not hit 'op'
            self.assertFalse(_is_song_label(label), label)

    def test_ass_timestamps(self):
        self.assertAlmostEqual(_ass_seconds("0:01:30.50"), 90.5)
        self.assertIsNone(_ass_seconds("not-a-time"))


class RangeTests(unittest.TestCase):
    def test_merge_and_min_length(self):
        merged = merge_ranges([
            (10.0, 12.0, "subtitle-style"),
            (13.0, 15.0, "subtitle-style"),   # within 2s gap -> merged
            (100.0, 100.4, "chapter"),        # below MIN_RANGE_SECONDS -> dropped
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0]["start"], merged[0]["end"]), (10.0, 15.0))

    def test_marking_respects_overlap_fraction_and_clears_stale(self):
        ranges = merge_ranges([(0.0, 60.0, "chapter")])
        segments = [
            {"i": 0, "start": 5.0, "end": 10.0},                       # fully inside
            {"i": 1, "start": 58.0, "end": 70.0},                      # <50% inside
            {"i": 2, "start": 200.0, "end": 205.0, "song_skip": True,  # stale flag
             "song_reason": "chapter"},
        ]
        marked = mark_song_segments(segments, ranges)
        self.assertEqual(marked, 1)
        self.assertTrue(segments[0].get("song_skip"))
        self.assertFalse(segments[1].get("song_skip", False))
        self.assertFalse(segments[2].get("song_skip", False))   # cleared, not sticky

    def test_detection_fails_open(self):
        with tempfile.TemporaryDirectory() as work:
            summary = detect_songs(Path(work) / "missing.mkv", Path(work), [])
        self.assertIn(summary["status"], ("failed", "none-found"))
        self.assertEqual(summary["segments_skipped"], 0)
        self.assertTrue(summary["experimental"])

    def test_detector_failure_clears_stale_flags(self):
        """Re-detection failure must not retain stale song_skip flags.
        Detector failure is MOCKED to raise (a missing-file path merely returns
        empty ranges and never reaches the exception handler)."""
        from unittest import mock
        from autodub import song_detect
        segments = [{"i": 0, "start": 1.0, "end": 2.0, "song_skip": True, "song_reason": "chapter"}]
        with mock.patch.object(song_detect, "chapter_song_ranges", side_effect=RuntimeError("boom")):
            with tempfile.TemporaryDirectory() as work:
                summary = song_detect.detect_songs(Path(work) / "x.mkv", Path(work), segments)
        self.assertEqual(summary["status"], "failed")
        self.assertNotIn("song_skip", segments[0])
        self.assertNotIn("song_reason", segments[0])

    def test_marking_uses_raw_ranges_not_merged_gaps(self):
        """A spoken aside between two karaoke cues must NOT be counted as song.
        Exercises detect_songs() itself (calling mark_song_segments directly also
        passed the OLD merged-range implementation — this must pin the real path)."""
        from unittest import mock
        from autodub import song_detect
        cues = [(10.0, 12.0, "subtitle-style"), (13.5, 15.5, "subtitle-style")]
        segments = [
            {"i": 0, "start": 12.1, "end": 13.4},   # spoken aside entirely inside the 1.5s gap
            {"i": 1, "start": 10.2, "end": 11.8},   # genuinely inside a cue
        ]
        with mock.patch.object(song_detect, "chapter_song_ranges", return_value=[]), \
             mock.patch.object(song_detect, "ass_song_ranges", return_value=cues):
            with tempfile.TemporaryDirectory() as work:
                summary = song_detect.detect_songs(Path(work) / "x.mkv", Path(work), segments)
        self.assertEqual(summary["segments_skipped"], 1)
        self.assertFalse(segments[0].get("song_skip", False),
                         "spoken aside was swallowed by merged song ranges")
        self.assertTrue(segments[1].get("song_skip"))


class PipelineGatingTests(unittest.TestCase):
    def _job(self, policy):
        return {"settings": {"song_policy": policy},
                "segments": [{"i": 7, "song_skip": True}, {"i": 8}]}

    def test_skip_requires_policy_on(self):
        job = self._job("skip-detected-v1")
        self.assertTrue(pipeline._song_skipped(job, job["segments"][0]))
        self.assertFalse(pipeline._song_skipped(job, job["segments"][1]))
        legacy = self._job("dub-all-v1")
        self.assertFalse(pipeline._song_skipped(legacy, legacy["segments"][0]))

    def test_legacy_jobs_default_to_dub_all(self):
        job = normalize_job({"settings": {}})
        self.assertEqual(job["settings"]["song_policy"], "dub-all-v1")

    def test_drop_song_lines_purges_cached_dubs(self):
        job = self._job("skip-detected-v1")
        with tempfile.TemporaryDirectory() as tmp:
            lines = Path(tmp)
            (lines / "line-00007.wav").write_bytes(b"cached song dub")
            (lines / "line-00008.wav").write_bytes(b"kept dialogue")
            dropped = pipeline._drop_song_lines(job, lines)
            self.assertEqual(dropped, 1)
            self.assertFalse((lines / "line-00007.wav").exists())
            self.assertTrue((lines / "line-00008.wav").exists())


class SongSourceContracts(unittest.TestCase):
    """Song fixes that need heavy runtime mocking are pinned at source level instead."""

    SRC = (Path(__file__).resolve().parents[1] / "src" / "autodub" / "pipeline.py").read_text(encoding="utf-8")

    def test_repair_refuses_song_skipped_lines(self):
        self.assertIn("turn Song skip off in Run settings", self.SRC)

    def test_remix_purges_and_skips_song_lines(self):
        remix = self.SRC.split("def remix_existing", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_drop_song_lines", remix)
        self.assertIn("_song_skipped", remix)
        self.assertIn("_restore_song_vocals", remix)

    def test_quality_render_restores_song_vocals(self):
        # The quality render is split into GPU and CPU halves: songs are restored in the CPU tail
        tail = self.SRC.split("def _render_quality_post", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_restore_song_vocals", tail)
        # and the single-job path must still run BOTH halves in order
        combined = self.SRC.split("def _render_quality(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_render_quality_synth", combined)
        self.assertIn("_render_quality_post", combined)

    def test_review_ui_shows_song_flags(self):
        app_js = (Path(__file__).resolve().parents[1] / "src" / "autodub" / "web" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("song_skip", app_js)
        self.assertIn("SONG SKIP (EXPERIMENTAL)", app_js)

    def test_manual_marking_is_wired_end_to_end(self):
        # server route + GUI control + replay-after-analyze.
        root = Path(__file__).resolve().parents[1]
        self.assertIn("mark-songs", (root / "src" / "autodub" / "server.py").read_text(encoding="utf-8"))
        app_js = (root / "src" / "autodub" / "web" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("mark-songs", app_js)
        self.assertIn("song-mark-mode", app_js)
        self.assertIn("song-mark-mode", (root / "src" / "autodub" / "web" / "static" / "index.html").read_text(encoding="utf-8"))
        self.assertEqual(self.SRC.count("replay_manual_marks(job)"), 2)  # both analyze paths


class ManualMarkTests(unittest.TestCase):
    """Manual range-marking + durable replay after re-analysis."""

    def _segments(self):
        return [{"i": i, "start": float(i), "end": float(i) + 1.0} for i in range(6)]

    def test_mark_sets_manual_flags_inclusively(self):
        from autodub.song_detect import apply_song_range
        segments = self._segments()
        self.assertEqual(apply_song_range(segments, 1, 3, True), 3)
        for segment in segments:
            if 1 <= segment["i"] <= 3:
                self.assertTrue(segment["song_skip"])
                self.assertEqual(segment["song_reason"], "manual")
            else:
                self.assertNotIn("song_skip", segment)
        self.assertEqual(apply_song_range(segments, 2, 2, True), 0)  # idempotent

    def test_mark_skips_already_flagged_and_keeps_detector_reason(self):
        from autodub.song_detect import apply_song_range
        segments = self._segments()
        segments[2].update({"song_skip": True, "song_reason": "chapter"})
        self.assertEqual(apply_song_range(segments, 1, 3, True), 2)
        self.assertEqual(segments[2]["song_reason"], "chapter")

    def test_unmark_clears_detector_flags_too(self):
        from autodub.song_detect import apply_song_range
        segments = self._segments()
        segments[1].update({"song_skip": True, "song_reason": "chapter"})
        segments[2].update({"song_skip": True, "song_reason": "manual"})
        self.assertEqual(apply_song_range(segments, 0, 5, False), 2)
        for segment in segments:
            self.assertNotIn("song_skip", segment)
            self.assertNotIn("song_reason", segment)

    def test_replay_restores_marks_after_wholesale_rebuild(self):
        from autodub.song_detect import replay_manual_marks
        job = {"segments": self._segments(),
               "manual_song_ranges": [{"start": 0, "end": 2, "mark": True},
                                     {"start": 1, "end": 1, "mark": False}]}
        self.assertEqual(replay_manual_marks(job), 4)  # 3 marked, then 1 unmarked
        flagged = [s["i"] for s in job["segments"] if s.get("song_skip")]
        self.assertEqual(flagged, [0, 2])

    def test_wrapper_validates_and_journals(self):
        import shutil
        from autodub.song_detect import mark_song_range
        from autodub.state import default_job, job_dir, load_job, save_job
        job_id = "dub-manual-song-mark-test"
        try:
            job = default_job(job_id, ".mp4", 1, "0" * 64)
            save_job(job)
            with self.assertRaisesRegex(ValueError, "analyze the job"):
                mark_song_range(job_id, start=0, end=1, mark=True)
            job["segments"] = self._segments()
            save_job(job)
            with self.assertRaisesRegex(ValueError, "invalid segment range"):
                mark_song_range(job_id, start=90, end=99, mark=True)
            result = mark_song_range(job_id, start=1, end=3, mark=True)
            self.assertEqual(result["changed"], 3)
            self.assertEqual(result["marked_total"], 3)
            self.assertTrue(result["policy_active"])  # new jobs default skip-detected-v1
            stored = load_job(job_id)
            self.assertEqual(stored["manual_song_ranges"],
                             [{"start": 1, "end": 3, "mark": True}])
            self.assertEqual(stored["song_summary"]["status"], "manually-marked")
            self.assertTrue(any("marked as song manually" in e["message"]
                                for e in stored["events"]))
            # no-op does not journal or churn
            again = mark_song_range(job_id, start=1, end=3, mark=True)
            self.assertEqual(again["changed"], 0)
            self.assertEqual(len(load_job(job_id)["manual_song_ranges"]), 1)
            # running jobs are refused
            stored = load_job(job_id)
            stored["status"] = "running"
            save_job(stored)
            with self.assertRaisesRegex(ValueError, "current stage"):
                mark_song_range(job_id, start=1, end=3, mark=False)
        finally:
            shutil.rmtree(job_dir(job_id), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
