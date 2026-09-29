"""Strict-timing contracts: silence trim, preserve-capped fit,
rendered-overlap QC. Uses real ffmpeg on tiny generated wavs (fast, local)."""
from __future__ import annotations

import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autodub.config import WORK_ROOT  # noqa: E402

from autodub.media import align_line, trim_edge_silence
from autodub.pipeline import _next_rendered_starts
from autodub.policies import get_timing_policy
from autodub.workflow import audit_timing
from tests import requires_ffmpeg  # noqa: E402


def _tone_wav(path: Path, seconds: float, *, lead_silence: float = 0.0,
              tail_silence: float = 0.0, rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = bytearray()
        frames += b"\x00\x00" * int(lead_silence * rate)
        for index in range(int(seconds * rate)):
            frames += struct.pack("<h", int(9000 * math.sin(2 * math.pi * 220 * index / rate)))
        frames += b"\x00\x00" * int(tail_silence * rate)
        handle.writeframes(bytes(frames))


@requires_ffmpeg
class SilenceTrimTests(unittest.TestCase):
    def test_edge_silence_is_removed_with_shoulders(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            source = tmp / "padded.wav"
            _tone_wav(source, 0.5, lead_silence=0.4, tail_silence=0.6)
            trimmed = trim_edge_silence(source, tmp / "trimmed.wav")
            self.assertLess(trimmed, 0.75)     # padding gone (was 1.5s total)
            self.assertGreater(trimmed, 0.45)  # tone intact


@requires_ffmpeg
class PreserveCappedTests(unittest.TestCase):
    def test_cap_never_crosses_next_line_start(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            source = tmp / "long.wav"
            _tone_wav(source, 2.0)
            result = align_line(source, tmp / "aligned.wav", 1.0, 0.9, 1.15,
                                fit_mode="preserve-capped", max_output_s=1.2)
            self.assertAlmostEqual(result["output_seconds"], 1.2, delta=0.08)
            self.assertGreater(result["capped_seconds"], 0.4)
            self.assertEqual(result["max_output_seconds"], 1.2)

    def test_fitting_line_is_not_capped(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            source = tmp / "short.wav"
            _tone_wav(source, 0.8)
            result = align_line(source, tmp / "aligned.wav", 1.0, 0.9, 1.15,
                                fit_mode="preserve-capped", max_output_s=3.0)
            self.assertNotIn("capped_seconds", result)
            self.assertGreater(result["output_seconds"], 0.6)

    def test_trim_and_cap_compose(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            source = tmp / "padded-long.wav"
            _tone_wav(source, 0.8, lead_silence=0.5, tail_silence=0.5)
            result = align_line(source, tmp / "aligned.wav", 1.0, 0.9, 1.15,
                                fit_mode="preserve-capped", trim_silence=True,
                                max_output_s=2.0)
            # trimming alone brings 1.8s under the 2.0 cap — no cap needed
            self.assertNotIn("capped_seconds", result)
            self.assertGreater(result["trimmed_silence_seconds"], 0.7)
            self.assertLess(result["output_seconds"], 1.1)

    def test_policy_registry_has_the_collision_proof_mode(self) -> None:
        policy = get_timing_policy("capped-breath-v1")
        self.assertEqual(policy["fit_mode"], "preserve-capped")
        self.assertTrue(policy["trim_silence"])
        self.assertLessEqual(policy["max_tempo"], 1.15)


class RenderedOverlapQcTests(unittest.TestCase):
    def test_rendered_overlap_and_capped_flags(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            lines = Path(temporary)
            _tone_wav(lines / "line-00000.wav", 3.0)
            _tone_wav(lines / "line-00001.wav", 1.0)
            segments = [
                {"i": 0, "start": 0.0, "end": 1.0, "translation": "a",
                 "alignment": {"output_seconds": 3.0, "capped_seconds": 0.5}},
                {"i": 1, "start": 2.0, "end": 3.0, "translation": "b",
                 "alignment": {"output_seconds": 1.0}},
            ]
            summary = audit_timing(segments, lines)
            self.assertEqual(summary["rendered_overlap"], 1)
            self.assertIn("rendered-overlap", segments[1]["qc"]["flags"])
            self.assertEqual(summary["capped"], 1)
            self.assertIn("hard-capped", segments[0]["qc"]["flags"])

    def test_no_false_overlap_when_lines_fit(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            lines = Path(temporary)
            _tone_wav(lines / "line-00000.wav", 0.9)
            _tone_wav(lines / "line-00001.wav", 0.9)
            segments = [
                {"i": 0, "start": 0.0, "end": 1.0, "translation": "a",
                 "alignment": {"output_seconds": 0.9}},
                {"i": 1, "start": 2.0, "end": 3.0, "translation": "b",
                 "alignment": {"output_seconds": 0.9}},
            ]
            summary = audit_timing(segments, lines)
            self.assertEqual(summary["rendered_overlap"], 0)
            self.assertEqual(summary["capped"], 0)


class NextStartBudgetTests(unittest.TestCase):
    def test_budgets_span_gaps_over_missing_lines(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            lines = Path(temporary)
            for index in (0, 2):
                _tone_wav(lines / f"line-{index:05d}.wav", 0.3)
            segments = [
                {"i": 0, "start": 0.0, "end": 1.0},
                {"i": 1, "start": 1.0, "end": 2.0},   # no rendered line (muted)
                {"i": 2, "start": 5.0, "end": 6.0},
            ]
            job = {"settings": {"song_policy": "skip-detected-v1"}, "segments": segments}
            budgets = _next_rendered_starts(job, segments, lines)
            # line 0's room runs to line 2's start (the muted line frees its slot)
            self.assertAlmostEqual(budgets[0], 4.99, places=2)
            self.assertNotIn(1, budgets)
            self.assertNotIn(2, budgets)  # last rendered line: unbounded

    def test_song_boundaries_and_tiny_gaps(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            lines = Path(temporary)
            for index in (0, 1, 3):
                _tone_wav(lines / f"line-{index:05d}.wav", 0.3)
            segments = [
                {"i": 0, "start": 0.0, "end": 1.0},
                # near-simultaneous next start: capping would delete content —
                # leave uncapped, QC flags the overlap instead
                {"i": 1, "start": 0.2, "end": 1.2},
                {"i": 2, "start": 6.0, "end": 9.0, "song_skip": True},  # restored singing
                {"i": 3, "start": 3.0, "end": 4.0},
            ]
            job = {"settings": {"song_policy": "skip-detected-v1"}, "segments": segments}
            budgets = _next_rendered_starts(job, segments, lines)
            self.assertNotIn(0, budgets)          # 0.2s gap < 0.35 -> uncapped
            # line 3's tail must stop at the SONG start (6.0), not run unbounded
            self.assertAlmostEqual(budgets[3], 2.99, places=2)


class SongSpanSourceContract(unittest.TestCase):
    def test_song_restore_groups_contiguous_cues(self) -> None:
        source = (ROOT / "src" / "autodub" / "pipeline.py").read_text(encoding="utf-8")
        body = source.split("def _restore_song_vocals", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("Whole-span restore", body)
        self.assertIn("<= 5.0", body)
        self.assertIn('songpass / f"span-', body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
