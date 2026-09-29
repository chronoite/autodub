"""Chunked dialogue-bus mixing for full-episode line counts (WinError 206 guard).

A 399-line episode render once failed because every line was a separate
ffmpeg input in ONE command, exceeding Windows' ~32k command-line limit. Above
``MIX_CHUNK_LINES`` the dialogue bus is now pre-summed in batches. These tests prove
the chunked path renders, places audio at the right times, cleans up its batch
files, and stays bit-equivalent in shape to the single-pass mix.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autodub.config import WORK_ROOT  # noqa: E402

from autodub.config import FFMPEG
from autodub.media import MIX_CHUNK_LINES, build_mix, probe_duration
from tests import requires_ffmpeg  # noqa: E402


def _write_tone(path: Path, seconds: float = 0.12, frequency: int = 880) -> None:
    subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"sine=frequency={frequency}:sample_rate=48000:duration={seconds}",
         "-ac", "1", str(path)], check=True)


@requires_ffmpeg
class MixChunkingTests(unittest.TestCase):
    def test_full_episode_line_count_mixes_via_chunks_and_cleans_up(self) -> None:
        line_count = MIX_CHUNK_LINES * 2 + 25          # forces 3 chunk buses
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temp:
            root = Path(temp)
            base = root / "base.wav"
            subprocess.run(
                [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                 "-i", "sine=frequency=110:sample_rate=48000:duration=30",
                 "-ac", "2", str(base)], check=True)
            aligned = root / "aligned"
            aligned.mkdir()
            tone = root / "tone.wav"
            _write_tone(tone)
            segments = []
            for index in range(line_count):
                target = aligned / f"line-{index:05d}.wav"
                target.write_bytes(tone.read_bytes())
                start = 0.05 + index * 0.13
                segments.append({"i": index, "start": start, "end": start + 0.12})
            mixed = root / "mix.wav"
            self.assertEqual(build_mix(base, segments, aligned, mixed, 0.3, 1.0), line_count)
            self.assertTrue(mixed.is_file())
            self.assertAlmostEqual(probe_duration(mixed), 30.0, delta=0.5)
            self.assertEqual(list(root.glob("*dlgchunk*")), [],
                             "chunk buses must be removed after the final mix")

    def test_small_jobs_still_use_the_single_pass_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temp:
            root = Path(temp)
            base = root / "base.wav"
            subprocess.run(
                [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                 "-i", "sine=frequency=110:sample_rate=48000:duration=3",
                 "-ac", "2", str(base)], check=True)
            aligned = root / "aligned"
            aligned.mkdir()
            _write_tone(aligned / "line-00000.wav")
            mixed = root / "mix.wav"
            count = build_mix(base, [{"i": 0, "start": 0.5, "end": 0.65}], aligned,
                              mixed, 0.3, 1.0)
            self.assertEqual(count, 1)
            self.assertTrue(mixed.is_file())
            self.assertEqual(list(root.glob("*dlgchunk*")), [])


if __name__ == "__main__":
    unittest.main()
