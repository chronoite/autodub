"""Hermetic contracts for non-verbal passthrough interval math (no ffmpeg)."""
from __future__ import annotations

import unittest

from autodub import nonverbal


class SilenceScanContracts(unittest.TestCase):
    def test_parse_and_invert(self):
        stderr = (
            "[silencedetect @ x] silence_start: 1.5\n"
            "[silencedetect @ x] silence_end: 4.0 | silence_duration: 2.5\n"
            "[silencedetect @ x] silence_start: 9.0\n"
            "[silencedetect @ x] silence_end: 10.0 | silence_duration: 1.0\n"
        )
        silences = nonverbal.parse_silences(stderr)
        self.assertEqual(silences, [(1.5, 4.0), (9.0, 10.0)])
        self.assertEqual(nonverbal.voiced_windows(silences, 12.0),
                         [(0.0, 1.5), (4.0, 9.0), (10.0, 12.0)])


class SubtractionContracts(unittest.TestCase):
    def test_leaves_the_scream(self):
        voiced = [(4.0, 9.0)]
        covers = [(4.0, 6.0), (7.0, 8.5)]
        self.assertEqual(nonverbal.subtract_covered(voiced, covers, guard=0.0),
                         [(6.0, 7.0), (8.5, 9.0)])
        self.assertEqual(nonverbal.subtract_covered(voiced, covers, guard=0.1),
                         [(6.1, 6.9), (8.6, 9.0)])

    def test_fully_covered_window_disappears(self):
        self.assertEqual(nonverbal.subtract_covered([(4.0, 6.0)], [(3.5, 6.5)], guard=0.0), [])


class WindowHygieneContracts(unittest.TestCase):
    def test_merge_and_filter(self):
        windows = [(1.0, 1.4), (1.5, 2.0), (5.0, 5.1), (8.0, 9.0)]
        self.assertEqual(nonverbal.merge_and_filter(windows, merge_gap=0.2, min_s=0.3),
                         [(1.0, 2.0), (8.0, 9.0)])

    def test_cap_total_reports_dropped(self):
        windows = [(0.0, 60.0), (100.0, 160.0), (200.0, 260.0)]
        kept, dropped = nonverbal.cap_total(windows, max_total_s=150.0)
        self.assertEqual(kept, [(0.0, 60.0), (100.0, 160.0)])
        self.assertEqual(dropped, 60.0)


if __name__ == "__main__":
    unittest.main()
