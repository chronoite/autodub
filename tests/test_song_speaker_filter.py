# -*- coding: utf-8 -*-
"""Hermetic contracts: intro/outro speakers and song-flagged lines never reach the
character boards (theme songs are never dubbed)."""
import unittest

from autodub.characters import dialogue_segments, song_speakers


class SongFilterContracts(unittest.TestCase):
    JOB = {
        "song_speakers": ["SPEAKER_09"],
        "segments": [
            {"i": 0, "speaker": "SPEAKER_00", "start": 1.0, "end": 3.0},
            {"i": 1, "speaker": "SPEAKER_09", "start": 4.0, "end": 9.0},   # marked singer
            {"i": 2, "speaker": "SPEAKER_00", "start": 10.0, "end": 12.0,
             "song_skip": True},                                            # insert song line
            {"i": 3, "speaker": "SPEAKER_01", "start": 13.0, "end": 15.0},
        ],
    }

    def test_marked_speaker_and_flagged_lines_are_excluded(self):
        kept = [s["i"] for s in dialogue_segments(self.JOB)]
        self.assertEqual(kept, [0, 3])

    def test_song_speaker_set(self):
        self.assertEqual(song_speakers(self.JOB), {"SPEAKER_09"})
        self.assertEqual(song_speakers({}), set())

    def test_job_without_flags_is_untouched(self):
        job = {"segments": [{"i": 0, "speaker": "A", "start": 0, "end": 1}]}
        self.assertEqual(len(dialogue_segments(job)), 1)


if __name__ == "__main__":
    unittest.main()
