from __future__ import annotations

import unittest

from autodub.subtitles import map_cues_to_segments, parse_ass, parse_srt, parse_webvtt


class SubtitleContractTests(unittest.TestCase):
    def test_srt_parser_removes_formatting_without_losing_words(self) -> None:
        cues = parse_srt(
            "1\n00:00:01,200 --> 00:00:02,500\n<i>Hello</i>\\Nthere.\n\n"
            "2\n00:00:03.000 --> 00:00:04.000\n{\\an8}Second line.\n"
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["text"], "Hello there.")
        self.assertEqual(cues[1]["text"], "Second line.")

    def test_overlap_mapping_preserves_unmatched_local_translation(self) -> None:
        segments = [
            {"i": 0, "start": 1.0, "end": 2.0, "translation": "Local fallback."},
            {"i": 1, "start": 5.0, "end": 6.0, "translation": "Keep me."},
        ]
        cues = [{"start": 1.1, "end": 2.1, "text": "Professional subtitle."}]
        self.assertEqual(map_cues_to_segments(segments, cues), 1)
        self.assertEqual(segments[0]["translation"], "Professional subtitle.")
        self.assertEqual(segments[0]["translation_source"], "embedded-english-subtitle")
        self.assertEqual(segments[1]["translation"], "Keep me.")

    def test_ass_name_is_preserved_but_blank_or_missing_names_are_omitted(self) -> None:
        cues = parse_ass(
            "[Script Info]\nTitle: Synthetic\n\n[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:01.20,0:00:02.50,Default,Alice,0,0,0,,{\\i1}Hello{\\i0}\\Nthere.\n"
            "Dialogue: 0,0:00:03.00,0:00:04.00,Default,   ,0,0,0,,Blank name.\n"
            "Dialogue: 0,0:00:05.00,0:00:06.00,Default,Bob,0,0,0,,Named.\n"
        )
        self.assertEqual(cues[0]["speaker"], "Alice")
        self.assertEqual(cues[0]["text"], "Hello there.")
        self.assertNotIn("speaker", cues[1])
        self.assertEqual(cues[2]["speaker"], "Bob")

    def test_webvtt_voice_is_preserved_but_blank_or_missing_voice_is_omitted(self) -> None:
        cues = parse_webvtt(
            "WEBVTT\n\n00:01.000 --> 00:02.000\n<v Alice>Hello &amp; welcome.</v>\n\n"
            "00:03.000 --> 00:04.000\n<v >Blank voice.</v>\n\n"
            "00:05.000 --> 00:06.000\nNo voice.\n"
        )
        self.assertEqual(cues[0], {"start": 1.0, "end": 2.0, "text": "Hello & welcome.", "speaker": "Alice"})
        self.assertNotIn("speaker", cues[1])
        self.assertNotIn("speaker", cues[2])

    def test_srt_stays_unattributed_and_subtitle_speaker_never_changes_diarization(self) -> None:
        cues = parse_srt("1\n00:00:01,000 --> 00:00:02,000\nPlain SRT.\n")
        self.assertNotIn("speaker", cues[0])
        segments = [{"i": 0, "start": 1.0, "end": 2.0, "speaker": "speaker-09"}]
        attributed = [{**cues[0], "speaker": "Source Character"}]
        self.assertEqual(map_cues_to_segments(segments, cues, attributed), 1)
        self.assertEqual(segments[0]["speaker"], "speaker-09")
        self.assertEqual(segments[0]["subtitle_speaker"], "Source Character")


if __name__ == "__main__":
    unittest.main()
