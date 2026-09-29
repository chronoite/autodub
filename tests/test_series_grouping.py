# -*- coding: utf-8 -*-
"""Hermetic contracts for the DubProjects cross-episode clustering (_cluster).

After single-link joining produced a mixed 15-speaker group, joins became complete-linkage at the identity (match) threshold. These tests pin the two
properties that rule bought: no transitive chaining through a shared middle member, and
no joins below the match threshold even when they clear the old ask threshold.
"""
import unittest

from autodub import series, voice_bank


def _inst(name, centroid):
    return {"job_id": f"job-{name}", "episode": f"S01E0{len(name)}", "speaker": name,
            "centroid": centroid, "cards": [], "weight": 1.0}


class SeriesClusterContracts(unittest.TestCase):
    def test_identical_centroids_form_one_group(self):
        got = series._cluster([_inst("a", [1.0, 0.0]), _inst("bb", [1.0, 0.0])])
        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0]["members"]), 2)
        self.assertEqual(got[0]["members"][1]["cohesion"], 1.0)

    def test_no_transitive_chaining_through_a_middle_member(self):
        # b sits between a and c: both match b well, but a-c is clearly two people.
        # Single-link at the rep would chain all three; complete linkage must not.
        a = [1.0, 0.0, 0.0]
        b = [0.98, 0.199, 0.0]     # cos(a,b) ~ 0.98
        c = [0.92, 0.392, 0.0]     # cos(b,c) ~ 0.98, cos(a,c) ~ 0.92 < match
        got = series._cluster([_inst("b", b), _inst("a", a), _inst("ccc", c)])
        sizes = sorted(len(cl["members"]) for cl in got)
        self.assertEqual(sizes, [1, 2], f"chained: {sizes}")

    def test_old_ask_band_scores_do_not_join(self):
        # cos ~ 0.92: above the old ask join (0.90), below identity grade (0.95).
        got = series._cluster([_inst("a", [1.0, 0.0]), _inst("bb", [0.92, 0.392])])
        self.assertEqual(len(got), 2)

    def test_cohesion_is_the_worst_link(self):
        a = [1.0, 0.0]
        b = [0.999, 0.0447]        # cos(a,b) ~ 0.999
        c = [0.996, 0.0894]        # cos(a,c) ~ 0.996, cos(b,c) ~ 0.999
        got = series._cluster([_inst("a", a), _inst("bb", b), _inst("ccc", c)])
        self.assertEqual(len(got), 1)
        worst = got[0]["members"][2]["cohesion"]
        self.assertLessEqual(worst, 0.997)   # min(cos(a,c), cos(b,c)) = cos(a,c)
        self.assertGreaterEqual(worst, voice_bank.VOICE_BANK_MATCH_THRESHOLD)


class GroupConfidenceContracts(unittest.TestCase):
    @staticmethod
    def _member(cohesion, solo_s):
        return {"cohesion": cohesion,
                "cards": [{"start": 0.0, "end": solo_s}] if solo_s else []}

    def test_solo_group_gets_no_light(self):
        got = series._confidence([self._member(1.0, 5.0)])
        self.assertEqual(got["level"], "solo")
        self.assertIsNone(got["worst_link"])

    def test_green_needs_deep_link_and_real_solo_audio(self):
        got = series._confidence([self._member(1.0, 6.0),
                                  self._member(0.97, 5.0), self._member(0.98, 4.0)])
        self.assertEqual(got["level"], "green")
        self.assertEqual(got["worst_link"], 0.97)

    def test_link_hugging_the_join_bar_is_red(self):
        got = series._confidence([self._member(1.0, 6.0), self._member(0.951, 5.0)])
        self.assertEqual(got["level"], "red")

    def test_thin_solo_audio_is_red_even_with_a_strong_link(self):
        got = series._confidence([self._member(1.0, 6.0), self._member(0.99, 1.0)])
        self.assertEqual(got["level"], "red")

    def test_between_bands_is_amber(self):
        got = series._confidence([self._member(1.0, 6.0), self._member(0.96, 5.0)])
        self.assertEqual(got["level"], "amber")


if __name__ == "__main__":
    unittest.main()
