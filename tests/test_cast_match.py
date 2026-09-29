# -*- coding: utf-8 -*-
"""Hermetic contracts for the cast-pack episode-signature matcher."""
import unittest

from autodub.cast_match import hints


CAST = [
    {"name": "Marco", "role": "main", "episodes": list(range(1, 13))},
    {"name": "Nora", "role": "minor", "episodes": [1, 2, 5, 6]},
    {"name": "Tobias", "role": "supporting", "episodes": [2, 6, 9, 10]},
    {"name": "NoData", "role": "minor", "episodes": []},
]


class CastMatchContracts(unittest.TestCase):
    def test_exact_signature_beats_the_everywhere_character(self):
        got = hints([1, 2, 5, 6], CAST)
        self.assertEqual(got[0]["name"], "Nora")   # exact set > Marco's superset

    def test_full_season_group_matches_the_lead(self):
        got = hints(list(range(1, 13)), CAST)
        self.assertEqual(got[0]["name"], "Marco")

    def test_poor_containment_is_silent(self):
        self.assertEqual(hints([3, 4, 7, 8], [CAST[1]]), [])   # 0/4 inside Nora

    def test_single_episode_groups_get_no_hints(self):
        self.assertEqual(hints([2], CAST), [])

    def test_characters_without_episode_data_are_skipped(self):
        self.assertEqual(hints([1, 2], [CAST[3]]), [])
