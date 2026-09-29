# -*- coding: utf-8 -*-
"""Hermetic contract for the prewarm card walk (pure part only; no media, no threads)."""
import unittest

from autodub.evidence_prewarm import card_keys


class PrewarmCardKeys(unittest.TestCase):
    def test_dedupes_and_keeps_view_order(self):
        view = {"groups": [
            {"cards": [{"job_id": "j1", "i": 3}, {"job_id": "j2", "i": 0}]},
            {"cards": [{"job_id": "j1", "i": 3}, {"job_id": "j3", "i": 7}]},
            {"cards": []},
        ]}
        self.assertEqual(card_keys(view), [("j1", 3), ("j2", 0), ("j3", 7)])

    def test_empty_view(self):
        self.assertEqual(card_keys({}), [])


if __name__ == "__main__":
    unittest.main()


class RosterCardWalk(unittest.TestCase):
    def test_member_face_cards_are_warmed_too(self):
        view = {"groups": [{
            "cards": [{"job_id": "j1", "i": 3}],
            "members": [
                {"job_id": "j1", "cards": [{"i": 3}, {"i": 9}]},   # i=3 dup, i=9 NOT face
                {"job_id": "j2", "cards": [{"i": 7}]},
                {"job_id": "j3", "cards": []},
            ]}]}
        from autodub.evidence_prewarm import card_keys
        self.assertEqual(card_keys(view), [("j1", 3), ("j2", 7)])
