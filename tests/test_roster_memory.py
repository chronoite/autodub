"""Hermetic contracts for the roster's reviewer memory: cannot-link pairs keep
separated speakers apart in clustering, and rejection lookups are exact."""
import unittest

from autodub import series, voice_bank


def _inst(name, job, centroid):
    return {"job_id": job, "episode": "S01E01", "speaker": name,
            "centroid": centroid, "cards": [], "weight": 1.0}


class RosterMemoryContracts(unittest.TestCase):
    def test_separated_pair_never_remerges_despite_perfect_cosine(self):
        a = _inst("SPEAKER_00", "j1", [1.0, 0.0])
        b = _inst("SPEAKER_03", "j2", [1.0, 0.0])   # identical voice, reviewer said NOT same
        sep = {tuple(sorted((voice_bank._member_key("j1", "SPEAKER_00"),
                             voice_bank._member_key("j2", "SPEAKER_03"))))}
        got = series._cluster([a, b], sep)
        self.assertEqual(len(got), 2)

    def test_unseparated_identical_pair_still_merges(self):
        a = _inst("SPEAKER_00", "j1", [1.0, 0.0])
        b = _inst("SPEAKER_03", "j2", [1.0, 0.0])
        self.assertEqual(len(series._cluster([a, b], set())), 1)

    def test_separation_blocks_the_whole_cluster_not_just_the_pair(self):
        # c matches both a and b, but is separated from a -> c may not join {a, b}.
        a = _inst("A", "j1", [1.0, 0.0])
        b = _inst("B", "j2", [1.0, 0.0])
        c = _inst("C", "j3", [1.0, 0.0])
        sep = {tuple(sorted((voice_bank._member_key("j1", "A"),
                             voice_bank._member_key("j3", "C"))))}
        got = series._cluster([a, b, c], sep)
        sizes = sorted(len(cl["members"]) for cl in got)
        self.assertEqual(sizes, [1, 2])

    def test_is_rejected_is_exact(self):
        bank = {"rejections": [{"character_id": "ch_1", "job_id": "j1", "speaker": "S0"}]}
        self.assertTrue(voice_bank.is_rejected(bank, "ch_1", "j1", "S0"))
        self.assertFalse(voice_bank.is_rejected(bank, "ch_1", "j1", "S1"))
        self.assertFalse(voice_bank.is_rejected(bank, "ch_2", "j1", "S0"))
        self.assertFalse(voice_bank.is_rejected({}, "ch_1", "j1", "S0"))


if __name__ == "__main__":
    unittest.main()


class UnionContracts(unittest.TestCase):
    def test_must_link_joins_despite_low_cosine_and_reads_honest_cohesion(self):
        a = _inst("A", "j1", [1.0, 0.0])
        b = _inst("B", "j2", [0.0, 1.0])   # orthogonal voices — normal path never merges
        uni = {tuple(sorted((voice_bank._member_key("j1", "A"),
                             voice_bank._member_key("j2", "B"))))}
        got = series._cluster([a, b], set(), uni)
        self.assertEqual(len(got), 1)
        self.assertLess(got[0]["members"][1]["cohesion"], 0.5)   # honest, not faked green

    def test_no_union_means_no_merge_for_unlike_voices(self):
        a = _inst("A", "j1", [1.0, 0.0])
        b = _inst("B", "j2", [0.0, 1.0])
        self.assertEqual(len(series._cluster([a, b], set(), set())), 2)
