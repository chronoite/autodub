from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autodub.speaker_evidence import derive_speaker_evidence, normalize_speaker_count, suggest_duplicate_pairs
from autodub.state import default_job, public_job
from autodub.workers import quality_worker


class _Turn:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _Annotation:
    def __init__(self, turns: list[tuple[float, float, str]]):
        self.turns = turns

    def itertracks(self, yield_label: bool = False):
        assert yield_label
        for start, end, label in self.turns:
            yield _Turn(start, end), None, label

    def labels(self):
        return sorted({label for _, _, label in self.turns})


class SpeakerEvidenceTests(unittest.TestCase):
    def test_exact_voiced_threshold_is_eligible_and_short_chunk_is_excluded(self) -> None:
        segments = [
            {"i": 0, "speaker": "speaker-01", "start": 0.0, "end": 2.0, "voiced_duration": 1.1999},
            {"i": 1, "speaker": "speaker-01", "start": 3.0, "end": 4.2, "voiced_duration": 1.2},
        ]
        derived = derive_speaker_evidence(
            [{"speaker": "speaker-01", "centroid": [9.0, 9.0]}],
            [
                {"segment_index": 0, "speaker": "speaker-01", "embedding": [8.0, 8.0]},
                {"segment_index": 1, "speaker": "speaker-01", "embedding": [1.0, 3.0]},
            ], segments,
        )
        self.assertEqual(derived["repair_embeddings"][0]["centroid"], [1.0, 3.0])
        self.assertEqual(derived["manifest"]["speakers"][0]["eligible_chunk_count"], 1)
        self.assertEqual([item["segment_index"] for item in derived["manifest"]["speakers"][0]["segments"]], [1])

    def test_manifest_is_deterministic_and_diverse_with_borderline_samples(self) -> None:
        segments = [
            {"i": i, "speaker": "speaker-01", "start": float(i * 10), "end": float(i * 10 + 2),
             "voiced_duration": 1.2, "speaker_confidence": confidence}
            for i, confidence in enumerate([0.9, 0.1, 0.8, 0.2, 0.7, 0.6])
        ]
        first = derive_speaker_evidence([], [], segments)["manifest"]
        second = derive_speaker_evidence([], [], copy.deepcopy(segments))["manifest"]
        self.assertEqual(first, second)
        selected = first["speakers"][0]["segments"]
        self.assertEqual(len(selected), 5)
        self.assertEqual({item["segment_index"] for item in selected if item["borderline"]}, {1, 3})
        self.assertIn(5, {item["segment_index"] for item in selected})

    def test_low_evidence_badge_boundary_is_three_eligible_chunks(self) -> None:
        segments = [
            {"i": i, "speaker": speaker, "start": i, "end": i + 2, "voiced_duration": 1.2}
            for i, speaker in enumerate(["speaker-01", "speaker-01", "speaker-02", "speaker-02", "speaker-02"])
        ]
        speakers = {item["speaker"]: item for item in derive_speaker_evidence([], [], segments)["manifest"]["speakers"]}
        self.assertTrue(speakers["speaker-01"]["low_evidence"])
        self.assertFalse(speakers["speaker-02"]["low_evidence"])

    def test_no_eligible_chunks_falls_back_without_mutating_assignments(self) -> None:
        segments = [{"i": 0, "speaker": "speaker-01", "start": 0.0, "end": 3.0, "voiced_duration": 1.19}]
        original = copy.deepcopy(segments)
        derived = derive_speaker_evidence(
            [{"speaker": "speaker-01", "centroid": [1.0, 0.0]}], [], segments
        )
        self.assertEqual(segments, original)
        self.assertEqual(derived["repair_embeddings"][0]["source"], "diarizer_fallback")
        self.assertEqual(derived["repair_embeddings"][0]["centroid"], [1.0, 0.0])

    def test_synthetic_over_split_suggests_correct_pair_without_mutation(self) -> None:
        segments = [
            {"i": 0, "speaker": "speaker-01"},
            {"i": 1, "speaker": "speaker-01"},
            {"i": 2, "speaker": "speaker-02"},
            {"i": 3, "speaker": "speaker-03"},
        ]
        original = copy.deepcopy(segments)
        suggestions = suggest_duplicate_pairs(
            [
                {"speaker": "speaker-01", "centroid": [1.0, 0.0]},
                {"speaker": "speaker-02", "centroid": [0.999, 0.01]},
                {"speaker": "speaker-03", "centroid": [0.0, 1.0]},
            ],
            [],
            segments,
        )
        self.assertEqual([(item["speaker_a"], item["speaker_b"]) for item in suggestions],
                         [("speaker-01", "speaker-02")])
        self.assertEqual(suggestions[0]["evidence_counts"]["speaker_a_segments"], 2)
        self.assertEqual(suggestions[0]["evidence_counts"]["speaker_b_segments"], 1)
        self.assertEqual(segments, original)

    def test_overlap_veto_boundary(self) -> None:
        embeddings = [
            {"speaker": "speaker-01", "centroid": [1.0, 0.0]},
            {"speaker": "speaker-02", "centroid": [1.0, 0.0]},
        ]
        segments = [{"speaker": "speaker-01"}, {"speaker": "speaker-02"}]
        overlap = {"speaker_a": "speaker-01", "speaker_b": "speaker-02", "longest_run_ms": 299}
        self.assertEqual(len(suggest_duplicate_pairs(
            embeddings, [{**overlap, "total_ms": 299, "run_count": 1}], segments
        )), 1)
        self.assertEqual(suggest_duplicate_pairs(
            embeddings, [{**overlap, "total_ms": 300, "run_count": 1}], segments
        ), [])

    def test_count_contract_reaches_pyannote_call_and_embeddings_return(self) -> None:
        raw = _Annotation([(0.0, 2.0, "A"), (1.0, 1.2, "B"), (1.5, 1.9, "B")])
        exclusive = _Annotation([(0.0, 1.5, "A"), (1.5, 2.0, "B")])
        output = SimpleNamespace(
            speaker_diarization=raw,
            exclusive_speaker_diarization=exclusive,
            speaker_embeddings=np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
        )
        calls = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _path):
                return cls()

            def to(self, _device):
                return None

            def __call__(self, _audio, **kwargs):
                calls.append(kwargs)
                return output

        torch_module = types.ModuleType("torch")
        torch_module.device = lambda value: value
        torch_module.from_numpy = lambda value: value
        pyannote_module = types.ModuleType("pyannote")
        pyannote_audio_module = types.ModuleType("pyannote.audio")
        pyannote_audio_module.Pipeline = FakePipeline
        pyannote_module.audio = pyannote_audio_module

        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.yaml").write_text("synthetic", encoding="utf-8")
            payload = {
                "audio": "synthetic.wav",
                "segments": [{"i": 0, "start": 0.0, "end": 1.0},
                             {"i": 1, "start": 1.6, "end": 2.0}],
                "device": "cpu",
            }
            with (
                patch.dict(sys.modules, {"torch": torch_module, "pyannote": pyannote_module,
                                         "pyannote.audio": pyannote_audio_module}),
                patch.object(quality_worker, "PYANNOTE_MODEL", model),
                patch.object(quality_worker, "_read_pcm_wav",
                             return_value=(np.zeros((16, 1), dtype=np.float32), 16000)),
            ):
                exact = quality_worker.diarize({**copy.deepcopy(payload),
                                                "speaker_count": {"mode": "exact", "count": 2}})
                quality_worker.diarize({**copy.deepcopy(payload),
                                        "speaker_count": {"mode": "min-max", "min": 2, "max": 4}})

        self.assertEqual(calls, [{"num_speakers": 2}, {"min_speakers": 2, "max_speakers": 4}])
        self.assertEqual([item["speaker"] for item in exact["speaker_embeddings"]],
                         ["speaker-01", "speaker-02"])
        self.assertEqual(exact["speaker_overlap_evidence"][0]["total_ms"], 600)
        self.assertEqual(exact["speaker_overlap_evidence"][0]["longest_run_ms"], 400)

    def test_worker_json_protocol_round_trip_keeps_embeddings(self) -> None:
        result = {
            "segments": [{"i": 0, "speaker": "speaker-01"}],
            "speaker_embeddings": [{"speaker": "speaker-01", "centroid": [0.1, 0.2]}],
            "speaker_overlap_evidence": [],
        }
        stdout = io.StringIO()
        with (
            patch.object(quality_worker, "diarize", return_value=result),
            patch.object(quality_worker.sys, "argv", ["quality_worker.py", "diarize"]),
            patch.object(quality_worker.sys, "stdin", io.StringIO(json.dumps({"speaker_count": {"mode": "automatic"}}))),
            patch.object(quality_worker.sys, "stdout", stdout),
        ):
            quality_worker.main()
        self.assertEqual(json.loads(stdout.getvalue())["speaker_embeddings"], result["speaker_embeddings"])

    def test_legacy_count_migrates_and_raw_embeddings_are_not_public(self) -> None:
        self.assertEqual(normalize_speaker_count(3), {"mode": "exact", "count": 3})
        job = default_job("dub-20260720-000000-abcd", ".mp4", 1, "0" * 64)
        job["speaker_evidence"] = {"speaker_embeddings": [{"speaker": "speaker-01", "centroid": [1.0]}]}
        job["speaker_evidence_manifest"] = {
            "schema": 1,
            "speakers": [{"speaker": "speaker-01", "low_evidence": True, "segments": [
                {"segment_index": 4, "start": 1.0, "end": 2.2, "voiced_duration": 1.2, "borderline": True}
            ]}],
        }
        job["speaker_duplicate_suggestions"] = [{"speaker_a": "speaker-01", "speaker_b": "speaker-02"}]
        public = public_job(job)
        self.assertNotIn("speaker_evidence", public)
        self.assertIn("speaker_duplicate_suggestions", public)
        encoded_manifest = json.dumps(public["speaker_evidence_manifest"])
        self.assertNotIn("centroid", encoded_manifest)
        self.assertNotIn("text", encoded_manifest)
        self.assertNotIn("path", encoded_manifest)

    def test_visual_card_modality_is_human_evidence_and_not_judged(self) -> None:
        manifest = derive_speaker_evidence([], [], [
            {"i": 0, "speaker": "speaker-01", "start": 0.0, "end": 2.0, "voiced_duration": 1.5}
        ])["manifest"]
        self.assertEqual(
            manifest["modalities"]["visual"],
            {"status": "not judged", "evidence_role": "human evidence"},
        )
        encoded = json.dumps(manifest["modalities"]["visual"])
        self.assertNotIn("score", encoded)
        self.assertNotIn("confidence", encoded)
        self.assertNotIn("detector", encoded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
