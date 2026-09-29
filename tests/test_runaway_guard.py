"""Runaway-TTS guard contracts.

A poisoned voice reference made Qwen3-TTS babble 327 s of audio for a 1.4 s slot
across an entire speaker (64 s/line average). These tests pin the two shipped
halves: the clean-window reference picker (autodub.reference_quality) and the
per-line worker guard (retry -> fallback reference -> kept-shortest). All worker
tests run GPU-free with fake torch/soundfile/qwen_tts modules (house pattern from
tests/test_speaker_evidence.py).
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from autodub.config import WORK_ROOT
from autodub.reference_quality import build_candidate_windows, pick_reference_windows
from autodub.workers import quality_worker


def _segment(i: int, start: float, end: float, *, confidence: float = 1.0,
             overlap: float = 0.0, text: str = "") -> dict:
    return {"i": i, "start": start, "end": end, "speaker_confidence": confidence,
            "overlap_ratio": overlap, "text": text}


class ReferencePickerTests(unittest.TestCase):
    def test_clean_window_beats_higher_scoring_dirty_window(self) -> None:
        # Regression: the 5s dirty window outscores the clean one on the raw
        # formula, but the clean pool gate must win.
        segments = [
            _segment(0, 0.0, 5.0, confidence=0.6, overlap=0.4, text="dirty"),
            _segment(1, 20.0, 23.0, confidence=1.0, overlap=0.0, text="clean"),
        ]
        picked = pick_reference_windows(segments, max_overlap=0.05,
                                        min_confidence=0.90, alternates=2)
        self.assertTrue(picked["clean"])
        self.assertEqual(picked["primary"]["source_segments"], [1])
        for spare in picked["alternates"]:
            self.assertLessEqual(spare["overlap"], 0.05)

    def test_all_dirty_falls_back_to_old_max_score_pick(self) -> None:
        segments = [
            _segment(0, 0.0, 5.0, confidence=0.6, overlap=0.30),
            _segment(1, 20.0, 22.0, confidence=0.5, overlap=0.35),
        ]
        old_style = max(build_candidate_windows(segments),
                        key=lambda w: (w["score"], w["duration"]))
        picked = pick_reference_windows(segments, max_overlap=0.05,
                                        min_confidence=0.90, alternates=2)
        self.assertFalse(picked["clean"])
        self.assertEqual(picked["primary"]["start"], old_style["start"])
        self.assertEqual(picked["primary"]["end"], old_style["end"])
        self.assertEqual(picked["alternates"], [])

    def test_alternates_are_disjoint_and_capped(self) -> None:
        segments = [_segment(i, i * 20.0, i * 20.0 + 4.0) for i in range(6)]
        picked = pick_reference_windows(segments, max_overlap=0.05,
                                        min_confidence=0.90, alternates=2)
        self.assertTrue(picked["clean"])
        self.assertEqual(len(picked["alternates"]), 2)
        seen = set(picked["primary"]["source_segments"])
        for spare in picked["alternates"]:
            self.assertTrue(seen.isdisjoint(spare["source_segments"]))
            seen.update(spare["source_segments"])


class RunawayMathTests(unittest.TestCase):
    def test_is_runaway_thresholds(self) -> None:
        self.assertTrue(quality_worker._is_runaway(12.0, 1.4))    # > 11.2
        self.assertFalse(quality_worker._is_runaway(10.0, 1.4))
        self.assertFalse(quality_worker._is_runaway(5.0, 0.3))    # floor: bar is 8.0
        self.assertTrue(quality_worker._is_runaway(9.0, 0.3))
        self.assertFalse(quality_worker._is_runaway(300.0, 0.0))  # guard-inert, no slot
        self.assertFalse(quality_worker._is_runaway(8.0, 1.0))    # strict >


class _FakeTorch(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.seed_calls: list[int] = []
        self.bfloat16 = "bfloat16"
        self.cuda = types.SimpleNamespace(manual_seed_all=lambda value: None)

    def manual_seed(self, value: int) -> None:
        self.seed_calls.append(int(value))


def _fake_soundfile() -> types.ModuleType:
    module = types.ModuleType("soundfile")

    def write(path: str, wav, rate: int) -> None:
        Path(path).write_bytes(b"RIFFfake" + bytes([len(wav) % 251]))

    module.write = write
    return module


class _FakeModel:
    """generate_voice_clone duration is scripted per reference path. Mirrors the real
    library contract: create_voice_clone_prompt returns a 1-ITEM LIST; generate
    accepts scalar text (one line) or list text (build-F batching)."""

    def __init__(self, script: dict[str, list[float]], rate: int = 100) -> None:
        self.script = {key: list(values) for key, values in script.items()}
        self.rate = rate
        self.generate_calls: list[str] = []
        self.batch_calls: list[tuple] = []
        self.prompt_calls: list[str] = []
        self.after_calls = None

    def create_voice_clone_prompt(self, *, ref_audio, ref_text, x_vector_only_mode):
        self.prompt_calls.append(str(ref_audio))
        return [str(ref_audio)]

    def _duration(self, key: str) -> float:
        queue = self.script[key]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def generate_voice_clone(self, *, text, language, voice_clone_prompt, **kwargs):
        if isinstance(text, list):
            self.batch_calls.append((list(text), list(voice_clone_prompt)))
            wavs = [[0.0] * int(self._duration(str(item)) * self.rate)
                    for item in voice_clone_prompt]
            result = (wavs, self.rate)
        else:
            key = (voice_clone_prompt[0] if isinstance(voice_clone_prompt, list)
                   else voice_clone_prompt)
            self.generate_calls.append(str(key))
            result = ([[0.0] * int(self._duration(str(key)) * self.rate)], self.rate)
        if self.after_calls:
            self.after_calls()
        return result


def _fake_qwen_tts(model: _FakeModel) -> types.ModuleType:
    module = types.ModuleType("qwen_tts")

    class Qwen3TTSModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return model

    module.Qwen3TTSModel = Qwen3TTSModel
    return module


class WorkerGuardTests(unittest.TestCase):
    def _run(self, model: _FakeModel, lines: list[dict], tmp: Path,
             fake_torch: _FakeTorch | None = None) -> dict:
        model_root = tmp / "model"
        model_root.mkdir(exist_ok=True)
        (model_root / "config.json").write_text("{}", encoding="ascii")
        torch_module = fake_torch or _FakeTorch()
        with (
            patch.dict(sys.modules, {
                "torch": torch_module,
                "soundfile": _fake_soundfile(),
                "qwen_tts": _fake_qwen_tts(model),
            }),
            patch.dict(quality_worker.QWEN_MODELS, {"qwen3-tts-1.7b": model_root}),
        ):
            return quality_worker.synthesize({"lines": lines, "seed": 1986})

    def _line(self, tmp: Path, i: int, *, slot: float | None, reference: str,
              fallbacks: list[str] | None = None) -> dict:
        line = {
            "text": "SYNTHETIC-MARKER-ALPHA",
            "language": "English",
            "reference_audio": reference,
            "reference_text": "SYNTHETIC-MARKER-BETA",
            "output": str(tmp / f"line-{i:05d}.wav"),
        }
        if slot is not None:
            line["slot_seconds"] = slot
            line["fallback_references"] = [
                {"audio": item, "text": "SYNTHETIC-MARKER-GAMMA"} for item in (fallbacks or [])
            ]
        return line

    def test_retry_then_fallback_and_demotion(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            model = _FakeModel({"primary.wav": [327.0, 300.0], "alt1.wav": [2.0]})
            lines = [
                self._line(tmp, 0, slot=1.4, reference="primary.wav", fallbacks=["alt1.wav"]),
                self._line(tmp, 1, slot=1.4, reference="primary.wav", fallbacks=["alt1.wav"]),
            ]
            result = self._run(model, lines, tmp)
            self.assertEqual(result["written"], 2)
            self.assertEqual(len(result["runaways"]), 1)
            entry = result["runaways"][0]
            self.assertEqual(entry["action"], "fallback-ref")
            self.assertEqual(entry["fallback"], "alt1.wav")
            self.assertEqual(entry["durations"], [327.0, 300.0, 2.0])
            # demotion: the second line generated exactly once, on the fallback
            self.assertEqual(model.generate_calls,
                             ["primary.wav", "primary.wav", "alt1.wav", "alt1.wav"])
            self.assertTrue((tmp / "line-00001.wav").is_file())

    def test_exhausted_keeps_shortest_and_continues(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            model = _FakeModel({"primary.wav": [30.0, 25.0, 28.0]})
            lines = [self._line(tmp, 0, slot=1.0, reference="primary.wav"),
                     self._line(tmp, 1, slot=1.0, reference="primary.wav")]
            result = self._run(model, lines, tmp)
            self.assertEqual(result["written"], 2)
            self.assertEqual(result["runaways"][0]["action"], "kept-shortest")
            self.assertEqual(result["runaways"][0]["durations"], [30.0, 25.0])
            self.assertIsNone(result["runaways"][0]["fallback"])
            self.assertTrue((tmp / "line-00001.wav").is_file())

    def test_slotless_payload_is_guard_inert(self) -> None:
        # Protects experiments/execute_voice_screen.py and the synthetic admission
        # tool: no slot_seconds means no guard, no reseed, exactly one generate.
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            fake_torch = _FakeTorch()
            model = _FakeModel({"primary.wav": [300.0]})
            result = self._run(model, [self._line(tmp, 0, slot=None, reference="primary.wav")],
                               tmp, fake_torch)
            self.assertEqual(result["runaways"], [])
            self.assertEqual(model.generate_calls, ["primary.wav"])
            self.assertEqual(fake_torch.seed_calls, [1986])  # batch seed only

    def test_per_line_reseed_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            fake_torch = _FakeTorch()
            model = _FakeModel({"primary.wav": [30.0, 2.0]})
            self._run(model, [self._line(tmp, 5, slot=1.0, reference="primary.wav")],
                      tmp, fake_torch)
            # batch seed, then line seed (1986+5), then retry seed (+104729)
            self.assertEqual(fake_torch.seed_calls, [1986, 1991, 1991 + 104729])

    def test_runaway_events_contain_no_dialogue_text(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            model = _FakeModel({"primary.wav": [30.0, 25.0, 28.0]})
            result = self._run(model, [self._line(tmp, 0, slot=1.0, reference="primary.wav")],
                               tmp)
            serialized = json.dumps(result["runaways"])
            self.assertNotIn("SYNTHETIC-MARKER", serialized)


class QwenLinePayloadTests(unittest.TestCase):
    def test_payload_slot_and_fallbacks_new_and_old_schema(self) -> None:
        from autodub.pipeline import _qwen_line
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            artifacts = Path(temporary)
            refs = artifacts / "voice-references"
            refs.mkdir()
            (refs / "speaker-01.wav").write_bytes(b"RIFF")
            (refs / "speaker-01-alt1.wav").write_bytes(b"RIFF")
            job = {
                "settings": {"source_language": "ja"},
                "speaker_references": {
                    "speaker-01": {
                        "file": "voice-references/speaker-01.wav",
                        "text": "ref",
                        "alternates": [
                            {"file": "voice-references/speaker-01-alt1.wav", "text": "alt"},
                            {"file": "voice-references/speaker-01-alt9.wav", "text": "gone"},
                        ],
                    },
                    "speaker-02": {  # old schema: no alternates/quality keys
                        "file": "voice-references/speaker-01.wav",
                        "text": "ref",
                    },
                },
            }
            segment = {"i": 3, "start": 10.0, "end": 11.5, "translation": "Hi."}
            payload = _qwen_line(job, artifacts, segment, "qwen-auto:speaker-01",
                                 artifacts / "line-00003.wav")
            self.assertEqual(payload["slot_seconds"], 1.5)
            self.assertEqual(len(payload["fallback_references"]), 1)  # missing alt filtered
            self.assertTrue(payload["fallback_references"][0]["audio"].endswith("speaker-01-alt1.wav"))
            legacy = _qwen_line(job, artifacts, segment, "qwen-auto:speaker-02",
                                artifacts / "line-00003.wav")
            self.assertEqual(legacy["fallback_references"], [])
            self.assertEqual(legacy["slot_seconds"], 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
