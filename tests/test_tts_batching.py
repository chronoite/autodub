"""Batched-TTS contracts (TTS_BATCH_SIZE — ships at batch_size 1, i.e. serial)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autodub.config import WORK_ROOT
from autodub.workers import quality_worker
from autodub.workers.quality_worker import _batch_groups

from tests.test_runaway_guard import _fake_qwen_tts, _fake_soundfile, _FakeModel, _FakeTorch


def _line(index: int, text: str, tmp: Path, *, slot: float | None = 1.0,
          reference: str = "primary.wav", fallbacks: list[str] | None = None) -> dict:
    line = {"text": text, "language": "English", "reference_audio": reference,
            "reference_text": "", "output": str(tmp / f"line-{index:05d}.wav")}
    if slot is not None:
        line["slot_seconds"] = slot
        line["fallback_references"] = [{"audio": f, "text": ""} for f in (fallbacks or [])]
    return line


class BatchGroupingTests(unittest.TestCase):
    def test_groups_are_length_sorted_and_complete(self) -> None:
        lines = [{"text": "x" * n} for n in (30, 5, 12, 40, 1, 22, 8)]
        groups = _batch_groups(lines, 3, True)
        self.assertEqual([len(g) for g in groups], [3, 3, 1])
        flattened = [l["text"] for g in groups for l in g]
        self.assertEqual(sorted(flattened), sorted(l["text"] for l in lines))
        lengths = [len(l["text"]) for g in groups for l in g]
        self.assertEqual(lengths, sorted(lengths))  # globally ascending when sorted

    def test_batch_size_one_is_original_order_singletons(self) -> None:
        lines = [{"text": "bbb"}, {"text": "a"}, {"text": "cc"}]
        for size in (1, 0, -2):
            groups = _batch_groups(lines, size, True)
            self.assertEqual(groups, [[lines[0]], [lines[1]], [lines[2]]])


class AdapterPayloadTests(unittest.TestCase):
    def test_batch_size_key_only_present_when_enabled(self) -> None:
        from autodub import adapters
        captured = {}
        with patch.object(adapters, "run_quality_worker",
                          side_effect=lambda cmd, payload, **kw: captured.update(payload) or {}):
            adapters.synthesize_quality_batch([{"text": "x"}], batch_size=1)
            self.assertNotIn("batch_size", captured)
            captured.clear()
            adapters.synthesize_quality_batch([{"text": "x"}], batch_size=3)
            self.assertEqual(captured["batch_size"], 3)


class BatchedWorkerTests(unittest.TestCase):
    def _run(self, model: _FakeModel, lines: list[dict], tmp: Path,
             batch_size: int, fake_torch: _FakeTorch | None = None) -> dict:
        model_root = tmp / "model"
        model_root.mkdir(exist_ok=True)
        (model_root / "config.json").write_text("{}", encoding="ascii")
        with (
            patch.dict(sys.modules, {"torch": fake_torch or _FakeTorch(),
                                     "soundfile": _fake_soundfile(),
                                     "qwen_tts": _fake_qwen_tts(model)}),
            patch.dict(quality_worker.QWEN_MODELS, {"qwen3-tts-1.7b": model_root}),
        ):
            return quality_worker.synthesize(
                {"lines": lines, "seed": 1986, "batch_size": batch_size})

    def test_batched_call_shape_and_all_lines_written(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            model = _FakeModel({"a.wav": [2.0], "b.wav": [2.0]})
            lines = [_line(0, "short", tmp, reference="a.wav"),
                     _line(1, "a much longer line of text", tmp, reference="b.wav"),
                     _line(2, "medium text", tmp, reference="a.wav")]
            result = self._run(model, lines, tmp, batch_size=2)
            self.assertEqual(result["written"], 3)
            self.assertEqual(result["runaways"], [])
            for index in range(3):
                self.assertTrue((tmp / f"line-{index:05d}.wav").is_file())
            # mixed prompt keys per group: batch calls carry list payloads
            self.assertTrue(model.batch_calls)
            for texts, prompts in model.batch_calls:
                self.assertEqual(len(texts), len(prompts))

    def test_runaway_inside_batch_is_rescued_singly(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            # batched generate yields a runaway for primary; single-line retries also
            # run away; the fallback reference cures it
            model = _FakeModel({"primary.wav": [30.0, 28.0, 27.0], "alt1.wav": [2.0]})
            lines = [_line(0, "aaa", tmp, fallbacks=["alt1.wav"]),
                     _line(1, "bbbbbb", tmp, fallbacks=["alt1.wav"])]
            result = self._run(model, lines, tmp, batch_size=2)
            self.assertEqual(result["written"], 2)
            self.assertGreaterEqual(len(result["runaways"]), 1)
            self.assertIn(result["runaways"][0]["action"], {"fallback-ref", "kept-shortest"})

    def test_cancel_between_groups_stops_cleanly(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            tmp = Path(temporary)
            cancel = tmp / "cancel.requested"
            model = _FakeModel({"a.wav": [2.0]})
            model.after_calls = lambda: cancel.write_text("stop", encoding="ascii")
            model_root = tmp / "model"
            model_root.mkdir()
            (model_root / "config.json").write_text("{}", encoding="ascii")
            lines = [_line(i, "text" * (i + 1), tmp, reference="a.wav") for i in range(4)]
            with (
                patch.dict(sys.modules, {"torch": _FakeTorch(),
                                         "soundfile": _fake_soundfile(),
                                         "qwen_tts": _fake_qwen_tts(model)}),
                patch.dict(quality_worker.QWEN_MODELS, {"qwen3-tts-1.7b": model_root}),
            ):
                result = quality_worker.synthesize(
                    {"lines": lines, "seed": 1986, "batch_size": 2,
                     "cancel_file": str(cancel)})
            self.assertTrue(result["cancelled"])
            self.assertEqual(result["written"], 2)  # first group landed, second refused


if __name__ == "__main__":
    unittest.main(verbosity=2)
