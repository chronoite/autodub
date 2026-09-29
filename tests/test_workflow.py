from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
import unittest.mock
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autodub.config import WORK_ROOT  # noqa: E402

from autodub.policies import public_policies
from autodub.state import job_dir
from autodub.state import default_job, save_job
from autodub.experiment_store import run_dir
from autodub.tts_experiment import plan_tts_experiment
from autodub.workflow import (
    apply_forced_alignment,
    apply_glossary,
    audit_timing,
    cancellation_requested,
    clear_cancel,
    clear_synth_progress,
    record_line,
    remove_stale_lines,
    render_srt,
    request_cancel,
    reusable_line,
    write_synth_progress,
)


def _silent_wav(path: Path, seconds: float, rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * max(1, round(seconds * rate)))


class WorkflowContractTests(unittest.TestCase):
    def test_forced_alignment_changes_only_timing_and_is_reversible(self) -> None:
        segments = [
            {"i": 0, "start": 1.0, "end": 2.0, "text": "a", "translation": "A", "speaker": "s1"},
            {"i": 1, "start": 2.2, "end": 3.0, "text": "b", "translation": "B", "speaker": "s2"},
        ]
        summary = apply_forced_alignment(
            segments,
            [
                {"i": 0, "start": 1.1, "end": 1.9, "words": [{"start": 1.1, "end": 1.9}]},
                {"i": 1, "start": 2.25, "end": 2.95, "words": [{"start": 2.25, "end": 2.95}]},
            ],
        )
        self.assertEqual(2, summary["changed"])
        self.assertEqual({"start": 1.0, "end": 2.0}, segments[0]["asr_window"])
        self.assertEqual((1.1, 1.9), (segments[0]["start"], segments[0]["end"]))
        self.assertEqual(("A", "s1"), (segments[0]["translation"], segments[0]["speaker"]))

    def test_forced_alignment_fails_closed_on_overlap_or_missing_ids(self) -> None:
        segments = [
            {"i": 0, "start": 0.0, "end": 1.0},
            {"i": 1, "start": 1.1, "end": 2.0},
        ]
        with self.assertRaises(ValueError):
            apply_forced_alignment(segments, [{"i": 0, "start": 0.0, "end": 1.0}])
        with self.assertRaises(ValueError):
            apply_forced_alignment(
                segments,
                [
                    {"i": 0, "start": 0.0, "end": 1.2},
                    {"i": 1, "start": 1.0, "end": 2.0},
                ],
            )

    def test_policy_catalog_has_stable_modular_defaults(self) -> None:
        policies = public_policies()
        self.assertEqual(policies["default_mix"], "balanced-v1")
        self.assertEqual(policies["default_timing"], "gentle-fit-v1")
        self.assertEqual(policies["default_space"], "dry-v1")
        self.assertIn("legacy-v1", {item["id"] for item in policies["mix"]})
        self.assertIn("segment-window-v1", {item["id"] for item in policies["timing"]})
        self.assertIn("light-room-v1", {item["id"] for item in policies["space"]})

    def test_line_cache_reuses_only_matching_content_and_voice(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            lines = Path(temporary)
            segment = {"i": 7, "speaker": "speaker-01", "translation": "First take."}
            settings = {"target_language": "en", "tts_backend": "test", "tts_seed": 1}
            _silent_wav(lines / "line-00007.wav", 0.2)
            record_line(lines, segment, "voice-a", settings)
            self.assertTrue(reusable_line(lines, segment, "voice-a", settings))
            self.assertFalse(reusable_line(lines, {**segment, "translation": "Changed."}, "voice-a", settings))
            self.assertFalse(reusable_line(lines, segment, "voice-b", settings))

            remove_stale_lines(lines, [])
            self.assertFalse((lines / "line-00007.wav").exists())
            self.assertEqual(json.loads((lines / "manifest.json").read_text(encoding="utf-8")), {})

    def test_cancellation_marker_is_explicit_and_reversible(self) -> None:
        job_id = "dub-workflow-cancel-test"
        root = job_dir(job_id)
        try:
            request_cancel(job_id)
            self.assertTrue(cancellation_requested(job_id))
            clear_cancel(job_id)
            self.assertFalse(cancellation_requested(job_id))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_progress_sidecar_contains_no_dialogue_and_can_be_cleared(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            lines = Path(temporary)
            started = time.monotonic() - 2.0
            write_synth_progress(
                lines,
                done=2,
                total=5,
                line_index=9,
                started_at=started,
                last_seconds=0.5,
            )
            payload = json.loads((lines / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["done"], 2)
            self.assertEqual(payload["total"], 5)
            self.assertEqual(payload["line_index"], 9)
            self.assertNotIn("text", payload)
            self.assertGreaterEqual(payload["eta_secs"], 2)
            (lines / "progress.json.tmp").write_text("stale", encoding="ascii")
            clear_synth_progress(lines)
            self.assertFalse((lines / "progress.json").exists())
            self.assertFalse((lines / "progress.json.tmp").exists())

    def test_clear_synth_progress_degrades_when_file_is_locked(self) -> None:
        # Regression contract: a reader holding the sidecar open
        # (WinError 32/5) must NEVER kill the render — retry, then degrade to a no-op.
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            lines = Path(temporary)
            (lines / "progress.json").write_text("{}", encoding="ascii")
            with unittest.mock.patch.object(
                Path, "unlink", side_effect=PermissionError("locked")
            ), unittest.mock.patch("autodub.workflow.time.sleep"):
                clear_synth_progress(lines)  # must return, not raise
            self.assertTrue((lines / "progress.json").exists())
            clear_synth_progress(lines)
            self.assertFalse((lines / "progress.json").exists())

    def test_renders_clear_stale_progress_before_synthesizing_stage(self) -> None:
        # Source contract: both the quality render and the
        # line repair clear the stale sidecar BEFORE flipping the stage to
        # "synthesizing" — public_job serves the sidecar from that flip onward.
        source = (ROOT / "src" / "autodub" / "pipeline.py").read_text(encoding="utf-8")
        for function in ("def _render_quality", "def repair_line"):
            body = source.split(function, 1)[1].split("\ndef ", 1)[0]
            before_stage_flip = body.split('event(job, "synthesizing"', 1)[0]
            self.assertIn("clear_synth_progress", before_stage_flip, function)

    def test_timing_qc_and_srt_are_human_reviewable(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            lines = Path(temporary)
            _silent_wav(lines / "line-00000.wav", 1.4)
            segments = [
                {
                    "i": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "translation": "Hello.",
                    "alignment": {"tempo_limited": True},
                },
                {"i": 1, "start": 0.8, "end": 1.2, "translation": "Overlap."},
            ]
            summary = audit_timing(segments, lines)
            self.assertEqual(summary["overrun"], 1)
            self.assertEqual(summary["missing"], 1)
            self.assertEqual(summary["overlap"], 1)
            self.assertEqual(summary["tempo_limited"], 1)
            srt = render_srt(segments)
            self.assertIn("00:00:00,000 --> 00:00:01,000", srt)
            self.assertIn("Hello.", srt)

    def test_runaway_flag_is_distinct_and_survives_exact_fit(self) -> None:
        # exact-fit pads/trims the aligned wav to the slot, so the
        # duration-based overrun check can never see a runaway — the RAW pre-tempo
        # synth duration (alignment.source_seconds) is the only reliable signal.
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            lines = Path(temporary)
            _silent_wav(lines / "line-00000.wav", 1.0)  # aligned exactly to slot
            segments = [
                {
                    "i": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "translation": "Runaway.",
                    "alignment": {"source_seconds": 327.0},
                },
            ]
            summary = audit_timing(segments, lines)
            self.assertEqual(summary["runaway"], 1)
            self.assertEqual(summary["overrun"], 0)
            self.assertEqual(summary["flagged"], 1)
            self.assertEqual(segments[0]["qc"]["flags"], ["runaway"])
            self.assertEqual(segments[0]["qc"]["raw_seconds"], 327.0)

    def test_runaway_threshold_floor_and_precedence(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temporary:
            lines = Path(temporary)
            for index in range(4):
                _silent_wav(lines / f"line-{index:05d}.wav", 0.5)
            segments = [
                # slot 1.0, raw exactly 8.0 -> NOT a runaway (strict >)
                {"i": 0, "start": 0.0, "end": 1.0, "translation": "a",
                 "alignment": {"source_seconds": 8.0}},
                # tiny 0.3s slot: the 1.0s floor makes the bar 8.0s, so a 2.5s
                # take is a normal long line, not a runaway
                {"i": 1, "start": 1.0, "end": 1.3, "translation": "b",
                 "alignment": {"source_seconds": 2.5}},
                # legacy segment without alignment fails open
                {"i": 2, "start": 2.0, "end": 3.0, "translation": "c"},
                # ratio override works without minting giant fixtures
                {"i": 3, "start": 3.0, "end": 4.0, "translation": "d",
                 "alignment": {"source_seconds": 2.5}},
            ]
            summary = audit_timing(segments, lines)
            self.assertEqual(summary["runaway"], 0)
            overridden = audit_timing(segments, lines, runaway_ratio=2.0)
            self.assertEqual(overridden["runaway"], 3)  # i=0, i=1 (floor), i=3
            # a missing line takes precedence over a stale alignment dict
            missing = [{"i": 9, "start": 0.0, "end": 1.0, "translation": "x",
                        "alignment": {"source_seconds": 400.0}}]
            summary = audit_timing(missing, lines)
            self.assertEqual(summary["missing"], 1)
            self.assertEqual(summary["runaway"], 0)
            self.assertEqual(missing[0]["qc"]["flags"], ["missing-line"])

    def test_glossary_prefers_longer_keys_first(self) -> None:
        self.assertEqual(
            apply_glossary("Lady Ada met Ada.", {"Ada": "A", "Lady Ada": "Lady A"}),
            "Lady A met A.",
        )

    def test_tts_experiment_plan_is_short_opaque_and_selective(self) -> None:
        job_id = "dub-workflow-tts-plan"
        root = job_dir(job_id)
        run_id = None
        try:
            job = default_job(job_id, ".mp4", 1, "0" * 64)
            job["segments"] = [
                {
                    "i": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "speaker-01",
                    "translation": "Synthetic line.",
                }
            ]
            save_job(job)
            run = plan_tts_experiment(
                job_id,
                ["windows-sapi"],
                start=2.0,
                duration=999.0,
            )
            run_id = run["id"]
            self.assertEqual(run["status"], "queued")
            self.assertEqual(run["selection"]["duration"], 120.0)
            self.assertEqual(run["selection"]["candidates"], ["windows-sapi"])
            self.assertNotIn("Synthetic line", json.dumps(run))
            excluded = [
                item for item in run["candidates"]
                if item["id"] != "windows-sapi"
            ]
            self.assertTrue(all(item["result"]["status"] == "not-selected" for item in excluded))
        finally:
            shutil.rmtree(root, ignore_errors=True)
            if run_id:
                shutil.rmtree(run_dir(run_id), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
