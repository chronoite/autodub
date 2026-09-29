"""Casting-studio contracts.

Merge verb, quarantine/mute, attribution flags, bank audition, code-SHA liveness.
"""
from __future__ import annotations

import shutil
import subprocess
import unittest

from autodub import casting, oplog
from autodub.state import default_job, job_dir, load_job, save_job
from autodub.workflow import load_line_manifest, save_line_manifest

from tests import ROOT


def _job_with_speakers(job_id: str) -> dict:
    job = default_job(job_id, ".mp4", 1, "0" * 64)
    job["status"] = "review"
    job["segments"] = [
        {"i": 0, "start": 0.0, "end": 2.0, "speaker": "speaker-01",
         "speaker_confidence": 1.0, "voiced_duration": 1.8, "translation": "A."},
        {"i": 1, "start": 2.0, "end": 4.0, "speaker": "speaker-02",
         "speaker_confidence": 0.6, "voiced_duration": 1.5, "translation": "B."},
        {"i": 2, "start": 4.0, "end": 5.0, "speaker": "speaker-01",
         "speaker_confidence": 1.0, "voiced_duration": 0.9, "translation": "C."},
        {"i": 3, "start": 5.0, "end": 5.4, "speaker": "speaker-05",
         "speaker_confidence": 0.0, "voiced_duration": 0.0, "translation": "D.",
         "attribution_suspect": True},
    ]
    job["speaker_references"] = {
        "speaker-01": {"file": "voice-references/speaker-01.wav", "text": "r1"},
        "speaker-02": {"file": "voice-references/speaker-02.wav", "text": "r2"},
    }
    job["speaker_voices"] = {"speaker-01": "qwen-auto:speaker-01",
                             "speaker-02": "qwen-auto:speaker-02",
                             "speaker-05": "mute:"}
    job["available_voices"] = ["qwen-auto:speaker-01", "qwen-auto:speaker-02", "mute:"]
    job["voice_labels"] = {"qwen-auto:speaker-01": "auto 1", "qwen-auto:speaker-02": "auto 2"}
    job["speaker_duplicate_suggestions"] = [
        {"speaker_a": "speaker-01", "speaker_b": "speaker-02", "cosine_similarity": 0.91},
    ]
    save_job(job)
    return job


class MergeSpeakersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.job_id = "dub-casting-merge-test"
        self.addCleanup(lambda: shutil.rmtree(job_dir(self.job_id), ignore_errors=True))
        _job_with_speakers(self.job_id)
        lines = job_dir(self.job_id) / "artifacts" / "lines"
        lines.mkdir(parents=True, exist_ok=True)
        (lines / "line-00001.wav").write_bytes(b"cached-old-voice")
        save_line_manifest(lines, {"1": "sig"})

    def test_merge_relabels_purges_and_drops_the_absorbed_voice(self) -> None:
        result = casting.merge_speakers(self.job_id, source="speaker-02", target="speaker-01")
        self.assertEqual(result["merged"], 1)
        self.assertEqual(result["purged_lines"], 1)
        stored = load_job(self.job_id)
        self.assertEqual(stored["segments"][1]["speaker"], "speaker-01")
        self.assertEqual(stored["segments"][1]["merged_from"], "speaker-02")
        self.assertNotIn("speaker-02", stored["speaker_references"])
        self.assertNotIn("speaker-02", stored["speaker_voices"])
        self.assertNotIn("qwen-auto:speaker-02", stored["available_voices"])
        self.assertEqual(stored["speaker_duplicate_suggestions"], [])
        self.assertEqual(stored["speaker_merges"], [
            {"from": "speaker-02", "to": "speaker-01", "lines": 1}])
        lines = job_dir(self.job_id) / "artifacts" / "lines"
        self.assertFalse((lines / "line-00001.wav").exists())
        self.assertNotIn("1", load_line_manifest(lines))
        self.assertTrue(any("merged into" in e["message"] for e in stored["events"]))

    def test_invalid_and_running_merges_refuse(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid"):
            casting.merge_speakers(self.job_id, source="speaker-01", target="speaker-01")
        with self.assertRaisesRegex(ValueError, "invalid"):
            casting.merge_speakers(self.job_id, source="speaker-99", target="speaker-01")
        job = load_job(self.job_id)
        job["status"] = "running"
        save_job(job)
        with self.assertRaisesRegex(ValueError, "current stage"):
            casting.merge_speakers(self.job_id, source="speaker-02", target="speaker-01")


class AttributionFlagsTests(unittest.TestCase):
    def test_flag_classes(self) -> None:
        job = {
            "segments": [
                {"i": 0, "start": 0.0, "end": 2.0, "speaker": "a", "speaker_confidence": 1.0},
                # ABA flap: b interrupts a within 6s at low confidence
                {"i": 1, "start": 2.0, "end": 3.0, "speaker": "b", "speaker_confidence": 0.5},
                {"i": 2, "start": 3.0, "end": 5.0, "speaker": "a", "speaker_confidence": 1.0},
                {"i": 3, "start": 6.0, "end": 6.4, "speaker": "ghost",
                 "speaker_confidence": 0.0, "attribution_suspect": True},
                {"i": 4, "start": 7.0, "end": 9.0, "speaker": "c", "speaker_confidence": 0.95},
            ],
            "speaker_duplicate_suggestions": [
                {"speaker_a": "a", "speaker_b": "c", "cosine_similarity": 0.90},
                {"speaker_a": "b", "speaker_b": "c", "cosine_similarity": 0.50},  # below bar
            ],
        }
        flags = {f["i"]: f["reasons"] for f in casting.attribution_flags(job)}
        self.assertIn("mid-scene-flap", flags[1])
        self.assertIn("low-confidence 0.50", flags[1])
        self.assertIn("quarantined-phantom", flags[3])
        self.assertIn("merge-suggested-label", flags[0])   # a is in a >=0.88 pair
        self.assertIn("merge-suggested-label", flags[4])   # c too
        self.assertNotIn(2, {i for i, r in flags.items() if "mid-scene-flap" in r} - {1})


class SourceContractTests(unittest.TestCase):
    PIPELINE = (ROOT / "src" / "autodub" / "pipeline.py").read_text(encoding="utf-8")

    def test_quarantine_and_mute_are_wired(self) -> None:
        self.assertIn('job["speaker_voices"][speaker] = "mute:"', self.PIPELINE)
        self.assertIn('"attribution_suspect"', self.PIPELINE)
        # both render paths skip muted voices BEFORE synthesis and purge stale wavs
        synth = self.PIPELINE.split("def _render_quality_synth", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('voice.startswith("mute")', synth)
        cpu = self.PIPELINE.split("def _render_cpu", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('voice.startswith("mute")', cpu)
        one = self.PIPELINE.split("def _synthesize_one", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("is muted", one)

    def test_bank_audition_branch_exists(self) -> None:
        review = (ROOT / "src" / "autodub" / "review.py").read_text(encoding="utf-8")
        body = review.split("def voice_reference", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('voice.startswith("bank:")', body)

    def test_server_wires_merge_and_flags(self) -> None:
        server = (ROOT / "src" / "autodub" / "server.py").read_text(encoding="utf-8")
        self.assertIn('parts[4] == "merge"', server)
        self.assertIn("attribution-flags", server)


class DemoSelectionTests(unittest.TestCase):
    def test_demo_lines_pick_emphatic_and_calm(self) -> None:
        job = {"segments": [
            {"i": 0, "speaker": "a", "translation": "A calm line that is long enough.",
             "delivery": {"label": "calm", "energy_dbfs": -30.0}},
            {"i": 1, "speaker": "a", "translation": "A big shouted line right here!",
             "delivery": {"label": "intense", "energy_dbfs": -8.0}},
            {"i": 2, "speaker": "a", "translation": "Short."},
            {"i": 3, "speaker": "b", "translation": "Someone else entirely speaking."},
        ]}
        picks = casting.demo_lines(job, "a")
        self.assertEqual(picks["emphatic"], 1)
        self.assertEqual(picks["calm"], 0)
        with self.assertRaises(ValueError):
            casting.demo_lines(job, "nobody")


class ApplyCastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.job_id = "dub-casting-lock-test"
        self.addCleanup(lambda: shutil.rmtree(job_dir(self.job_id), ignore_errors=True))
        _job_with_speakers(self.job_id)
        job = load_job(self.job_id)
        job["speaker_references"]["speaker-02"]["text"] = ""  # transcript missing
        save_job(job)
        lines = job_dir(self.job_id) / "artifacts" / "lines"
        lines.mkdir(parents=True, exist_ok=True)
        (lines / "line-00001.wav").write_bytes(b"old-voice")
        save_line_manifest(lines, {"1": "sig"})

    def test_lock_applies_voices_purges_and_records(self) -> None:
        aligned = job_dir(self.job_id) / "artifacts" / "aligned"
        aligned.mkdir(parents=True, exist_ok=True)
        (aligned / "line-00001.wav").write_bytes(b"stale-aligned")
        result = casting.apply_cast(self.job_id, cast={
            "speaker-01": {"character": "Hero", "voice": "qwen-auto:speaker-01",
                           "tier": "main"},
            "speaker-02": {"character": "Friend", "voice": "qwen-auto-xv:speaker-02",
                           "second_choice": "qwen-auto:speaker-01", "tier": "minor"},
        })
        self.assertEqual(sorted(result["changed"]), ["speaker-02"])
        self.assertEqual(result["purged_lines"], 2)  # lines/ AND aligned/ copies
        stored = load_job(self.job_id)
        self.assertEqual(stored["speaker_voices"]["speaker-02"], "qwen-auto-xv:speaker-02")
        self.assertIn("qwen-auto-xv:speaker-02", stored["available_voices"])
        # names live in cast_lock; speaker_characters stays an ID-only contract
        # and is untouched without a series enrollment
        self.assertEqual(stored["cast_lock"]["cast"]["speaker-01"]["character"], "Hero")
        self.assertIsNone(stored.get("speaker_characters"))
        self.assertEqual(stored["cast_lock"]["changed_labels"], ["speaker-02"])
        self.assertFalse((job_dir(self.job_id) / "artifacts" / "lines" / "line-00001.wav").exists())
        self.assertFalse((aligned / "line-00001.wav").exists())

    def test_studio_muted_labels_lock_cleanly(self) -> None:
        # frontend sends muted labels as {"voice": "mute:"} with no character —
        # an earlier version deadlocked here
        result = casting.apply_cast(self.job_id, cast={
            "speaker-01": {"character": "Hero", "voice": "qwen-auto:speaker-01"},
            "speaker-02": {"voice": "mute:"},
        })
        stored = load_job(self.job_id)
        self.assertEqual(stored["speaker_voices"]["speaker-02"], "mute:")
        self.assertEqual(result["cast"], 2)

    def test_voice_reuse_across_labels_validates_the_embedded_reference(self) -> None:
        # casting speaker-02 to speaker-01's ICL voice is deliberate doubling and
        # must validate speaker-01's transcript, not speaker-02's (which is empty)
        result = casting.apply_cast(self.job_id, cast={
            "speaker-01": {"character": "Hero", "voice": "qwen-auto:speaker-01"},
            "speaker-02": {"character": "Extra", "voice": "qwen-auto:speaker-01"},
        })
        self.assertEqual(result["cast"], 2)

    def test_icl_without_transcript_is_a_hard_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "transcript missing"):
            casting.apply_cast(self.job_id, cast={
                "speaker-01": {"character": "Hero", "voice": "qwen-auto:speaker-01"},
                "speaker-02": {"character": "Friend", "voice": "qwen-auto:speaker-02"},
            })

    def test_incomplete_or_invalid_cast_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            casting.apply_cast(self.job_id, cast={
                "speaker-01": {"character": "Hero", "voice": "qwen-auto:speaker-01"}})
        with self.assertRaisesRegex(ValueError, "unknown voice option"):
            casting.apply_cast(self.job_id, cast={
                "speaker-01": {"character": "Hero", "voice": "sapi:whatever"},
                "speaker-02": {"character": "Friend", "voice": "qwen-auto-xv:speaker-02"},
            })
        # muted quarantined labels do not need a decision (speaker-05 absent above)


class EmbeddingVerificationTests(unittest.TestCase):
    def test_disagreements_flag_only_confident_losses(self) -> None:
        job = {
            "segments": [{"i": 0, "start": 0.0, "end": 1.0, "speaker": "a"},
                         {"i": 1, "start": 1.0, "end": 2.0, "speaker": "a"},
                         {"i": 2, "start": 2.0, "end": 3.0, "speaker": "b"}],
            "speaker_evidence": {
                "speaker_embeddings": [
                    {"speaker": "a", "centroid": [1.0, 0.0]},
                    {"speaker": "b", "centroid": [0.0, 1.0]},
                ],
                "speaker_chunk_embeddings": [
                    # segment 0 assigned 'a' but its embedding IS b's direction
                    {"segment_index": 0, "speaker": "a", "embedding": [0.05, 1.0]},
                    # segment 1 agrees with its label
                    {"segment_index": 1, "speaker": "a", "embedding": [1.0, 0.1]},
                ],
            },
        }
        findings = casting.embedding_disagreements(job)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["i"], 0)
        self.assertEqual(findings[0]["suggests"], "b")

    def test_no_evidence_means_no_findings(self) -> None:
        self.assertEqual(casting.embedding_disagreements({"segments": []}), [])


class StageThreeSourceContracts(unittest.TestCase):
    def test_server_wires_cast_demos_and_flags(self) -> None:
        server = (ROOT / "src" / "autodub" / "server.py").read_text(encoding="utf-8")
        self.assertIn('parts[3] == "apply-cast"', server)
        self.assertIn('parts[4] == "prerender"', server)
        self.assertIn('consume_arm(job_id, "demo")', server)
        self.assertIn("embedding_disagreements", server)
        gpu = (ROOT / "src" / "autodub" / "gpu_session.py").read_text(encoding="utf-8")
        self.assertIn('"demo"', gpu)

    def test_qwen_line_supports_xvector_variant(self) -> None:
        pipeline_src = (ROOT / "src" / "autodub" / "pipeline.py").read_text(encoding="utf-8")
        body = pipeline_src.split("def _qwen_line", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('voice.startswith("qwen-auto-xv:")', body)
        self.assertIn('"reference_text": ""', body)


class StudioFrontendContracts(unittest.TestCase):
    STATIC = ROOT / "src" / "autodub" / "web" / "static"

    def test_studio_screen_is_wired(self) -> None:
        index = (self.STATIC / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="open-studio"', index)
        self.assertIn('src="studio.js"', index)
        page = (self.STATIC / "studio-page.js").read_text(encoding="utf-8")
        for token in ("apply-cast", "speakers/merge", "reject-merge", "demos/prerender",
                      "evidence-frame", "evidence-video", "qwen-auto-xv:"):
            self.assertIn(token, page)
        # "Not sure" never merges: it is its own verdict next to Same / Different
        self.assertIn("Not sure", (self.STATIC / "studio.html").read_text(encoding="utf-8"))

    def test_no_inline_style_attributes(self) -> None:
        # The server's CSP (style-src 'self') ignores style="..." attributes, so pages must use
        # classes or CSSOM writes instead.
        for path in sorted(self.STATIC.iterdir()):
            if path.suffix in {".html", ".js"}:
                self.assertNotIn('style="', path.read_text(encoding="utf-8"), path.name)


class CodeShaTests(unittest.TestCase):
    def test_code_sha_matches_git_and_is_stamped_into_renders(self) -> None:
        sha = oplog.code_sha()
        expected = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True, cwd=str(ROOT)).stdout.strip()
        if not expected:
            self.skipTest("not a git checkout")
        self.assertEqual(sha, expected)
        self.assertNotEqual(sha, "unknown")
        source = (ROOT / "src" / "autodub" / "pipeline.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("code_sha()"), 2)  # analyze + render stamps
        server = (ROOT / "src" / "autodub" / "server.py").read_text(encoding="utf-8")
        self.assertIn('server_info("startup"', server)


if __name__ == "__main__":
    unittest.main(verbosity=2)
