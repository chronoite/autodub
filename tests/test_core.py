from __future__ import annotations

import io
import json
import shutil
import subprocess
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock

from autodub import adapters, gpu_session
from autodub.adapters import sapi_voices, synthesize_sapi
from autodub.config import (
    FFMPEG,
    MEDIA_PYTHON,
    runtime_env,
    TRANSLATION_MODEL,
    WHISPER_MODEL,
    WORK_ROOT,
)
from autodub.experiments.timing_screen import _render_comparison
from autodub.media import _atempo_chain, align_line, build_mix, mux, probe_duration
from autodub.state import default_job, job_dir, new_job_id, save_job
from autodub.voice_profiles import import_profile, list_profiles, profile_path

from tests import requires_ffmpeg, requires_model, requires_windows, requires_worker_modules, ROOT


class CoreContractTests(unittest.TestCase):
    def test_forced_realign_is_an_explicit_single_use_gpu_action(self) -> None:
        safe = {"safe": True, "coordinator": "local", "busy_ports": [], "reasons": []}
        with mock.patch("autodub.gpu_session.preflight_status", return_value=safe):
            result = gpu_session.arm("dub-realign-contract", "realign")
        self.assertTrue(result["armed"])
        self.assertTrue(gpu_session.consume_arm("dub-realign-contract", "realign"))
        self.assertFalse(gpu_session.consume_arm("dub-realign-contract", "realign"))

    def test_worker_log_context_is_isolated_between_threads(self) -> None:
        barrier = threading.Barrier(2)
        observed = {}

        def capture(job_id: str) -> None:
            adapters.set_job_context(job_id)
            barrier.wait(timeout=5)
            observed[job_id] = adapters._JOB_CONTEXT.get()

        threads = [
            threading.Thread(target=capture, args=("dub-thread-a",)),
            threading.Thread(target=capture, args=("dub-thread-b",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(observed, {
            "dub-thread-a": "dub-thread-a",
            "dub-thread-b": "dub-thread-b",
        })

    def test_new_jobs_use_balanced_modular_audio_defaults(self) -> None:
        job = default_job("dub-20260716-volume-test", ".mp4", 1, "0" * 64)
        self.assertEqual(job["settings"]["mix_policy"], "balanced-v1")
        self.assertEqual(job["settings"]["timing_policy"], "gentle-fit-v1")
        self.assertEqual(job["settings"]["space_policy"], "dry-v1")
        self.assertEqual(job["settings"]["dialogue_gain"], 1.0)
        self.assertEqual(job["settings"]["source_bed_gain"], 0.85)

    def test_atempo_chain_stays_in_ffmpeg_range(self) -> None:
        self.assertEqual(_atempo_chain(1.25), "atempo=1.250000")
        self.assertEqual(_atempo_chain(4.0), "atempo=2.000000,atempo=2.000000")
        self.assertEqual(_atempo_chain(0.25), "atempo=0.500000,atempo=0.500000")

    @requires_windows
    @requires_ffmpeg
    def test_windows_voice_adapter_writes_real_wav(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temp:
            output = Path(temp) / "line.wav"
            synthesize_sapi("AutoDub local verification line.", sapi_voices()[0], output)
            with wave.open(str(output), "rb") as handle:
                self.assertGreater(handle.getnframes(), 1000)
                self.assertGreater(handle.getframerate(), 0)

    @requires_windows
    @requires_ffmpeg
    def test_local_clone_profile_is_opaque_and_retrievable(self) -> None:
        profile_id = None
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temp:
            audio = Path(temp) / "reference.wav"
            synthesize_sapi("Synthetic local clone reference sentence.", sapi_voices()[0], audio)
            data = audio.read_bytes()
            try:
                result = import_profile(io.BytesIO(data), len(data), ".wav", "Synthetic local clone reference sentence.", "en")
                profile_id = result["id"]
                self.assertTrue(profile_id.startswith("voice-"))
                self.assertTrue(profile_path(profile_id).is_file())
                self.assertIn(result["option"], {item["option"] for item in list_profiles()})
            finally:
                if profile_id:
                    shutil.rmtree(profile_path(profile_id).parent, ignore_errors=True)

    @requires_windows
    @requires_ffmpeg
    def test_synthetic_mix_and_mux(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temp:
            root = Path(temp)
            video = root / "fixture.mp4"
            base = root / "base.wav"
            lines = root / "lines"
            aligned = root / "aligned"
            lines.mkdir(); aligned.mkdir()
            subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=0x182338:s=640x360:d=4:r=24", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)], check=True)
            subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vn", "-ar", "48000", "-ac", "2", str(base)], check=True)
            voices = sapi_voices()
            segments = [
                {"i": 0, "start": 0.4, "end": 1.7, "speaker": "speaker-01", "translation": "First local test line."},
                {"i": 1, "start": 2.1, "end": 3.5, "speaker": "speaker-02", "translation": "Second local test line."},
            ]
            for item in segments:
                raw = lines / f"line-{item['i']:05d}.wav"
                synthesize_sapi(item["translation"], voices[item["i"] % len(voices)], raw)
                align_line(raw, aligned / raw.name, item["end"] - item["start"], 0.6, 1.75)
            mixed = root / "mix.wav"
            self.assertEqual(build_mix(base, segments, aligned, mixed, 0.3, 1.0), 2)
            room_mix = root / "mix-light-room.wav"
            self.assertEqual(
                build_mix(
                    base,
                    segments,
                    aligned,
                    room_mix,
                    0.85,
                    1.0,
                    policy={
                        "dialogue_filter": "highpass=f=70,lowpass=f=15500,aecho=0.8:0.9:18:0.035"
                    },
                ),
                2,
            )
            self.assertTrue(room_mix.is_file())
            output = root / "output.mp4"
            mux(video, mixed, output)
            self.assertTrue(output.is_file())
            self.assertGreaterEqual(probe_duration(output), 3.9)

    def test_mix_splits_dialogue_for_sidechain_and_audible_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temp:
            root = Path(temp)
            base = root / "base.wav"
            aligned = root / "aligned"
            aligned.mkdir()
            line = aligned / "line-00000.wav"
            base.touch()
            line.touch()
            segments = [{"i": 0, "start": 0.5, "end": 1.5}]
            captured = []

            with (
                mock.patch("autodub.media.probe_duration", return_value=4.0),
                mock.patch("autodub.media.run_ffmpeg", side_effect=lambda args, _stage: captured.extend(args)),
            ):
                self.assertEqual(build_mix(base, segments, aligned, root / "mix.wav", 0.3, 1.0), 1)

            graph = captured[captured.index("-filter_complex") + 1]
            self.assertIn("asplit=2[dialogue_sidechain][dialogue_mix]", graph)
            self.assertIn("[bed][dialogue_sidechain]sidechaincompress=", graph)
            self.assertIn("[ducked][dialogue_mix]amix=", graph)
            self.assertNotIn("[bed][dialogue]sidechaincompress=", graph)
            self.assertNotIn("[ducked][dialogue]amix=", graph)

    @requires_windows
    @requires_ffmpeg
    def test_timing_experiment_builds_playable_opaque_comparison(self) -> None:
        job_id = new_job_id()
        root = job_dir(job_id)
        run_root = None
        try:
            artifacts = root / "artifacts"
            lines = artifacts / "lines"
            lines.mkdir(parents=True)
            source = root / "source.mp4"
            base = artifacts / "audio-full.wav"
            subprocess.run(
                [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                 "color=c=0x182338:s=320x180:d=3:r=24", "-f", "lavfi", "-i",
                 "sine=frequency=220:sample_rate=48000:duration=3", "-c:v", "libx264", "-pix_fmt",
                 "yuv420p", "-c:a", "aac", "-shortest", str(source)],
                check=True,
            )
            subprocess.run(
                [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                 "-vn", "-ar", "48000", "-ac", "2", str(base)],
                check=True,
            )
            synthesize_sapi("Synthetic timing comparison line.", sapi_voices()[0], lines / "line-00000.wav")
            job = default_job(job_id, ".mp4", source.stat().st_size, "0" * 64)
            job["segments"] = [{"i": 0, "start": 0.4, "end": 1.8, "speaker": "speaker-01", "translation": "Synthetic timing comparison line."}]
            job["artifacts"]["full_audio"] = "audio-full.wav"
            save_job(job)
            run_root = Path(tempfile.mkdtemp(dir=WORK_ROOT, prefix="timing-proof-"))
            video, evidence = _render_comparison(
                job,
                [{"i": 0, "start": 0.5, "end": 2.0, "boundary_count": 2}],
                run_root,
                "segment-window-atempo",
            )
            self.assertTrue(video.is_file())
            self.assertGreaterEqual(probe_duration(video), 2.9)
            self.assertEqual(len(evidence), 1)
            self.assertIn("fit", evidence[0])
        finally:
            shutil.rmtree(root, ignore_errors=True)
            if run_root:
                shutil.rmtree(run_root, ignore_errors=True)

    @requires_ffmpeg
    @requires_worker_modules("librosa", "sklearn")
    def test_acoustic_cluster_worker_schema(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temp:
            audio = Path(temp) / "tones.wav"
            subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=180:sample_rate=16000:duration=2", "-f", "lavfi", "-i", "sine=frequency=680:sample_rate=16000:duration=2", "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1", str(audio)], check=True)
            payload = {"audio": str(audio), "speaker_count": {"mode": "exact", "count": 2}, "segments": [{"i": 0, "start": 0, "end": 2, "text": "a"}, {"i": 1, "start": 2, "end": 4, "text": "b"}]}
            result = subprocess.run([str(MEDIA_PYTHON), str(ROOT / "src" / "autodub" / "workers" / "model_worker.py"), "cluster"], input=json.dumps(payload), capture_output=True, text=True, env=runtime_env(), timeout=300)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout)
            self.assertEqual(len(output["segments"]), 2)
            self.assertTrue(all(item["speaker"].startswith("speaker-") for item in output["segments"]))

    @requires_model(TRANSLATION_MODEL)
    @requires_worker_modules("transformers", "sentencepiece", "sacremoses")
    def test_pinned_translation_runs_offline_with_utf8(self) -> None:
        payload = {
            "source_language": "ja",
            "target_language": "en",
            "segments": [
                {"i": 0, "start": 0.0, "end": 1.0, "text": "こんにちは。"},
                {"i": 1, "start": 1.0, "end": 2.0, "text": "元気ですか？"},
            ],
        }
        result = subprocess.run(
            [str(MEDIA_PYTHON), str(ROOT / "src" / "autodub" / "workers" / "model_worker.py"), "translate"],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=runtime_env(),
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        translated = json.loads(result.stdout)["segments"]
        self.assertTrue(all(item.get("translation") for item in translated))
        self.assertNotEqual(translated[0]["translation"], translated[0]["text"])

    @requires_windows
    @requires_ffmpeg
    @requires_model(WHISPER_MODEL)
    def test_whisper_transcribes_synthetic_voice_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temp:
            audio = Path(temp) / "speech.wav"
            synthesize_sapi("This is a synthetic AutoDub transcription test.", sapi_voices()[0], audio)
            payload = {"audio": str(audio), "language": "en"}
            result = subprocess.run(
                [str(MEDIA_PYTHON), str(ROOT / "src" / "autodub" / "workers" / "model_worker.py"), "transcribe"],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=runtime_env(),
                timeout=600,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            segments = json.loads(result.stdout)["segments"]
            self.assertTrue(segments)
            combined = " ".join(item["text"].lower() for item in segments).replace("-", "")
            self.assertIn("autodub", combined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
