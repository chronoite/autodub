from __future__ import annotations

import base64
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

from autodub.adapters import sapi_voices, synthesize_sapi
from autodub.config import WORK_ROOT
from autodub.experiment_store import load_run, run_dir, save_run
from autodub.server import AutoDubHandler, serve
from autodub.state import default_job, job_dir, new_job_id, public_job, save_job
from autodub.voice_profiles import profile_path

from tests import requires_ffmpeg, requires_windows


class LoopbackServerTests(unittest.TestCase):
    def test_public_job_never_exposes_traceback_or_source_filename(self) -> None:
        job = default_job("dub-20260717-redaction", ".mp4", 1, "0" * 64)
        job["error"] = "short user-facing error"
        job["source"]["original_name"] = "private-original-name.mkv"
        job["error_detail"] = "Traceback: C:\\private\\host-path-marker"
        public = public_job(job)
        encoded = json.dumps(public)
        self.assertNotIn("error_detail", public)
        self.assertNotIn("host-path-marker", encoded)
        self.assertNotIn("source.mp4", encoded)
        self.assertNotIn("private-original-name", encoded)
        self.assertEqual(public["error"], "short user-facing error")

    def test_refuses_lan_bind(self) -> None:
        with self.assertRaises(ValueError):
            serve(host="0.0.0.0", port=0)

    def test_health_and_ui_security_headers(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), AutoDubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            with urllib.request.urlopen(base + "/api/health", timeout=5) as response:
                health = response.read()
                self.assertIn(b'"ok": true', health)
                self.assertIn(b'"default_profile": "quality-gpu-v1"', health)
                self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
            with urllib.request.urlopen(base + "/api/profiles", timeout=5) as response:
                profiles = json.load(response)
                self.assertEqual(profiles["default"], "quality-gpu-v1")
                self.assertTrue(any(not item["requires_gpu"] for item in profiles["profiles"]))
            with urllib.request.urlopen(base + "/api/policies", timeout=5) as response:
                policies = json.load(response)
                self.assertEqual(policies["default_mix"], "balanced-v1")
            with urllib.request.urlopen(base + "/api/tts-candidates", timeout=5) as response:
                candidates = json.load(response)["candidates"]
                self.assertEqual(len(candidates), 6)
                self.assertTrue(any(not item["requires_gpu"] for item in candidates))
            with urllib.request.urlopen(base + "/api/episode-queue", timeout=5) as response:
                queue = json.load(response)
                self.assertIn(queue["status"], {"idle", "stopped", "running"})
            with urllib.request.urlopen(base + "/", timeout=5) as response:
                page = response.read().decode("utf-8")
                self.assertIn("AutoDub", page)
                self.assertIn('href="theme.css"', page)
                self.assertIn("Quality first", page)
                self.assertIn('id="stage-checklist"', page)
                self.assertIn('id="build-tts-experiment"', page)
                self.assertIn('id="realign"', page)
                self.assertNotIn('href="experiments.css"', page)
                self.assertNotIn('id="experiment-view"', page)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_segment_evidence_media_is_lazy_opaque_and_range_capable(self) -> None:
        job_id = new_job_id()
        root = job_dir(job_id)
        job = default_job(job_id, ".private-source-name.mp4", 1, "0" * 64)
        job["segments"] = [{"i": 7, "start": 1.0, "end": 2.0, "speaker": "speaker-01"}]
        audio = root / "artifacts" / "previews" / "source-00007.wav"
        frame = root / "artifacts" / "evidence" / "frames" / "frame-00007.jpg"
        audio.parent.mkdir(parents=True)
        frame.parent.mkdir(parents=True)
        audio.write_bytes(b"synthetic-audio")
        frame.write_bytes(b"synthetic-frame")
        save_job(job)
        server = ThreadingHTTPServer(("127.0.0.1", 0), AutoDubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}/api/jobs/{job_id}/segments/7"
            ranged = urllib.request.Request(base + "/evidence-audio", headers={"Range": "bytes=1-4"})
            with urllib.request.urlopen(ranged, timeout=5) as response:
                headers = str(response.headers)
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), b"ynth")
                self.assertEqual(response.headers["Content-Type"], "audio/wav")
                self.assertEqual(response.headers["Content-Range"], "bytes 1-4/15")
                self.assertIn('filename="segment-audio-00007.wav"', headers)
                self.assertNotIn("private-source-name", headers)
                self.assertNotIn(str(root), headers)
            with urllib.request.urlopen(base + "/evidence-frame", timeout=5) as response:
                headers = str(response.headers)
                self.assertEqual(response.read(), b"synthetic-frame")
                self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                self.assertIn('filename="segment-frame-00007.jpg"', headers)
                self.assertNotIn('filename="frame-00007.jpg"', headers)
                self.assertNotIn("private-source-name", headers)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)
            shutil.rmtree(root, ignore_errors=True)

    @requires_windows
    @requires_ffmpeg
    def test_clone_reference_upload_updates_opaque_job(self) -> None:
        job_id = new_job_id()
        profile_id = None
        root = job_dir(job_id)
        root.mkdir(parents=True)
        save_job(default_job(job_id, ".mp4", 1, "0" * 64))
        server = ThreadingHTTPServer(("127.0.0.1", 0), AutoDubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temp:
                audio = Path(temp) / "reference.wav"
                transcript = "Synthetic clone upload reference."
                synthesize_sapi(transcript, sapi_voices()[0], audio)
                body = audio.read_bytes()
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_address[1]}/api/jobs/{job_id}/voice-profile?ext=.wav",
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-AutoDub-Transcript": base64.b64encode(transcript.encode("utf-8")).decode("ascii"),
                        "X-AutoDub-Language": "en",
                    },
                )
                with urllib.request.urlopen(request, timeout=20) as response:
                    updated = json.load(response)
                clone_options = [option for option in updated["available_voices"] if option.startswith("gsv:voice-")]
                self.assertEqual(len(clone_options), 1)
                profile_id = clone_options[0].removeprefix("gsv:")
                self.assertNotIn("reference.wav", json.dumps(updated))
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)
            shutil.rmtree(root, ignore_errors=True)
            if profile_id:
                shutil.rmtree(profile_path(profile_id).parent, ignore_errors=True)

    def test_experiment_review_api_is_opaque_range_capable_and_traversal_safe(self) -> None:
        job_id = new_job_id()
        job_root = job_dir(job_id)
        run_id = None
        job_root.mkdir(parents=True)
        save_job(default_job(job_id, ".mp4", 1, "0" * 64))
        server = ThreadingHTTPServer(("127.0.0.1", 0), AutoDubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            body = json.dumps({"experiment": "speaker-detection-v1", "job": job_id}).encode("utf-8")
            request = urllib.request.Request(
                base + "/api/experiments",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                created = json.load(response)
            run_id = created["id"]
            manifest = load_run(run_id)
            candidate = manifest["candidates"][0]
            evidence = run_dir(run_id) / "assignments" / "synthetic.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text('{"schema":1,"segments":[]}', encoding="utf-8")
            candidate["result"] = {
                "status": "ready-for-review",
                "evidence": "assignments/synthetic.json",
                "error": "synthetic C:\\private\\host-path",
            }
            save_run(manifest)

            with urllib.request.urlopen(base + f"/api/experiments/{run_id}", timeout=5) as response:
                public = json.load(response)
            self.assertNotIn("private", json.dumps(public))
            self.assertEqual(public["candidates"][0]["artifacts"][0]["path"], "assignments/synthetic.json")

            query = urlencode({"candidate": candidate["id"], "path": "assignments/synthetic.json"})
            ranged = urllib.request.Request(
                base + f"/api/experiments/{run_id}/artifact?{query}",
                headers={"Range": "bytes=0-4"},
            )
            with urllib.request.urlopen(ranged, timeout=5) as response:
                self.assertEqual(response.status, 206)
                self.assertTrue(response.headers["Content-Range"].startswith("bytes 0-4/"))
                self.assertEqual(len(response.read()), 5)

            bad_query = urlencode({"candidate": candidate["id"], "path": "../RUN.json"})
            with self.assertRaises(urllib.error.HTTPError) as blocked:
                urllib.request.urlopen(base + f"/api/experiments/{run_id}/artifact?{bad_query}", timeout=5)
            self.assertEqual(blocked.exception.code, 404)

            criterion = manifest["criteria"][0]
            review_body = json.dumps(
                {"status": "complete", "winner": candidate["id"], "notes": ["Synthetic review."],
                 "scores": {candidate["id"]: {criterion: 5}}}
            ).encode("utf-8")
            review_request = urllib.request.Request(
                base + f"/api/experiments/{run_id}",
                data=review_body,
                method="PATCH",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(review_request, timeout=5) as response:
                reviewed = json.load(response)
            self.assertEqual(reviewed["human_review"]["winner"], candidate["id"])
            self.assertEqual(load_run(run_id)["human_review"]["scores"][candidate["id"]][criterion], 5)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)
            shutil.rmtree(job_root, ignore_errors=True)
            if run_id:
                shutil.rmtree(run_dir(run_id), ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
