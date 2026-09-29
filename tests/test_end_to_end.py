from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from autodub.adapters import sapi_voices, synthesize_sapi
from autodub.cli import import_source
from autodub.config import FFMPEG, TRANSLATION_MODEL, WHISPER_MODEL, WORK_ROOT
from autodub.pipeline import analyze, render
from autodub.quality_profiles import CPU_PROFILE, get_profile
from autodub.state import job_dir, load_job, output_path, save_job

from tests import requires_ffmpeg, requires_model, requires_windows


class EndToEndSyntheticTest(unittest.TestCase):
    @requires_windows
    @requires_ffmpeg
    @requires_model(WHISPER_MODEL)
    @requires_model(TRANSLATION_MODEL)
    def test_analyze_review_render_export(self) -> None:
        job_id = None
        with tempfile.TemporaryDirectory(dir=WORK_ROOT) as temp:
            fixture = Path(temp)
            speech = fixture / "speech.wav"
            video = fixture / "source.mp4"
            synthesize_sapi("This is the complete local AutoDub pipeline test.", sapi_voices()[0], speech)
            subprocess.run(
                [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=0x111827:s=640x360:d=5:r=24", "-i", str(speech), "-filter_complex", "[1:a]apad=whole_dur=5[a]", "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", "5", str(video)],
                check=True,
            )
            try:
                job = import_source(video)
                job_id = job["id"]
                job["settings"]["source_language"] = "en"
                job["settings"]["target_language"] = "en"
                job["settings"]["speaker_count"] = {"mode": "exact", "count": 1}
                job["settings"]["quality_profile"] = CPU_PROFILE
                job["settings"]["quality_stack"] = get_profile(CPU_PROFILE)
                job["settings"]["source_bed_gain"] = 0.3
                save_job(job)

                analyze(job_id)
                analyzed = load_job(job_id)
                self.assertEqual(analyzed["status"], "review", analyzed.get("error"))
                self.assertTrue(analyzed["segments"])
                self.assertTrue(all(item.get("translation") for item in analyzed["segments"]))

                render(job_id)
                rendered = load_job(job_id)
                self.assertEqual(rendered["status"], "complete", rendered.get("error"))
                self.assertGreater(rendered["artifacts"]["rendered_lines"], 0)
                self.assertTrue(output_path(job_id).is_file())

                # A rerender must reflect current review state, never reuse an obsolete line WAV.
                for item in rendered["segments"]:
                    item["translation"] = ""
                save_job(rendered)
                render(job_id)
                rerendered = load_job(job_id)
                self.assertEqual(rerendered["status"], "failed")
                self.assertIn("no rendered dialogue", rerendered["error"])
            finally:
                if job_id:
                    # Test owns both paths and cleans only its opaque synthetic job/export.
                    shutil.rmtree(job_dir(job_id), ignore_errors=True)
                    output_path(job_id).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
