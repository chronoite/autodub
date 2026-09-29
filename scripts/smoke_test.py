"""End-to-end smoke test of an installed AutoDub checkout, using the real models and ffmpeg.

Runs the whole CPU pipeline — import, analyze, render — in a throwaway data directory and checks
that the export is a playable video with an audio track of the expected length. Nothing touches
the GPU and nothing leaves the machine.

    python scripts/smoke_test.py                      # synthetic English clip (Windows voices)
    python scripts/smoke_test.py --source clip.mkv    # your own short Japanese clip
    python scripts/smoke_test.py --keep               # keep the data directory to inspect

Prerequisites: `python -m autodub doctor` reports "core": true.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_clip(folder: Path, ffmpeg: Path) -> Path:
    from autodub.adapters import sapi_voices, synthesize_sapi

    voices = sapi_voices()
    if not voices:
        raise SystemExit("no Windows SAPI voice available: pass --source <short video with speech>")
    speech = folder / "speech.wav"
    synthesize_sapi("This short clip checks the complete local AutoDub pipeline from start to finish.",
                    voices[0], speech)
    video = folder / "synthetic.mp4"
    subprocess.run(
        [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=s=640x360:d=8:r=24",
         "-f", "lavfi", "-i", "sine=frequency=110:sample_rate=48000:duration=8",
         "-i", str(speech),
         "-filter_complex", "[1:a]volume=0.08[bed];[2:a]adelay=1000|1000,apad[voice];"
                            "[bed][voice]amix=inputs=2:duration=first[a]",
         "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-t", "8", str(video)],
        check=True,
    )
    return video


def _probe(ffprobe: Path, path: Path) -> dict:
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries", "format=duration:stream=codec_type",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoDub end-to-end smoke test")
    parser.add_argument("--source", type=Path, help="short local video with speech (default: synthetic)")
    parser.add_argument("--language", default=None, help="source language code (default: ja for --source, en for synthetic)")
    parser.add_argument("--keep", action="store_true", help="keep the temporary data directory")
    args = parser.parse_args()

    home = Path(tempfile.mkdtemp(prefix="autodub-smoke-"))
    os.environ["AUTODUB_HOME"] = str(home)          # must be set before autodub is imported
    sys.path.insert(0, str(ROOT / "src"))

    from autodub.cli import import_source
    from autodub.config import FFMPEG, FFPROBE, ensure_layout
    from autodub.pipeline import analyze, render
    from autodub.quality_profiles import CPU_PROFILE, get_profile
    from autodub.state import load_job, output_path, save_job

    ensure_layout()
    started = time.monotonic()
    try:
        source = args.source or _synthetic_clip(home, FFMPEG)
        language = args.language or ("ja" if args.source else "en")
        job = import_source(source, CPU_PROFILE)
        job_id = job["id"]
        job["settings"].update({
            "source_language": language,
            "target_language": "en",
            "quality_profile": CPU_PROFILE,
            "quality_stack": get_profile(CPU_PROFILE),
        })
        save_job(job)
        print(f"[1/3] imported {job_id}", flush=True)

        analyze(job_id)
        analyzed = load_job(job_id)
        if analyzed["status"] != "review":
            raise SystemExit(f"analyze failed: {analyzed.get('error')}")
        print(f"[2/3] analyzed: {len(analyzed['segments'])} segment(s), "
              f"{len({s.get('speaker') for s in analyzed['segments']})} speaker(s)", flush=True)
        for segment in analyzed["segments"][:5]:
            print(f"      {segment['start']:6.2f}s  {segment.get('speaker')}: {segment.get('translation')}")

        render(job_id)
        rendered = load_job(job_id)
        if rendered["status"] != "complete":
            raise SystemExit(f"render failed: {rendered.get('error')}")
        export = output_path(job_id)
        info = _probe(FFPROBE, export)
        streams = sorted(s["codec_type"] for s in info.get("streams", []))
        duration = float(info["format"]["duration"])
        source_duration = float(_probe(FFPROBE, source)["format"]["duration"])
        ok = streams == ["audio", "video"] and abs(duration - source_duration) < 1.0
        print(f"[3/3] rendered {export.name}: streams={streams} duration={duration:.2f}s "
              f"(source {source_duration:.2f}s)", flush=True)
        print(json.dumps({"ok": ok, "job": job_id, "seconds": round(time.monotonic() - started, 1),
                          "export": str(export) if args.keep else None}, indent=2))
        return 0 if ok else 1
    finally:
        if args.keep:
            print(f"data kept in {home}")
        else:
            shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
