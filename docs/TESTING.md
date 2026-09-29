# Testing

## Unit and contract tests

```bash
pip install -r requirements/test.txt      # NumPy, used by worker helpers under test
python -m unittest discover -s tests -t .
```

- **Isolation.** `tests/__init__.py` runs before any test module: it puts `src/` on the path and
  points `AUTODUB_HOME` at a fresh temporary directory, so the suite can never read or modify real
  jobs. Set `AUTODUB_TEST_KEEP_HOME=1` to keep that directory for debugging.
- **Capability gates.** Tests that need ffmpeg, Windows speech voices, model weights, or worker
  packages skip with a stated reason when those are missing. On a bare machine (as in CI) the
  pure-logic suite runs; on a configured workstation the same command also exercises real media
  processing and model inference on synthetic audio.
- **What is covered.** Timing budgets and fit modes, mixing and chunked ffmpeg graphs, song
  detection, speaker evidence and voice-bank matching, cross-episode clustering constraints,
  adaptation validation, runaway-TTS guard, GPU arm/lease semantics, thermal fail-closed behaviour,
  queue overlap, HTTP security headers, opaque public job views, path-traversal safety of artifact
  routes, and source-level contracts that pin wiring between UI, server and pipeline.

## End-to-end smoke test

```bash
python scripts/smoke_test.py            # synthetic clip, CPU pipeline, real models
python scripts/smoke_test.py --source clip.mkv --keep
```

Runs import → analyze → render in a throwaway data directory and verifies the export has audio and
video streams and the source's duration. Requires `python -m autodub doctor` to report `core: true`.

## Model integrity

```bash
python scripts/verify_models.py          # full SHA-256 check against provenance records
python scripts/verify_models.py --quick  # sizes only
```

## Continuous integration

`.github/workflows/ci.yml` runs the unit suite on Ubuntu and Windows with Python 3.11 and 3.12,
byte-compiles every module, and syntax-checks the web UI scripts. CI has no models or GPU, so
inference tests skip there; the smoke test is the local counterpart.
