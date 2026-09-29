# Changelog

## 0.2.0 — 2026-09-29

First public release of an experimental personal research project.

- Local, human-in-the-loop Japanese-to-English dubbing pipeline: separation, transcription,
  alignment, diarization, translation, voice-cloned synthesis, timing fit, mix and mux.
- Standard-library orchestrator, loopback-only HTTP API and web studio; model workers isolated in
  their own Python environments.
- Series voice bank with three-zone matching and an append-only decision log.
- Optional slot-aware dialogue adaptation with a meaning-preservation validator.
- GPU arm / preflight / lease gates and a fail-closed thermal guard for the episode queue.
- Experiment harness with a blind review page.
- Pinned, verifiable model downloads; 241 unit tests, an end-to-end smoke test and CI.
