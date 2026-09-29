# AutoDub quality experiments

This package holds the experiment registry (`registry.json`) and the executors that run it. The
registry contains plans and judging contracts only — never videos, transcripts, speaker names, or
rendered media. Runs live under `$AUTODUB_HOME/work/experiments/<opaque-run-id>/` and refer to an
opaque AutoDub job ID.

Each experiment changes **one** component (voice model, diarizer, aligner, separator, translator,
mix or timing policy) while holding the source scene and the review criteria constant. CPU is a
first-class candidate whenever it may match GPU quality; the same model can appear as both a CPU
and a GPU variant.

## Running an experiment

```bash
# 1. Materialize a run manifest (no inference, no GPU)
python -m autodub.experiments.create_run --experiment voice-quality-v1 --job <job-id>

# 2. Check readiness without loading any model
python -m autodub.experiments.voice_screen --run <run-id> --speaker speaker-01 --dry-run

# 3. Execute (reads local job media, so it must be acknowledged explicitly)
python -m autodub.experiments.voice_screen --run <run-id> --speaker speaker-01 \
    --ack-private-local-material
```

Executors: `speaker_screen`, `voice_screen`, `timing_screen`, `separation_screen`,
`translation_screen`. Every executor supports `--dry-run`. GPU candidates additionally require
`--arm-gpu` and acquire the GPU lease (see `docs/GPU-COORDINATION.md`). The timing and separation
executors write no source text, source filenames, or host paths into their evidence.

The contextual translation candidate starts the same loopback-only KoboldCpp child that dialogue
adaptation uses and stops its whole process tree when done.

Mix and timing comparisons (`dub-*` experiments) and TTS comparisons can also be built from the web
UI (Experiment mode on a reviewed job).

## Judging

Open **Experiment review** (`/experiments.html`). Candidates are shown under neutral letters in a
per-run shuffled order; each gets a pass / maybe / fail verdict and an optional one-word main
problem (timing, voice, mix, text). No winner is ever promoted from automated metrics alone.
