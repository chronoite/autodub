# Setup

AutoDub is one small standard-library application plus a few *worker environments* that hold the
machine-learning stacks. Workers are separate because the models need different, mutually
incompatible PyTorch/CUDA builds; each worker environment is just a Python virtual environment
whose interpreter path you give AutoDub.

## 1. Application

```bash
git clone <this repository> autodub
cd autodub
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -e ".[download]"      # [download] adds huggingface_hub for scripts/download_models.py
```

Install **FFmpeg** (with ffprobe) and either put it on `PATH` or set `AUTODUB_FFMPEG` and
`AUTODUB_FFPROBE`.

## 2. Worker environments

| Environment | Setting | Requirements | Needed for |
|---|---|---|---|
| media (CPU) | `AUTODUB_PYTHON_MEDIA` | `requirements/media.txt` | both profiles |
| analysis (GPU) | `AUTODUB_PYTHON_ANALYSIS` | `requirements/analysis.txt` | quality profile |
| tts (GPU) | `AUTODUB_PYTHON_QUALITY` | `requirements/tts.txt` | quality profile |
| chatterbox | `AUTODUB_PYTHON_CHATTERBOX` | `requirements/chatterbox.txt` | TTS experiments only |
| cosyvoice | `AUTODUB_PYTHON_COSYVOICE` | `requirements/cosyvoice.txt` | TTS experiments only |

Example for the CPU environment on Windows:

```bash
python -m venv envs\media
envs\media\Scripts\pip install -r requirements\media.txt
set AUTODUB_PYTHON_MEDIA=%CD%\envs\media\Scripts\python.exe
```

For the GPU environments, install the CUDA build of PyTorch that matches the version pinned in the
requirements file first (from https://pytorch.org), then the rest of the file. Any environment
left unset falls back to the interpreter running AutoDub.

## 3. Models

```bash
python scripts/download_models.py --accept-online-download core      # Whisper + Marian (CPU pipeline)
python scripts/download_models.py --accept-online-download quality   # Demucs, pyannote, WhisperX, Qwen3-TTS
python scripts/download_models.py --accept-online-download optional  # Qwen3-14B GGUF, Chatterbox, CosyVoice
python scripts/verify_models.py                                       # offline SHA-256 check
```

Models go to `AUTODUB_MODELS_DIR` (default `<repo>/models`). Every model is pinned to an upstream
revision and gets a provenance file listing each file's hash. **pyannote is gated**: accept its
conditions on its Hugging Face page and run `huggingface-cli login` before downloading the
`quality` group. Check each model's license before use — see `THIRD-PARTY-NOTICES.md`.

## 4. Optional components

- **Dialogue adaptation** needs [KoboldCpp](https://github.com/LostRuins/koboldcpp) (set
  `AUTODUB_KOBOLDCPP`) and the `qwen3-14b-gguf` model.
- **GPT-SoVITS** voices (CPU profile) need a running GPT-SoVITS API server
  (`AUTODUB_GPT_SOVITS_URL`, default `http://127.0.0.1:9880`).
- **CosyVoice** experiments need the CosyVoice source tree (`AUTODUB_COSYVOICE_SOURCE`).

## 5. Check and run

```bash
python -m autodub doctor     # reports each check; "ready": {"core": true, ...}
python -m autodub serve      # http://127.0.0.1:8030
python scripts/smoke_test.py # optional end-to-end check with the real models
```

## Configuration reference

All settings are environment variables; `.env.example` lists every one with its default.

| Variable | Default | Purpose |
|---|---|---|
| `AUTODUB_HOME` | `<repo>/data` | jobs, uploads, outputs, voice bank, logs |
| `AUTODUB_MODELS_DIR` | `<repo>/models` | model weights |
| `AUTODUB_HOST`, `AUTODUB_PORT` | `127.0.0.1`, `8030` | web UI (loopback only) |
| `AUTODUB_FFMPEG`, `AUTODUB_FFPROBE` | from `PATH` | media tools |
| `AUTODUB_PYTHON_*` | current interpreter | worker environments (table above) |
| `AUTODUB_KOBOLDCPP` | from `PATH` | adaptation LLM server |
| `AUTODUB_ADAPT_PORT` | `5031` | loopback port for the adaptation LLM child |
| `AUTODUB_ADAPT_CONFLICT_PORTS` | none | ports that must be free before GPU work (e.g. another LLM server) |
| `AUTODUB_GPU_BROKER_URL` | none | optional shared GPU lease broker |
| `AUTODUB_THERMAL_GUARD` | `1` | `0` disables the episode-queue temperature abort (temps are still logged) |
| `AUTODUB_GPU_TEMP_COMMAND` | nvidia-smi | custom temperature command |
| `AUTODUB_GPT_SOVITS_URL` | `http://127.0.0.1:9880` | GPT-SoVITS API |
| `AUTODUB_RUNTIME_DEPS` | `<repo>/runtime_deps` | optional extra site-packages for CPU workers |

Tuning constants (voice-bank thresholds, timing budgets, mix and thermal limits) are named constants
in `src/autodub/config.py`, each documented with the measurement it came from.
