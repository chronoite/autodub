"""Runtime configuration.

Every path and external endpoint is resolved from environment variables so AutoDub runs from any
checkout without editing code. Defaults assume a self-contained layout next to the repository
(or ``~/.autodub`` when AutoDub is installed as a package)::

    <repo>/data/      jobs, uploads, outputs, voice references, logs   (AUTODUB_HOME)
    <repo>/models/    model weights, fetched by scripts/                 (AUTODUB_MODELS_DIR)

Tunable thresholds live here as named constants rather than being scattered through the code, so a
behavioural change is always a reviewed one-line diff.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


def _env_python(name: str, default: Path) -> Path:
    """Interpreter for one worker environment. Unset means "use the interpreter running AutoDub"."""
    return _env_path(name, default)


# ---- locations -------------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]            # repository root (src/autodub -> repo)
# Running from a checkout keeps data and models beside the repository; an installed package
# (no pyproject.toml above it) defaults to a per-user folder instead of site-packages.
_BASE = PROJECT_ROOT if (PROJECT_ROOT / "pyproject.toml").is_file() else Path.home() / ".autodub"
STATIC_ROOT = PACKAGE_ROOT / "web" / "static"

DATA_ROOT = _env_path("AUTODUB_HOME", _BASE / "data")
INPUT_ROOT = DATA_ROOT / "input"
WORK_ROOT = DATA_ROOT / "work"
OUTPUT_ROOT = DATA_ROOT / "out"
VOICES_ROOT = DATA_ROOT / "voices"
# Optional extra site-packages prepended for CPU workers (pure-Python packages installed with
# ``pip install --target``); ignored when the directory does not exist.
RUNTIME_DEPS = _env_path("AUTODUB_RUNTIME_DEPS", _BASE / "runtime_deps")

MODELS_ROOT = _env_path("AUTODUB_MODELS_DIR", _BASE / "models")
WHISPER_MODEL = MODELS_ROOT / "stt" / "faster-whisper-large-v3"
TRANSLATION_MODEL = MODELS_ROOT / "translation" / "opus-mt-ja-en"
PYANNOTE_MODEL = MODELS_ROOT / "diarization" / "pyannote-community-1"
WHISPERX_JA_ALIGN_MODEL = MODELS_ROOT / "alignment" / "whisperx-ja-wav2vec2"
QWEN3_CONTEXTUAL_GGUF = MODELS_ROOT / "translation" / "qwen3-14b-contextual" / "Qwen3-14B-Q8_0.gguf"
QWEN_TTS_17B = MODELS_ROOT / "tts" / "qwen3-tts" / "1.7B-Base"
QWEN_TTS_06B = MODELS_ROOT / "tts" / "qwen3-tts" / "0.6B-Base"
CHATTERBOX_V3 = MODELS_ROOT / "tts" / "chatterbox-multilingual"
COSYVOICE3 = MODELS_ROOT / "tts" / "cosyvoice3"
AUTODUB_MODEL_CACHE = MODELS_ROOT / "cache"

# ---- worker interpreters (each model family can live in its own virtualenv) ------------------
_SELF = Path(sys.executable)
MEDIA_PYTHON = _env_python("AUTODUB_PYTHON_MEDIA", _SELF)
QUALITY_PYTHON = _env_python("AUTODUB_PYTHON_QUALITY", MEDIA_PYTHON)
ANALYSIS_PYTHON = _env_python("AUTODUB_PYTHON_ANALYSIS", MEDIA_PYTHON)
CHATTERBOX_PYTHON = _env_python("AUTODUB_PYTHON_CHATTERBOX", MEDIA_PYTHON)
COSYVOICE_PYTHON = _env_python("AUTODUB_PYTHON_COSYVOICE", MEDIA_PYTHON)
COSYVOICE_SOURCE = _env_path("AUTODUB_COSYVOICE_SOURCE", _BASE / "third_party" / "CosyVoice")
_extra_site = os.environ.get("AUTODUB_COSYVOICE_EXTRA_SITE", "").strip()
COSYVOICE_EXTRA_SITE = Path(_extra_site).expanduser() if _extra_site else None

# ---- external tools and services --------------------------------------------------------------
FFMPEG = _env_path("AUTODUB_FFMPEG", Path(shutil.which("ffmpeg") or "ffmpeg"))
FFPROBE = _env_path("AUTODUB_FFPROBE", Path(shutil.which("ffprobe") or FFMPEG.with_name(
    "ffprobe.exe" if FFMPEG.suffix == ".exe" else "ffprobe")))
KOBOLDCPP = _env_path("AUTODUB_KOBOLDCPP", Path(shutil.which("koboldcpp") or "koboldcpp"))
GPT_SOVITS_URL = os.environ.get("AUTODUB_GPT_SOVITS_URL", "http://127.0.0.1:9880").rstrip("/")
# Optional HTTP lease broker shared with other GPU applications on the same machine. Empty means
# AutoDub coordinates GPU work with an in-process lock only (see gpu_session.py).
GPU_BROKER_URL = os.environ.get("AUTODUB_GPU_BROKER_URL", "").rstrip("/")

HOST = os.environ.get("AUTODUB_HOST", "127.0.0.1")
PORT = int(os.environ.get("AUTODUB_PORT", "8030"))
MAX_UPLOAD_BYTES = 80 * 1024**3
ALLOWED_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}

# ---- speaker evidence --------------------------------------------------------------------------
# Duplicate-speaker hints are advisory only; they never trigger an automatic merge.
SPEAKER_DUPLICATE_COSINE_THRESHOLD = 0.75
SPEAKER_OVERLAP_VETO_MS = 300
# Short-utterance eligibility is measured from voiced diarization time, never the transcript window.
SPEAKER_EVIDENCE_MIN_VOICED_SECONDS = 1.2
SPEAKER_EVIDENCE_LOW_CHUNK_COUNT = 3
SPEAKER_EVIDENCE_MANIFEST_LIMIT = 5
SPEAKER_EVIDENCE_BORDERLINE_LIMIT = 2

# ---- voice bank --------------------------------------------------------------------------------
# Three-zone matching: cosine >= MATCH auto-assigns (logged); >= ASK asks the reviewer; below ASK
# proposes a new character. Calibrated on real episodes: different speakers scored 0.67-0.91
# against each other (worst pair 0.914) while the same character across episodes scored 0.94-0.99,
# so MATCH sits above the worst different-speaker pair with margin and ASK covers the ambiguous band.
VOICE_BANK_MATCH_THRESHOLD = 0.95
VOICE_BANK_ASK_THRESHOLD = 0.90
# The embedding space centroids live in. Changing it quarantines every stored centroid (cross-space
# cosine is meaningless) until refresh_centroid rebuilds each one.
VOICE_BANK_EMBEDDER = "pyannote-embedding-v1"
# Confidence lights for multi-member character groups: GREEN when the worst pairwise link sits deep
# in same-character territory and every member has real solo audio; RED when the worst link hugs the
# join threshold or a member has too little solo evidence; AMBER otherwise.
GROUP_GREEN_COHESION = 0.965
GROUP_RED_COHESION = 0.955
GROUP_GREEN_MIN_SOLO_S = 3.0
GROUP_RED_MIN_SOLO_S = 1.5

# ---- evidence cards ----------------------------------------------------------------------------
EVIDENCE_CARD_CLIPS = 3
EVIDENCE_CARD_CLIP_PAD_S = 0.15
EVIDENCE_CARD_MIN_CLIP_S = 1.2
EVIDENCE_CARD_MAX_CLIP_S = 8.0
# Click-to-play clips: short segments are expanded around their midpoint so one-liners still show
# enough motion to identify the speaker; long ones clamp to the maximum.
EVIDENCE_VIDEO_MIN_S = 5.0
EVIDENCE_VIDEO_MAX_S = 30.0
EVIDENCE_VIDEO_HEIGHT = 360

# ---- dialogue adaptation -----------------------------------------------------------------------
# The rewriter is a local GGUF model served by a loopback-only KoboldCpp child process on ADAPT_PORT.
ADAPT_PORT = int(os.environ.get("AUTODUB_ADAPT_PORT", "5031"))
# Ports that must be free before the adaptation model starts (for example another LLM server that
# would compete for GPU memory). Comma-separated; empty disables the check.
ADAPT_CONFLICT_PORTS = tuple(int(p) for p in os.environ.get("AUTODUB_ADAPT_CONFLICT_PORTS", "").split(",") if p.strip())
ADAPT_CPS = 10.8                 # characters/second the synthesized voices actually speak (measured
                                 # with scripts/measure_cps.py; re-measure per voice cast)
ADAPT_TAIL_SLACK_S = 0.15        # tolerated late tail (ITU-R BT.1359: late audio up to ~125-185 ms is
                                 # imperceptible; early is not, so lines never start early)
ADAPT_GUARD_S = 0.08             # inter-line guard when another line follows closely
ADAPT_MAX_REWRITES = 2           # shortening attempts per line before it is flagged
ADAPT_SHORT_RATIO = 0.60         # estimate under this fraction of the slot counts as "too short"...
ADAPT_SHORT_MIN_SLOT_S = 1.5     # ...but only for slots at least this long
ADAPT_EXPAND_FILL = 0.80         # expansion rewrites aim at ~80% of the slot, never 100%
ADAPT_BATCH_LINES = 10           # lines per LLM request
ADAPT_LLM_TIMEOUT_S = 900
ADAPT_LLM_TEMPERATURE = 0.3      # fidelity over flair
ADAPT_LEASE_WAIT_S = 10800       # maximum wait for a GPU lease

# ---- non-verbal passthrough --------------------------------------------------------------------
# Quality renders replace the vocal stem, so vocalisations the transcript does not cover (screams,
# laughs, gasps) would go silent. Voiced windows that no transcript segment covers are sliced from
# the original vocals and mixed back at unity gain. Original audio is never removed.
NONVERBAL_SILENCE_DB = -30.0
NONVERBAL_SILENCE_MIN_S = 0.35
NONVERBAL_COVER_GUARD_S = 0.05
NONVERBAL_MIN_S = 0.30
NONVERBAL_MERGE_GAP_S = 0.20
NONVERBAL_MAX_TOTAL_S = 150.0    # sanity cap per episode; overflow is logged, never silent

# ---- runaway-TTS guard -------------------------------------------------------------------------
# A corrupted voice reference can make an autoregressive TTS model babble minutes of audio for a
# one-second slot. One knob pair is shared by the QC detector and the synthesis-time guard so their
# verdicts can never disagree.
TTS_RUNAWAY_FACTOR = 8.0         # raw synthesized audio > FACTOR x slot length => runaway
TTS_RUNAWAY_MIN_SLOT_S = 1.0     # floor so a normal overrun in a tiny slot is not called a runaway
TTS_RUNAWAY_RETRIES = 1          # same-reference retries before switching to the next clean reference
SPEAKER_REF_MAX_OVERLAP = 0.05   # reference windows must be this free of overlapping speech...
SPEAKER_REF_MIN_CONFIDENCE = 0.90  # ...and this confidently attributed
SPEAKER_REF_ALTERNATES = 2       # spare clean windows extracted per speaker for the guard

# ---- thermal-aware episode queue ---------------------------------------------------------------
# The guard fails closed: an unreadable GPU stops the queue rather than risking the hardware.
# AUTODUB_GPU_TEMP_COMMAND may point at a script that prints "core NN C" / "memory junction NN C"
# lines (useful where nvidia-smi cannot see memory-junction temperature); otherwise nvidia-smi is used.
GPU_TEMP_COMMAND = os.environ.get("AUTODUB_GPU_TEMP_COMMAND", "").strip()
THERMAL_GUARD_ENABLED = os.environ.get("AUTODUB_THERMAL_GUARD", "1") != "0"
THERMAL_ABORT_C = 95.0           # abort the queue before an episode at/above this temperature
THERMAL_WARN_C = 93.0            # log a warning but continue in this band
THERMAL_COOLDOWN_S = 120         # interruptible pause between GPU episodes
THERMAL_READ_TIMEOUT_S = 60

# ---- TTS throughput levers (both default to the proven serial behaviour) ------------------------
TTS_BATCH_SIZE = 1               # lines per batched generate call; 1 = serial
TTS_BATCH_SORT = True            # length-sort batch groups to minimise padding (only when > 1)
QUEUE_OVERLAP_POST_STAGES = False  # run the CPU tail (align/mix/mux) while the next episode synthesizes
QUEUE_MAX_PENDING_TAILS = 1


def ensure_layout() -> None:
    for path in (INPUT_ROOT, WORK_ROOT, OUTPUT_ROOT, VOICES_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def runtime_env(*, gpu: bool = False, portable_deps: bool = True) -> dict[str, str]:
    """Return a fail-offline environment for every model worker.

    GPU visibility is an explicit opt-in: starting the UI or running ``doctor`` keeps CUDA hidden,
    and quality workers only see the GPU inside a coordinated lease.
    """
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "PYANNOTE_METRICS_ENABLED": "0",
            "HF_HOME": str(AUTODUB_MODEL_CACHE),
            "AUTODUB_MODELS_DIR": str(MODELS_ROOT),
            "AUTODUB_COSYVOICE_SOURCE": str(COSYVOICE_SOURCE),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    if FFMPEG.parent != Path("."):
        env["PATH"] = str(FFMPEG.parent) + os.pathsep + env.get("PATH", "")
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    if portable_deps and RUNTIME_DEPS.exists():
        env["PYTHONPATH"] = str(RUNTIME_DEPS) + os.pathsep + env.get("PYTHONPATH", "")
    elif not portable_deps:
        env.pop("PYTHONPATH", None)
    return env
