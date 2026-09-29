"""Pinned model manifest shared by download_models.py and verify_models.py.

Every entry pins an exact upstream revision so an install is reproducible, and records the
license declared by the upstream model card. Destinations are the paths autodub.config expects
under AUTODUB_MODELS_DIR. Nothing here is imported by the AutoDub runtime, which never downloads.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autodub import config  # noqa: E402

# kind "hf": a Hugging Face snapshot. kind "url": direct files (sha256 prefix is in the filename).
MODELS: dict[str, dict] = {
    "whisper": {
        "kind": "hf",
        "group": "core",
        "repo_id": "Systran/faster-whisper-large-v3",
        "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
        "destination": config.WHISPER_MODEL,
        "license": "MIT",
        "purpose": "speech recognition (CTranslate2 conversion of OpenAI Whisper large-v3)",
    },
    "opus-mt-ja-en": {
        "kind": "hf",
        "group": "core",
        "repo_id": "Helsinki-NLP/opus-mt-ja-en",
        "revision": "0770961a39ba6bd66305b149c3f4110bcafca2e6",
        "destination": config.TRANSLATION_MODEL,
        "license": "Apache-2.0",
        "purpose": "baseline line-by-line Japanese-to-English translation (CPU)",
        "allow_patterns": ["*.json", "*.txt", "*.spm", "*.bin", "*.safetensors", "README.md", "LICENSE*"],
    },
    "pyannote-community-1": {
        "kind": "hf",
        "group": "quality",
        "repo_id": "pyannote/speaker-diarization-community-1",
        "revision": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
        "destination": config.PYANNOTE_MODEL,
        "license": "CC-BY-4.0",
        "purpose": "speaker diarization (gated: accept the conditions on Hugging Face and log in first)",
        "token": True,
    },
    "whisperx-ja-align": {
        "kind": "hf",
        "group": "quality",
        "repo_id": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        "revision": "2785e99ab97df77a32b5bd0ece5c9fa188a02f19",
        "destination": config.WHISPERX_JA_ALIGN_MODEL,
        "license": "Apache-2.0",
        "purpose": "Japanese forced alignment for WhisperX",
    },
    "qwen3-tts-1.7b": {
        "kind": "hf",
        "group": "quality",
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "revision": "fd4b254389122332181a7c3db7f27e918eec64e3",
        "destination": config.QWEN_TTS_17B,
        "license": "Apache-2.0",
        "purpose": "default voice-cloning TTS",
    },
    "qwen3-tts-0.6b": {
        "kind": "hf",
        "group": "optional",
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "revision": "5d83992436eae1d760afd27aff78a71d676296fc",
        "destination": config.QWEN_TTS_06B,
        "license": "Apache-2.0",
        "purpose": "smaller voice-cloning TTS (experiments)",
    },
    "demucs-htdemucs-ft": {
        "kind": "url",
        "group": "quality",
        "base_url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/",
        "files": ["f7e0c4bc-ba3fe64a.th", "d12395a8-e57c48e6.th", "92cfc3b6-ef3bcb9c.th", "04573f0d-f3cf25b2.th"],
        "destination": config.AUTODUB_MODEL_CACHE / "torch" / "hub" / "checkpoints",
        "license": "MIT",
        "purpose": "dialogue/music separation (htdemucs_ft)",
    },
    "qwen3-14b-gguf": {
        "kind": "hf",
        "group": "optional",
        "repo_id": "Qwen/Qwen3-14B-GGUF",
        "revision": "530227a7d994db8eca5ab5ced2fb692b614357fd",
        "destination": config.QWEN3_CONTEXTUAL_GGUF.parent,
        "license": "Apache-2.0",
        "purpose": "dialogue adaptation and contextual translation (served by KoboldCpp)",
        "allow_patterns": ["Qwen3-14B-Q8_0.gguf", "README.md", "LICENSE", "params"],
    },
    "chatterbox-multilingual": {
        "kind": "hf",
        "group": "optional",
        "repo_id": "ResembleAI/chatterbox",
        "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
        "destination": config.CHATTERBOX_V3,
        "license": "MIT",
        "purpose": "alternative voice-cloning TTS (experiments)",
        "allow_patterns": ["ve.pt", "t3_mtl23ls_v3.safetensors", "s3gen.pt",
                           "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json", "README.md"],
    },
    "cosyvoice3": {
        "kind": "hf",
        "group": "optional",
        "repo_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "revision": "29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
        "destination": config.COSYVOICE3,
        "license": "Apache-2.0",
        "purpose": "alternative voice-cloning TTS (experiments)",
        "allow_patterns": ["CosyVoice-BlankEN/*", "README.md", "campplus.onnx", "config.json", "configuration.json",
                           "cosyvoice3.yaml", "flow.pt", "hift.pt", "llm.pt", "speech_tokenizer_v3.onnx"],
    },
}

GROUPS = ("core", "quality", "optional")
PROVENANCE = "MODEL-PROVENANCE.json"
