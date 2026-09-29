# Third-party notices

AutoDub's own source code is released under the MIT License (see `LICENSE`).

**This repository contains no third-party code, model weights, or binaries.** Every component
below is installed separately by the user — Python packages with `pip`, model weights with
`scripts/download_models.py`, and external programs from their own distributors — and each is
governed by its own license. By installing and using a component you accept its license terms.
The table records the license each project declares so users can check compatibility with their
intended use.

Licenses were taken from the installed package metadata, the model cards at the pinned revisions,
the LICENSE files shipped with each component, and each upstream repository's declared license
(checked September 2026). Licenses can change between versions; recheck when upgrading.

## Model weights (downloaded by `scripts/download_models.py`)

| Model | Used for | License | Source |
|---|---|---|---|
| faster-whisper large-v3 (CTranslate2 conversion of OpenAI Whisper large-v3) | speech recognition | MIT | https://huggingface.co/Systran/faster-whisper-large-v3 |
| Helsinki-NLP opus-mt-ja-en | baseline Japanese→English translation | Apache-2.0 | https://huggingface.co/Helsinki-NLP/opus-mt-ja-en |
| pyannote speaker-diarization-community-1 | speaker diarization | CC-BY-4.0 (gated: accept the conditions on Hugging Face before downloading) | https://huggingface.co/pyannote/speaker-diarization-community-1 |
| wav2vec2-large-xlsr-53-japanese | Japanese forced alignment (WhisperX) | Apache-2.0 | https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese |
| Qwen3-TTS 12Hz 1.7B / 0.6B Base | voice-cloning speech synthesis | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base, https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base |
| Demucs htdemucs_ft | dialogue / music separation | MIT | https://github.com/facebookresearch/demucs |
| Qwen3-14B GGUF (Q8_0), optional | dialogue adaptation, contextual translation | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-14B-GGUF |
| Chatterbox multilingual, optional | alternative TTS (experiments) | MIT | https://huggingface.co/ResembleAI/chatterbox |
| Fun-CosyVoice3 0.5B, optional | alternative TTS (experiments) | Apache-2.0 | https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 |

**CC-BY-4.0 attribution (pyannote):** speaker diarization uses *pyannote speaker-diarization-community-1*
by pyannoteAI / CNRS, licensed under CC-BY-4.0.

**Voice cloning:** the TTS models clone voices from reference audio. Only clone voices you have the
right to use, and respect the rights of the performers and owners of any source media.

## Python packages (installed into the worker environments from `requirements/`)

| Package | License |
|---|---|
| faster-whisper, CTranslate2 | MIT |
| transformers, huggingface_hub, lightning, qwen-tts, modelscope, sentencepiece | Apache-2.0 |
| sacremoses | MIT |
| pyannote.audio | MIT |
| whisperx | BSD-2-Clause |
| demucs | MIT |
| PyTorch, torchaudio | BSD-3-Clause |
| NumPy, scikit-learn, soundfile | BSD-3-Clause |
| librosa | ISC |
| chatterbox-tts, spacy-pkuseg, onnxruntime | MIT |

## External programs (never bundled; configured by path)

| Program | Used for | License | Source |
|---|---|---|---|
| FFmpeg / ffprobe | all media decoding, mixing, muxing | LGPL-2.1+ or GPL-2.0+/GPL-3.0 depending on the build you install | https://ffmpeg.org/legal.html |
| KoboldCpp, optional | serves the Qwen3-14B GGUF for dialogue adaptation, run as a separate local process | AGPL-3.0 | https://github.com/LostRuins/koboldcpp |
| CosyVoice source, optional | CosyVoice3 inference code | Apache-2.0 | https://github.com/FunAudioLLM/CosyVoice |
| GPT-SoVITS, optional | alternative clone voice via its local HTTP API | MIT | https://github.com/RVC-Boss/GPT-SoVITS |
| nvidia-smi | GPU temperature for the thermal guard | NVIDIA driver utility (proprietary, ships with the driver) | https://developer.nvidia.com/nvidia-system-management-interface |
| Windows SAPI voices | CPU-profile speech synthesis | Windows operating-system component | https://learn.microsoft.com/previous-versions/windows/desktop/ms723627(v=vs.85) |

AutoDub runs these as separate programs — KoboldCpp and GPT-SoVITS over local HTTP (loopback by
default) — and does not include, link, or modify their code.
