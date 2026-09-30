# Third-party notices

AutoDub's own source code is released under the MIT License (see `LICENSE`).

**This repository contains no third-party model weights, binaries, or vendored source files.** One
short loader routine (`_load` in `src/autodub/workers/chatterbox_worker.py`) follows
`ChatterboxMultilingualTTS.from_local` from chatterbox-tts, Copyright (c) 2025 Resemble AI, MIT
License; that notice is reproduced at the end of this file.

Every component below is installed separately by the user — Python packages with `pip`, model
weights with `scripts/download_models.py`, and external programs from their own distributors — and
each is governed by its own license. By installing and using a component you accept its license
terms. The tables record the license each project declares so users can check compatibility with
their intended use.

Licenses were taken from the installed package metadata, PyPI, the model cards at the pinned
revisions, the LICENSE files shipped with each component, and each upstream repository's declared
license (checked September 2026). Licenses can change between versions; recheck when upgrading.

## Model weights (downloaded by `scripts/download_models.py`)

| Model | Used for | License | Source |
|---|---|---|---|
| faster-whisper large-v3 (CTranslate2 conversion of OpenAI Whisper large-v3) | speech recognition | MIT as declared by the Systran conversion (OpenAI's whisper repository is MIT; the openai/whisper-large-v3 Hugging Face card declares Apache-2.0) | https://huggingface.co/Systran/faster-whisper-large-v3 |
| Helsinki-NLP opus-mt-ja-en | baseline Japanese→English translation | Apache-2.0 | https://huggingface.co/Helsinki-NLP/opus-mt-ja-en |
| pyannote speaker-diarization-community-1 | speaker diarization | CC-BY-4.0 (gated: accept the access conditions on Hugging Face before downloading) | https://huggingface.co/pyannote/speaker-diarization-community-1 |
| wav2vec2-large-xlsr-53-japanese | Japanese forced alignment (WhisperX) | Apache-2.0 | https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese |
| Qwen3-TTS 12Hz 1.7B / 0.6B Base | voice-cloning speech synthesis | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base, https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base |
| Demucs htdemucs_ft | dialogue / music separation | MIT (weights are fetched from dl.fbaipublicfiles.com and have no separate license; the repository's MIT license covers them) | https://github.com/adefossez/demucs (maintained home of the pip package; the original facebookresearch/demucs is archived) |
| Qwen3-14B GGUF (Q8_0), optional | dialogue adaptation, contextual translation | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-14B-GGUF |
| Chatterbox multilingual, optional | alternative TTS (experiments) | MIT; outputs carry Resemble AI's imperceptible Perth watermark, which AutoDub leaves in place | https://huggingface.co/ResembleAI/chatterbox |
| Fun-CosyVoice3 0.5B, optional | alternative TTS (experiments) | Apache-2.0 | https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 |

**CC-BY-4.0 attribution (pyannote):** speaker diarization uses the *speaker-diarization-community-1*
pipeline by pyannote (https://huggingface.co/pyannote/speaker-diarization-community-1), licensed
under the Creative Commons Attribution 4.0 International License
(https://creativecommons.org/licenses/by/4.0/). AutoDub does not include or modify the model; you
download it unmodified from Hugging Face at a pinned revision after accepting its access conditions
(which ask for your organisation and use case and consent to occasional email from pyannote). The
model card lists papers to cite for research use.

**Training data:** some models were trained on data with non-commercial terms (for example the JSUT
corpus audio used for wav2vec2-large-xlsr-53-japanese, and MedleyDB tracks in MUSDB used for
Demucs). The weight licenses above are as declared by their publishers; whether dataset terms affect
use of the weights is legally unsettled. Check before any commercial use.

**Voice cloning:** the TTS models clone voices from reference audio. Only clone voices you have the
right to use, and respect the rights of the performers and owners of any source media.

## Python packages (installed into the worker environments from `requirements/`)

| Package | License |
|---|---|
| faster-whisper, CTranslate2 | MIT |
| transformers, huggingface_hub, safetensors, lightning, qwen-tts, modelscope, sentencepiece | Apache-2.0 |
| sacremoses | MIT |
| pyannote.audio | MIT |
| whisperx | BSD-2-Clause |
| demucs | MIT |
| PyTorch | BSD-3-Clause (BSD-style; see the PyTorch LICENSE) |
| torchaudio | BSD-2-Clause |
| NumPy, scikit-learn, soundfile | BSD-3-Clause |
| librosa | ISC |
| chatterbox-tts, spacy-pkuseg, onnxruntime | MIT |

### Notable dependencies pulled in by the packages above

| Package | Pulled in by | License | Note |
|---|---|---|---|
| pykakasi | chatterbox-tts | GPL-3.0-or-later | loaded only for Japanese synthesis text; AutoDub synthesizes English |
| lameenc | demucs | LGPL-3.0-or-later | MP3 encoding; AutoDub writes WAV |
| soxr | librosa | LGPL-2.1-or-later | |
| av (PyAV) | faster-whisper | BSD-3-Clause | binary wheels include FFmpeg libraries |
| torchcodec, torchvision | pyannote.audio, whisperx | BSD-3-Clause | torchcodec loads the FFmpeg you install |
| tokenizers, torchmetrics, accelerate, gradio, diffusers, s3tokenizer, nltk | various | Apache-2.0 | |
| pyannote.core, pyannote.pipeline, pyannote.metrics, pyannote.database | pyannote.audio | MIT | |
| pyannoteai-sdk | pyannote.audio | not declared | client for pyannote's hosted service; AutoDub never calls it |
| resemble-perth | chatterbox-tts | MIT | watermarks every Chatterbox output |
| sox (Python wrapper) | qwen-tts | BSD-3-Clause | uses the SoX program if installed |
| NVIDIA CUDA libraries (nvidia-* wheels), TensorRT (CosyVoice, Linux) | CUDA builds of PyTorch; CosyVoice | NVIDIA proprietary license | |
| CosyVoice requirements (openai-whisper, x-transformers, pyworld, HyperPyYAML, wetext, gdown), Matcha-TTS submodule | CosyVoice source | MIT / Apache-2.0 | |

None of these are redistributed by this repository; they are installed by pip into your own
environments. The GPL/LGPL packages above are not imported by AutoDub's own code.

## External programs (never bundled; configured by path)

| Program | Used for | License | Source |
|---|---|---|---|
| FFmpeg / ffprobe | all media decoding, mixing, muxing | LGPL-2.1+ or GPL-2.0+/GPL-3.0 depending on the build you install | https://ffmpeg.org/legal.html |
| KoboldCpp, optional | serves the Qwen3-14B GGUF for dialogue adaptation | AGPL-3.0 | https://github.com/LostRuins/koboldcpp |
| CosyVoice source, optional | CosyVoice3 inference code | Apache-2.0 | https://github.com/FunAudioLLM/CosyVoice |
| GPT-SoVITS, optional | alternative clone voice via its local HTTP API | MIT | https://github.com/RVC-Boss/GPT-SoVITS |
| nvidia-smi | GPU temperature for the thermal guard | NVIDIA driver utility (proprietary, ships with the driver) | https://developer.nvidia.com/system-management-interface |
| Windows SAPI voices | CPU-profile speech synthesis | Windows operating-system component | https://learn.microsoft.com/previous-versions/windows/desktop/ms723627(v=vs.85) |

AutoDub runs FFmpeg, nvidia-smi and PowerShell (Windows SAPI) as separate processes, launches
KoboldCpp as a loopback child process and talks to it over HTTP, and reaches GPT-SoVITS over local
HTTP; it does not include, link, or modify their code. The CosyVoice source is different:
`cosyvoice_worker.py` imports it as a Python library from your own checkout
(`AUTODUB_COSYVOICE_SOURCE`) and replaces a few of its download and data-loading functions in memory
at run time. No CosyVoice file is copied into this repository or changed on disk.

## chatterbox-tts (MIT License)

The `_load` routine in `src/autodub/workers/chatterbox_worker.py` follows
`ChatterboxMultilingualTTS.from_local` in chatterbox-tts (https://github.com/resemble-ai/chatterbox).

Copyright (c) 2025 Resemble AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
