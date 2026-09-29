"""Pinned Chatterbox Multilingual V3 experiment worker.

PyPI 0.1.7 predates the small `t3_model="v3"` loader addition now present upstream, so this worker
constructs the official V3 stack from the same installed library components and the exact local
official weights.  It never calls `from_pretrained` or any downloader.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path


os.environ.update(
    {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
)

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autodub.config import CHATTERBOX_V3 as MODEL_ROOT  # noqa: E402

REQUIRED = (
    "ve.pt",
    "t3_mtl23ls_v3.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
)


class _OfflineCjkSegmenter:
    """Deterministic fallback for the tokenizer's otherwise eager pkuseg download."""

    @staticmethod
    def cut(text: str):
        return [character for character in text if character.strip()]


def _load(device: str):
    import torch
    from safetensors.torch import load_file as load_safetensors
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    import chatterbox.models.tokenizers.tokenizer as tokenizer_module
    import spacy_pkuseg
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder

    missing = [name for name in REQUIRED if not (MODEL_ROOT / name).is_file()]
    if missing:
        raise RuntimeError("pinned Chatterbox V3 files are missing")
    pkuseg_root = MODEL_ROOT / "pkuseg"
    pkuseg_root.mkdir(exist_ok=True)
    os.environ["PKUSEG_HOME"] = str(pkuseg_root)
    # MTLTokenizer constructs its CJK segmenter even for English inference. Do not let that import
    # bootstrap an external model; target-English AutoDub does not use it, and the local fallback is
    # deterministic if CJK normalization is encountered.
    spacy_pkuseg.pkuseg = lambda *_args, **_kwargs: _OfflineCjkSegmenter()
    tokenizer_module.hf_hub_download = lambda **_kwargs: str(MODEL_ROOT / "Cangjie5_TC.json")
    map_location = torch.device("cpu") if device == "cpu" else None
    voice_encoder = VoiceEncoder()
    voice_encoder.load_state_dict(torch.load(MODEL_ROOT / "ve.pt", map_location=map_location, weights_only=True))
    voice_encoder.to(device).eval()
    t3 = T3(T3Config.multilingual())
    t3.load_state_dict(load_safetensors(MODEL_ROOT / "t3_mtl23ls_v3.safetensors"))
    t3.to(device).eval()
    s3gen = S3Gen()
    s3gen.load_state_dict(torch.load(MODEL_ROOT / "s3gen.pt", map_location=map_location, weights_only=True))
    s3gen.to(device).eval()
    tokenizer = MTLTokenizer(str(MODEL_ROOT / "grapheme_mtl_merged_expanded_v1.json"))
    conds = None
    if (MODEL_ROOT / "conds.pt").is_file():
        conds = Conditionals.load(MODEL_ROOT / "conds.pt", map_location=map_location).to(device)
    return ChatterboxMultilingualTTS(t3, s3gen, voice_encoder, tokenizer, device, conds=conds)


def synthesize(payload: dict) -> dict:
    import soundfile as sf
    import torch

    device = str(payload.get("device") or "cuda")
    if device not in {"cuda", "cpu"}:
        raise RuntimeError("Chatterbox device must be cuda or cpu")
    seed = int(payload.get("seed", 1986))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _load(device)
    written = 0
    for line in payload["lines"]:
        destination = Path(line["output"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        waveform = model.generate(
            text=line["text"],
            language_id=line.get("language_id") or "en",
            audio_prompt_path=line["reference_audio"],
            exaggeration=float(line.get("exaggeration", 0.5)),
            cfg_weight=float(line.get("cfg_weight", 0.5)),
            temperature=float(line.get("temperature", 0.8)),
        )
        sf.write(str(destination), waveform.squeeze(0).detach().cpu().numpy(), model.sr)
        written += 1
    return {"model": "chatterbox-multilingual-v3", "device": device, "seed": seed, "written": written}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "synthesize":
        raise SystemExit("usage: chatterbox_worker.py synthesize")
    payload = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result = synthesize(payload)
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
