"""Pinned official CosyVoice3 zero-shot experiment worker.

The official source and model paths are local.  Downloader functions are replaced with a hard error
before importing the runtime, so even an accidental non-local model argument cannot reach the network.
"""
from __future__ import annotations

import contextlib
import json
import os
import site
import sys
import types
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

from autodub.config import COSYVOICE3 as MODEL_ROOT  # noqa: E402
from autodub.config import COSYVOICE_EXTRA_SITE, COSYVOICE_SOURCE as SOURCE_ROOT  # noqa: E402

REQUIRED = (
    "cosyvoice3.yaml",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "CosyVoice-BlankEN/model.safetensors",
)


def _deny_download(*_args, **_kwargs):
    raise RuntimeError("CosyVoice runtime downloads are disabled; use the pinned local payload")


def _load():
    missing = [name for name in REQUIRED if not (MODEL_ROOT / name).is_file()]
    if missing:
        raise RuntimeError("pinned CosyVoice3 files are missing")
    if not (SOURCE_ROOT / "cosyvoice" / "cli" / "cosyvoice.py").is_file():
        raise RuntimeError("pinned CosyVoice source is missing")
    if COSYVOICE_EXTRA_SITE is not None:
        # Optional extra site-packages (for example one providing ``lightning``) is appended after
        # this environment's own packages, so it is a fallback and cannot replace the pinned
        # torch/transformers stack.
        site.addsitedir(str(COSYVOICE_EXTRA_SITE))
    sys.path.insert(0, str(SOURCE_ROOT / "third_party" / "Matcha-TTS"))
    sys.path.insert(0, str(SOURCE_ROOT))
    # Matcha imports gdown at module load even though inference never downloads. Supply a hard-offline
    # compatibility module so merely importing the pinned local runtime cannot add a network client.
    offline_gdown = types.ModuleType("gdown")
    offline_gdown.download = _deny_download
    sys.modules.setdefault("gdown", offline_gdown)
    offline_wget = types.ModuleType("wget")
    offline_wget.download = _deny_download
    sys.modules.setdefault("wget", offline_wget)
    # The inference YAML names training data processors, which makes HyperPyYAML import pyarrow even
    # though AutoModel never executes those processors. Resolve those names to hard-fail placeholders
    # without installing or exposing the training/data stack.
    import cosyvoice.dataset as dataset_package

    offline_processor = types.ModuleType("cosyvoice.dataset.processor")
    for name in (
        "parquet_opener", "tokenize", "filter", "resample", "truncate", "compute_fbank",
        "compute_f0", "parse_embedding", "shuffle", "sort", "batch", "padding",
    ):
        setattr(offline_processor, name, _deny_download)
    dataset_package.processor = offline_processor
    sys.modules["cosyvoice.dataset.processor"] = offline_processor
    import modelscope

    modelscope.snapshot_download = _deny_download
    from cosyvoice.cli.cosyvoice import AutoModel

    return AutoModel(model_dir=str(MODEL_ROOT), load_trt=False, load_vllm=False, fp16=False)


def synthesize(payload: dict) -> dict:
    import torch
    import torchaudio

    seed = int(payload.get("seed", 1986))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _load()
    written = 0
    for line in payload["lines"]:
        destination = Path(line["output"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        prompt = "You are a helpful assistant.<|endofprompt|>" + str(line.get("reference_text") or "")
        chunks = [
            item["tts_speech"].detach().cpu()
            for item in model.inference_zero_shot(
                line["text"], prompt, line["reference_audio"], stream=False, speed=1.0, text_frontend=True
            )
        ]
        if not chunks:
            raise RuntimeError("CosyVoice3 produced no audio")
        waveform = torch.cat(chunks, dim=1)
        torchaudio.save(str(destination), waveform, model.sample_rate)
        written += 1
    return {"model": "cosyvoice3-0.5b", "seed": seed, "written": written}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "synthesize":
        raise SystemExit("usage: cosyvoice_worker.py synthesize")
    payload = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result = synthesize(payload)
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
