"""CPU-only model worker. Input and output are JSON over stdio; runtime is forced offline."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autodub.config import TRANSLATION_MODEL, WHISPER_MODEL  # noqa: E402


def transcribe(payload: dict) -> dict:
    from faster_whisper import WhisperModel

    model = WhisperModel(str(WHISPER_MODEL), device="cpu", compute_type="int8", cpu_threads=max(2, (os.cpu_count() or 8) // 2))
    language = payload.get("language") or None
    segments, info = model.transcribe(payload["audio"], language=language, word_timestamps=True, vad_filter=True, beam_size=5)
    output = []
    for index, segment in enumerate(segments):
        text = segment.text.strip()
        if text:
            output.append({"i": index, "start": round(segment.start, 3), "end": round(segment.end, 3), "text": text})
    return {"language": info.language, "segments": output}


def cluster(payload: dict) -> dict:
    """Cluster real acoustic features per ASR slot; deterministic CPU fallback for diarization.

    This is deliberately conservative: the review UI exposes every assignment.  A pyannote adapter
    can replace it later without changing the job schema.
    """
    import librosa
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    audio, sample_rate = librosa.load(payload["audio"], sr=16000, mono=True)
    segments = payload["segments"]
    vectors = []
    valid = []
    for index, segment in enumerate(segments):
        start = max(0, int(float(segment["start"]) * sample_rate))
        end = min(len(audio), int(float(segment["end"]) * sample_rate))
        clip = audio[start:end]
        if len(clip) < sample_rate // 4:
            continue
        mfcc = librosa.feature.mfcc(y=clip, sr=sample_rate, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        vector = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1), delta.std(axis=1)])
        norm = np.linalg.norm(vector)
        vectors.append(vector / norm if norm else vector)
        valid.append(index)

    speaker_count = payload.get("speaker_count") or {"mode": "automatic"}
    mode = str(speaker_count.get("mode") or "automatic")
    labels = [0] * len(vectors)
    if len(vectors) >= 2:
        if mode == "exact":
            clusters = min(int(speaker_count["count"]), len(vectors))
            labels = AgglomerativeClustering(n_clusters=clusters, metric="cosine", linkage="average").fit_predict(vectors)
        else:
            labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.32, metric="cosine", linkage="average").fit_predict(vectors)
            if mode == "min-max":
                automatic_count = len(set(int(label) for label in labels))
                bounded_count = min(
                    max(automatic_count, int(speaker_count["min"])),
                    int(speaker_count["max"]),
                    len(vectors),
                )
                if bounded_count != automatic_count:
                    labels = AgglomerativeClustering(n_clusters=bounded_count, metric="cosine", linkage="average").fit_predict(vectors)
    assigned = {index: int(label) for index, label in zip(valid, labels)}
    previous = 0
    for index, segment in enumerate(segments):
        previous = assigned.get(index, previous)
        segment["speaker"] = f"speaker-{previous + 1:02d}"
    return {"method": "acoustic-mfcc-reviewable", "segments": segments}


def translate(payload: dict) -> dict:
    if payload.get("source_language") == payload.get("target_language"):
        for item in payload["segments"]:
            item["translation"] = str(item.get("text") or "").strip()
        return {"model": "identity-dev-path", "segments": payload["segments"]}
    if payload.get("target_language") != "en" or payload.get("source_language") != "ja":
        raise RuntimeError("the installed prototype translation model supports ja -> en; other language pairs require another pinned local adapter")
    if not TRANSLATION_MODEL.exists():
        raise RuntimeError("local translation model missing; run scripts/download_models.py --accept-online-download core")

    from transformers import MarianMTModel, MarianTokenizer
    import torch

    tokenizer = MarianTokenizer.from_pretrained(str(TRANSLATION_MODEL), local_files_only=True)
    model = MarianMTModel.from_pretrained(str(TRANSLATION_MODEL), local_files_only=True).to("cpu")
    model.eval()
    segments = payload["segments"]
    texts = [item.get("text", "") for item in segments]
    for offset in range(0, len(texts), 16):
        batch_text = texts[offset : offset + 16]
        tokens = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.inference_mode():
            generated = model.generate(**tokens, max_new_tokens=256, num_beams=4)
        translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for item, English in zip(segments[offset : offset + 16], translations):
            item["translation"] = English.strip()
    return {"model": "opus-mt-ja-en", "segments": segments}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"transcribe", "cluster", "translate"}:
        raise SystemExit("usage: model_worker.py transcribe|cluster|translate")
    payload = json.load(sys.stdin)
    result = globals()[sys.argv[1]](payload)
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
