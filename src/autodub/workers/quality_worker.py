"""GPU-capable quality workers for AutoDub; JSON over stdio and local models only.

The parent exposes CUDA only after a fresh GPU preflight and lease.  CPU
challengers remain valid in separate workers; this module exists for models that benefit from GPU.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path


os.environ.update(
    {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "PYANNOTE_METRICS_ENABLED": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
)

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autodub.config import (  # noqa: E402
    AUTODUB_MODEL_CACHE,
    PYANNOTE_MODEL,
    QWEN_TTS_06B,
    QWEN_TTS_17B,
    SPEAKER_EVIDENCE_MIN_VOICED_SECONDS,
    TTS_RUNAWAY_FACTOR,
    TTS_RUNAWAY_MIN_SLOT_S,
    TTS_RUNAWAY_RETRIES,
    WHISPER_MODEL,
    WHISPERX_JA_ALIGN_MODEL,
)
from autodub.speaker_evidence import speaker_count_pipeline_kwargs  # noqa: E402

QWEN_MODELS = {"qwen3-tts-1.7b": QWEN_TTS_17B, "qwen3-tts-0.6b": QWEN_TTS_06B}
TORCH_HOME = AUTODUB_MODEL_CACHE / "torch"
DEMUCS_FILES = (
    "f7e0c4bc-ba3fe64a.th",
    "d12395a8-e57c48e6.th",
    "92cfc3b6-ef3bcb9c.th",
    "04573f0d-f3cf25b2.th",
)
os.environ["TORCH_HOME"] = str(TORCH_HOME)
os.environ["HF_HOME"] = str(AUTODUB_MODEL_CACHE / "huggingface")


def _is_runaway(duration_s: float, slot_s: float,
                ratio: float = TTS_RUNAWAY_FACTOR,
                floor: float = TTS_RUNAWAY_MIN_SLOT_S) -> bool:
    """A poisoned clone reference once babbled 327 s of audio into a 1.4 s slot.
    Guard-inert for payload lines without slot_seconds (experiment lanes)."""
    return slot_s > 0 and duration_s > ratio * max(slot_s, floor)


def _line_index(stem: str) -> int:
    """Line number from a wav stem; 0 for non-dialogue outputs (casting demos use
    demo-<label>-<hash>-<register> names — int() on those crashed the whole batch)."""
    try:
        return int(stem.split("-")[-1])
    except ValueError:
        return 0


def _batch_groups(lines: list[dict], batch_size: int, sort: bool) -> list[list[dict]]:
    """Pure grouping for batched TTS. batch_size<=1 returns
    singleton groups in ORIGINAL order — the contract that TTS_BATCH_SIZE=1 is
    byte-identical scheduling to the serial loop. Length-sorted grouping minimizes
    padding waste because a batch's wall time is its longest member."""
    if batch_size <= 1:
        return [[line] for line in lines]
    ordered = sorted(lines, key=lambda l: len(str(l.get("text") or ""))) if sort else list(lines)
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]


def transcribe(payload: dict) -> dict:
    from faster_whisper import WhisperModel

    if not (WHISPER_MODEL / "model.bin").is_file():
        raise RuntimeError("pinned faster-whisper Large V3 model is missing")
    model = WhisperModel(str(WHISPER_MODEL), device="cuda", compute_type="float16")
    segments, info = model.transcribe(
        payload["audio"],
        language=payload.get("language") or None,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=True,
    )
    output = []
    for index, segment in enumerate(segments):
        text = segment.text.strip()
        if not text:
            continue
        words = [
            {"start": round(word.start, 3), "end": round(word.end, 3), "word": word.word}
            for word in (segment.words or [])
            if word.start is not None and word.end is not None
        ]
        output.append(
            {"i": index, "start": round(segment.start, 3), "end": round(segment.end, 3), "text": text, "words": words}
        )
    return {"model": "faster-whisper-large-v3-cuda-fp16", "language": info.language, "segments": output}


def _turns(annotation) -> list[tuple[float, float, str]]:
    return [
        (float(turn.start), float(turn.end), str(label))
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


def _pairwise_overlap_evidence(
    timeline: list[tuple[float, float, str]], label_order: dict[str, str]
) -> list[dict]:
    """Summarize simultaneous-speech intervals for every opaque cluster pair."""
    by_label: dict[str, list[tuple[float, float]]] = {label: [] for label in label_order}
    for start, end, label in timeline:
        if label in by_label and end > start:
            by_label[label].append((start, end))

    evidence = []
    for first, second in combinations(label_order, 2):
        intersections = []
        for first_start, first_end in by_label[first]:
            for second_start, second_end in by_label[second]:
                start = max(first_start, second_start)
                end = min(first_end, second_end)
                if end > start:
                    intersections.append((start, end))
        merged = []
        for start, end in sorted(intersections):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        durations = [end - start for start, end in merged]
        evidence.append(
            {
                "speaker_a": label_order[first],
                "speaker_b": label_order[second],
                "total_ms": int(round(sum(durations) * 1000)),
                "longest_run_ms": int(round(max(durations, default=0.0) * 1000)),
                "run_count": len(merged),
            }
        )
    return evidence


def _read_pcm_wav(path: str):
    """Read the pipeline's PCM WAV without adding a second compiled audio stack."""
    import wave

    import numpy as np

    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        samples = values.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported PCM sample width: {sample_width}")
    if samples.size % channels:
        raise RuntimeError("invalid PCM WAV frame layout")
    return samples.reshape(-1, channels), int(sample_rate)


def _voiced_ranges(
    start: float, end: float, label: str, timeline: list[tuple[float, float, str]]
) -> list[tuple[float, float]]:
    ranges = sorted(
        (max(start, turn_start), min(end, turn_end))
        for turn_start, turn_end, turn_label in timeline
        if turn_label == label and min(end, turn_end) > max(start, turn_start)
    )
    merged: list[tuple[float, float]] = []
    for range_start, range_end in ranges:
        if merged and range_start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], range_end))
        else:
            merged.append((range_start, range_end))
    return merged


def _embed_voiced_chunk(pipeline, waveform, sample_rate: int, ranges: list[tuple[float, float]]):
    """Reuse pyannote's loaded embedding model; unavailable internals fall back to diarizer centroids."""
    import numpy as np

    embedder = getattr(pipeline, "_embedding", None)
    if not callable(embedder):
        return None
    pieces = []
    for start, end in ranges:
        first = max(0, int(round(start * sample_rate)))
        last = min(len(waveform), int(round(end * sample_rate)))
        if last > first:
            pieces.append(waveform[first:last])
    if not pieces:
        return None
    batch = np.concatenate(pieces, axis=0).T[None, :, :]
    try:
        # pyannote's embedder calls .to(device) on its input: it must be a torch
        # tensor, not numpy.
        import torch

        embedded = embedder(torch.from_numpy(np.ascontiguousarray(batch)).float())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # any internals mismatch degrades to the diarizer-centroid fallback instead of
        # killing the whole analyze job
        return None
    if hasattr(embedded, "detach"):
        embedded = embedded.detach().cpu().numpy()
    vector = np.asarray(embedded, dtype=np.float64).reshape(-1)
    if not vector.size or not np.isfinite(vector).all():
        return None
    return [float(value) for value in vector]


def diarize(payload: dict) -> dict:
    import torch
    from pyannote.audio import Pipeline

    if not (PYANNOTE_MODEL / "config.yaml").is_file():
        raise RuntimeError("pinned pyannote Community-1 model is missing")
    device = str(payload.get("device") or "cuda")
    if device not in {"cuda", "cpu"}:
        raise RuntimeError("pyannote device must be cuda or cpu")
    pipeline = Pipeline.from_pretrained(str(PYANNOTE_MODEL))
    pipeline.to(torch.device(device))
    kwargs = speaker_count_pipeline_kwargs(payload.get("speaker_count"))
    waveform, sample_rate = _read_pcm_wav(payload["audio"])
    audio = {
        "waveform": torch.from_numpy(waveform.T.copy()),
        "sample_rate": int(sample_rate),
    }
    output = pipeline(audio, **kwargs)
    raw_annotation = getattr(output, "speaker_diarization", None)
    if raw_annotation is None:
        raw_annotation = output
    exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)
    if exclusive_annotation is None:
        exclusive_annotation = raw_annotation
    raw_timeline = _turns(raw_annotation)
    exclusive_timeline = _turns(exclusive_annotation)
    label_order: dict[str, str] = {}
    assigned_chunks = []
    for segment in payload["segments"]:
        start, end = float(segment["start"]), float(segment["end"])
        duration = max(0.001, end - start)
        overlap: dict[str, float] = {}
        for turn_start, turn_end, label in exclusive_timeline:
            amount = max(0.0, min(end, turn_end) - max(start, turn_start))
            if amount:
                overlap[label] = overlap.get(label, 0.0) + amount
        label = max(overlap, key=overlap.get) if overlap else "unknown"
        if label not in label_order:
            label_order[label] = f"speaker-{len(label_order) + 1:02d}"
        segment["speaker"] = label_order[label]
        ranges = _voiced_ranges(start, end, label, raw_timeline)
        voiced_duration = sum(range_end - range_start for range_start, range_end in ranges)
        segment["voiced_duration"] = round(voiced_duration, 4)
        segment["speaker_evidence_eligible"] = voiced_duration >= SPEAKER_EVIDENCE_MIN_VOICED_SECONDS
        assigned_chunks.append((segment, ranges))
        raw_overlap = 0.0
        raw_primary = 0.0
        for turn_start, turn_end, raw_label in raw_timeline:
            amount = max(0.0, min(end, turn_end) - max(start, turn_start))
            raw_overlap += amount
            if raw_label == label:
                raw_primary += amount
        segment["speaker_confidence"] = round(min(1.0, raw_primary / duration), 4)
        segment["overlap_ratio"] = round(max(0.0, (raw_overlap - duration) / duration), 4)

    raw_labels = [str(label) for label in raw_annotation.labels()]
    for label in raw_labels:
        if label not in label_order:
            label_order[label] = f"speaker-{len(label_order) + 1:02d}"
    cluster_label_order = {label: label_order[label] for label in raw_labels}
    raw_embeddings = getattr(output, "speaker_embeddings", None)
    speaker_embeddings = []
    if raw_embeddings is not None:
        for index, label in enumerate(raw_labels):
            if index >= len(raw_embeddings):
                break
            speaker_embeddings.append(
                {"speaker": label_order[label], "centroid": [float(value) for value in raw_embeddings[index]]}
            )

    speaker_chunk_embeddings = []
    for segment, ranges in assigned_chunks:
        if not segment["speaker_evidence_eligible"]:
            continue
        vector = _embed_voiced_chunk(pipeline, waveform, sample_rate, ranges)
        if vector is not None:
            speaker_chunk_embeddings.append({
                "segment_index": int(segment.get("i", len(speaker_chunk_embeddings))),
                "speaker": segment["speaker"],
                "embedding": vector,
            })

    return {
        "method": "pyannote-community-1-exclusive",
        "device": device,
        "segments": payload["segments"],
        "speaker_embeddings": speaker_embeddings,
        "speaker_chunk_embeddings": speaker_chunk_embeddings,
        "speaker_overlap_evidence": _pairwise_overlap_evidence(raw_timeline, cluster_label_order),
    }


def separate(payload: dict) -> dict:
    checkpoints = TORCH_HOME / "hub" / "checkpoints"
    missing = [name for name in DEMUCS_FILES if not (checkpoints / name).is_file()]
    if missing:
        raise RuntimeError("pinned Demucs htdemucs_ft weights are missing")
    output = Path(payload["output"])
    output.mkdir(parents=True, exist_ok=True)
    device = str(payload.get("device") or "cuda")
    if device not in {"cuda", "cpu"}:
        raise RuntimeError("Demucs device must be cuda or cpu")
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs_ft",
        "--two-stems",
        "vocals",
        "-d",
        device,
        "-o",
        str(output),
        payload["audio"],
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10800)
    if result.returncode:
        raise RuntimeError("Demucs separation failed: " + result.stderr[-1200:])
    stem = Path(payload["audio"]).stem
    root = output / "htdemucs_ft" / stem
    vocals, bed = root / "vocals.wav", root / "no_vocals.wav"
    if not vocals.is_file() or not bed.is_file():
        raise RuntimeError("Demucs did not produce the expected local stems")
    return {"model": "htdemucs_ft", "device": device, "vocals": str(vocals), "bed": str(bed)}


class _WholeSegmentSentenceSplitter:
    """Avoid WhisperX's otherwise-online NLTK fallback for Japanese source segments."""

    @staticmethod
    def span_tokenize(text: str):
        return [(0, len(text))] if text else []


def align(payload: dict) -> dict:
    import whisperx.alignment as whisperx_alignment

    if not (WHISPERX_JA_ALIGN_MODEL / "config.json").is_file():
        raise RuntimeError("pinned Japanese WhisperX alignment model is missing")
    language = str(payload.get("language") or "ja")
    if language != "ja":
        raise RuntimeError("the pinned forced-alignment challenger currently supports Japanese only")
    device = str(payload.get("device") or "cuda")
    if device not in {"cuda", "cpu"}:
        raise RuntimeError("alignment device must be cuda or cpu")

    # WhisperX calls nltk.download() if punkt data is absent. Japanese alignment does not need an
    # English Punkt model, so provide a deterministic local splitter before the library can attempt
    # any network fallback. Model loading is also forced to a local path with cache-only semantics.
    whisperx_alignment.nltk_load = lambda _path: _WholeSegmentSentenceSplitter()
    model, metadata = whisperx_alignment.load_align_model(
        language,
        device,
        model_name=str(WHISPERX_JA_ALIGN_MODEL),
        model_cache_only=True,
    )
    aligned = whisperx_alignment.align(
        payload["segments"],
        model,
        metadata,
        payload["audio"],
        device,
        return_char_alignments=False,
        print_progress=False,
    )
    public_segments = []
    for index, segment in enumerate(aligned["segments"]):
        words = [
            {"start": round(float(word["start"]), 3), "end": round(float(word["end"]), 3)}
            for word in segment.get("words", [])
            if word.get("start") is not None and word.get("end") is not None
        ]
        public_segments.append(
            {
                "i": int(payload["segments"][index].get("i", index)),
                "start": round(float(segment["start"]), 3),
                "end": round(float(segment["end"]), 3),
                "words": words,
            }
        )
    return {
        "model": "whisperx-ja-wav2vec2",
        "device": device,
        "segments": public_segments,
    }


def _write_progress(lines_dir: Path, done: int, total: int, line_index: int,
                    elapsed: float, last_secs: float) -> None:
    """Crash-safe per-line progress sidecar (a long synthesis once sat at one percentage
    for 90 minutes with no per-line feedback). The server merges this into public
    job state; a stall is diagnosable from disk even if everything else dies."""
    import time as _time
    state = {
        "done": done, "total": total, "line_index": line_index,
        "avg_secs": round(elapsed / done, 2) if done else None,
        "last_line_secs": round(last_secs, 2),
        "eta_secs": round((total - done) * (elapsed / done)) if done else None,
        "at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = lines_dir / "progress.json.tmp"
    try:
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, lines_dir / "progress.json")
    except OSError:
        pass  # progress is best-effort; never fail the render over it


def synthesize(payload: dict) -> dict:
    import time as _time

    cancel_file = Path(payload["cancel_file"]) if payload.get("cancel_file") else None
    if cancel_file is not None and cancel_file.is_file():
        return {
            "model": str(payload.get("model") or "qwen3-tts-1.7b"),
            "seed": int(payload.get("seed", 1986)),
            "written": 0,
            "cancelled": True,
        }

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    model_id = str(payload.get("model") or "qwen3-tts-1.7b")
    if model_id not in QWEN_MODELS:
        raise RuntimeError("unknown pinned Qwen3-TTS model")
    model_root = QWEN_MODELS[model_id]
    if not (model_root / "config.json").is_file():
        raise RuntimeError(f"pinned {model_id} Base model is missing")
    model = Qwen3TTSModel.from_pretrained(
        str(model_root), device_map="cuda:0", dtype=torch.bfloat16, local_files_only=True
    )
    seed = int(payload.get("seed", 1986))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prompts = {}

    def _prompt(key: tuple) -> object:
        if key not in prompts:
            prompts[key] = model.create_voice_clone_prompt(
                ref_audio=key[0],
                ref_text=key[1] or None,
                x_vector_only_mode=not bool(key[1]),
            )
        return prompts[key]

    # runaway-TTS guard state: once a reference is demoted, EVERY later line of that
    # speaker starts on the fallback (a poisoned reference poisons all its lines —
    # a runaway averages about a minute per line; re-proving the primary per line would
    # burn hours).
    demoted: dict[tuple, tuple] = {}
    runaway_events = []
    written = 0
    total = len(payload["lines"])
    t_start = _time.time()

    batch_size = int(payload.get("batch_size", 1) or 1)
    if batch_size > 1:
        # DARK path (config TTS_BATCH_SIZE, ships at 1): one generate_voice_clone call
        # per length-sorted group — text/language/voice_clone_prompt as parallel lists
        # (the installed qwen_tts wrapper batches them into one talker.generate).
        # Per-line wall time is unobservable inside a batch, so the runaway guard
        # checks AUDIO seconds vs slot after the group and rescues lines singly.
        class _CancelBatch(Exception):
            pass

        def _reseed(value: int) -> None:
            torch.manual_seed(value)
            torch.cuda.manual_seed_all(value)

        def _single(line: dict, key: tuple, salt: int) -> tuple:
            if float(line.get("slot_seconds") or 0.0) > 0:
                _reseed(seed + _line_index(Path(line["output"]).stem) + salt)
            wavs, rate = model.generate_voice_clone(
                text=line["text"],
                language=line.get("language") or "English",
                voice_clone_prompt=_prompt(key),
                non_streaming_mode=True,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.05,
                max_new_tokens=4096,
            )
            return wavs[0], rate, len(wavs[0]) / float(rate)

        def _rescue(line: dict, index: int, slot: float, wav, rate, duration: float):
            original_key = (line["reference_audio"], line.get("reference_text") or "")
            key = demoted.get(original_key, original_key)
            best = (duration, wav, rate)
            durations = [round(duration, 2)]
            action, fallback_used = None, None
            for attempt in range(1, TTS_RUNAWAY_RETRIES + 1):
                if cancel_file is not None and cancel_file.is_file():
                    raise _CancelBatch()
                wav, rate, duration = _single(line, key, attempt * 104729)
                durations.append(round(duration, 2))
                if duration < best[0]:
                    best = (duration, wav, rate)
                if not _is_runaway(duration, slot):
                    action = "retry-ok"
                    break
            if action is None:
                for fallback in line.get("fallback_references") or []:
                    if cancel_file is not None and cancel_file.is_file():
                        raise _CancelBatch()
                    fallback_key = (fallback["audio"], fallback.get("text") or "")
                    wav, rate, duration = _single(line, fallback_key, 0)
                    durations.append(round(duration, 2))
                    if duration < best[0]:
                        best = (duration, wav, rate)
                    if not _is_runaway(duration, slot):
                        action = "fallback-ref"
                        fallback_used = Path(fallback["audio"]).name
                        demoted[original_key] = fallback_key
                        break
            if action is None:
                action = "kept-shortest"
                duration, wav, rate = best
            runaway_events.append({"line_index": index, "slot_seconds": slot,
                                   "durations": durations, "action": action,
                                   "fallback": fallback_used})
            return wav, rate, duration

        groups = _batch_groups(payload["lines"], batch_size,
                               bool(payload.get("batch_sort", True)))
        for group in groups:
            if cancel_file is not None and cancel_file.is_file():
                return {"model": model_id, "seed": seed, "written": written,
                        "runaways": runaway_events, "cancelled": True}
            keys = [demoted.get((l["reference_audio"], l.get("reference_text") or ""),
                                (l["reference_audio"], l.get("reference_text") or ""))
                    for l in group]
            prompt_items = [_prompt(key)[0] for key in keys]
            first_index = _line_index(Path(group[0]["output"]).stem)
            _reseed(seed + first_index)
            t_group = _time.time()
            wavs, rate = model.generate_voice_clone(
                text=[l["text"] for l in group],
                language=[l.get("language") or "English" for l in group],
                voice_clone_prompt=prompt_items,
                non_streaming_mode=True,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.05,
                max_new_tokens=4096,
            )
            group_secs = _time.time() - t_group
            audio_secs = []
            try:
                for line, wav in zip(group, wavs):
                    destination = Path(line["output"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    index = _line_index(destination.stem)
                    slot = float(line.get("slot_seconds") or 0.0)
                    duration = len(wav) / float(rate)
                    line_rate = rate
                    if _is_runaway(duration, slot):
                        wav, line_rate, duration = _rescue(line, index, slot, wav, rate, duration)
                    sf.write(str(destination), wav, line_rate)
                    written += 1
                    audio_secs.append(round(duration, 1))
                    _write_progress(destination.parent, written, total, index,
                                    _time.time() - t_start,
                                    group_secs / max(1, len(group)))
            except _CancelBatch:
                return {"model": model_id, "seed": seed, "written": written,
                        "runaways": runaway_events, "cancelled": True}
            # stderr: indices + audio seconds only (no dialogue text)
            print(f"[synthesize] batch x{len(group)} in {group_secs:.1f}s "
                  f"(i={[int(Path(l['output']).stem.split('-')[-1]) for l in group]} "
                  f"audio={audio_secs}s) {written}/{total}", file=sys.stderr, flush=True)
        return {"model": model_id, "seed": seed, "written": written,
                "runaways": runaway_events}

    for line in payload["lines"]:
        if cancel_file is not None and cancel_file.is_file():
            return {
                "model": model_id,
                "seed": seed,
                "written": written,
                "runaways": runaway_events,
                "cancelled": True,
            }
        destination = Path(line["output"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        line_number = _line_index(destination.stem)
        slot = float(line.get("slot_seconds") or 0.0)
        original_key = (line["reference_audio"], line.get("reference_text") or "")
        prompt_key = demoted.get(original_key, original_key)

        def _attempt(key: tuple, salt: int) -> tuple:
            if slot > 0:
                # Per-line reseed: the same seed
                # gives the same take regardless of cache hits, retries, or batch
                # composition. Slotless payloads (experiment lanes) keep the old
                # sequential-stream behaviour untouched.
                torch.manual_seed(seed + line_number + salt)
                torch.cuda.manual_seed_all(seed + line_number + salt)
            wavs, rate = model.generate_voice_clone(
                text=line["text"],
                language=line.get("language") or "English",
                voice_clone_prompt=_prompt(key),
                non_streaming_mode=True,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.05,
                max_new_tokens=4096,
            )
            return wavs[0], rate, len(wavs[0]) / float(rate)

        t_line = _time.time()
        wav, sample_rate, duration = _attempt(prompt_key, 0)
        best = (duration, wav, sample_rate)
        durations = [round(duration, 2)]
        action = None
        fallback_used = None
        if _is_runaway(duration, slot):
            for attempt in range(1, TTS_RUNAWAY_RETRIES + 1):
                if cancel_file is not None and cancel_file.is_file():
                    return {"model": model_id, "seed": seed, "written": written,
                            "runaways": runaway_events, "cancelled": True}
                print(f"[synthesize] RUNAWAY line i={line_number}: {duration:.1f}s into a "
                      f"{slot:.1f}s slot — retry {attempt}/{TTS_RUNAWAY_RETRIES}",
                      file=sys.stderr, flush=True)
                wav, sample_rate, duration = _attempt(prompt_key, attempt * 104729)
                durations.append(round(duration, 2))
                if duration < best[0]:
                    best = (duration, wav, sample_rate)
                if not _is_runaway(duration, slot):
                    action = "retry-ok"
                    break
            if action is None:
                for fallback in line.get("fallback_references") or []:
                    if cancel_file is not None and cancel_file.is_file():
                        return {"model": model_id, "seed": seed, "written": written,
                                "runaways": runaway_events, "cancelled": True}
                    fallback_key = (fallback["audio"], fallback.get("text") or "")
                    print(f"[synthesize] RUNAWAY line i={line_number}: trying fallback "
                          f"reference {Path(fallback['audio']).name}",
                          file=sys.stderr, flush=True)
                    wav, sample_rate, duration = _attempt(fallback_key, 0)
                    durations.append(round(duration, 2))
                    if duration < best[0]:
                        best = (duration, wav, sample_rate)
                    if not _is_runaway(duration, slot):
                        action = "fallback-ref"
                        fallback_used = Path(fallback["audio"]).name
                        demoted[original_key] = fallback_key
                        break
            if action is None:
                # Exhausted: write the shortest take — align_line tempo-crushes it and
                # audit_timing flags it downstream; degraded beats silent or babbling.
                action = "kept-shortest"
                duration, wav, sample_rate = best
            runaway_events.append({
                "line_index": line_number,
                "slot_seconds": slot,
                "durations": durations,
                "action": action,
                "fallback": fallback_used,
            })
        sf.write(str(destination), wav, sample_rate)
        written += 1
        line_secs = _time.time() - t_line
        # stderr: line index + timing only (no dialogue text) — the parent preserves this
        # as worker diagnostics; a runaway line (like the 288s ED wander) is visible live.
        print(f"[synthesize] line {written}/{total} (i={destination.stem.split('-')[-1]}) "
              f"{line_secs:.1f}s", file=sys.stderr, flush=True)
        _write_progress(destination.parent, written, total,
                        _line_index(destination.stem), _time.time() - t_start, line_secs)
    return {"model": model_id, "seed": seed, "written": written, "runaways": runaway_events}


def main() -> None:
    commands = {
        "transcribe": transcribe,
        "diarize": diarize,
        "separate": separate,
        "align": align,
        "synthesize": synthesize,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("usage: quality_worker.py transcribe|diarize|separate|align|synthesize")
    payload = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result = commands[sys.argv[1]](payload)
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
