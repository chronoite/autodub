"""Pure speaker-count, short-utterance, manifest, and duplicate-evidence contracts.

Raw centroids stay job-local. This module only compares caller-provided values and never
mutates diarization labels, segments, or speaker assignments.

Evidence manifests are deterministic. For each speaker they take up to two eligible
segments with the lowest diarizer confidence (the borderline/hard samples), then fill the
five-slot default by greedy temporal farthest-point selection. Ties are resolved by
``(start, end, segment index)`` and the final manifest is chronological. Thus identical
inputs always produce identical diverse-plus-borderline selections.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Mapping, Sequence

from .config import (
    SPEAKER_DUPLICATE_COSINE_THRESHOLD,
    SPEAKER_EVIDENCE_BORDERLINE_LIMIT,
    SPEAKER_EVIDENCE_LOW_CHUNK_COUNT,
    SPEAKER_EVIDENCE_MANIFEST_LIMIT,
    SPEAKER_EVIDENCE_MIN_VOICED_SECONDS,
    SPEAKER_OVERLAP_VETO_MS,
)


def derive_speaker_evidence(
    speaker_embeddings: Sequence[Mapping[str, Any]],
    chunk_embeddings: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    *,
    min_voiced_seconds: float = SPEAKER_EVIDENCE_MIN_VOICED_SECONDS,
    low_chunk_count: int = SPEAKER_EVIDENCE_LOW_CHUNK_COUNT,
    manifest_limit: int = SPEAKER_EVIDENCE_MANIFEST_LIMIT,
    borderline_limit: int = SPEAKER_EVIDENCE_BORDERLINE_LIMIT,
) -> dict[str, Any]:
    """Derive eligible-only repair centroids and a public-safe evidence manifest.

    ``speaker_embeddings`` are the diarizer's all-speech centroids and are retained only
    as a cross-check/fallback. A repair centroid is an arithmetic mean of caller-provided
    chunk embeddings whose measured voiced duration meets the inclusive threshold. When
    no eligible chunk vector exists for a speaker, its original centroid is copied with
    ``source=diarizer_fallback`` so assignment behavior remains exactly as before.
    """
    threshold = float(min_voiced_seconds)
    if threshold < 0 or low_chunk_count < 1 or manifest_limit < 1 or borderline_limit < 0:
        raise ValueError("speaker evidence limits must be non-negative and non-zero where required")

    ordered = sorted(segments, key=_segment_order)
    eligible_by_speaker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    eligible_by_index: dict[int, tuple[str, Mapping[str, Any]]] = {}
    speakers = {str(item.get("speaker") or "") for item in speaker_embeddings}
    speakers.discard("")
    for position, segment in enumerate(ordered):
        speaker = str(segment.get("speaker") or "")
        if not speaker:
            continue
        speakers.add(speaker)
        voiced = _finite_float(segment.get("voiced_duration"), 0.0)
        if voiced >= threshold:
            eligible_by_speaker[speaker].append(segment)
            eligible_by_index[_segment_index(segment, position)] = (speaker, segment)

    vectors: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    dimensions: dict[str, int] = {}
    for item in chunk_embeddings:
        try:
            index = int(item.get("segment_index"))
        except (TypeError, ValueError):
            continue
        eligible = eligible_by_index.get(index)
        speaker = str(item.get("speaker") or "")
        raw = item.get("embedding")
        if eligible is None or speaker != eligible[0] or not _vector(raw):
            continue
        vector = tuple(float(value) for value in raw)
        dimension = dimensions.setdefault(speaker, len(vector))
        if len(vector) == dimension:
            vectors[speaker].append(vector)

    originals = {}
    for item in speaker_embeddings:
        speaker = str(item.get("speaker") or "")
        raw = item.get("centroid")
        if speaker and _vector(raw):
            originals[speaker] = [float(value) for value in raw]

    repair_embeddings = []
    for speaker in sorted(speakers):
        speaker_vectors = vectors.get(speaker, [])
        if speaker_vectors:
            repair_embeddings.append({
                "speaker": speaker,
                "centroid": [sum(values) / len(values) for values in zip(*speaker_vectors)],
                "source": "eligible_chunks",
                "eligible_embedding_count": len(speaker_vectors),
            })
        elif speaker in originals:
            repair_embeddings.append({
                "speaker": speaker,
                "centroid": originals[speaker],
                "source": "diarizer_fallback",
                "eligible_embedding_count": 0,
            })

    manifest_speakers = []
    for speaker in sorted(speakers):
        candidates = eligible_by_speaker.get(speaker, [])
        selected, borderline_keys = _select_manifest_segments(candidates, manifest_limit, borderline_limit)
        manifest_speakers.append({
            "speaker": speaker,
            "eligible_chunk_count": len(candidates),
            "low_evidence": len(candidates) < low_chunk_count,
            "segments": [
                {
                    "segment_index": _segment_index(item, position),
                    "start": round(_finite_float(item.get("start"), 0.0), 3),
                    "end": round(_finite_float(item.get("end"), 0.0), 3),
                    "voiced_duration": round(_finite_float(item.get("voiced_duration"), 0.0), 3),
                    "borderline": _segment_order(item) in borderline_keys,
                }
                for position, item in enumerate(selected)
            ],
        })
    return {
        "repair_embeddings": repair_embeddings,
        "manifest": {
            "schema": 1,
            "eligibility_voiced_seconds": threshold,
            "low_evidence_chunk_count": low_chunk_count,
            "modalities": {
                "visual": {"status": "not judged", "evidence_role": "human evidence"},
            },
            "speakers": manifest_speakers,
        },
    }


def normalize_speaker_count(value: Any = None) -> dict[str, Any]:
    """Return the canonical automatic/exact/min-max speaker-count contract."""
    if value is None or value == {}:
        return {"mode": "automatic"}
    if isinstance(value, bool):
        raise ValueError("speaker count must not be boolean")
    if isinstance(value, int):
        return {"mode": "automatic"} if value <= 0 else {"mode": "exact", "count": value}
    if not isinstance(value, Mapping):
        raise ValueError("speaker_count must be an object")

    mode = str(value.get("mode") or "automatic").strip().lower()
    if mode in {"auto", "automatic"}:
        return {"mode": "automatic"}
    if mode == "exact":
        return {"mode": "exact", "count": _positive_int(value.get("count"), "exact speaker count")}
    if mode in {"range", "min-max", "min_max"}:
        minimum = _positive_int(value.get("min"), "minimum speaker count")
        maximum = _positive_int(value.get("max"), "maximum speaker count")
        if minimum > maximum:
            raise ValueError("minimum speaker count must not exceed maximum")
        return {"mode": "min-max", "min": minimum, "max": maximum}
    raise ValueError("speaker_count mode must be automatic, exact, or min-max")


def speaker_count_pipeline_kwargs(value: Any = None) -> dict[str, int]:
    """Translate the public contract to pyannote Community-1 keyword arguments."""
    contract = normalize_speaker_count(value)
    if contract["mode"] == "exact":
        return {"num_speakers": contract["count"]}
    if contract["mode"] == "min-max":
        return {"min_speakers": contract["min"], "max_speakers": contract["max"]}
    return {}


def suggest_duplicate_pairs(
    speaker_embeddings: Sequence[Mapping[str, Any]],
    overlap_evidence: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    *,
    cosine_threshold: float = SPEAKER_DUPLICATE_COSINE_THRESHOLD,
    overlap_veto_ms: int = SPEAKER_OVERLAP_VETO_MS,
    evidence_manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Suggest centroid-similar pairs unless simultaneous speech reaches the veto.

    The return value contains derived similarities and evidence counts only. Inputs are
    treated as immutable and no merge or relabel operation exists in this function.
    """
    threshold = float(cosine_threshold)
    veto_ms = int(overlap_veto_ms)
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("cosine threshold must be between -1 and 1")
    if veto_ms < 0:
        raise ValueError("overlap veto must be non-negative")

    centroids: dict[str, tuple[float, ...]] = {}
    for item in speaker_embeddings:
        speaker = str(item.get("speaker") or "")
        raw = item.get("centroid")
        if not speaker or not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        vector = tuple(float(value) for value in raw)
        if vector and all(math.isfinite(value) for value in vector):
            centroids[speaker] = vector

    overlap_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in overlap_evidence:
        first = str(item.get("speaker_a") or "")
        second = str(item.get("speaker_b") or "")
        if first and second and first != second:
            overlap_by_pair[tuple(sorted((first, second)))] = item

    segment_counts = Counter(str(item.get("speaker") or "") for item in segments)
    speaker_flags = {
        str(item.get("speaker") or ""): item
        for item in (evidence_manifest or {}).get("speakers", [])
        if isinstance(item, Mapping)
    }
    suggestions = []
    for first, second in combinations(sorted(centroids), 2):
        similarity = _cosine_similarity(centroids[first], centroids[second])
        if similarity is None or similarity < threshold:
            continue
        overlap = overlap_by_pair.get((first, second), {})
        total_ms = max(0, int(overlap.get("total_ms") or 0))
        if total_ms >= veto_ms:
            continue
        suggestions.append(
            {
                "speaker_a": first,
                "speaker_b": second,
                "cosine_similarity": round(similarity, 6),
                "simultaneous_speech_total_ms": total_ms,
                "simultaneous_speech_longest_run_ms": max(0, int(overlap.get("longest_run_ms") or 0)),
                "low_evidence": bool(speaker_flags.get(first, {}).get("low_evidence"))
                or bool(speaker_flags.get(second, {}).get("low_evidence")),
                "evidence_counts": {
                    "speaker_a_segments": segment_counts[first],
                    "speaker_b_segments": segment_counts[second],
                    "speaker_a_eligible_chunks": int(speaker_flags.get(first, {}).get("eligible_chunk_count") or 0),
                    "speaker_b_eligible_chunks": int(speaker_flags.get(second, {}).get("eligible_chunk_count") or 0),
                    "speaker_centroids": 2,
                    "simultaneous_speech_runs": max(0, int(overlap.get("run_count") or 0)),
                },
            }
        )
    return suggestions


def _select_manifest_segments(
    candidates: Sequence[Mapping[str, Any]], limit: int, borderline_limit: int
) -> tuple[list[Mapping[str, Any]], set[tuple[float, float, int]]]:
    ordered = sorted(candidates, key=_segment_order)
    borderline = sorted(
        ordered,
        key=lambda item: (_finite_float(item.get("speaker_confidence"), 1.0), _segment_order(item)),
    )[:min(borderline_limit, limit)]
    selected = list(borderline)
    while len(selected) < min(limit, len(ordered)):
        remaining = [item for item in ordered if item not in selected]
        if not selected:
            selected.append(remaining[0])
            continue
        centers = [_segment_center(item) for item in selected]
        selected.append(min(
            remaining,
            key=lambda item: (-min(abs(_segment_center(item) - center) for center in centers), _segment_order(item)),
        ))
    return sorted(selected, key=_segment_order), {_segment_order(item) for item in borderline}


def _segment_index(segment: Mapping[str, Any], fallback: int) -> int:
    try:
        return int(segment.get("i", fallback))
    except (TypeError, ValueError):
        return fallback


def _segment_order(segment: Mapping[str, Any]) -> tuple[float, float, int]:
    return (
        _finite_float(segment.get("start"), 0.0),
        _finite_float(segment.get("end"), 0.0),
        _segment_index(segment, 0),
    )


def _segment_center(segment: Mapping[str, Any]) -> float:
    return (_finite_float(segment.get("start"), 0.0) + _finite_float(segment.get("end"), 0.0)) / 2.0


def _finite_float(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _vector(raw: Any) -> bool:
    try:
        return (
            isinstance(raw, Sequence)
            and not isinstance(raw, (str, bytes))
            and bool(raw)
            and all(math.isfinite(float(value)) for value in raw)
        )
    except (TypeError, ValueError):
        return False


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if converted <= 0 or converted != value:
        raise ValueError(f"{label} must be a positive integer")
    return converted


def _cosine_similarity(first: tuple[float, ...], second: tuple[float, ...]) -> float | None:
    if len(first) != len(second):
        return None
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return None
    return sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)
