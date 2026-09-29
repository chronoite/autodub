from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from .state import replace_retry  # noqa: E402 (shared atomic-write finisher)
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PACKAGE_ROOT, WORK_ROOT
from .state import load_job


EXPERIMENT_ROOT = WORK_ROOT / "experiments"
REGISTRY_PATH = PACKAGE_ROOT / "experiments" / "registry.json"
REVIEW_STATES = {"pending", "in-progress", "complete"}
# Judgment scale: one pass/maybe/fail verdict per candidate. Per-criterion 1-5 grids
# proved to be judging overload; numeric scores remain accepted for older runs but are
# no longer the primary review surface.
VERDICTS = {"pass", "maybe", "fail"}
# Optional one-word blocker on a maybe/fail: turns
# a soft verdict into the next bench variable without restoring criteria grids.
BLOCKERS = {"timing", "voice", "mix", "text"}
# Candidate IDs render in the review UI and become artifact-URL components; constrain
# them at run creation (defense in depth).
CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,119}$")
# Reviews are load-modify-save; the lock plus a client-supplied base revision stop
# two open tabs (phone + desktop) from silently erasing each other's judgment.
_REVIEW_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def valid_run_id(run_id: str) -> bool:
    return run_id.startswith("adx-") and all(
        character in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_id
    )


def run_dir(run_id: str) -> Path:
    if not valid_run_id(run_id):
        raise ValueError("invalid experiment run ID")
    return EXPERIMENT_ROOT / run_id


def run_path(run_id: str) -> Path:
    return run_dir(run_id) / "RUN.json"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_retry(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def create_run(experiment_id: str, job_id: str) -> dict[str, Any]:
    load_job(job_id)
    experiment = next(
        (item for item in registry()["experiments"] if item["id"] == experiment_id),
        None,
    )
    if experiment is None:
        raise ValueError("unknown experiment")
    candidate_ids = [str(candidate.get("id", "")) for candidate in experiment["candidates"]]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("experiment has duplicate candidate IDs")
    for candidate_id in candidate_ids:
        if not CANDIDATE_ID.fullmatch(candidate_id):
            raise ValueError("experiment candidate ID is not display/URL safe")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"adx-{stamp}-{secrets.token_hex(2)}"
    root = run_dir(run_id)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "id": run_id,
        "experiment": experiment["id"],
        "job": job_id,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "status": "planned",
        "candidates": [{**candidate, "result": "pending"} for candidate in experiment["candidates"]],
        "criteria": experiment["criteria"],
        "human_review": {"status": "pending", "winner": None, "notes": [], "scores": {},
                         "verdicts": {}, "blockers": {}, "revision": 0},
    }
    _atomic_json(run_path(run_id), manifest)
    return manifest


def load_run(run_id: str) -> dict[str, Any]:
    return json.loads(run_path(run_id).read_text(encoding="utf-8"))


def save_run(manifest: dict[str, Any]) -> None:
    if not valid_run_id(str(manifest.get("id") or "")):
        raise ValueError("invalid experiment manifest")
    manifest["updated_at"] = _utc_now()
    _atomic_json(run_path(manifest["id"]), manifest)


def _candidate_artifact_refs(candidate: dict) -> list[str]:
    result = candidate.get("result")
    if not isinstance(result, dict):
        return []
    refs = []
    evidence = result.get("evidence")
    if isinstance(evidence, str):
        refs.append(evidence)
    for value in result.get("artifacts", []):
        if isinstance(value, str):
            refs.append(value)
    evidence_dir = result.get("evidence_dir")
    if isinstance(evidence_dir, str):
        refs.append(evidence_dir)
    return refs


def _within(root: Path, candidate: Path) -> bool:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def candidate_artifacts(manifest: dict, candidate_id: str) -> list[dict[str, Any]]:
    candidate = next((item for item in manifest.get("candidates", []) if item.get("id") == candidate_id), None)
    if candidate is None:
        raise ValueError("unknown experiment candidate")
    root = run_dir(manifest["id"])
    files: list[Path] = []
    for reference in _candidate_artifact_refs(candidate):
        relative = Path(reference)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        path = root / relative
        if not _within(root, path):
            continue
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in sorted(path.iterdir()) if item.is_file() and _within(root, item))
    unique = []
    seen = set()
    for path in files:
        relative = path.relative_to(root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        unique.append({"path": relative, "name": path.name, "bytes": path.stat().st_size})
    return unique


def resolve_artifact(run_id: str, candidate_id: str, relative_path: str) -> Path:
    manifest = load_run(run_id)
    allowed = {item["path"] for item in candidate_artifacts(manifest, candidate_id)}
    if relative_path not in allowed:
        raise FileNotFoundError(relative_path)
    path = run_dir(run_id) / Path(relative_path)
    if not path.is_file() or not _within(run_dir(run_id), path):
        raise FileNotFoundError(relative_path)
    return path


def public_run(manifest: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(manifest))
    for candidate in value.get("candidates", []):
        result = candidate.get("result")
        if isinstance(result, dict) and result.get("error"):
            result["error"] = "candidate failed; inspect the local run from the host"
        candidate["artifacts"] = candidate_artifacts(manifest, candidate["id"])
    return value


def list_runs() -> list[dict[str, Any]]:
    if not EXPERIMENT_ROOT.exists():
        return []
    output = []
    for path in EXPERIMENT_ROOT.glob("adx-*/RUN.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            public = public_run(manifest)
            output.append(
                {
                    "id": public["id"],
                    "experiment": public["experiment"],
                    "job": public["job"],
                    "status": public.get("status", "planned"),
                    "created_at": public.get("created_at"),
                    "updated_at": public.get("updated_at", public.get("created_at")),
                    "review_status": public.get("human_review", {}).get("status", "pending"),
                    "candidate_count": len(public.get("candidates", [])),
                }
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return sorted(output, key=lambda item: item.get("created_at") or "", reverse=True)


def update_review(run_id: str, update: dict[str, Any]) -> dict[str, Any]:
    with _REVIEW_LOCK:
        return _update_review_locked(run_id, update)


def _update_review_locked(run_id: str, update: dict[str, Any]) -> dict[str, Any]:
    manifest = load_run(run_id)
    candidate_ids = {item["id"] for item in manifest.get("candidates", [])}
    criteria = set(manifest.get("criteria", []))
    review = manifest.setdefault("human_review", {})
    if "revision" in update and update["revision"] is not None:
        if int(update["revision"]) != int(review.get("revision", 0)):
            raise ValueError("review changed elsewhere - reload before saving")
    status = str(update.get("status", review.get("status", "pending")))
    if status not in REVIEW_STATES:
        raise ValueError("invalid review status")
    winner = update.get("winner", review.get("winner"))
    if winner in {"", None}:
        winner = None
    elif winner not in candidate_ids:
        raise ValueError("winner is not part of this experiment")
    notes = update.get("notes", review.get("notes", []))
    if not isinstance(notes, list) or len(notes) > 50:
        raise ValueError("review notes must be a short list")
    notes = [str(note)[:1000] for note in notes if str(note).strip()]
    raw_verdicts = update.get("verdicts", review.get("verdicts", {}))
    if not isinstance(raw_verdicts, dict):
        raise ValueError("invalid review verdicts")
    verdicts = {}
    for candidate_id, verdict in raw_verdicts.items():
        if candidate_id not in candidate_ids:
            raise ValueError("verdict for unknown candidate")
        value = str(verdict).strip().lower()
        if value not in VERDICTS:
            raise ValueError("verdicts must be pass, maybe, or fail")
        verdicts[candidate_id] = value
    raw_blockers = update.get("blockers", review.get("blockers", {}))
    if not isinstance(raw_blockers, dict):
        raise ValueError("invalid review blockers")
    blockers = {}
    for candidate_id, blocker in raw_blockers.items():
        if candidate_id not in candidate_ids:
            raise ValueError("blocker for unknown candidate")
        if blocker in {None, ""}:
            continue
        value = str(blocker).strip().lower()
        if value not in BLOCKERS:
            raise ValueError("blockers must be timing, voice, mix, or text")
        blockers[candidate_id] = value
    raw_scores = update.get("scores", review.get("scores", {}))
    if not isinstance(raw_scores, dict):
        raise ValueError("invalid review scores")
    scores = {}
    for candidate_id, candidate_scores in raw_scores.items():
        if candidate_id not in candidate_ids or not isinstance(candidate_scores, dict):
            raise ValueError("invalid candidate scores")
        scores[candidate_id] = {}
        for criterion, score in candidate_scores.items():
            if criterion not in criteria:
                raise ValueError("invalid review criterion")
            numeric = int(score)
            if not 1 <= numeric <= 5:
                raise ValueError("review scores must be between 1 and 5")
            scores[candidate_id][criterion] = numeric
    review.update({"status": status, "winner": winner, "notes": notes, "scores": scores,
                   "verdicts": verdicts, "blockers": blockers,
                   "revision": int(review.get("revision", 0)) + 1, "updated_at": _utc_now()})
    save_run(manifest)
    return public_run(manifest)
