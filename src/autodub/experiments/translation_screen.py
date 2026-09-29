"""Compare local linewise and contextual translation for one opaque AutoDub job.

Actual execution requires an explicit acknowledgement because source transcript text is private. The
contextual candidate uses the same loopback-only KoboldCpp child as dialogue adaptation
(``adaptation_runner.context_server``). RUN.json receives status and path metadata, never transcript
text or translations.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path


from autodub import adapters
from autodub.adaptation_runner import context_server
from autodub.config import ADAPT_PORT as CONTEXT_PORT
from autodub.config import WORK_ROOT
from autodub.gpu_session import gpu_lease
from autodub.state import load_job


CONTEXTUAL = "qwen3-14b-contextual-gguf"
IMPLEMENTED = {"opus-mt-ja-en-linewise", CONTEXTUAL}


def _valid_id(value: str, prefix: str) -> bool:
    return value.startswith(prefix) and all(char in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in value)


def _atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _chat(prompt: str) -> str:
    body = {
        "model": "local-qwen3-14b-q8",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Japanese-to-English audiovisual translator. Preserve meaning, names, "
                    "pronouns, relationships, tone, and scene continuity. Prefer natural speakable English "
                    "that fits the source timing. Return only the requested JSON array. /no_think"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 4096,
        "stream": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{CONTEXT_PORT}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["choices"][0]["message"]["content"])


def _json_array(text: str) -> list[dict]:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise RuntimeError("contextual translator returned no JSON array")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, list):
        raise RuntimeError("contextual translator returned an invalid result")
    return value


def _contextual(segments: list[dict]) -> list[dict]:
    output = []
    for offset in range(0, len(segments), 12):
        chunk = segments[offset : offset + 12]
        rows = [
            {
                "id": int(item["i"]),
                "start": round(float(item["start"]), 3),
                "end": round(float(item["end"]), 3),
                "japanese": str(item.get("text") or ""),
                "literal_baseline": str(item.get("translation") or ""),
            }
            for item in chunk
        ]
        prompt = (
            "Translate/revise these consecutive subtitle lines as one scene. Return a JSON array with "
            "exactly one object per input in the same order: {\"id\": integer, \"english\": string}.\n"
            + json.dumps(rows, ensure_ascii=False)
        )
        translated = _json_array(_chat(prompt))
        by_id = {int(item["id"]): str(item["english"]).strip() for item in translated}
        if set(by_id) != {int(item["i"]) for item in chunk}:
            raise RuntimeError("contextual translator changed the requested line IDs")
        output.extend({"i": int(item["i"]), "translation": by_id[int(item["i"])]} for item in chunk)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="run AutoDub contextual translation screen")
    parser.add_argument("--run", required=True)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--context-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--arm-gpu", action="store_true")
    parser.add_argument("--ack-private-local-material", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not _valid_id(args.run, "adx-"):
        raise SystemExit("invalid opaque experiment run ID")
    run_root = WORK_ROOT / "experiments" / args.run
    run_path = run_root / "RUN.json"
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "translation-context-v1":
        raise SystemExit("run is not a translation-context experiment")
    job_id = str(manifest.get("job") or "")
    if not _valid_id(job_id, "dub-"):
        raise SystemExit("run has an invalid opaque job ID")
    job = load_job(job_id)
    if not job.get("segments"):
        raise SystemExit("job has no reviewed speech segments")
    known = {item["id"] for item in manifest["candidates"]}
    selected = args.candidates or [item["id"] for item in manifest["candidates"]]
    if set(selected) - known:
        raise SystemExit("candidate is not part of this run")
    readiness = {
        candidate: {
            "implemented": candidate in IMPLEMENTED,
            "requires_gpu": candidate == CONTEXTUAL and args.context_device == "cuda",
            "requires_private_ack": True,
            "device": args.context_device if candidate == CONTEXTUAL else "cpu",
        }
        for candidate in selected
    }
    if args.dry_run:
        print(json.dumps({"ok": True, "run": args.run, "readiness": readiness}, indent=2))
        return
    if not args.ack_private_local_material:
        raise SystemExit("actual translation requires --ack-private-local-material")
    gpu_selected = CONTEXTUAL in selected and args.context_device == "cuda"
    if gpu_selected and not args.arm_gpu:
        raise SystemExit("GPU contextual translation requires --arm-gpu and a clear AutoDub preflight")

    output_root = run_root / "translations"
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}

    def execute(lease=None) -> None:
        for candidate in selected:
            try:
                if candidate == "opus-mt-ja-en-linewise":
                    translated = adapters.translate(copy.deepcopy(job["segments"]), "ja", "en")
                    rows = [{"i": int(item["i"]), "translation": str(item.get("translation") or "")} for item in translated]
                elif candidate == CONTEXTUAL:
                    if lease is not None and args.context_device == "cuda":
                        lease.ensure_active()
                    with context_server(args.context_device):
                        rows = _contextual(job["segments"])
                else:
                    raise RuntimeError("candidate adapter is not implemented")
                destination = output_root / f"{candidate}.json"
                _atomic_json(destination, {"schema": 1, "candidate": candidate, "lines": rows})
                results[candidate] = {
                    "status": "ready-for-review",
                    "lines": len(rows),
                    "evidence": destination.relative_to(run_root).as_posix(),
                }
            except Exception as exc:
                results[candidate] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}

    if gpu_selected:
        with gpu_lease(f"translation-experiment:{args.run}", wait_seconds=10800) as lease:
            execute(lease)
    else:
        execute()
    by_id = {item["id"]: item for item in manifest["candidates"]}
    for candidate, result in results.items():
        by_id[candidate]["result"] = result
    manifest["status"] = "awaiting-review" if any(
        result["status"] == "ready-for-review" for result in results.values()
    ) else "failed"
    _atomic_json(run_path, manifest)
    print(json.dumps({"ok": manifest["status"] == "awaiting-review", "run": args.run, "results": results}, indent=2))


if __name__ == "__main__":
    main()
