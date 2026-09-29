"""Explicit, one-time model download for AutoDub.

The AutoDub runtime never touches the network; this script is the only thing that does, and only
with ``--accept-online-download``. Each model is fetched at its pinned revision and receives a
``MODEL-PROVENANCE.json`` listing every file's size and SHA-256, which ``verify_models.py`` checks
offline later.

    python scripts/download_models.py --accept-online-download core
    python scripts/download_models.py --accept-online-download quality
    python scripts/download_models.py --accept-online-download qwen3-14b-gguf

Groups: ``core`` (CPU pipeline), ``quality`` (GPU pipeline), ``optional`` (adaptation and
experiments). You are responsible for complying with each model's license; see
THIRD-PARTY-NOTICES.md. pyannote models are gated: accept their conditions on Hugging Face and run
``huggingface-cli login`` before downloading them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_manifest import GROUPS, MODELS, PROVENANCE  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_hf(spec: dict, destination: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=spec["repo_id"],
        revision=spec.get("revision"),
        local_dir=str(destination),
        token=True if spec.get("token") else None,
        allow_patterns=spec.get("allow_patterns"),
    )


def _fetch_urls(spec: dict, destination: Path) -> None:
    for name in spec["files"]:
        target = destination / name
        if not target.is_file():
            partial = target.with_suffix(target.suffix + ".partial")
            urllib.request.urlretrieve(spec["base_url"] + name, partial)
            partial.replace(target)
        expected_prefix = name.split("-", 1)[1].split(".", 1)[0]
        if not sha256(target).startswith(expected_prefix):
            raise RuntimeError(f"hash prefix mismatch for {name}")


def acquire(name: str) -> dict:
    spec = MODELS[name]
    destination: Path = spec["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    (_fetch_hf if spec["kind"] == "hf" else _fetch_urls)(spec, destination)
    wanted = set(spec.get("files") or [])
    files = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or ".cache" in path.parts or path.name == PROVENANCE:
            continue
        relative = path.relative_to(destination).as_posix()
        if wanted and relative not in wanted:
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    record = {
        "schema": 1,
        "name": name,
        "acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": (f"https://huggingface.co/{spec['repo_id']}" if spec["kind"] == "hf" else spec["base_url"]),
        "revision": spec.get("revision"),
        "license": spec["license"],
        "purpose": spec["purpose"],
        "files": files,
    }
    (destination / PROVENANCE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {"name": name, "folder": str(destination), "files": len(files),
            "bytes": sum(item["bytes"] for item in files)}


def main() -> None:
    parser = argparse.ArgumentParser(description="download pinned AutoDub models")
    parser.add_argument("targets", nargs="+", help=f"model names or groups ({', '.join(GROUPS)})")
    parser.add_argument("--accept-online-download", action="store_true",
                        help="required acknowledgement that this command uses the network")
    args = parser.parse_args()
    if not args.accept_online_download:
        raise SystemExit("refusing network access without --accept-online-download")
    names: list[str] = []
    for target in args.targets:
        if target in GROUPS:
            names.extend(name for name, spec in MODELS.items() if spec["group"] == target)
        elif target in MODELS:
            names.append(target)
        else:
            raise SystemExit(f"unknown model or group: {target} (choose from {sorted(MODELS)} or {GROUPS})")
    results = [acquire(name) for name in dict.fromkeys(names)]
    print(json.dumps({"ok": True, "models": results}, indent=2))


if __name__ == "__main__":
    main()
