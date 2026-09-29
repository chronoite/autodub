"""Offline verification of installed model payloads against their provenance records.

Reports every model in the manifest as verified, missing (not downloaded), or failed (a file is
missing, resized, or its SHA-256 changed). Exit code 1 on any failure; missing models are listed
but do not fail, because only the ``core`` group is required for the CPU pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_manifest import MODELS, PROVENANCE  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(name: str, *, quick: bool) -> dict:
    spec = MODELS[name]
    record_path = Path(spec["destination"]) / PROVENANCE
    if not record_path.is_file():
        return {"name": name, "group": spec["group"], "status": "missing"}
    record = json.loads(record_path.read_text(encoding="utf-8"))
    problems = []
    for item in record.get("files", []):
        path = record_path.parent / item["path"]
        if not path.is_file():
            problems.append(f"missing file {item['path']}")
        elif path.stat().st_size != item["bytes"]:
            problems.append(f"size mismatch {item['path']}")
        elif not quick and sha256(path) != item["sha256"]:
            problems.append(f"hash mismatch {item['path']}")
    return {"name": name, "group": spec["group"], "status": "failed" if problems else "verified",
            "files": len(record.get("files", [])), "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description="verify installed AutoDub models offline")
    parser.add_argument("--quick", action="store_true", help="check sizes only, skip hashing")
    args = parser.parse_args()
    results = [verify(name, quick=args.quick) for name in MODELS]
    print(json.dumps({"ok": not any(r["status"] == "failed" for r in results), "models": results}, indent=2))
    if any(r["status"] == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
