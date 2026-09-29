"""Materialize a source-free AutoDub experiment run manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


from autodub.config import WORK_ROOT
from autodub.experiment_store import create_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    if not args.job.startswith("dub-") or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in args.job
    ):
        raise SystemExit("job must be an opaque AutoDub job ID")
    try:
        manifest = create_run(args.experiment, args.job)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"ok": True, "run": manifest["id"], "manifest": str(WORK_ROOT / "experiments" / manifest["id"] / "RUN.json")}, indent=2))


if __name__ == "__main__":
    main()
