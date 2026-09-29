"""Dialogue adaptation runner: pluggable rewrite engines over a job's reviewed script.

The impure half of adaptation.py. Rewrite engines are modular: an engine is just a candidate file
(``work/adapt-candidates/<job>-<engine>.json`` mapping segment i -> {"text": ...}).
``apply_best()`` validates every candidate against the anchor translation, estimates spoken
duration, and picks per line, recording which engine won. Built-in engines:

  qwen      the pinned Qwen3-14B Q8 GGUF served by a loopback KoboldCpp child on ADAPT_PORT,
            batched meaning-locked rewrite prompts, up to ADAPT_MAX_REWRITES rounds.
  external  any other rewriter (a human ADR writer, a larger model) that fills the work file
            produced by ``--mode dump-work`` and saves it in the same candidate format.

Accepted text goes into segment["translation"]; the pre-adaptation anchor is preserved in
segment["adapt"] with the losing candidates, so re-runs re-derive from the original and the
review screen still shows and edits the final word.

Rewrite policy encoded in the prompt is the industry edit-operator ladder: reorder, synonym swap,
contractions, drop redundancy. Names, numbers, negation, question form, tone and profanity strength
never change (the validator enforces the checkable subset).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from . import adaptation, oplog
from .config import (
    ADAPT_BATCH_LINES,
    ADAPT_LEASE_WAIT_S,
    ADAPT_LLM_TEMPERATURE,
    ADAPT_LLM_TIMEOUT_S,
    ADAPT_MAX_REWRITES,
    ADAPT_CONFLICT_PORTS,
    ADAPT_PORT,
    KOBOLDCPP,
    QWEN3_CONTEXTUAL_GGUF,
    WORK_ROOT,
    runtime_env,
)
from .state import event, load_job, save_job


CANDIDATES_DIR = WORK_ROOT / "adapt-candidates"

_SYSTEM_PROMPT = (
    "You are a professional English dub-script adapter (ADR writer) for anime. You rewrite "
    "existing English lines so they can be SPOKEN naturally inside a timing slot. Rules, in "
    "order: never change meaning, plot facts, names, numbers, negation/polarity, "
    "question-vs-statement, tone, or profanity strength. Allowed edits, cheapest first: "
    "reorder the sentence, swap synonyms, use contractions, drop redundant words. Prefer "
    "natural spoken English over subtitle English. Return only the requested JSON array. "
    "/no_think"
)


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.35):
            return True
    except OSError:
        return False


@contextmanager
def context_server(device: str = "cuda"):
    """Loopback KoboldCpp child serving the pinned Qwen3-14B Q8; terminates only itself."""
    for port in ADAPT_CONFLICT_PORTS:
        if _port_open(port):
            raise RuntimeError(f"port {port} is active; stop that service before adaptation")
    if _port_open(ADAPT_PORT):
        raise RuntimeError(f"temporary local port {ADAPT_PORT} is already active")
    if not KOBOLDCPP.is_file() or not QWEN3_CONTEXTUAL_GGUF.is_file():
        raise RuntimeError("KoboldCpp (AUTODUB_KOBOLDCPP) or the Qwen3-14B Q8 model is missing")
    command = [
        str(KOBOLDCPP),
        "--model", str(QWEN3_CONTEXTUAL_GGUF),
        "--host", "127.0.0.1",
        "--port", str(ADAPT_PORT),
        "--contextsize", "16384",
        "--quiet",
    ]
    if device == "cuda":
        command.extend(["--usecublas", "mmq", "--gpulayers", "-1"])
    else:
        command.extend(["--gpulayers", "0"])
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=runtime_env(gpu=device == "cuda", portable_deps=False),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        start_new_session=os.name != "nt",
    )
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("adaptation translator exited during startup")
            if _port_open(ADAPT_PORT):
                break
            time.sleep(1)
        else:
            raise RuntimeError("adaptation translator did not become ready")
        yield
    finally:
        if process.poll() is None:
            # KoboldCpp is a PyInstaller launcher that spawns a worker child; terminating
            # only the launcher would leave the model resident in GPU memory, so the whole
            # process tree is stopped.
            _kill_tree(process)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def _kill_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, timeout=30)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _chat(prompt: str) -> str:
    body = {
        "model": "local-qwen3-14b-q8",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": ADAPT_LLM_TEMPERATURE,
        "top_p": 0.9,
        "max_tokens": 4096,
        "stream": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{ADAPT_PORT}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=ADAPT_LLM_TIMEOUT_S) as response:
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
        raise RuntimeError("adapter returned no JSON array")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, list):
        raise RuntimeError("adapter returned an invalid result")
    return value


def _rewrite_batch(rows: list[dict]) -> dict[int, str]:
    """One LLM call over up to ADAPT_BATCH_LINES rows -> {segment i: candidate text}.

    A failed batch (bad JSON, network hiccup) returns {} — callers treat every row in it
    as "no candidate this round" rather than aborting the episode.
    """
    payload = []
    for row in rows:
        entry = {
            "id": int(row["i"]),
            "japanese": row["japanese"],
            "english": row["current"],
        }
        if row["kind"] == "long":
            entry["task"] = f"shorten to at most {int(row['max_chars'])} characters"
        else:
            entry["task"] = (f"expand naturally to about {int(row['target_chars'])} characters; "
                             "no invented facts, just fuller natural phrasing")
        payload.append(entry)
    prompt = (
        "Rewrite each line per its task. The japanese field is the original meaning; the "
        "english field is the current line. Return a JSON array with exactly one object per "
        "input, same order: {\"id\": integer, \"english\": string}.\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    try:
        result = _json_array(_chat(prompt))
        return {int(item["id"]): str(item["english"]).strip()
                for item in result if isinstance(item, dict) and "id" in item}
    except Exception:
        return {}


def _work_rows(job: dict) -> tuple[list[dict], list[dict]]:
    """(all plan rows, the long/short subset that needs an engine)."""
    rows = adaptation.plan(job["segments"])
    work = [dict(row) for row in rows if row["kind"] in ("long", "short")]
    return rows, work


def _save_json(path: Path, value: dict) -> None:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")


def dump_work(job_id: str) -> Path:
    """Work file for the external engine: everything a rewriter needs per line and nothing
    else (no speaker evidence, no media paths)."""
    job = load_job(job_id)
    _, work = _work_rows(job)
    lines = []
    for row in work:
        keys = ["i", "kind", "japanese", "anchor", "budget_s", "base_s",
                "max_chars" if row["kind"] == "long" else "target_chars"]
        lines.append({key: row[key] for key in keys})
    payload = {"job": job_id, "lines": lines}
    path = CANDIDATES_DIR / f"{job_id}-work.json"
    _save_json(path, payload)
    return path


def collect_qwen(job_id: str, *, log=print) -> Path:
    """qwen-local engine: run the rewrite rounds, save best-per-line, NO job mutation.
    Requires context_server() to be up. Iterates like a real adapter: a still-long line
    gets its previous attempt as the new starting text for the next round.
    """
    job = load_job(job_id)
    if not job.get("segments"):
        raise RuntimeError("job has no analyzed segments")
    _, work = _work_rows(job)
    for row in work:
        row.update(current=row["anchor"], best="", best_est=None, rewrites=0, done=False)
    shorts = [row for row in work if row["kind"] == "short"]
    longs = [row for row in work if row["kind"] == "long"]

    def run_round(rounds_rows: list[dict]) -> None:
        for offset in range(0, len(rounds_rows), ADAPT_BATCH_LINES):
            chunk = rounds_rows[offset:offset + ADAPT_BATCH_LINES]
            candidates = _rewrite_batch(chunk)
            for row in chunk:
                candidate = candidates.get(int(row["i"]), "")
                row["rewrites"] += 1
                if not candidate:
                    continue
                ok, _reason = adaptation.validate_rewrite(row["anchor"], candidate)
                if not ok:
                    continue
                est = adaptation.estimate_seconds(candidate)
                if row["kind"] == "long":
                    if row["best_est"] is None or est < row["best_est"]:
                        row["best"], row["best_est"] = adaptation.normalize_text(candidate), est
                    row["current"] = adaptation.normalize_text(candidate)
                    row["done"] = est <= row["budget_s"]
                elif est <= row["budget_s"] and est > row["est_s"]:
                    row["best"], row["best_est"] = adaptation.normalize_text(candidate), est
                    row["done"] = True

    if shorts:
        run_round(shorts)
    round_no = 0
    while round_no < ADAPT_MAX_REWRITES:
        pending = [row for row in longs if not row["done"]]
        if not pending:
            break
        round_no += 1
        log(f"  round {round_no}: {len(pending)} long lines")
        run_round(pending)

    path = CANDIDATES_DIR / f"{job_id}-qwen.json"
    _save_json(path, {
        "job": job_id,
        "engine": "qwen",
        "candidates": {str(row["i"]): {"text": row["best"], "rewrites": row["rewrites"]}
                       for row in work if row["best"]},
    })
    log(f"  qwen candidates: {sum(1 for row in work if row['best'])}/{len(work)} lines")
    return path


def _load_candidates(job_id: str, engine: str) -> dict[int, str]:
    path = CANDIDATES_DIR / f"{job_id}-{engine}.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): str(value.get("text") or "")
            for key, value in (data.get("candidates") or {}).items()}


def apply_best(job_id: str, engines: list[str], *, picks: dict[str, str] | None = None) -> dict:
    """Merge candidate files into the job: validate, estimate, pick per line, write back.

    Deterministic pick order per line: (1) an explicit picks[str(i)] = engine override
    (a reviewer's per-line quality judgment) if that candidate is valid; (2) among valid
    candidates that FIT the budget, engine list order breaks ties; (3) nothing fits ->
    shortest valid candidate wins, line flagged residue. Anchor always survives when no
    candidate is usable. Every losing candidate is kept in the adapt block for the diff.
    """
    job = load_job(job_id)
    rows, work = _work_rows(job)
    by_i = {int(segment["i"]): segment for segment in job["segments"]}
    candidate_sets = {engine: _load_candidates(job_id, engine) for engine in engines}
    picks = picks or {}

    changed = 0
    residue_lines = []
    wins: dict[str, int] = {}
    work_by_i = {int(row["i"]): row for row in work}
    for row in rows:
        segment = by_i.get(int(row["i"]))
        if segment is None or row["kind"].startswith("skip"):
            continue
        outcome = work_by_i.get(int(row["i"]))
        evaluated = {}
        if outcome:
            for engine in engines:
                text = candidate_sets[engine].get(int(row["i"]), "")
                if not text:
                    continue
                ok, reason = adaptation.validate_rewrite(row["anchor"], text)
                est = adaptation.estimate_seconds(text)
                grew = est > row["est_s"]
                fits = (ok and est <= row["budget_s"] and (row["kind"] == "long" or grew))
                evaluated[engine] = {"text": adaptation.normalize_text(text),
                                     "est_s": round(est, 3), "valid": ok,
                                     "reject": reason, "fits": fits}
        chosen_engine, verdict = "anchor", ("fit" if not outcome else
                                            "short-kept" if row["kind"] == "short" else "residue")
        fit_verdict = "adapted" if row["kind"] == "long" else "expanded"
        forced = picks.get(str(row["i"]))
        if forced and evaluated.get(forced, {}).get("text"):
            # An explicit pick is a REVIEWED judgment and outranks the auto-validator —
            # the validator guards unsupervised merges (e.g. anchors polluted with OCR'd
            # on-screen text flunk every honest rewrite on "dropped" caps/digits).
            chosen_engine = forced
            verdict = (fit_verdict if evaluated[forced]["fits"]
                       else "residue" if row["kind"] == "long" else "short-kept")
        else:
            fitting = [engine for engine in engines if evaluated.get(engine, {}).get("fits")]
            if fitting:
                chosen_engine = fitting[0]
                verdict = fit_verdict
            elif outcome and row["kind"] == "long":
                valid = [(evaluated[engine]["est_s"], engine) for engine in engines
                         if evaluated.get(engine, {}).get("valid")]
                if valid and min(valid)[0] < row["est_s"]:
                    chosen_engine = min(valid)[1]
                verdict = "residue"
        accepted = (evaluated[chosen_engine]["text"] if chosen_engine != "anchor"
                    else row["anchor"])
        accepted_est = (evaluated[chosen_engine]["est_s"] if chosen_engine != "anchor"
                        else row["est_s"])
        wins[chosen_engine] = wins.get(chosen_engine, 0) + (1 if outcome else 0)
        if verdict == "residue":
            residue_lines.append(int(row["i"]))
        segment["adapt"] = {
            "schema": 2,
            "anchor": row["anchor"],
            "anchor_source": row["anchor_source"],
            "base_s": row["base_s"],
            "budget_s": row["budget_s"],
            "anchor_est_s": row["est_s"],
            "est_s": accepted_est,
            "verdict": verdict,
            "engine": chosen_engine,
            "candidates": {engine: value["text"] for engine, value in evaluated.items()},
        }
        if accepted and accepted != adaptation.normalize_text(segment.get("translation") or ""):
            segment["translation"] = accepted
            segment["translation_source"] = "adapted-v1"
            changed += 1

    counts: dict[str, int] = {}
    for row in rows:
        if row["kind"].startswith("skip"):
            counts[row["kind"]] = counts.get(row["kind"], 0) + 1
        else:
            verdict = by_i[int(row["i"])]["adapt"]["verdict"]
            counts[verdict] = counts.get(verdict, 0) + 1
    job["adapt_summary"] = {
        "schema": 2,
        "policy": "adapt-slot-v1",
        "engines": engines,
        "lines": len(rows),
        "changed": changed,
        "counts": counts,
        "engine_wins": wins,
        "residue_lines": residue_lines,
    }
    event(job, "review",
          "Dialogue adaptation: %d lines, %d rewritten; wins %s."
          % (len(rows), changed, json.dumps(wins)), 72)
    save_job(job)
    oplog.job_event(job_id, "adaptation",
                    f"adapt-slot-v1 applied ({'+'.join(engines)}): {json.dumps(counts)}")
    return job["adapt_summary"]


def adapt_job(job_id: str, *, log=print) -> dict:
    """Single-engine path: capture qwen candidates, then apply them."""
    collect_qwen(job_id, log=log)
    return apply_best(job_id, ["qwen"])


def adapt_reviewed(job_id: str, *, gpu_authorized: bool = False,
                   engines: list[str] | None = None,
                   picks: dict[str, str] | None = None,
                   capture: bool = True) -> dict:
    """Adaptation stage on a reviewed job: job status, events and GPU lease around the engines.

    Engines default to every candidate file present (external + qwen when an external candidate
    file exists, qwen only otherwise)."""
    from .gpu_session import GpuSafetyError, gpu_lease

    job = load_job(job_id)
    if not job.get("segments"):
        raise ValueError("analyze and review the job before adapting dialogue")
    previous_status = job.get("status")
    job["status"] = "running"
    event(job, "adapting", "Slot-aware dialogue adaptation: fitting English lines "
          "to their speech windows.", 72)
    try:
        if capture:
            if not gpu_authorized:
                raise GpuSafetyError("dialogue adaptation requires a fresh GPU arm")
            with gpu_lease(f"adapt:{job_id}", wait_seconds=ADAPT_LEASE_WAIT_S) as lease:
                lease.ensure_active()
                with context_server("cuda"):
                    collect_qwen(job_id,
                                 log=lambda m: oplog.job_event(job_id, "adapting", str(m)))
        chosen = engines or [e for e in ("external", "qwen")
                             if (CANDIDATES_DIR / f"{job_id}-{e}.json").is_file()]
        if not chosen:
            raise ValueError("no adaptation candidate files exist for this job")
        summary = apply_best(job_id, chosen, picks=picks)
        # apply_best loads/saves the job itself — reload before touching status, or
        # this stale pre-apply dict would clobber the merge on save.
        job = load_job(job_id)
        job["status"] = "review"
        event(job, "review", "Dialogue adaptation applied; review the adapted lines "
              "before rendering.", 72)
        return summary
    except Exception:
        job = load_job(job_id)
        job["status"] = previous_status or "review"
        job["error"] = "dialogue adaptation failed"
        job["error_detail"] = traceback.format_exc()[-2000:]
        save_job(job)
        oplog.job_error(job_id, "adapting", traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="run slot-aware dialogue adaptation")
    parser.add_argument("--job", action="append", dest="jobs", required=True)
    parser.add_argument("--mode", choices=("full", "dump-work", "capture", "apply"),
                        default="full",
                        help="full = capture qwen + apply qwen (single-engine); dump-work = "
                             "write work files for an external engine; capture = qwen "
                             "candidates only; apply = merge candidate files into the jobs")
    parser.add_argument("--engines", default="qwen",
                        help="apply mode: comma list in tie-break priority order")
    parser.add_argument("--picks", default="",
                        help="apply mode: JSON file {job_id: {segment_i: engine}} of per-line overrides")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ack-private-local-material", action="store_true")
    args = parser.parse_args()
    if not args.ack_private_local_material:
        raise SystemExit("adaptation reads transcript text; pass --ack-private-local-material")
    results = {}

    if args.mode == "dump-work":
        for job_id in args.jobs:
            path = dump_work(job_id)
            results[job_id] = {"work": str(path)}
            print(f"work file: {path}", flush=True)
    elif args.mode == "apply":
        engines = [engine.strip() for engine in args.engines.split(",") if engine.strip()]
        all_picks = json.loads(Path(args.picks).read_text(encoding="utf-8")) if args.picks else {}
        for job_id in args.jobs:
            summary = apply_best(job_id, engines, picks=all_picks.get(job_id) or {})
            results[job_id] = summary
            print(f"{job_id}: wins {json.dumps(summary['engine_wins'])} "
                  f"counts {json.dumps(summary['counts'])}", flush=True)
    else:
        target = collect_qwen if args.mode == "capture" else adapt_job

        def execute() -> None:
            with context_server(args.device):
                for job_id in args.jobs:
                    started = time.monotonic()
                    print(f"{args.mode} {job_id} ...", flush=True)
                    try:
                        outcome = target(job_id)
                        seconds = round(time.monotonic() - started, 1)
                        results[job_id] = {"result": str(outcome), "seconds": seconds}
                        print(f"  done in {seconds}s", flush=True)
                    except Exception as exc:
                        results[job_id] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                        print(f"  FAILED: {results[job_id]['error']}", flush=True)

        if args.device == "cuda":
            from .gpu_session import gpu_lease
            with gpu_lease("adaptation:" + args.jobs[0], wait_seconds=ADAPT_LEASE_WAIT_S) as lease:
                lease.ensure_active()
                execute()
        else:
            execute()
    print(json.dumps({"results": results}, indent=2))
    if any("error" in value for value in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
