"""series view: the series-level view.

Episodes of one show are analyzed as separate jobs, but a character should be named once for the
whole show. The voice bank is already per-series; this module is the surface that treats a season
as one thing. This module groups analyzed jobs by series, merges every episode's unassigned
speakers, and proposes CROSS-EPISODE CLUSTERS — "these N speakers across M episodes look
like one character" — so the reviewer names a character once and the mapping lands in every
member episode at the same moment.

Clustering is greedy COMPLETE-LINKAGE at the identity (match) threshold: a speaker joins
a cluster only if it scores >= VOICE_BANK_MATCH_THRESHOLD against EVERY current member.
(Complete linkage replaced an earlier single-link join at the ASK threshold, which chained
different people into one unjudgeable 15-speaker group: calibration put different-people pairs
as high as 0.914, so 0.90 sat inside that range, and representative-only matching let A~B~C
chains form. The cheap error direction is a
real character split across two groups: that costs one extra "same" click at the bank.)
It is a PROPOSAL generator, not a decider — every group is confirmed or rejected by the
reviewer, and each member carries its worst-link cosine ("cohesion") so a weak join is
visible. Suggestions never relabel a speaker; only recorded answers do.

Reads job state; the only writes go through the same paths the
per-episode panel uses (``speaker_characters`` + the series decision log).
"""
from __future__ import annotations

import re
from typing import Any

from . import voice_bank
from .characters import _centroids, _seed_reference, dialogue_segments, guess_series, song_speakers
from .cast_match import hints as cast_hints, load_cast
from .evidence_cards import plan_cards
from .state import load_job, save_job

_EP_TAG = re.compile(r"S(\d{1,2})E(\d{1,3})", re.IGNORECASE)


def _episode_label(job: dict[str, Any]) -> str:
    name = str(job.get("source", {}).get("original_name") or "")
    match = _EP_TAG.search(name)
    if match:
        return f"S{int(match.group(1)):02d}E{int(match.group(2)):02d}"
    return job.get("id", "?")[-4:]


def _job_series(job: dict[str, Any]) -> str:
    return str(job.get("voice_bank_series") or "").strip() or guess_series(
        str(job.get("source", {}).get("original_name")
            or job.get("source", {}).get("file") or ""))


def _iter_jobs() -> list[dict[str, Any]]:
    from .config import WORK_ROOT
    jobs = []
    root = WORK_ROOT / "jobs"
    if not root.exists():
        return jobs
    for path in sorted(root.glob("dub-*/job.json")):
        try:
            jobs.append(load_job(path.parent.name))
        except Exception:
            continue
    return jobs


def list_projects() -> dict[str, Any]:
    """Every series with its member episodes — the series view landing list."""
    groups: dict[str, dict[str, Any]] = {}
    for job in _iter_jobs():
        name = _job_series(job)
        try:
            slug = voice_bank.series_slug(name)
        except ValueError:
            continue
        group = groups.setdefault(slug, {"series": name, "slug": slug, "episodes": []})
        speakers = sorted({str(s.get("speaker") or "") for s in job.get("segments") or []} - {""})
        assigned = job.get("speaker_characters") or {}
        group["episodes"].append({
            "job_id": job["id"],
            "episode": _episode_label(job),
            "status": job.get("status"),
            "stage": job.get("stage"),
            "analyzed": bool(speakers),
            "speakers": len(speakers),
            "assigned": sum(1 for s in speakers if s in assigned),
        })
    out = []
    for slug, group in sorted(groups.items()):
        group["episodes"].sort(key=lambda e: e["episode"])
        # Season subfolders are purely a view grouping: the BANK stays per-show, so a character named in season 1
        # keeps the same voice in season 4 - which is the whole cross-season requirement.
        seasons: dict[str, list] = {}
        for ep in group["episodes"]:
            m = _EP_TAG.match(ep["episode"]) or _EP_TAG.search(ep["episode"])
            key = f"Season {int(m.group(1))}" if m else "Other"
            seasons.setdefault(key, []).append(ep)
        group["seasons"] = [{"season": k, "episodes": v}
                            for k, v in sorted(seasons.items())]
        bank = voice_bank.load_bank(slug)
        group["bank_characters"] = len(bank["characters"])
        out.append(group)
    return {"projects": out}


def _cosine(a, b) -> float:
    import math
    if len(a) != len(b) or not a:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else -1.0


def _confidence(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure: the red/amber/green confidence light for one group.

    Complete linkage checks every joiner against every member, so the min over join-time
    cohesions IS the exact all-pairs worst link. Thin solo audio can make a confident
    cosine a lie, so evidence depth gates the color too. Solo groups -> level "solo"
    (nothing to cross-check; a one-scene character is not a red flag)."""
    from .config import (GROUP_GREEN_COHESION, GROUP_GREEN_MIN_SOLO_S,
                         GROUP_RED_COHESION, GROUP_RED_MIN_SOLO_S)
    solo = [round(sum(c["end"] - c["start"] for c in m.get("cards", [])), 2)
            for m in members]
    min_solo = min(solo) if solo else 0.0
    if len(members) < 2:
        return {"level": "solo", "worst_link": None, "min_solo_s": min_solo}
    worst = min(m["cohesion"] for m in members[1:])
    if worst < GROUP_RED_COHESION or min_solo < GROUP_RED_MIN_SOLO_S:
        level = "red"
    elif worst >= GROUP_GREEN_COHESION and min_solo >= GROUP_GREEN_MIN_SOLO_S:
        level = "green"
    else:
        level = "amber"
    return {"level": level, "worst_link": worst, "min_solo_s": min_solo}


def _cluster(instances: list[dict[str, Any]],
             separated: set[tuple[str, str]] | None = None,
             unified: set[tuple[str, str]] | None = None,
             threshold: float | None = None) -> list[dict[str, Any]]:
    """Greedy complete-linkage grouping (pure; hermetically tested).

    A candidate joins the cluster where its WORST cosine against the existing members is
    best, and only when that worst link clears the identity-grade match threshold. The
    first (heaviest solo material) member stays the representative for bank matching and
    the card face; ``cohesion`` records each member's worst link at join time.

    ``separated`` holds reviewer cannot-link pairs (sorted ``job::speaker`` key tuples,
    roster "Not them" clicks): a candidate never joins a cluster containing a speaker
    the reviewer said is a different person, no matter the cosine.
    """
    separated = separated or set()
    unified = unified or set()
    threshold = threshold if threshold is not None else voice_bank.VOICE_BANK_MATCH_THRESHOLD
    clusters: list[dict[str, Any]] = []
    for inst in instances:
        key = voice_bank._member_key(inst["job_id"], inst["speaker"])
        forced = None
        for cluster in clusters:
            if any(tuple(sorted((key, other))) in unified for other in cluster["keys"]):
                forced = cluster
                break
        if forced is not None:
            # Reviewer must-link: join regardless of cosine; cohesion stays honest (the
            # real worst link), so a forced merge of unlike voices reads red, not green.
            score = min(_cosine(inst["centroid"], c) for c in forced["centroids"])
            forced["members"].append({**_public(inst), "cohesion": round(score, 3)})
            forced["centroids"].append(inst["centroid"])
            forced["keys"].append(key)
            continue
        best, best_score = None, -1.0
        for cluster in clusters:
            if any(tuple(sorted((key, other))) in separated for other in cluster["keys"]):
                continue
            score = min(_cosine(inst["centroid"], c) for c in cluster["centroids"])
            if score > best_score:
                best, best_score = cluster, score
        if best is not None and best_score >= threshold:
            best["members"].append({**_public(inst), "cohesion": round(best_score, 3)})
            best["centroids"].append(inst["centroid"])
            best["keys"].append(key)
        else:
            clusters.append({
                "rep_centroid": inst["centroid"],
                "centroids": [inst["centroid"]],
                "keys": [key],
                "members": [{**_public(inst), "cohesion": 1.0}],
            })
    return clusters


def series_characters(slug: str) -> dict[str, Any]:
    """The merged season view: bank + every analyzed episode's UNASSIGNED speakers,
    clustered across episodes, each cluster matched against the bank.

    A cluster's zone: ``clear`` (mean centroid matches a bank character at the match
    threshold — one click applies it to every member), ``ask`` (bank candidate shown, the
    reviewer decides), ``new`` (no bank candidate — name it once, map everywhere).
    """
    jobs = [j for j in _iter_jobs()
            if voice_bank.series_slug(_job_series(j)) == slug and (j.get("segments"))]
    if not jobs:
        raise ValueError("no analyzed episodes in this series yet")
    name = _job_series(jobs[0])
    bank = voice_bank.load_bank(slug)

    # every unassigned speaker instance, with its centroid and its evidence cards
    instances = []
    for job in jobs:
        assigned = job.get("speaker_characters") or {}
        songs = song_speakers(job)
        cents = _centroids(job)
        cards = plan_cards(dialogue_segments(job))
        for speaker, centroid in cents.items():
            if speaker in assigned or speaker in songs:
                continue
            instances.append({
                "job_id": job["id"], "episode": _episode_label(job),
                "speaker": speaker, "centroid": centroid,
                "cards": cards.get(speaker, []),
                # more solo card material first: those become cluster representatives,
                # so the group's face is its clearest-voiced member
                "weight": sum(c["end"] - c["start"] for c in cards.get(speaker, [])),
            })
    instances.sort(key=lambda i: (-i["weight"], i["episode"], i["speaker"]))

    _separations = voice_bank.separation_set(bank)
    _sep_keys = {key for pair in _separations for key in pair}
    _cast = load_cast(slug)
    clusters = _cluster(instances, _separations, voice_bank.union_set(bank))

    result = []
    for n, cluster in enumerate(clusters, 1):
        match = voice_bank.match_speakers(
            slug, {"g": cluster["rep_centroid"]}).get("g") or {}
        # Roster memory: if the reviewer already said "not that character" for ANY member,
        # the whole proposal is dead — do not re-ask on every reload.
        cid = match.get("character_id")
        if cid and any(voice_bank.is_rejected(bank, cid, m["job_id"], m["speaker"])
                       for m in cluster["members"]):
            match = {}
        result.append({
            "group": n,
            "zone": match.get("zone", "new"),
            "bank_candidate": {"id": match.get("character_id"),
                               "name": match.get("character_name"),
                               "score": match.get("score")},
            "episodes": sorted({m["episode"] for m in cluster["members"]}),
            "confidence": _confidence(cluster["members"]),
            # Cast-pack hint: "likely: <name>" from episode-signature matching when a
            # reviewer-supplied cast.json exists. Presentation only.
            "cast_hints": cast_hints(_ep_ints({m["episode"] for m in cluster["members"]}),
                                     _cast) if _cast else [],
            # A reviewer "different person" click must not vanish into the minor-voices
            # pile: solos carrying a separation surface in the roster's TO SORT queue.
            "reviewer_split": any(key in _sep_keys for key in cluster["keys"]),
            "members": cluster["members"],
            # cards from up to three DIFFERENT episodes so the reviewer judges the group
            # across the season, not one scene three times
            "cards": _spread_cards(cluster["members"]),
        })
    result.sort(key=lambda g: (-len(g["members"]), g["group"]))
    return {"series": name, "slug": slug,
            "bank": [{"id": c["id"], "name": c["name"]} for c in bank["characters"]],
            "episodes_analyzed": len(jobs),
            "minor_suggestions": _minor_suggestions(clusters, result, _separations, _cast),
            "groups": result}


def _ep_ints(labels) -> list[int]:
    out = []
    for lab in labels:
        m = _EP_TAG.search(str(lab))
        if m:
            out.append(int(m.group(2)))
    return out


def _minor_suggestions(clusters: list[dict[str, Any]], result: list[dict[str, Any]],
                       separated: set[tuple[str, str]], cast=None) -> list[dict[str, Any]]:
    """Ask-band COMPLETE-LINKAGE groupings among solo groups, because some "minor voices"
    are really recurring characters. The 0.95 join
    bar deliberately refuses 0.90-0.94 links, so a real recurring character can land as
    several solos - this proposes those re-joins. Complete linkage at the ask threshold
    (every pair must clear 0.90), because the first cut used single-link components and
    immediately rebuilt a useless 28-voice chained blob. Proposals only; nothing moves."""
    solos = [(i, cl) for i, cl in enumerate(clusters) if len(cl["members"]) == 1]
    # Group numbers were assigned as enumerate(clusters, 1) BEFORE result was sorted, so
    # derive them from cluster position — zipping against the sorted result mislabeled
    # every suggestion (one group number appeared as both a multi and a solo group).
    group_no = {id(cl): i + 1 for i, cl in enumerate(clusters)}
    pseudo = [{"job_id": cl["members"][0]["job_id"], "episode": cl["members"][0]["episode"],
               "speaker": cl["members"][0]["speaker"], "centroid": cl["rep_centroid"],
               "cards": cl["members"][0]["cards"], "weight": 0.0, "_src": (i, cl)}
              for i, cl in solos]
    joined = _cluster(pseudo, separated, set(),
                      threshold=voice_bank.VOICE_BANK_ASK_THRESHOLD)
    out = []
    for sug in joined:
        if len(sug["members"]) < 2:
            continue
        links = [m["cohesion"] for m in sug["members"][1:]]
        by_key = {(m["job_id"], m["speaker"]) for m in sug["members"]}
        src = [cl for i, cl in solos if (cl["members"][0]["job_id"],
                                         cl["members"][0]["speaker"]) in by_key]
        eps = sorted({cl["members"][0]["episode"] for cl in src})
        out.append({
            "groups": sorted(group_no[id(cl)] for cl in src),
            "score_min": round(min(links), 3), "score_max": round(max(links), 3),
            "members": [cl["members"][0] for cl in src],
            "episodes": eps,
            "cast_hints": cast_hints(_ep_ints(eps), cast) if cast else [],
        })
    out.sort(key=lambda s: (-len(s["members"]), -s["score_max"]))
    return out


def _public(inst: dict[str, Any]) -> dict[str, Any]:
    return {k: inst[k] for k in ("job_id", "episode", "speaker", "cards")}


def _spread_cards(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out, seen_eps = [], set()
    for member in members:                       # members arrive best-evidence-first
        if member["episode"] in seen_eps:
            continue
        for card in member["cards"][:1]:
            out.append({**card, "job_id": member["job_id"], "episode": member["episode"]})
        seen_eps.add(member["episode"])
        if len(out) >= 3:
            break
    return out


def apply_group_answer(slug: str, *, members: list[dict[str, Any]], answer: str,
                       character_id: str | None = None,
                       new_name: str | None = None) -> dict[str, Any]:
    """One reviewer verdict for a whole cross-episode group.

    same       -> map EVERY member speaker in its own job to `character_id`.
    different  -> with `new_name`: seed the character from the first member that can
                  provide a reference (best evidence first), then map every member.
    skip       -> one logged decision, nothing mapped.
    """
    if answer not in {"same", "different", "skip"}:
        raise ValueError("answer must be same, different or skip")
    if not members:
        raise ValueError("a group answer needs members")
    if answer == "skip":
        voice_bank.log_decision(slug, kind="group_skip",
                                members=[f"{m['job_id']}:{m['speaker']}" for m in members])
        return {"ok": True, "answer": "skip", "mapped": 0}

    created = None
    if answer == "different":
        if not (new_name or "").strip():
            raise ValueError("'different' needs the new character's name")
        seed_error = None
        for member in members:                   # first member able to give a clean clip wins
            job = load_job(str(member["job_id"]))
            try:
                clip, transcript = _seed_reference(job["id"], job, str(member["speaker"]),
                                                   None, slug)
                centroid = _centroids(job).get(str(member["speaker"]))
                if not centroid:
                    raise ValueError("no embedding evidence")
                created = voice_bank.add_character(
                    slug, new_name.strip(), centroid=centroid, reference_clip=clip,
                    reference_transcript=transcript, source_job=job["id"])
                character_id = created["id"]
                break
            except (ValueError, FileNotFoundError) as exc:
                seed_error = exc
                continue
        if created is None:
            raise ValueError(f"no member could seed a reference ({seed_error})")
    elif not character_id:
        raise ValueError("'same' needs the bank character_id")

    mapped = 0
    for member in members:
        job = load_job(str(member["job_id"]))
        job.setdefault("voice_bank_series", _job_series(job))
        job.setdefault("speaker_characters", {})[str(member["speaker"])] = character_id
        save_job(job)
        mapped += 1
    voice_bank.log_decision(
        slug, kind="group_answer", answer=answer, character_id=character_id,
        character_name=(created or {}).get("name") or next(
            (c["name"] for c in voice_bank.load_bank(slug)["characters"]
             if c["id"] == character_id), None),
        members=[f"{m['job_id']}:{m['speaker']}" for m in members])
    return {"ok": True, "answer": answer, "character_id": character_id,
            "created": created, "mapped": mapped}


def mark_song_group(slug: str, *, members: list[dict[str, Any]]) -> dict[str, Any]:
    """Reviewer verdict: this whole group is the opening/ending singer, not a character.
    Theme songs are not dubbed.

    For each member: the speaker joins its job's additive ``song_speakers`` list and
    every one of that speaker's segments gets ``song_skip=True`` — the render's existing
    song policy then skips those lines, and both review boards stop proposing the
    speaker. Diarization labels are never touched; the decision is logged append-only.
    """
    if not members:
        raise ValueError("marking a group as intro/outro needs members")
    marked_segments = 0
    for member in members:
        job = load_job(str(member["job_id"]))
        speaker = str(member["speaker"])
        songs = {str(s) for s in (job.get("song_speakers") or [])}
        songs.add(speaker)
        job["song_speakers"] = sorted(songs)
        for segment in job.get("segments") or []:
            if str(segment.get("speaker") or "") == speaker:
                segment["song_skip"] = True
                marked_segments += 1
        save_job(job)
    voice_bank.log_decision(
        slug, kind="group_song",
        members=[{"job_id": str(m["job_id"]), "speaker": str(m["speaker"])} for m in members],
        segments=marked_segments)
    return {"marked_members": len(members), "marked_segments": marked_segments}


def separate_member(slug: str, *, member: dict[str, Any],
                    others: list[dict[str, Any]]) -> dict[str, Any]:
    """Roster click: "this clip is NOT the same person as the rest of this group."
    Records cannot-link pairs so the next clustering pass never re-merges them; the
    member resurfaces as its own entry. Labels and existing mappings untouched."""
    if not others:
        raise ValueError("separating a member needs the rest of its group")
    added = voice_bank.add_separations(slug, member=member, others=others)
    return {"ok": True, "separations_added": added}


def reject_candidate(slug: str, *, character_id: str,
                     members: list[dict[str, Any]]) -> dict[str, Any]:
    """Roster click: "this group/clip is NOT that named character." Remembered in
    the bank so the also-them proposal never comes back for these speakers."""
    if not character_id or not members:
        raise ValueError("rejecting needs a character and members")
    for m in members:
        voice_bank.add_rejection(slug, character_id=str(character_id),
                                 job_id=str(m["job_id"]), speaker=str(m["speaker"]))
    return {"ok": True, "rejected": len(members)}


def unite_members(slug: str, *, member: dict[str, Any],
                  target: dict[str, Any]) -> dict[str, Any]:
    """Roster reassign: "this clip IS that candidate group's person." Must-link
    remembered in the bank; the next clustering pass joins them (cohesion stays the real
    score, so a forced merge of unlike voices reads red). Labels untouched."""
    voice_bank.add_union(slug, member=member, target=target)
    return {"ok": True}
