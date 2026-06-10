"""Routing ground-truth builder for ``graph.ask()`` (v0.29 E1 / T3).

Builds the per-query routing-label table that the E2 signal-AUC harness and
the E3 router gates consume. Output is JSONL, exactly one row per
namespaced qid:

    qid              "<corpus>:<source_qid>" — raw qids collide across
                     corpora (three conversational sheets all use
                     c001..c030; krra_hard and x2bee_hard both use h001..)
    source_qid       qid as it appears in the source file
    corpus           source corpus / bench name (xlsx sheet name, finreg
                     file stem, "krra_graph", "autorag")
    query            query text when the source carries it, else null
    type             query-type label enumerated from the source data —
                     never hardcoded (assort_hard alone carries 18 types)
    single_shot_hit  bool | null — the cheap single-shot path succeeded.
                     The *meaning* differs per source (see
                     ``single_shot_basis``); E2/T5 consumers must not
                     conflate the judge-based and deterministic variants.
    single_shot_basis  "judge" | "design" | "id_hit" | null — what kind of
                     measurement the single_shot_hit axis is:
                       judge   finreg JSONL — judge-correct majority over
                               the naive-RAG arm (answer-level; diverges
                               from the plan's deterministic hit@5 — the
                               T5 proxy-validity check compares the two)
                       design  krra_graph — ``current_answerable``
                               annotation, not a measurement
                       id_hit  hits JSONL — deterministic single-shot
                               top-k id-hit
    single_shot_source  loader that supplied the single_shot axis (differs
                     from ``source`` after an axis merge), null when the
                     axis is unmeasured
    agent_solve      bool | null — the agent loop solved the query
    agent_source     loader that supplied the agent axis, null when
                     unmeasured
    label            2x2 routing label, see ``routing_label()``
    tier             measurement confidence: confirmed | provisional |
                     hit_only | unmeasured
    source           loader that produced the (base) row
    split            train | heldout (deterministic, hash-parity per qid)

Label semantics (``single_shot_hit`` x ``agent_solve``):

    True  x True   -> both              (either path works)
    True  x False  -> cheap_sufficient  (escalation would be waste)
    False x True   -> agent_required    (the routing payoff class)
    False x False  -> unsolved
    one axis None  -> partial label: single_shot_hit / single_shot_miss /
                      agent_solved / agent_failed
    both axes None -> unlabeled

Tier discipline (agent benches have a ±8/120 run-to-run noise floor, so a
single run must never masquerade as a confirmed label):

    confirmed    >=3 runs per arm, outcome = per-arm majority (>=2 wins)
    provisional  single measurement (one agent run / design annotation)
    hit_only     single-shot axis only, no agent measurement at all
    unmeasured   no measurement on either axis

Input sources — each one is optional; a missing file or missing optional
dependency skips that source with a warning instead of failing:

(a) ``--finreg-jsonl``: per-(query, arm, run) JSONL written by
    ``examples/ablation/rag_vs_agent_answer.py --out-jsonl`` (v0.29 T1/T2
    rerun). Records are deduped by (qid, arm, run) first — keeping the
    last occurrence, so concatenated/resumed JSONL files cannot mint
    extra runs. ``agent_solve`` and ``single_shot_hit`` are the per-arm
    strict majorities over distinct runs (``eval.unified.majority_solve``
    with min_wins = n_runs // 2 + 1, i.e. >=2/3 for the canonical 3-run
    protocol; a 2/4 tie is NOT solved); tier is confirmed only when the
    qid has >=3 distinct runs on *both* arms.
(b) ``--gt-xlsx``: ``eval/data/gt_datasets.xlsx`` — every sheet with a
    ``qid`` header row contributes rows; ``type`` labels are read from the
    sheet. The agent axis is joined from ``--agent-log`` (the v0.17.1
    single-run bench log) at tier=provisional when present; rows without a
    log entry stay unmeasured (this is where the assort Hard OOM queries
    land — they are separated, not relabeled). This source carries no
    single-shot axis itself; pair it with a ``--hits-jsonl`` file for the
    same corpus to complete the 2x2 label via axis merge (see below).
(c) ``--krra-graph``: ``eval/data/queries/krra_graph.json`` — queries
    *designed* to be unanswerable by top-k retrieval alone
    (``current_answerable`` false plus a per-query ``topk_inadequacy``
    rationale). Such rows are labeled agent_required at tier=provisional:
    a design annotation, not an agent measurement.
(d) ``--hits-jsonl CORPUS=PATH`` (repeatable): deterministic single-shot
    hit file at tier=hit_only. With CORPUS=autorag this is the large
    negative-mass source for the agent-required class; with
    CORPUS=assort_hard / x2bee_hard etc. it supplies the single-shot
    axis the xlsx source lacks (CORPUS must match the xlsx sheet name
    for the merge to key on the same namespaced qid). Format: JSONL, one
    ``{"qid": str, "hit": bool}`` object per line (an optional
    ``"query"`` key is carried through). ``hit`` must be the
    deterministic single-shot top-k id-hit for that query — not a judge
    outcome. Generate it from any harness that scores the bench per
    query, e.g. the per-query hits inside ``eval/run_all.py --quick``
    dumped one JSON object per line. ``--autorag-hits PATH`` is kept as
    shorthand for ``--hits-jsonl autorag=PATH``.

Duplicate qids across sources resolve by tier (higher wins, ties keep
loader order). When the two rows carry *complementary* axes, the loser's
measured axis is merged into the winner instead of being dropped — e.g.
an xlsx agent-axis row plus a deterministic hits row combine into one
full 2x2 label; the winner keeps its tier and the filled axis records
its own ``*_source`` / ``single_shot_basis`` provenance. Conflicting
values on the *same* axis are never merged — the winner's value stands.

CLI::

    uv run python eval/routing_gt.py \
        --finreg-jsonl finreg.jsonl finreg_multihop.jsonl \
        --hits-jsonl autorag=autorag_hits.jsonl \
        --hits-jsonl assort_hard=assort_hard_hits.jsonl \
        --out routing_gt.jsonl

``--gt-xlsx`` / ``--agent-log`` / ``--krra-graph`` default to the files
checked into the repo; pass an empty string to disable a source.

Determinism: same inputs -> byte-identical output. Rows are sorted, and
the train/heldout split is the blake2b-hash parity of the namespaced qid
(see ``assign_splits``), independent of input order and of which other
qids are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.unified import load_bench_log, load_perquery_jsonl, majority_solve

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_GT_XLSX = EVAL_DIR / "data" / "gt_datasets.xlsx"
DEFAULT_KRRA_GRAPH = EVAL_DIR / "data" / "queries" / "krra_graph.json"
DEFAULT_AGENT_LOG = EVAL_DIR / "baselines" / "agent_20260419_020747.log"

TIER_RANK = {"unmeasured": 0, "hit_only": 1, "provisional": 2, "confirmed": 3}

# Corpora that never enter the train split. assort (Easy) is a
# precision-gate-only set — signal vocabulary / threshold tuning must
# never see it (plan E1, held-out clause).
HELDOUT_ONLY_CORPORA = frozenset({"assort"})


def _warn(msg: str) -> None:
    print(f"[routing_gt] WARNING: {msg}", file=sys.stderr)


# --- Labels --------------------------------------------------------


def routing_label(single_shot_hit: bool | None, agent_solve: bool | None) -> str:
    """2x2 routing label with partial fallbacks for unmeasured axes."""
    if single_shot_hit is None and agent_solve is None:
        return "unlabeled"
    if agent_solve is None:
        return "single_shot_hit" if single_shot_hit else "single_shot_miss"
    if single_shot_hit is None:
        return "agent_solved" if agent_solve else "agent_failed"
    if single_shot_hit and agent_solve:
        return "both"
    if single_shot_hit:
        return "cheap_sufficient"
    if agent_solve:
        return "agent_required"
    return "unsolved"


def _row(
    corpus: str,
    qid: str,
    *,
    tier: str,
    source: str,
    query: str | None = None,
    qtype: str | None = None,
    single_shot_hit: bool | None = None,
    single_shot_basis: str | None = None,
    agent_solve: bool | None = None,
    label: str | None = None,
) -> dict:
    has_ss = single_shot_hit is not None
    return {
        "qid": f"{corpus}:{qid}",
        "source_qid": qid,
        "corpus": corpus,
        "query": query,
        "type": qtype,
        "single_shot_hit": single_shot_hit,
        "single_shot_basis": single_shot_basis if has_ss else None,
        "single_shot_source": source if has_ss else None,
        "agent_solve": agent_solve,
        "agent_source": source if agent_solve is not None else None,
        "label": label if label is not None else routing_label(single_shot_hit, agent_solve),
        "tier": tier,
        "source": source,
    }


# --- Source (a): finreg per-query JSONL (T1/T2 reruns) -------------


def _finreg_corpus_name(path: Path) -> str:
    stem = path.stem
    if "finreg_multihop" in stem:
        return "finreg_multihop"
    if "finreg" in stem:
        return "finreg"
    return stem


def load_finreg_source(paths: Sequence[Path]) -> list[dict]:
    """Rows from ``rag_vs_agent_answer.py --out-jsonl`` files.

    Records are deduped by (qid, arm, run) first, keeping the last
    occurrence — concatenated or resumed JSONL files must not present
    repeated records of the same run as extra runs (that would mint
    confirmed labels from single-run outcomes). Per arm, the outcome is
    the strict majority over distinct runs (``majority_solve`` with
    min_wins = n_runs // 2 + 1 — >=2/3 for the canonical 3-run protocol;
    a 2/4 tie is not a majority). tier=confirmed requires >=3 distinct
    runs on both arms; anything thinner degrades to provisional rather
    than silently claiming a confirmed label.

    ``single_shot_hit`` here is the judge-correct majority of the
    naive-RAG arm (``single_shot_basis="judge"``) — an answer-level
    outcome, not the deterministic top-k id-hit.
    """
    rows: list[dict] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            _warn(f"finreg per-query JSONL missing, source skipped: {path}")
            continue
        records = load_perquery_jsonl(path)
        if not records:
            _warn(f"finreg per-query JSONL empty, source skipped: {path}")
            continue
        corpus = _finreg_corpus_name(path)
        deduped: dict[tuple[str, str, object], dict] = {}
        for rec in records:
            deduped[(rec["qid"], rec.get("arm", ""), rec.get("run"))] = rec
        if len(deduped) < len(records):
            _warn(
                f"{path}: {len(records) - len(deduped)} duplicate (qid, arm, run) "
                "records collapsed (kept last occurrence)"
            )
        records = list(deduped.values())
        by_qid: dict[str, list[dict]] = {}
        run_ids: dict[tuple[str, str], set] = {}
        queries: dict[str, str | None] = {}
        for rec in records:
            by_qid.setdefault(rec["qid"], []).append(rec)
            run_ids.setdefault((rec["qid"], rec.get("arm", "")), set()).add(rec.get("run"))
            queries.setdefault(rec["qid"], rec.get("query"))
        for qid in sorted(by_qid):
            recs = by_qid[qid]
            n_rag = len(run_ids.get((qid, "rag"), ()))
            n_agent = len(run_ids.get((qid, "agent"), ()))
            rag = (
                majority_solve(recs, "rag", min_wins=n_rag // 2 + 1).get(qid) if n_rag else None
            )
            agent = (
                majority_solve(recs, "agent", min_wins=n_agent // 2 + 1).get(qid)
                if n_agent
                else None
            )
            rows.append(
                _row(
                    corpus,
                    qid,
                    query=queries.get(qid),
                    single_shot_hit=rag,
                    single_shot_basis="judge",
                    agent_solve=agent,
                    tier="confirmed" if min(n_rag, n_agent) >= 3 else "provisional",
                    source="finreg_jsonl",
                )
            )
    return rows


# --- Source (b): gt_datasets.xlsx + v0.17.1 agent log --------------

_QID_PAT = re.compile(r"([a-z]+)(\d+)")
_SECTION_PAT = re.compile(r"^\s+(.+?) \(agent")


def _normalize_bench_name(name: str) -> str:
    """Map a log section header to its xlsx sheet name.

    "KRRA Hard" -> "krra_hard", "assort Conv" -> "assort_conversational".
    """
    words = [("conversational" if w == "conv" else w) for w in name.strip().lower().split()]
    return "_".join(words)


def _segment_qid_stream(entries: list[tuple[str, bool]]) -> list[list[tuple[str, bool]]]:
    """Split an ordered (qid, hit) stream into bench segments.

    A new segment starts whenever the qid prefix changes or the numeric
    part stops increasing (qids restart at 001 in each bench section; gaps
    from skipped queries keep the sequence strictly increasing within one
    section).
    """
    segments: list[list[tuple[str, bool]]] = []
    cur: list[tuple[str, bool]] = []
    prev_prefix, prev_num = None, -1
    for qid, hit in entries:
        m = _QID_PAT.fullmatch(qid)
        prefix, num = (m.group(1), int(m.group(2))) if m else (qid, 0)
        if cur and (prefix != prev_prefix or num <= prev_num):
            segments.append(cur)
            cur = []
        cur.append((qid, hit))
        prev_prefix, prev_num = prefix, num
    if cur:
        segments.append(cur)
    return segments


def load_agent_log_axis(log_path: Path) -> dict[tuple[str, str], bool]:
    """Parse a single-run agent bench log into {(corpus, qid): hit}.

    ``eval.unified.load_bench_log`` yields ordered (stem, qid, hit) tuples
    but drops the bench-section information, and raw qids collide across
    sections. Sections are recovered deterministically: (1) read the
    section headers ("KRRA Hard (agent, ...)" lines) in file order,
    (2) segment the qid stream at prefix changes / numeric resets,
    (3) zip headers with segments. On any count mismatch the whole source
    is dropped (with a warning) rather than risking misattributed labels.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        _warn(f"agent bench log missing, agent axis skipped: {log_path}")
        return {}
    hits = load_bench_log(log_path)
    if not hits:
        _warn(f"agent bench log has no parseable hit lines, agent axis skipped: {log_path}")
        return {}
    headers: list[str] = []
    with log_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _SECTION_PAT.match(line)
            if m:
                headers.append(_normalize_bench_name(m.group(1)))
    segments = _segment_qid_stream([(qid, hit) for _, qid, hit in hits])
    if len(headers) != len(segments):
        _warn(
            f"agent log section/segment mismatch ({len(headers)} headers vs "
            f"{len(segments)} qid segments), agent axis skipped: {log_path}"
        )
        return {}
    axis: dict[tuple[str, str], bool] = {}
    for corpus, segment in zip(headers, segments):
        for qid, hit in segment:
            axis[(corpus, qid)] = hit
    return axis


def load_gt_datasets_source(xlsx_path: Path, agent_log: Path | None) -> list[dict]:
    """Rows from gt_datasets.xlsx, agent axis joined from the bench log.

    Every sheet containing a ``qid`` header row contributes rows; the
    ``type`` column is enumerated from the data (sheets without type
    values yield null). Rows matched in the agent log get
    tier=provisional (single run); unmatched rows stay unmeasured —
    including the assort Hard queries the v0.17.1 run could not execute.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        _warn(f"gt_datasets xlsx missing, source skipped: {xlsx_path}")
        return []
    try:
        import openpyxl
    except ImportError:
        _warn("openpyxl not installed, gt_datasets xlsx source skipped")
        return []
    agent_axis = load_agent_log_axis(agent_log) if agent_log else {}
    rows: list[dict] = []
    joined = 0
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    try:
        for ws in wb.worksheets:
            sheet_rows = list(ws.iter_rows(values_only=True))
            header_idx = next(
                (i for i, r in enumerate(sheet_rows) if r and r[0] == "qid"), None
            )
            if header_idx is None:
                continue  # Summary / non-query sheet
            col = {name: i for i, name in enumerate(sheet_rows[header_idx]) if name}
            for r in sheet_rows[header_idx + 1 :]:
                if not r or not r[0]:
                    continue
                qid = str(r[0]).strip()
                query = _cell(r, col.get("query"))
                qtype = _cell(r, col.get("type"))
                solve = agent_axis.get((ws.title, qid))
                if solve is not None:
                    joined += 1
                rows.append(
                    _row(
                        ws.title,
                        qid,
                        query=query,
                        qtype=qtype,
                        agent_solve=solve,
                        tier="provisional" if solve is not None else "unmeasured",
                        source="gt_datasets_xlsx",
                    )
                )
    finally:
        wb.close()
    # The join is by (sheet name, qid); a silent zero-join would degrade
    # the whole agent axis to unmeasured without any signal — the one
    # failure mode every other path in this module warns about.
    if agent_axis and joined == 0:
        _warn(
            f"agent log axis joined 0/{len(agent_axis)} entries — section-name "
            f"normalization no longer matches the xlsx sheet names: {xlsx_path}"
        )
    elif agent_axis and joined < len(agent_axis):
        _warn(f"agent log axis joined only {joined}/{len(agent_axis)} entries to xlsx rows")
    return rows


def _cell(row: tuple, idx: int | None) -> str | None:
    if idx is None or idx >= len(row) or row[idx] is None:
        return None
    return str(row[idx])


# --- Source (c): krra_graph design GT ------------------------------


def load_krra_graph_source(path: Path) -> list[dict]:
    """Rows from krra_graph.json (designed routing GT).

    ``current_answerable`` maps onto the single_shot_hit axis
    (``single_shot_basis="design"`` — an annotation, not a measurement);
    a query that is not answerable AND carries a ``topk_inadequacy``
    rationale is labeled agent_required by design. tier=provisional
    throughout — these are design annotations, not agent runs.
    """
    path = Path(path)
    if not path.exists():
        _warn(f"krra_graph json missing, source skipped: {path}")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for q in data.get("queries", []):
        answerable = bool(q.get("current_answerable", False))
        designed_agent_required = not answerable and bool(q.get("topk_inadequacy"))
        rows.append(
            _row(
                "krra_graph",
                str(q["qid"]),
                query=q.get("query"),
                qtype=q.get("category"),
                single_shot_hit=answerable,
                single_shot_basis="design",
                agent_solve=None,
                label="agent_required" if designed_agent_required else None,
                tier="provisional",
                source="krra_graph",
            )
        )
    return rows


# --- Source (d): deterministic single-shot hits --------------------


def load_hits_source(path: Path, *, corpus: str = "autorag") -> list[dict]:
    """Rows from an optional per-query single-shot hit JSONL.

    Format (defined here; see module docstring for how to generate it):
    one ``{"qid": str, "hit": bool}`` object per line, optional ``query``
    carried through. ``hit`` is the deterministic single-shot top-k
    id-hit (``single_shot_basis="id_hit"``). No agent axis exists in this
    source, so a row is tier=hit_only with a partial label on its own —
    when another source carries the agent axis for the same
    ``corpus:qid``, the dedupe pass merges this row's axis into it
    (which is how the xlsx benches get their missing single-shot axis).
    """
    path = Path(path)
    if not path.exists():
        _warn(f"hits JSONL missing, source skipped: {corpus}={path}")
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(
                _row(
                    corpus,
                    str(rec["qid"]),
                    query=rec.get("query"),
                    single_shot_hit=bool(rec["hit"]),
                    single_shot_basis="id_hit",
                    agent_solve=None,
                    tier="hit_only",
                    source="hits_jsonl",
                )
            )
    return rows


# --- Assembly ------------------------------------------------------


def _merge_axes(winner: dict, loser: dict) -> bool:
    """Fill the winner's unmeasured axes (in place) from the loser row.

    Complementary-axis duplicates — e.g. an xlsx agent-axis row vs a
    deterministic hits row for the same qid — are *independent*
    measurements; dropping the loser would silently discard an axis and
    make full 2x2 labels (agent_required in particular) unreachable for
    the xlsx benches. The winner keeps its tier (it covers the axis it
    measured); each filled axis records its own source/basis so per-axis
    provenance survives the merge. Conflicting values on the *same* axis
    are never merged — the winner's value stands. Returns True when an
    axis was filled (the label is then recomputed).
    """
    merged = False
    if winner.get("single_shot_hit") is None and loser.get("single_shot_hit") is not None:
        winner["single_shot_hit"] = loser["single_shot_hit"]
        winner["single_shot_basis"] = loser.get("single_shot_basis")
        winner["single_shot_source"] = loser.get("single_shot_source", loser.get("source"))
        merged = True
    if winner.get("agent_solve") is None and loser.get("agent_solve") is not None:
        winner["agent_solve"] = loser["agent_solve"]
        winner["agent_source"] = loser.get("agent_source", loser.get("source"))
        merged = True
    if merged:
        winner["label"] = routing_label(winner["single_shot_hit"], winner["agent_solve"])
    # Carry metadata the winner lacks (hits rows have no type column).
    if winner.get("query") is None and loser.get("query") is not None:
        winner["query"] = loser["query"]
    if winner.get("type") is None and loser.get("type") is not None:
        winner["type"] = loser["type"]
    return merged


def _dedupe(rows: list[dict]) -> list[dict]:
    """One row per namespaced qid; the higher measurement tier wins.

    Ties keep the earlier source (loader order: finreg, xlsx, krra_graph,
    hits). The loser is not simply dropped: any axis it measured that the
    winner did not is merged into the winner (``_merge_axes``) — this is
    what lets a tier=hit_only single-shot row complete a tier=provisional
    xlsx agent-axis row into a full 2x2 label. The known same-axis
    collision is krra_graph, present both as an xlsx sheet (unmeasured)
    and as the design json (provisional) — the design rows win.
    """
    best: dict[str, dict] = {}
    for r in rows:
        prev = best.get(r["qid"])
        if prev is None:
            best[r["qid"]] = r
            continue
        if TIER_RANK[r["tier"]] > TIER_RANK[prev["tier"]]:
            winner, loser = r, prev
            best[r["qid"]] = r
        else:
            winner, loser = prev, r
        merged = _merge_axes(winner, loser)
        action = "merged complementary axis from" if merged else "dropping"
        _warn(
            f"duplicate qid {r['qid']}: keeping {winner['source']} (tier={winner['tier']}), "
            f"{action} {loser['source']} (tier={loser['tier']})"
        )
    return list(best.values())


def _qid_hash(corpus: str, qid: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{corpus}:{qid}".encode(), digest_size=8).digest(), "big"
    )


def assign_splits(rows: list[dict]) -> None:
    """Assign a deterministic per-qid train/heldout split by hash parity.

    Each row's membership is the blake2b parity of its namespaced qid:
    even -> train, odd -> heldout. Membership depends only on the qid
    itself, so it is *stable under qid-set changes* — rebuilding the GT
    from a query subset, or a rerun that adds/removes queries (the assort
    Hard OOM retries, a --limit slice of the T2 JSONL), never moves an
    existing qid between splits. A rank-based exact-50/50 split would:
    that silently leaks previously-heldout qids into train across GT
    versions, which is a tuning-contamination path for E2. The price is
    an approximate (binomial) per-corpus balance instead of an exact one
    — acceptable at the 10-720q corpus sizes this GT carries. Corpora in
    ``HELDOUT_ONLY_CORPORA`` go entirely to heldout (tuning must never
    see them).
    """
    for r in rows:
        if r["corpus"] in HELDOUT_ONLY_CORPORA:
            r["split"] = "heldout"
        else:
            r["split"] = "train" if _qid_hash(r["corpus"], r["source_qid"]) % 2 == 0 else "heldout"


def build_routing_gt(
    out: Path | None,
    *,
    finreg_jsonl: Sequence[Path] = (),
    gt_xlsx: Path | None = None,
    agent_log: Path | None = None,
    krra_graph: Path | None = None,
    hits_jsonl: Sequence[tuple[str, Path]] = (),
) -> list[dict]:
    """Assemble the routing GT table and optionally write it as JSONL.

    ``hits_jsonl`` is a sequence of (corpus, path) pairs — one
    deterministic single-shot hit file per corpus (the corpus name must
    match the other sources' naming, e.g. the xlsx sheet name, for the
    axis merge to apply). Missing sources are skipped with a warning;
    given the same input files the output is byte-identical (sorted
    rows, hash-derived split).
    """
    rows: list[dict] = []
    rows += load_finreg_source(finreg_jsonl)
    if gt_xlsx is not None:
        rows += load_gt_datasets_source(gt_xlsx, agent_log)
    if krra_graph is not None:
        rows += load_krra_graph_source(krra_graph)
    for corpus, path in hits_jsonl:
        rows += load_hits_source(path, corpus=corpus)
    rows = _dedupe(rows)
    assign_splits(rows)
    rows.sort(key=lambda r: (r["corpus"], r["source_qid"]))
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def _print_summary(rows: list[dict]) -> None:
    print(f"routing GT: {len(rows)} rows")
    for key in ("label", "tier", "split"):
        counts = Counter(r[key] for r in rows)
        parts = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  by {key:<6} {parts}")
    corpus_counts = Counter(r["corpus"] for r in rows)
    parts = "  ".join(f"{k}={v}" for k, v in sorted(corpus_counts.items()))
    print(f"  by corpus {parts}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the v0.29 routing ground-truth table.")
    ap.add_argument(
        "--finreg-jsonl",
        nargs="*",
        default=[],
        help="per-(query,arm,run) JSONL files from rag_vs_agent_answer.py --out-jsonl",
    )
    ap.add_argument(
        "--gt-xlsx",
        default=str(DEFAULT_GT_XLSX),
        help="gt_datasets.xlsx path ('' disables the source)",
    )
    ap.add_argument(
        "--agent-log",
        default=str(DEFAULT_AGENT_LOG),
        help="single-run agent bench log for the xlsx agent axis ('' disables)",
    )
    ap.add_argument(
        "--krra-graph",
        default=str(DEFAULT_KRRA_GRAPH),
        help="krra_graph.json design-GT path ('' disables the source)",
    )
    ap.add_argument(
        "--hits-jsonl",
        action="append",
        default=[],
        metavar="CORPUS=PATH",
        help="repeatable: deterministic single-shot {'qid','hit'} JSONL for one corpus "
        "(e.g. autorag=hits.jsonl assort_hard=ah_hits.jsonl); CORPUS must match the "
        "xlsx sheet name for the axis merge to complete that bench's 2x2 label",
    )
    ap.add_argument(
        "--autorag-hits",
        default="",
        help="shorthand for --hits-jsonl autorag=PATH ('' disables)",
    )
    ap.add_argument("--out", required=True, help="output JSONL path")
    args = ap.parse_args(argv)

    hits: list[tuple[str, Path]] = []
    for spec in args.hits_jsonl:
        corpus, sep, path = spec.partition("=")
        if not sep or not corpus or not path:
            ap.error(f"--hits-jsonl expects CORPUS=PATH, got {spec!r}")
        hits.append((corpus, Path(path)))
    if args.autorag_hits:
        hits.append(("autorag", Path(args.autorag_hits)))

    rows = build_routing_gt(
        Path(args.out),
        finreg_jsonl=[Path(p) for p in args.finreg_jsonl],
        gt_xlsx=Path(args.gt_xlsx) if args.gt_xlsx else None,
        agent_log=Path(args.agent_log) if args.agent_log else None,
        krra_graph=Path(args.krra_graph) if args.krra_graph else None,
        hits_jsonl=hits,
    )
    _print_summary(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
