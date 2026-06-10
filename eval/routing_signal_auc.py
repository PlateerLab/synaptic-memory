"""Tier-0 routing-signal AUC harness (v0.29 E2 — go/no-go gate).

Measures whether DETERMINISTIC, zero-LLM signals predict the
``agent_required`` routing label *before* any router code is written
(docs/PLAN-v0.29-ask-routing.md §E2). Input is the routing GT JSONL
that ``eval/routing_gt.py`` (T3) emits — one record per query with
qid / label / tier / split / corpus — plus an optional per-query
retrieval-results JSONL for the score-shape signals (s2/s3).

Measurement discipline, enforced by code structure:

  - AUC / recall verdicts use the *confirmed* label tier only;
    provisional / hit-only tiers appear in a reference section with
    no verdict weight. Hit-only rows carry partial labels (agent axis
    unmeasured): 'single_shot_hit' maps onto the negative class in the
    reference section only; 'single_shot_miss' stays unlabelled.
  - Threshold tuning reads split=train only (``tune_thresholds``
    raises on anything else); the go/no-go numbers come from
    split=heldout only (``evaluate_heldout`` raises on anything else).
  - Every signal is a pure ``(query, SignalContext) -> float`` — no
    LLM calls, no I/O. AUC is rank-based Mann-Whitney, pure stdlib
    (no scipy/sklearn — the zero-dependency rule covers eval too).

Go/no-go (single deterministic run, held-out split):

    agent-required recall >= 0.90 (confirmed labels only)
    AND escalation budgets: AutoRAG <= 15 %, assort Easy <= 20 %.

On NO-GO, E3 ships the conservative router (promote high-confidence
positives only; defer the rest to the tier-1 sufficiency gate) — that
decision is this harness's output.

CLI:
    uv run python eval/routing_signal_auc.py gt.jsonl \\
        [--retrieval retrieval.jsonl] [--out report.md]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.unified import RecallType, classify_query

# --- Routing GT records ---------------------------------------------

# 2x2 label space from eval/routing_gt.py: single_shot_hit x agent_solve.
# ``agent_required`` is the positive class the router must catch.
# ``cheap_sufficient`` and ``both`` are negatives (the cheap path
# suffices — for ``both`` it wins on cost). ``unsolved`` (neither arm
# solves) carries no routing information and is excluded from AUC /
# recall; it still counts toward escalation rates, which are
# prediction-only.
POSITIVE_LABEL = "agent_required"
NEGATIVE_LABELS = frozenset({"cheap_sufficient", "both"})

# T3's hit-only rows (agent axis unmeasured, e.g. AutoRAG) carry the
# partial label 'single_shot_hit' / 'single_shot_miss'. A hit means the
# cheap path succeeded, so for the REFERENCE section only it maps onto
# the negative class — AutoRAG's role as the mass cheap-sufficient
# negative source. A miss stays unlabelled (could be agent_required or
# unsolved). The mapping is keyed on tier == 'hit_only', which the
# verdict tiers never include, so it cannot touch the go/no-go numbers.
HIT_ONLY_NEGATIVE_LABEL = "single_shot_hit"

VERDICT_TIERS = ("confirmed",)
REFERENCE_TIERS = ("provisional", "hit_only")

# Corpora whose graph carries typed ``_table_name`` rows — fallback for
# s1's table gate when the GT record doesn't say ``has_table_nodes``.
TABLE_CORPUS_PREFIXES = ("assort", "x2bee")


def _warn(msg: str) -> None:
    print(f"[routing_signal_auc] WARNING: {msg}", file=sys.stderr)


def _norm_token(raw: object) -> str:
    """Lowercase and strip everything non-alphanumeric so 'hit-only',
    'hit_only' and 'Hit Only' all normalise to the same token."""
    return "".join(ch for ch in str(raw or "").lower() if ch.isalnum())


def _norm_tier(raw: object) -> str:
    tok = _norm_token(raw)
    return {
        "confirmed": "confirmed",
        "provisional": "provisional",
        "hitonly": "hit_only",
        "unmeasured": "unmeasured",
        "": "unmeasured",
    }.get(tok, tok)


def _norm_split(raw: object) -> str:
    return _norm_token(raw)  # "held-out" / "held_out" -> "heldout"


def _norm_label(raw: object) -> str:
    """Same leniency for labels as for tier/split: 'agent-required' and
    'Agent Required' must count as positives, not silently fall out of
    both AUC classes. Unknown labels keep their normalised token (they
    are excluded from AUC either way)."""
    tok = _norm_token(raw)
    return {
        "agentrequired": "agent_required",
        "cheapsufficient": "cheap_sufficient",
        "both": "both",
        "unsolved": "unsolved",
        # T3 partial labels (one axis unmeasured)
        "singleshothit": "single_shot_hit",
        "singleshotmiss": "single_shot_miss",
        "agentsolved": "agent_solved",
        "agentfailed": "agent_failed",
        "unlabeled": "unlabeled",
    }.get(tok, tok)


@dataclass(slots=True)
class GTRecord:
    """One routing-GT row. Loader is lenient: only qid+label are
    required; tier defaults to unmeasured, split/corpus to empty."""

    qid: str
    query: str = ""
    label: str = ""
    tier: str = "unmeasured"
    split: str = ""
    corpus: str = ""
    has_table_nodes: bool | None = None


def load_routing_gt(path: Path) -> list[GTRecord]:
    """Load the T3 routing GT JSONL. Tolerates blank lines, non-JSON
    lines, unknown keys and alternate tier/split/label spellings; skips
    records that lack qid or label (they can't be scored). Duplicate
    qids keep the LAST record and warn — contexts and signal values key
    by qid (last wins), so keeping duplicates in the list would double-
    count them in recall and escalation rates."""
    out: list[GTRecord] = []
    index_of: dict[str, int] = {}
    dups: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid") or "").strip()
        label = str(row.get("label") or "").strip()
        if not qid or not label:
            continue
        rec = GTRecord(
            qid=qid,
            query=str(row.get("query") or ""),
            label=_norm_label(label),
            tier=_norm_tier(row.get("tier")),
            split=_norm_split(row.get("split")),
            corpus=str(row.get("corpus") or "").strip().lower(),
            has_table_nodes=(
                bool(row["has_table_nodes"]) if "has_table_nodes" in row else None
            ),
        )
        if qid in index_of:
            dups.append(qid)
            out[index_of[qid]] = rec
        else:
            index_of[qid] = len(out)
            out.append(rec)
    if dups:
        _warn(
            f"{len(dups)} duplicate qid(s) in {path} — kept last occurrence "
            f"(e.g. {dups[:3]})"
        )
    return out


# --- Signal context -------------------------------------------------


@dataclass(slots=True)
class RetrievalInfo:
    """Per-query single-shot retrieval observation (top-k pass).

    ``scores``: top-k retrieval/rerank scores, best first not required.
    ``hit``: whether the pass hit GT (optional — when absent, s2_no_hit
    falls back to "returned zero results").
    ``has_table_row``: whether any top-k row is a ``_table_name``
    property row (structured-corpus contact).
    """

    scores: list[float] = field(default_factory=list)
    hit: bool | None = None
    has_table_row: bool | None = None


def load_retrieval_results(path: Path) -> dict[str, RetrievalInfo]:
    """Load the optional per-query retrieval JSONL → qid → RetrievalInfo.

    Lenient keys: ``scores`` | ``rerank_scores`` | ``top_scores``;
    ``hit``; ``has_table_row`` | ``table_row``. Malformed records
    (non-numeric scores) are skipped and warned, like non-JSON lines —
    one bad line must not abort the whole s2/s3 pass.
    """
    out: dict[str, RetrievalInfo] = {}
    n_skipped = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid") or "").strip()
        if not qid:
            continue
        raw_scores = row.get("scores") or row.get("rerank_scores") or row.get("top_scores") or []
        try:
            scores = [float(s) for s in raw_scores]
        except (ValueError, TypeError):
            n_skipped += 1
            continue
        hit = row.get("hit")
        table = row.get("has_table_row", row.get("table_row"))
        out[qid] = RetrievalInfo(
            scores=scores,
            hit=None if hit is None else bool(hit),
            has_table_row=None if table is None else bool(table),
        )
    if n_skipped:
        _warn(f"{n_skipped} retrieval record(s) in {path} skipped (non-numeric scores)")
    return out


@dataclass(slots=True)
class SignalContext:
    """Everything a signal may consult besides the query text."""

    has_table_nodes: bool = False
    retrieval: RetrievalInfo | None = None


def build_contexts(
    records: Sequence[GTRecord],
    retrieval: Mapping[str, RetrievalInfo] | None = None,
) -> dict[str, SignalContext]:
    """qid → SignalContext. ``has_table_nodes`` honours the GT record's
    own flag when present, else falls back to the corpus-prefix map."""
    retrieval = retrieval or {}
    out: dict[str, SignalContext] = {}
    for r in records:
        has_tables = (
            r.has_table_nodes
            if r.has_table_nodes is not None
            else r.corpus.startswith(TABLE_CORPUS_PREFIXES)
        )
        out[r.qid] = SignalContext(has_table_nodes=has_tables, retrieval=retrieval.get(r.qid))
    return out


# --- Signals ---------------------------------------------------------
#
# Protocol: (query, SignalContext) -> float. Higher = more
# agent-required. NaN = signal not computable for this query (dropped
# from AUC, never fires in the combo).

Signal = Callable[[str, SignalContext], float]

# Aggregation / filter lexicons — the structured-operation families
# assort Hard is built from (eval/unified.py's classify_query covers
# the enumeration / multi-hop families but has no aggregation/filter
# axis, so they live here).
_AGGREGATION_HINTS = (
    "평균",
    "합계",
    "총합",
    "총 ",
    "개수",
    "몇 개",
    "몇 건",
    "몇 명",
    "최대",
    "최소",
    "가장 많",
    "가장 적",
    "count",
    "sum of",
    "average",
    "how many",
    "total ",
    "maximum",
    "minimum",
)
_FILTER_HINTS = (
    "이상",
    "이하",
    "초과",
    "미만",
    "보다 큰",
    "보다 작은",
    "조건",
    "해당하는",
    "greater than",
    "less than",
    "at least",
    "more than",
    "fewer than",
    "between ",
    "filter",
)


def s1_structured_lexicon(query: str, ctx: SignalContext) -> float:
    """Structured-operation vocabulary x typed-node availability.

    Reuses ``eval/unified.py:classify_query`` for the enumeration and
    multi-hop families, plus aggregation / filter lexicons. Gated on
    ``ctx.has_table_nodes``: without typed table rows the structured
    tools don't exist, so vocabulary alone must not escalate. Graded
    0..1 by matched-family count for AUC rank resolution. Embedder-
    independent — the first-line signal.
    """
    if not query or not ctx.has_table_nodes:
        return 0.0
    dim = classify_query(query)
    q = query.lower()
    families = 0
    if dim.enumeration or dim.recall_type == RecallType.ENUMERATION.value:
        families += 1
    if dim.recall_type == RecallType.MULTI_HOP.value or dim.hop_count >= 2:
        families += 1
    if any(tok in q for tok in _AGGREGATION_HINTS):
        families += 1
    if any(tok in q for tok in _FILTER_HINTS):
        families += 1
    return families / 4.0


def s2_no_hit(query: str, ctx: SignalContext) -> float:
    """1.0 when the single-shot pass produced no hit. Uses the explicit
    ``hit`` field when the retrieval JSONL carries one; otherwise falls
    back to "the pass returned zero results". NaN without a retrieval
    pass."""
    r = ctx.retrieval
    if r is None:
        return math.nan
    if r.hit is not None:
        return 0.0 if r.hit else 1.0
    return 1.0 if not r.scores else 0.0


def _population_std(xs: Sequence[float]) -> float:
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))


def s2_score_flatness(query: str, ctx: SignalContext) -> float:
    """Negated std of the top-k scores — a flat score distribution means
    retrieval couldn't discriminate, so flatter ranks as more
    agent-required. Auxiliary: score-shape thresholds are embedder-
    dependent (rerank-deadzone precedent, project_v028)."""
    r = ctx.retrieval
    if r is None or len(r.scores) < 2:
        return math.nan
    return -_population_std(r.scores)


def s2_margin_deficit(query: str, ctx: SignalContext) -> float:
    """Negated top1−top2 margin — a small margin means an ambiguous
    winner. Auxiliary, embedder-dependent like s2_score_flatness."""
    r = ctx.retrieval
    if r is None or len(r.scores) < 2:
        return math.nan
    top = sorted(r.scores, reverse=True)
    return -(top[0] - top[1])


def s3_table_row_in_topk(query: str, ctx: SignalContext) -> float:
    """1.0 when a ``_table_name`` property row appears in the top-k —
    the query made contact with structured rows, where single-shot is
    weakest (assort Hard 0.0 single-shot). NaN without a retrieval
    pass or when the pass didn't report row kinds."""
    r = ctx.retrieval
    if r is None or r.has_table_row is None:
        return math.nan
    return 1.0 if r.has_table_row else 0.0


SIGNALS: dict[str, Signal] = {
    "s1_structured_lexicon": s1_structured_lexicon,
    "s2_no_hit": s2_no_hit,
    "s2_score_flatness": s2_score_flatness,
    "s2_margin_deficit": s2_margin_deficit,
    "s3_table_row_in_topk": s3_table_row_in_topk,
}


def compute_signal_values(
    records: Sequence[GTRecord],
    contexts: Mapping[str, SignalContext],
    signals: Mapping[str, Signal] | None = None,
) -> dict[str, dict[str, float]]:
    """signal name → (qid → value) over every record."""
    signals = signals or SIGNALS
    return {
        name: {r.qid: fn(r.query, contexts[r.qid]) for r in records}
        for name, fn in signals.items()
    }


# --- AUC (rank-based Mann-Whitney, stdlib) ---------------------------


def rank_auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """AUC via the Mann-Whitney U statistic with average ranks for ties.

    Equals P(score_pos > score_neg) + 0.5 * P(tie). Pure stdlib.
    Raises ValueError when either class is empty (AUC undefined).
    """
    if not positives or not negatives:
        raise ValueError("rank_auc needs at least one score in each class")
    combined = sorted(
        [(v, True) for v in positives] + [(v, False) for v in negatives],
        key=lambda t: t[0],
    )
    n = len(combined)
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average rank of the tie block
        rank_sum_pos += avg_rank * sum(1 for k in range(i, j + 1) if combined[k][1])
        i = j + 1
    n_pos, n_neg = len(positives), len(negatives)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


@dataclass(slots=True)
class SignalEval:
    """AUC of one signal over one (tier subset, split subset)."""

    name: str
    auc: float | None = None
    n_pos: int = 0
    n_neg: int = 0
    n_nan: int = 0
    note: str = ""


def _labelled_pairs(
    records: Sequence[GTRecord],
    values: Mapping[str, float],
    *,
    tiers: Sequence[str],
) -> tuple[list[tuple[float, bool]], int]:
    """(value, is_positive) pairs for the tier subset; NaN values are
    dropped and counted. ``unsolved`` records never enter. Hit-only
    rows map their partial 'single_shot_hit' label onto the negative
    class (see ``HIT_ONLY_NEGATIVE_LABEL``)."""
    pairs: list[tuple[float, bool]] = []
    n_nan = 0
    for r in records:
        if r.tier not in tiers:
            continue
        if r.label == POSITIVE_LABEL:
            is_pos = True
        elif r.label in NEGATIVE_LABELS:
            is_pos = False
        elif r.tier == "hit_only" and r.label == HIT_ONLY_NEGATIVE_LABEL:
            is_pos = False
        else:
            continue
        v = values.get(r.qid, math.nan)
        if math.isnan(v):
            n_nan += 1
            continue
        pairs.append((v, is_pos))
    return pairs, n_nan


def evaluate_signal(
    records: Sequence[GTRecord],
    values: Mapping[str, float],
    name: str,
    *,
    tiers: Sequence[str] = VERDICT_TIERS,
) -> SignalEval:
    """AUC of one signal on the labelled subset of ``records``."""
    pairs, n_nan = _labelled_pairs(records, values, tiers=tiers)
    pos = [v for v, y in pairs if y]
    neg = [v for v, y in pairs if not y]
    ev = SignalEval(name=name, n_pos=len(pos), n_neg=len(neg), n_nan=n_nan)
    if not pairs:
        ev.note = "requires retrieval pass" if n_nan else "no labelled records"
    elif not pos or not neg:
        ev.note = "single-class subset"
    else:
        ev.auc = rank_auc(pos, neg)
        if n_nan:
            ev.note = f"{n_nan} NaN dropped"
    return ev


# --- recall @ precision ----------------------------------------------


def precision_recall_curve(
    pairs: Sequence[tuple[float, bool]],
) -> list[tuple[float, float, float]]:
    """(threshold, precision, recall) at every distinct observed value,
    predicting positive when value >= threshold, descending. Callers
    must pre-drop NaN pairs (``_labelled_pairs`` does)."""
    n_pos_total = sum(1 for _, y in pairs if y)
    pts: list[tuple[float, float, float]] = []
    for thr in sorted({v for v, _ in pairs}, reverse=True):
        tp = sum(1 for v, y in pairs if y and v >= thr)
        fp = sum(1 for v, y in pairs if not y and v >= thr)
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / n_pos_total if n_pos_total else 0.0
        pts.append((thr, precision, recall))
    return pts


def recall_at_precision(
    pairs: Sequence[tuple[float, bool]], min_precision: float
) -> float | None:
    """Best recall among thresholds whose precision >= ``min_precision``;
    None when no threshold reaches it."""
    best: float | None = None
    for _, p, r in precision_recall_curve(pairs):
        if p >= min_precision and (best is None or r > best):
            best = r
    return best


# --- OR-combo + escalation budgets -----------------------------------

# Plan E2 escalation budgets — cheap-sufficient precision gates on both
# a document corpus (AutoRAG) and a structured corpus (assort Easy) so
# a degenerate "_table_name => always escalate" router can't pass.
DEFAULT_BUDGETS: dict[str, float] = {"autorag": 0.15, "assort_easy": 0.20}


def budget_key(corpus: str) -> str | None:
    """Map a GT corpus name onto an escalation-budget bucket. 'assort'
    (the Easy query-file stem) and 'assort_easy' hit the assort-Easy
    budget; 'assort_hard' / 'assort_conversational' deliberately do
    not — they are agent-gain territory, not cheap traffic."""
    c = corpus.strip().lower()
    if c.startswith("autorag"):
        return "autorag"
    if c in ("assort", "assort_easy"):
        return "assort_easy"
    return None


@dataclass(slots=True)
class ComboThresholds:
    """OR-combination router: escalate iff ANY signal value >= its
    threshold. ``math.inf`` disables a signal; NaN never fires. No
    weights — plan E2 mandates the simplest combination so the verdict
    stays interpretable."""

    thresholds: dict[str, float] = field(default_factory=dict)

    def fires(self, values: Mapping[str, Mapping[str, float]], qid: str) -> bool:
        for name, thr in self.thresholds.items():
            if math.isinf(thr):
                continue
            v = values.get(name, {}).get(qid, math.nan)
            if not math.isnan(v) and v >= thr:
                return True
        return False


def _confirmed_recall(
    records: Sequence[GTRecord],
    values: Mapping[str, Mapping[str, float]],
    combo: ComboThresholds,
) -> tuple[float, int, int]:
    """(recall, n_escalated_pos, n_pos) over confirmed agent_required."""
    pos = [r for r in records if r.tier == "confirmed" and r.label == POSITIVE_LABEL]
    n_fire = sum(1 for r in pos if combo.fires(values, r.qid))
    return (n_fire / len(pos) if pos else 0.0, n_fire, len(pos))


def _escalation_rates(
    records: Sequence[GTRecord],
    values: Mapping[str, Mapping[str, float]],
    combo: ComboThresholds,
    budgets: Mapping[str, float],
) -> dict[str, tuple[float, int, int]]:
    """budget bucket → (rate, n_fired, n). Label-free: every record of
    a budget corpus counts regardless of tier — the budget protects
    cheap traffic, not labelled traffic."""
    counts: dict[str, list[int]] = {k: [0, 0] for k in budgets}
    for r in records:
        k = budget_key(r.corpus)
        if k is None or k not in counts:
            continue
        counts[k][1] += 1
        if combo.fires(values, r.qid):
            counts[k][0] += 1
    return {k: ((fired / n if n else 0.0), fired, n) for k, (fired, n) in counts.items()}


def _within_budgets(
    rates: Mapping[str, tuple[float, int, int]], budgets: Mapping[str, float]
) -> bool:
    return all(rate <= budgets[k] for k, (rate, _f, n) in rates.items() if n)


def _candidate_thresholds(vals: Sequence[float], max_candidates: int = 24) -> list[float]:
    """Distinct non-NaN values, quantile-thinned for continuous signals
    so the sweep stays bounded and deterministic."""
    uniq = sorted({v for v in vals if not math.isnan(v)})
    if len(uniq) <= max_candidates:
        return uniq
    step = (len(uniq) - 1) / (max_candidates - 1)
    return sorted({uniq[round(i * step)] for i in range(max_candidates)})


def tune_thresholds(
    train_records: Sequence[GTRecord],
    values: Mapping[str, Mapping[str, float]],
    *,
    budgets: Mapping[str, float] | None = None,
) -> ComboThresholds:
    """Greedy OR-threshold sweep on the TRAIN split only.

    Raises ValueError if any record is not split=train — held-out data
    must never leak into threshold selection (plan E2). Greedy: start
    all-disabled; repeatedly apply the single threshold lowering that
    adds the most confirmed recall while keeping every escalation
    budget satisfied on train (ties broken by fewer escalations, then
    by SIGNALS order — s1 is the embedder-independent first line —
    then by the higher, more conservative threshold); stop when no
    feasible move adds recall. OR is monotone in each threshold, so
    only lowerings need consideration. Deterministic.
    """
    budgets = dict(DEFAULT_BUDGETS if budgets is None else budgets)
    leaked = [r.qid for r in train_records if _norm_split(r.split) != "train"]
    if leaked:
        raise ValueError(
            f"tune_thresholds accepts split=train records only; "
            f"got {len(leaked)} others (e.g. {leaked[:3]})"
        )
    names = list(values.keys())
    candidates = {
        name: _candidate_thresholds([values[name].get(r.qid, math.nan) for r in train_records])
        for name in names
    }
    combo = ComboThresholds(thresholds=dict.fromkeys(names, math.inf))
    cur_recall, _, _ = _confirmed_recall(train_records, values, combo)
    while True:
        # (recall, -total_escalations, -signal_index, threshold) —
        # max() picks best recall, then fewest escalations, then the
        # earliest signal in SIGNALS order, then the most conservative
        # (highest) threshold.
        best: tuple[float, int, int, float] | None = None
        best_name = ""
        for idx, name in enumerate(names):
            for thr in candidates[name]:
                if thr >= combo.thresholds[name]:
                    continue
                trial = ComboThresholds(thresholds={**combo.thresholds, name: thr})
                rates = _escalation_rates(train_records, values, trial, budgets)
                if not _within_budgets(rates, budgets):
                    continue
                recall, _, _ = _confirmed_recall(train_records, values, trial)
                if recall <= cur_recall:
                    continue
                esc_total = sum(f for _r, f, _n in rates.values())
                cand = (recall, -esc_total, -idx, thr)
                if best is None or cand > best:
                    best = cand
                    best_name = name
        if best is None:
            return combo
        cur_recall, thr = best[0], best[3]
        combo.thresholds[best_name] = thr


# --- Held-out verdict -------------------------------------------------


@dataclass(slots=True)
class HeldoutVerdict:
    """Frozen-combo evaluation on the held-out split — the only numbers
    the E2 go/no-go may quote."""

    recall: float
    n_pos: int
    n_escalated_pos: int
    precision: float | None
    escalation: dict[str, tuple[float, int, int]]
    go: bool
    min_recall: float
    budgets: dict[str, float]
    notes: list[str] = field(default_factory=list)


def evaluate_heldout(
    heldout_records: Sequence[GTRecord],
    values: Mapping[str, Mapping[str, float]],
    combo: ComboThresholds,
    *,
    budgets: Mapping[str, float] | None = None,
    min_recall: float = 0.90,
) -> HeldoutVerdict:
    """Apply a FROZEN combo to the held-out split.

    Raises ValueError if any record is not split=heldout — verdict
    numbers must come from data the tuner never saw (plan E2). GO
    additionally requires every escalation-budget corpus to be present
    in held-out: an unverifiable budget fails closed (NO-GO), never
    open.
    """
    budgets = dict(DEFAULT_BUDGETS if budgets is None else budgets)
    leaked = [r.qid for r in heldout_records if _norm_split(r.split) != "heldout"]
    if leaked:
        raise ValueError(
            f"evaluate_heldout accepts split=heldout records only; "
            f"got {len(leaked)} others (e.g. {leaked[:3]})"
        )
    recall, n_fire_pos, n_pos = _confirmed_recall(heldout_records, values, combo)
    n_fire_neg = sum(
        1
        for r in heldout_records
        if r.tier == "confirmed" and r.label in NEGATIVE_LABELS and combo.fires(values, r.qid)
    )
    fired = n_fire_pos + n_fire_neg
    precision = n_fire_pos / fired if fired else None
    rates = _escalation_rates(heldout_records, values, combo, budgets)
    notes: list[str] = []
    if n_pos == 0:
        notes.append("no confirmed agent_required records in held-out — recall gate vacuous")
    # GO requires every budget bucket to be POPULATED on held-out: the
    # plan gate is "recall AND AutoRAG AND assort Easy", and an absent
    # corpus must fail closed, not skip its budget. (The train-side
    # tuner keeps the n==0 skip — assort is heldout-only by T3 design,
    # so the tuner legitimately never sees that bucket.)
    missing_budgets = [k for k, (_rate, _f, n) in rates.items() if n == 0]
    for k in missing_budgets:
        notes.append(f"budget corpus '{k}' absent from held-out — budget unverifiable, forces NO-GO")
    go = (
        n_pos > 0
        and recall >= min_recall
        and _within_budgets(rates, budgets)
        and not missing_budgets
    )
    return HeldoutVerdict(
        recall=recall,
        n_pos=n_pos,
        n_escalated_pos=n_fire_pos,
        precision=precision,
        escalation=rates,
        go=go,
        min_recall=min_recall,
        budgets=dict(budgets),
        notes=notes,
    )


# --- Markdown report --------------------------------------------------

_PRECISION_LEVELS = (0.90, 0.80, 0.70, 0.50)


def _fmt(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _signal_table(evals: Sequence[SignalEval]) -> list[str]:
    lines = [
        "| signal | AUC | n_pos | n_neg | NaN | note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for ev in evals:
        lines.append(
            f"| {ev.name} | {_fmt(ev.auc)} | {ev.n_pos} | {ev.n_neg} | "
            f"{ev.n_nan} | {ev.note} |"
        )
    return lines


def generate_report(
    records: Sequence[GTRecord],
    values: Mapping[str, Mapping[str, float]],
    *,
    budgets: Mapping[str, float] | None = None,
    min_recall: float = 0.90,
    gt_path: Path | None = None,
    retrieval_path: Path | None = None,
) -> str:
    """Full E2 markdown report: per-signal AUC (confirmed, held-out),
    recall@precision, reference tiers, train-tuned OR-combo, held-out
    go/no-go verdict."""
    budgets = dict(DEFAULT_BUDGETS if budgets is None else budgets)
    train = [r for r in records if _norm_split(r.split) == "train"]
    heldout = [r for r in records if _norm_split(r.split) == "heldout"]
    n_other = len(records) - len(train) - len(heldout)

    tier_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for r in records:
        tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1
        label_counts[r.label] = label_counts.get(r.label, 0) + 1

    lines: list[str] = []
    lines.append("# Tier-0 routing-signal AUC report (v0.29 E2)")
    lines.append("")
    lines.append(f"- GT: `{gt_path}` — {len(records)} records "
                 f"(train {len(train)} / heldout {len(heldout)} / other-split {n_other})")
    if retrieval_path:
        lines.append(f"- Retrieval results: `{retrieval_path}`")
    else:
        lines.append("- Retrieval results: absent — s2/s3 signals require a retrieval pass")
    lines.append(f"- Tiers: {dict(sorted(tier_counts.items()))}")
    lines.append(f"- Labels: {dict(sorted(label_counts.items()))} "
                 f"(positive={POSITIVE_LABEL}; negative={sorted(NEGATIVE_LABELS)}; "
                 f"unsolved excluded from AUC/recall)")
    lines.append("")

    lines.append("## Signal AUC — confirmed tier, held-out split (verdict basis)")
    lines.append("")
    verdict_evals = [
        evaluate_signal(heldout, values[name], name, tiers=VERDICT_TIERS) for name in values
    ]
    lines.extend(_signal_table(verdict_evals))
    lines.append("")

    lines.append("## recall @ precision — confirmed tier, held-out split")
    lines.append("")
    header = "| signal | " + " | ".join(f"R@P≥{p:.2f}" for p in _PRECISION_LEVELS) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(_PRECISION_LEVELS))
    for name in values:
        pairs, _ = _labelled_pairs(heldout, values[name], tiers=VERDICT_TIERS)
        cells = " | ".join(_fmt(recall_at_precision(pairs, p)) for p in _PRECISION_LEVELS)
        lines.append(f"| {name} | {cells} |")
    lines.append("")

    lines.append("## Reference — provisional / hit-only tiers, held-out split (no verdict weight)")
    lines.append("")
    reference_evals = [
        evaluate_signal(heldout, values[name], name, tiers=REFERENCE_TIERS) for name in values
    ]
    lines.extend(_signal_table(reference_evals))
    lines.append("")

    lines.append("## Tuned OR-combo (thresholds selected on train split only)")
    lines.append("")
    if train:
        combo = tune_thresholds(train, values, budgets=budgets)
        lines.append("| signal | threshold |")
        lines.append("|---|---:|")
        for name, thr in combo.thresholds.items():
            lines.append(f"| {name} | {'disabled' if math.isinf(thr) else f'{thr:.4f}'} |")
        t_recall, t_fire, t_pos = _confirmed_recall(train, values, combo)
        lines.append("")
        lines.append(f"- train recall (confirmed): {t_recall:.3f} ({t_fire}/{t_pos})")
        for k, (rate, f, n) in _escalation_rates(train, values, combo, budgets).items():
            lines.append(f"- train escalation {k}: {_fmt(rate)} ({f}/{n})")
    else:
        combo = ComboThresholds(thresholds=dict.fromkeys(values, math.inf))
        lines.append("(no train records — combo disabled)")
    lines.append("")

    lines.append("## Held-out verdict")
    lines.append("")
    verdict = evaluate_heldout(
        heldout, values, combo, budgets=budgets, min_recall=min_recall
    )
    lines.append(
        f"- agent-required recall (confirmed): {verdict.recall:.3f} "
        f"({verdict.n_escalated_pos}/{verdict.n_pos}) — gate ≥ {min_recall:.2f}"
    )
    lines.append(f"- precision on confirmed labels: {_fmt(verdict.precision)}")
    for k, (rate, f, n) in verdict.escalation.items():
        lines.append(
            f"- escalation {k}: {rate:.1%} ({f}/{n}) — budget ≤ {budgets[k]:.0%}"
        )
    for note in verdict.notes:
        lines.append(f"- note: {note}")
    lines.append("")
    if verdict.go:
        lines.append("**VERDICT: GO** — tier-0 signals clear the E2 gate on held-out data.")
    else:
        lines.append(
            "**VERDICT: NO-GO** — adopt conservative routing for E3: promote "
            "high-confidence positives only; defer the rest to the tier-1 "
            "sufficiency gate."
        )
    lines.append("")
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("gt", type=Path, help="routing GT JSONL (eval/routing_gt.py output)")
    p.add_argument(
        "--retrieval",
        type=Path,
        default=None,
        help="optional per-query retrieval JSONL (qid, scores, hit, has_table_row) for s2/s3",
    )
    p.add_argument(
        "--out", type=Path, default=None, help="write the markdown report here (default: stdout)"
    )
    p.add_argument(
        "--min-recall",
        type=float,
        default=0.90,
        help="agent-required recall gate on confirmed held-out labels (default 0.90)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records = load_routing_gt(args.gt)
    if not records:
        print(f"!! no usable records in {args.gt}", file=sys.stderr)
        return 2
    retrieval = load_retrieval_results(args.retrieval) if args.retrieval else {}
    contexts = build_contexts(records, retrieval)
    values = compute_signal_values(records, contexts)
    report = generate_report(
        records,
        values,
        min_recall=args.min_recall,
        gt_path=args.gt,
        retrieval_path=args.retrieval,
    )
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"written: {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
