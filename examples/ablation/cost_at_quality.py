"""E4 keystone — mixed-pool cost-at-quality report (PLAN-v0.29 §E4).

The one report that proves or kills the v0.29 direction: over a mixed
pool (cheap-sufficient mass + agent-required mass), four arms are placed
on the (tokens/query, solve) plane:

    always-RAG     every query answered by the cheap single-shot path
    always-agent   every query answered by the multi-turn agent loop
    oracle-router  virtual router that picks the per-query best arm from
                   the routing-GT label — computed by RECOMBINING the
                   per-query JSONL records, zero execution
    ask()          graph.ask() honest routing (tier-0 signals + tier-1
                   sufficiency gate)

Design principle (plan §E4): the pool must let BOTH degenerate routers
lose visibly — always-RAG loses on solve where single-shot is ~0
(assort Hard class), always-agent loses on cost where the cheap path is
enough (finreg class). ask() passes only by separating from both.

Input reuse first — NO re-measurement of what E1 already measured:

  * always-RAG / always-agent coordinates come from the T1/T2 per-query
    JSONL files (``rag_vs_agent_answer.py --out-jsonl``), recombined via
    ``eval.unified.load_perquery_jsonl`` + ``majority_solve``.
  * oracle-router needs no execution at all: label=agent_required picks
    the agent arm's records, anything else picks the rag arm's records
    (for ``both`` the rag arm solves at lower cost; for ``unsolved``
    neither solves, so the oracle takes the cheap loss).
  * Only the ask() arm (and any corpus without an E1 JSONL) needs new
    GPU time — gated behind ``--run-live``; ``--dry-run`` prints the
    live plan without touching the LLM server.

Acceptance gates (plan §E4, evaluated on the common paired pool):

  1. quality   [3-run + McNemar]  ask solve >= max(always-RAG,
                                  always-agent) - 2 queries
  2. cost      [deterministic]    ask tokens/query <= 0.35x always-agent
  3. separation [3-run + McNemar] mixed-pool ask solve > always-RAG
                                  solve, discordant-pair p <= alpha

Gate verdicts are three-valued: PASS / FAIL / "not evaluable" — an
unmeasured ask arm or <3 ask runs reports honestly instead of passing
vacuously (single-run conclusions are banned; ±8/120 noise floor).

qid pairing contract: the live ask() arm derives qids exactly like
``rag_vs_agent_answer.py`` (``q.get("id") or f"q{i:03d}"`` over the
queries that carry a gold ``answer``), so its records pair 1:1 with the
E1 JSONL records of the same dataset.

Usage (offline — today's mode, CPU only)::

    uv run python examples/ablation/cost_at_quality.py \
      --perquery finreg_multihop=examples/ablation/diagnostics/rag_vs_agent_perquery_finreg_multihop_20260610.jsonl \
      --gt eval/results/routing_gt.jsonl \
      --out examples/ablation/diagnostics/cost_at_quality.md

Usage (live ask() arm — 5th-week GPU slot, NOT today)::

    uv run python examples/ablation/cost_at_quality.py \
      --perquery finreg_multihop=... --perquery finreg=... \
      --run-live --runs 3 \
      --live-spec finreg_multihop=eval/data/finreg_graph.sqlite,eval/data/queries/finreg_multihop.json \
      --ask-out examples/ablation/diagnostics/ask_perquery.jsonl

A previously recorded live run is reloaded with ``--ask-jsonl PATH``
(same shape, ``arm="ask"``), so the report never re-burns GPU time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.routing_gt import routing_label
from eval.unified import load_perquery_jsonl, majority_solve, mcnemar_paired
from examples.ablation.rag_vs_agent_answer import _JUDGE_SYSTEM as JUDGE_SYSTEM

# Plan §E4 acceptance-gate constants.
QUALITY_TOLERANCE_Q = 2  # ask solve >= max(arms) - 2 queries
COST_BUDGET_RATIO = 0.35  # ask tokens/query <= 0.35x always-agent
MCNEMAR_ALPHA = 0.05  # separation gate significance
MIN_RUNS = 3  # ±8/120 noise floor: no single-run verdicts

ARM_DISPLAY = {
    "rag": "always-RAG",
    "agent": "always-agent",
    "oracle": "oracle-router",
    "ask": "ask()",
}


def _warn(msg: str) -> None:
    print(f"[cost_at_quality] WARNING: {msg}", file=sys.stderr)


# --- Pool ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmCell:
    """One (query, arm) measurement: majority outcome over its runs."""

    outcome: bool
    runs: int
    mean_tokens: float


def tag_records(records: list[dict], corpus: str | None) -> list[dict]:
    """Attach a ``corpus`` field to per-query records that lack one.

    The E1 JSONL records carry no corpus (one file per dataset), so the
    CLI source name supplies it; live ask() records carry their own.
    Raw qids collide across corpora — every pool key is namespaced
    ``corpus:qid`` downstream.
    """
    out = []
    for rec in records:
        corp = rec.get("corpus") or corpus
        if not corp:
            raise ValueError(f"record without corpus and no default given: {rec.get('qid')}")
        out.append({**rec, "corpus": corp})
    return out


def summarize_cells(records: list[dict]) -> dict[tuple[str, str], ArmCell]:
    """Per-(namespaced qid, arm) cells from raw per-(query, arm, run) records.

    Records are deduped by (corpus, qid, arm, run) keeping the last
    occurrence — concatenated / resumed JSONL files must not mint extra
    runs (same discipline as ``eval.routing_gt.load_finreg_source``).
    The outcome is the strict per-arm majority over distinct runs
    (``majority_solve`` with min_wins = n_runs // 2 + 1 — >=2/3 on the
    canonical 3-run protocol; a 1-run cell is that run's verdict).
    ``mean_tokens`` is the mean of (prompt + completion) over the cell's
    deduped records — the per-query cost of routing every run of this
    query down this arm.
    """
    deduped: dict[tuple[str, str, str, object], dict] = {}
    for rec in records:
        deduped[(rec["corpus"], str(rec["qid"]), rec.get("arm", ""), rec.get("run"))] = rec
    if len(deduped) < len(records):
        _warn(
            f"{len(records) - len(deduped)} duplicate (corpus, qid, arm, run) records "
            "collapsed (kept last occurrence)"
        )
    groups: dict[tuple[str, str], list[dict]] = {}
    for (corpus, qid, arm, _run), rec in deduped.items():
        groups.setdefault((f"{corpus}:{qid}", arm), []).append(rec)
    cells: dict[tuple[str, str], ArmCell] = {}
    for (nqid, arm), recs in groups.items():
        n_runs = len({r.get("run") for r in recs})
        raw_qid = str(recs[0]["qid"])
        outcome = majority_solve(recs, arm, min_wins=n_runs // 2 + 1)[raw_qid]
        tokens = [
            int(r.get("prompt_tokens", 0) or 0) + int(r.get("completion_tokens", 0) or 0)
            for r in recs
        ]
        cells[(nqid, arm)] = ArmCell(
            outcome=outcome, runs=n_runs, mean_tokens=sum(tokens) / len(tokens)
        )
    return cells


def load_gt_labels(path: Path) -> dict[str, str]:
    """{namespaced qid: label} from an ``eval/routing_gt.py`` output JSONL."""
    labels: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                labels[row["qid"]] = row["label"]
    return labels


# --- Arms ------------------------------------------------------------


def oracle_choice(label: str | None) -> str:
    """Which arm the oracle router picks for a GT label.

    ``agent_required`` is the only label where the agent arm is the
    per-query optimum; everywhere else the cheap arm either also solves
    (``cheap_sufficient`` / ``both``) or nothing solves (``unsolved``) —
    the oracle then takes the cheaper arm. Partial / missing labels fall
    back to the cheap arm too (a routing oracle must not invent
    information the GT does not carry).
    """
    return "agent" if label == "agent_required" else "rag"


@dataclass(slots=True)
class ArmStats:
    """One arm's coordinates over the common evaluable pool."""

    name: str
    outcomes: dict[str, bool] = field(default_factory=dict)  # nqid -> solved
    tokens: dict[str, float] = field(default_factory=dict)  # nqid -> mean tokens
    min_runs: int = 0  # weakest measurement backing any outcome

    @property
    def solved(self) -> int:
        return sum(1 for v in self.outcomes.values() if v)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def solve_rate(self) -> float:
        return self.solved / self.total if self.total else 0.0

    @property
    def tokens_per_query(self) -> float:
        return sum(self.tokens.values()) / len(self.tokens) if self.tokens else 0.0


def compute_arms(
    cells: Mapping[tuple[str, str], ArmCell],
    labels: Mapping[str, str] | None = None,
) -> tuple[dict[str, ArmStats], dict]:
    """Build the 4 arms (3 when ask is unmeasured) over the common pool.

    The common pool is the qid set every compared arm actually measured:
    qids with BOTH a rag and an agent cell, intersected with the ask
    cells when an ask arm exists. Dropped qids are counted in the meta —
    never silently.

    The oracle arm is pure recombination: per qid, ``oracle_choice`` on
    the GT label (or, when no GT file was given, on the label derived
    from this pool's own rag/agent majorities via
    ``eval.routing_gt.routing_label`` — the identical rule routing_gt
    applies to the same JSONL) selects whose cell to copy.
    """
    rag_qids = {q for (q, a) in cells if a == "rag"}
    agent_qids = {q for (q, a) in cells if a == "agent"}
    ask_qids = {q for (q, a) in cells if a == "ask"}
    paired = rag_qids & agent_qids
    common = sorted(paired & ask_qids) if ask_qids else sorted(paired)

    meta: dict = {
        "pool": len(common),
        "unpaired_dropped": len((rag_qids | agent_qids) - paired),
        "ask_only_dropped": len(ask_qids - paired),
        "paired_without_ask_dropped": len(paired - ask_qids) if ask_qids else 0,
        "label_source": "routing_gt" if labels is not None else "derived_from_pool",
        "labels_missing_fallback": 0,
        "by_corpus": {},
    }

    arms: dict[str, ArmStats] = {
        "rag": ArmStats("rag"),
        "agent": ArmStats("agent"),
        "oracle": ArmStats("oracle"),
    }
    if ask_qids:
        arms["ask"] = ArmStats("ask")
    runs_seen: dict[str, list[int]] = {name: [] for name in arms}

    for nqid in common:
        rag_cell = cells[(nqid, "rag")]
        agent_cell = cells[(nqid, "agent")]
        label = labels.get(nqid) if labels is not None else None
        if label is None:
            if labels is not None:
                meta["labels_missing_fallback"] += 1
            label = routing_label(rag_cell.outcome, agent_cell.outcome)
        chosen = oracle_choice(label)
        oracle_cell = agent_cell if chosen == "agent" else rag_cell

        for name, cell in (("rag", rag_cell), ("agent", agent_cell), ("oracle", oracle_cell)):
            arms[name].outcomes[nqid] = cell.outcome
            arms[name].tokens[nqid] = cell.mean_tokens
            runs_seen[name].append(cell.runs)
        if ask_qids:
            ask_cell = cells[(nqid, "ask")]
            arms["ask"].outcomes[nqid] = ask_cell.outcome
            arms["ask"].tokens[nqid] = ask_cell.mean_tokens
            runs_seen["ask"].append(ask_cell.runs)

        corpus = nqid.split(":", 1)[0]
        meta["by_corpus"].setdefault(corpus, 0)
        meta["by_corpus"][corpus] += 1

    for name, runs in runs_seen.items():
        arms[name].min_runs = min(runs) if runs else 0
    return arms, meta


# --- Gates -----------------------------------------------------------


@dataclass(slots=True)
class GateResult:
    name: str
    passed: bool | None  # None = not evaluable (honest, not vacuous pass)
    detail: str

    @property
    def verdict(self) -> str:
        if self.passed is None:
            return "NOT EVALUABLE"
        return "PASS" if self.passed else "FAIL"


def _fmt_mcnemar(name_a: str, name_b: str, a_only: int, b_only: int, p: float) -> str:
    return f"{name_a} vs {name_b}: +{a_only}/-{b_only} discordant, p={p:.4g}"


def evaluate_gates(
    arms: Mapping[str, ArmStats],
    *,
    quality_tolerance: int = QUALITY_TOLERANCE_Q,
    cost_budget: float = COST_BUDGET_RATIO,
    alpha: float = MCNEMAR_ALPHA,
    min_runs: int = MIN_RUNS,
) -> list[GateResult]:
    """Three-valued verdicts for the plan §E4 acceptance gates.

    All comparisons run on the common pool ``compute_arms`` built (every
    arm holds the identical qid set). Quality and separation are
    multi-run gates: with fewer than ``min_runs`` ask runs they report
    NOT EVALUABLE — a single run sits inside the ±8/120 noise floor and
    must not produce a verdict. Cost is deterministic given the records.
    """
    ask = arms.get("ask")
    rag, agent = arms["rag"], arms["agent"]
    if ask is None or not ask.total:
        detail = "ask arm unmeasured — run the live arm (--run-live) or load --ask-jsonl"
        return [
            GateResult("quality", None, detail),
            GateResult("cost", None, detail),
            GateResult("separation", None, detail),
        ]

    qids = sorted(ask.outcomes)
    ask_o = [ask.outcomes[q] for q in qids]
    rag_o = [rag.outcomes[q] for q in qids]
    agent_o = [agent.outcomes[q] for q in qids]
    vs_rag = mcnemar_paired(ask_o, rag_o)
    vs_agent = mcnemar_paired(ask_o, agent_o)
    runs_ok = ask.min_runs >= min_runs
    runs_note = f"ask min runs/query = {ask.min_runs} (gate needs >= {min_runs})"

    gates: list[GateResult] = []

    # 1. quality: ask solve >= max(always-RAG, always-agent) - tolerance
    best = max(rag.solved, agent.solved)
    quality_detail = (
        f"ask {ask.solved}/{ask.total} vs max(always-RAG {rag.solved}, "
        f"always-agent {agent.solved}) - {quality_tolerance} = {best - quality_tolerance}; "
        f"McNemar {_fmt_mcnemar('ask', 'always-RAG', *vs_rag)}; "
        f"{_fmt_mcnemar('ask', 'always-agent', *vs_agent)}"
    )
    if not runs_ok:
        gates.append(GateResult("quality", None, f"{runs_note}; {quality_detail}"))
    else:
        gates.append(GateResult("quality", ask.solved >= best - quality_tolerance, quality_detail))

    # 2. cost: ask tokens/query <= budget x always-agent (deterministic)
    if agent.tokens_per_query <= 0:
        gates.append(
            GateResult("cost", None, "always-agent arm carries no token counts — cannot ratio")
        )
    else:
        ratio = ask.tokens_per_query / agent.tokens_per_query
        gates.append(
            GateResult(
                "cost",
                ratio <= cost_budget,
                f"ask {ask.tokens_per_query:,.0f} tok/q = {ratio:.2f}x always-agent "
                f"{agent.tokens_per_query:,.0f} tok/q (budget <= {cost_budget:.2f}x)",
            )
        )

    # 3. separation: mixed-pool ask solve > always-RAG, discordant-proven
    a_only, b_only, p = vs_rag
    sep_detail = (
        f"ask {ask.solved} vs always-RAG {rag.solved}; "
        f"{_fmt_mcnemar('ask', 'always-RAG', a_only, b_only, p)} (alpha={alpha})"
    )
    if not runs_ok:
        gates.append(GateResult("separation", None, f"{runs_note}; {sep_detail}"))
    else:
        gates.append(GateResult("separation", ask.solved > rag.solved and p <= alpha, sep_detail))
    return gates


# --- Report ----------------------------------------------------------


def render_markdown(
    arms: Mapping[str, ArmStats], gates: Sequence[GateResult], meta: Mapping
) -> str:
    agent_tpq = arms["agent"].tokens_per_query
    lines = [
        "# cost-at-quality — mixed pool (v0.29 E4 keystone)",
        "",
        f"common paired pool: {meta['pool']} queries "
        f"({', '.join(f'{c}={n}' for c, n in sorted(meta['by_corpus'].items()))})",
        f"oracle labels: {meta['label_source']}"
        + (
            f" ({meta['labels_missing_fallback']} qids missing from GT, derived from pool)"
            if meta.get("labels_missing_fallback")
            else ""
        ),
    ]
    dropped = [
        f"{meta[k]} {k.replace('_', ' ')}"
        for k in ("unpaired_dropped", "ask_only_dropped", "paired_without_ask_dropped")
        if meta.get(k)
    ]
    if dropped:
        lines.append(f"dropped from the common pool: {'; '.join(dropped)}")
    lines += [
        "",
        "| arm | solve | rate | tokens/query | cost vs always-agent | min runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("rag", "agent", "oracle", "ask"):
        arm = arms.get(key)
        if arm is None:
            lines.append(f"| {ARM_DISPLAY[key]} | — | — | — | — | — (unmeasured) |")
            continue
        ratio = f"{arm.tokens_per_query / agent_tpq:.2f}x" if agent_tpq > 0 else "n/a"
        lines.append(
            f"| {ARM_DISPLAY[key]} | {arm.solved}/{arm.total} | {arm.solve_rate:.3f} "
            f"| {arm.tokens_per_query:,.0f} | {ratio} | {arm.min_runs} |"
        )
    lines += ["", "## acceptance gates (plan §E4)", ""]
    for g in gates:
        lines.append(f"- **{g.name}**: {g.verdict} — {g.detail}")
    ask = arms.get("ask")
    oracle = arms["oracle"]
    lines.append("")
    if ask is not None and ask.total:
        otpq = oracle.tokens_per_query
        cost_gap = f"{ask.tokens_per_query / otpq:.2f}x oracle cost" if otpq > 0 else "n/a"
        lines.append(
            f"oracle gap (disclosed, not a gate): ask {ask.solved}/{ask.total} vs "
            f"oracle {oracle.solved}/{oracle.total} solve ({ask.solved - oracle.solved:+d}q); "
            f"{cost_gap}"
        )
    else:
        lines.append(
            f"oracle ceiling: {oracle.solved}/{oracle.total} solve at "
            f"{oracle.tokens_per_query:,.0f} tok/q — the routing headroom ask() competes for"
        )
    lines += [
        "",
        "claim scope: per plan §E4 — only the agent-required-class delta is claimed "
        "(noise floor multiples); domain generalization (finreg+assort, 2 domains) "
        "is NOT established by this report.",
        "",
    ]
    return "\n".join(lines)


# --- Live ask() arm (--run-live; 5th-week GPU slot) -------------------


def load_gold_queries(path: Path, answer_field: str = "answer") -> list[dict]:
    """Queries that carry a gold reference, with the E1-pairing qid.

    Filter and qid derivation replicate ``rag_vs_agent_answer.py``
    exactly (``q.get("id") or f"q{i:03d}"`` over the gold-bearing
    queries, positional) so live ask() records pair 1:1 with that
    harness's JSONL for the same dataset. The derived qid is stored as
    ``_qid``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = [q for q in data.get("queries", []) if q.get("query") and q.get(answer_field)]
    return [{**q, "_qid": q.get("id") or f"q{qi:03d}"} for qi, q in enumerate(queries)]


async def run_ask_arm(
    specs: Sequence[tuple[str, Path, Path]],
    *,
    client: object,
    model: str,
    runs: int = 3,
    k: int = 10,
    max_turns: int = 5,
    mode: str = "auto",
    answer_field: str = "answer",
    embedder: object | None = None,
    concurrency: int = 4,
    out_jsonl: Path | None = None,
) -> list[dict]:
    """Measure the ask() arm: per-(query, run) records in the E1 JSONL shape.

    Per query, ``graph.ask()`` runs with honest routing and the answer
    is graded by the same LLM judge as the E1 arms (``_JUDGE_SYSTEM``
    from rag_vs_agent_answer.py, temperature 0). Records carry the
    AskResult route / escalated flags and the ask-internal token usage
    (synthesis + tier-1 judge + escalated agent loop); the external
    correctness judge is measurement cost and is excluded — the same
    accounting as the E1 arms. The JSONL is rewritten after every run so
    a multi-hour run that dies keeps its completed runs.
    """
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.graph import SynapticGraph

    records: list[dict] = []

    def _flush() -> None:
        if out_jsonl is None:
            return
        out = Path(out_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def judge(answer: str, gold: str, query: str) -> bool:
        if not answer.strip():
            return False
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"QUESTION:\n{query}\n\nREFERENCE:\n{gold}\n\nCANDIDATE:\n{answer}",
                },
            ],
            max_tokens=8,
            temperature=0.0,
        )
        out = resp.choices[0].message.content or ""
        return out.strip().upper().startswith("YES")

    for corpus, graph_path, queries_path in specs:
        queries = load_gold_queries(queries_path, answer_field)
        if not queries:
            _warn(
                f"{corpus}: no queries with a gold {answer_field!r} field in "
                f"{queries_path} — spec skipped (the judge needs a reference)"
            )
            continue
        backend = SqliteGraphBackend(str(graph_path))
        await backend.connect()
        graph = SynapticGraph(backend, embedder=embedder)
        sem = asyncio.Semaphore(concurrency)
        try:
            for r in range(runs):

                async def one(
                    q: dict,
                    r: int = r,
                    corpus: str = corpus,
                    graph: object = graph,
                    sem: asyncio.Semaphore = sem,
                ) -> dict:
                    async with sem:
                        t0 = time.time()
                        res = await graph.ask(
                            q["query"],
                            llm_client=client,
                            model=model,
                            mode=mode,
                            k=k,
                            max_turns=max_turns,
                        )
                        ok = await judge(res.answer, q[answer_field], q["query"])
                        return {
                            "qid": q["_qid"],
                            "corpus": corpus,
                            "query": q["query"],
                            "arm": "ask",
                            "run": r + 1,
                            "judge_correct": ok,
                            "empty": not res.answer.strip(),
                            "elapsed_s": round(time.time() - t0, 1),
                            "answer": res.answer,
                            "route": res.route,
                            "escalated": res.escalated,
                            "route_reasons": list(res.route_reasons),
                            "prompt_tokens": res.prompt_tokens,
                            "completion_tokens": res.completion_tokens,
                        }

                run_recs = await asyncio.gather(*[one(q) for q in queries])
                records.extend(run_recs)
                _flush()
                solved = sum(1 for rec in run_recs if rec["judge_correct"])
                escalated = sum(1 for rec in run_recs if rec["escalated"])
                print(
                    f"ask() {corpus} run {r + 1}: {solved}/{len(queries)} solved, "
                    f"{escalated} escalated",
                    flush=True,
                )
        finally:
            await backend.close()
    return records


def print_dry_run_plan(specs: Sequence[tuple[str, Path, Path]], runs: int) -> None:
    """The live plan, zero LLM/GPU contact — what --run-live WOULD do."""
    print(f"DRY RUN — live ask() arm plan ({runs} runs, no LLM calls made):")
    total = 0
    for corpus, graph_path, queries_path in specs:
        try:
            n_gold = len(load_gold_queries(queries_path))
        except FileNotFoundError:
            print(f"  {corpus}: queries file missing: {queries_path}")
            continue
        status = "" if graph_path.exists() else "  [graph sqlite MISSING]"
        print(
            f"  {corpus}: {n_gold} gold-answer queries x {runs} runs (graph={graph_path}{status})"
        )
        if n_gold == 0:
            print(
                f"    -> would be SKIPPED: no gold answers in {queries_path} "
                "(the judge needs a reference)"
            )
        total += n_gold * runs
    print(
        f"  total: {total} ask() calls + {total} judge calls minimum "
        "(escalations add agent-loop turns)"
    )


# --- CLI ---------------------------------------------------------------


def _parse_kv(spec: str, flag: str) -> tuple[str, str]:
    key, sep, value = spec.partition("=")
    if not sep or not key or not value:
        raise SystemExit(f"{flag} expects NAME=VALUE, got {spec!r}")
    return key, value


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v0.29 E4 mixed-pool cost-at-quality report")
    ap.add_argument(
        "--perquery",
        action="append",
        default=[],
        metavar="CORPUS=PATH",
        help="repeatable: E1 per-query JSONL (rag_vs_agent_answer.py --out-jsonl) for one corpus",
    )
    ap.add_argument(
        "--gt",
        default="",
        help="routing GT JSONL (eval/routing_gt.py output) for the oracle arm; "
        "without it labels derive from the pool's own rag/agent majorities (same rule)",
    )
    ap.add_argument(
        "--ask-jsonl",
        default="",
        help="previously recorded ask() arm per-query JSONL (from --run-live --ask-out)",
    )
    ap.add_argument("--out", default="", help="write the markdown report here (also printed)")
    # --- live arm (GPU) — plan 5th-week slot, never implicit ---------
    ap.add_argument("--run-live", action="store_true", help="measure the ask() arm live (GPU)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="with --run-live: print the live plan and exit without any LLM call",
    )
    ap.add_argument(
        "--live-spec",
        action="append",
        default=[],
        metavar="CORPUS=GRAPH.sqlite,QUERIES.json",
        help="repeatable: one live ask() dataset (graph sqlite + gold-answer queries json)",
    )
    ap.add_argument("--runs", type=int, default=3, help="live ask() runs (gates need >=3)")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--llm-base-url", default="http://localhost:8012/v1")
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--embed-url", default="http://localhost:8013/v1")
    ap.add_argument("--embed-model", default="Qwen3-Embedding-4B")
    ap.add_argument(
        "--ask-out",
        default="",
        help="where --run-live writes the ask() per-query JSONL (flushed every run)",
    )
    args = ap.parse_args(argv)

    if args.dry_run and not args.run_live:
        ap.error("--dry-run only applies to --run-live")

    live_specs: list[tuple[str, Path, Path]] = []
    for spec in args.live_spec:
        corpus, value = _parse_kv(spec, "--live-spec")
        graph_str, sep, queries_str = value.partition(",")
        if not sep or not graph_str or not queries_str:
            raise SystemExit(f"--live-spec expects CORPUS=GRAPH.sqlite,QUERIES.json, got {spec!r}")
        live_specs.append((corpus, Path(graph_str), Path(queries_str)))

    if args.run_live and args.dry_run:
        if not live_specs:
            ap.error("--run-live needs at least one --live-spec")
        print_dry_run_plan(live_specs, args.runs)
        return 0

    if not args.perquery:
        ap.error("at least one --perquery CORPUS=PATH is required (E1 JSONL reuse — no rerun)")

    records: list[dict] = []
    for spec in args.perquery:
        corpus, path_str = _parse_kv(spec, "--perquery")
        path = Path(path_str)
        if not path.exists():
            _warn(f"per-query JSONL missing, source skipped: {corpus}={path}")
            continue
        records += tag_records(load_perquery_jsonl(path), corpus)

    if args.ask_jsonl:
        path = Path(args.ask_jsonl)
        if path.exists():
            records += tag_records(load_perquery_jsonl(path), None)
        else:
            _warn(f"ask per-query JSONL missing, skipped: {path}")

    if args.run_live:
        if not live_specs:
            ap.error("--run-live needs at least one --live-spec")
        from openai import AsyncOpenAI

        from synaptic.extensions.embedder import OpenAIEmbeddingProvider

        client = AsyncOpenAI(base_url=args.llm_base_url, api_key="ignored")
        embedder = OpenAIEmbeddingProvider(api_base=args.embed_url, model=args.embed_model)
        ask_records = asyncio.run(
            run_ask_arm(
                live_specs,
                client=client,
                model=args.model,
                runs=args.runs,
                k=args.k,
                max_turns=args.max_turns,
                embedder=embedder,
                concurrency=args.concurrency,
                out_jsonl=Path(args.ask_out) if args.ask_out else None,
            )
        )
        records += ask_records

    if not records:
        _warn("no records loaded — nothing to report")
        return 1

    cells = summarize_cells(records)
    labels = load_gt_labels(Path(args.gt)) if args.gt else None
    arms, meta = compute_arms(cells, labels)
    gates = evaluate_gates(arms)
    report = render_markdown(arms, gates, meta)
    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
