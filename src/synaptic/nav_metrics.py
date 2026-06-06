"""Agent-navigation-efficiency metrics — how FEW hops / LLM-calls an agent
needs to *reach the gold evidence* in a corpus.

The measurement the literature lacks. GR-score / greedy-routing measure how a
GRAPH routes between nodes; RAG benchmarks measure final-answer accuracy. Neither
measures the thing that actually matters for an agent over a large corpus: *given
a turn/call budget, did the agent's traversal surface the gold evidence, and in
how few steps?* This module owns that measurement so every structure change can
be judged by it (fewer hops to evidence at equal success = a better structure).

Pure functions over the per-turn trace produced by
``run_agent_loop(record_trace=True)``. No LLM, no embedder, corpus-agnostic.
The caller supplies already-normalized gold ids (matching is the eval harness's
job, kept out of the metric so it stays general).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def navigation_efficiency(
    trace: Sequence[dict],
    gold_ids: Iterable[str],
    *,
    call_budget: int | None = None,
) -> dict:
    """Per-query navigation efficiency from one agent trace.

    Args:
        trace: ``AgentSearchResult.trace`` — ordered per-turn snapshots, each
            ``{"turn": int, "tool_calls": int, "found_ids": set[str]}``.
        gold_ids: the query's gold evidence ids (already normalized to match the
            ids the agent's tools emit).
        call_budget: if given, also report whether the *first* gold id was
            reached within this many cumulative tool calls (success@budget).

    Returns a dict with:
        found_any / found_all: did the trace ever surface any / all gold.
        hops_to_first_gold / hops_to_all_gold: 1-based turn at which the first /
            all gold first appeared (None if never).
        calls_to_first_gold / calls_to_all_gold: cumulative tool_calls at that
            point (None if never) — the real cost-to-evidence.
        total_turns: trace length.
        success_within_budget: only when call_budget is given.
    """
    gold = set(gold_ids)
    out: dict = {
        "found_any": False,
        "found_all": False,
        "hops_to_first_gold": None,
        "hops_to_all_gold": None,
        "calls_to_first_gold": None,
        "calls_to_all_gold": None,
        "total_turns": len(trace),
    }
    if not gold:
        if call_budget is not None:
            out["success_within_budget"] = False
        return out

    for snap in trace:
        fids = set(snap.get("found_ids") or ())
        turn = snap.get("turn")
        calls = snap.get("tool_calls")
        if out["hops_to_first_gold"] is None and (fids & gold):
            out["found_any"] = True
            out["hops_to_first_gold"] = turn
            out["calls_to_first_gold"] = calls
        if out["hops_to_all_gold"] is None and gold <= fids:
            out["found_all"] = True
            out["hops_to_all_gold"] = turn
            out["calls_to_all_gold"] = calls
            break  # all gold reached — nothing earlier can beat this

    if call_budget is not None:
        c = out["calls_to_first_gold"]
        out["success_within_budget"] = c is not None and c <= call_budget
    return out


def aggregate(per_query: Sequence[dict]) -> dict:
    """Aggregate per-query ``navigation_efficiency`` dicts into corpus-level
    means. Hop/call means are taken over *reached* queries only (None skipped),
    reported alongside the reach rate so a high mean-hops isn't hidden by a low
    reach rate.
    """
    n = len(per_query) or 1

    def _mean(key: str) -> float | None:
        vals = [q[key] for q in per_query if q.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    agg = {
        "n_queries": len(per_query),
        "reach_any_rate": sum(1 for q in per_query if q.get("found_any")) / n,
        "reach_all_rate": sum(1 for q in per_query if q.get("found_all")) / n,
        "mean_hops_to_first_gold": _mean("hops_to_first_gold"),
        "mean_hops_to_all_gold": _mean("hops_to_all_gold"),
        "mean_calls_to_first_gold": _mean("calls_to_first_gold"),
        "mean_calls_to_all_gold": _mean("calls_to_all_gold"),
    }
    if per_query and "success_within_budget" in per_query[0]:
        agg["success_within_budget_rate"] = (
            sum(1 for q in per_query if q.get("success_within_budget")) / n
        )
    return agg
