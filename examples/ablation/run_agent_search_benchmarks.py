"""Agent tool-surface retrieval benchmark runner.

This complements ``run_tier1_benchmarks.py``. The tier-1 runner measures
single-call ``graph.search()`` quality; this script measures deterministic
agent-facing retrieval surfaces that keep session state across tool calls:

- ``graph_search``: baseline ``SynapticGraph.search``.
- ``agent_search``: intent-aware ``SynapticGraph.agent_search``.
- ``deep_search``: compound agent tool, search -> expand -> optional document read.
- ``scripted_session``: deterministic multi-turn session, ``deep_search`` followed
  by one or more paginated ``search`` calls with ``exclude_seen=True``.

The scripted mode is LLM-free on purpose. It tests whether the tool/session
layer can accumulate useful evidence over consecutive exploration turns without
mixing in model-planning noise or API cost.

For the full LLM-planned loop where the agent changes follow-up queries based on
earlier evidence, use ``run_agent_loop_benchmarks.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synaptic.agent_tools import search_tool
from synaptic.agent_tools_v2 import deep_search_tool
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.graph import SynapticGraph
from synaptic.search_session import SearchSession

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "diagnostics"
DEFAULT_MSMARCO_PATH = REPO_ROOT / "tests" / "benchmark" / "data" / "msmarco_passage.json"
TOP_K = 10


@dataclass(slots=True)
class ModeReport:
    mode: str
    n_docs: int
    n_queries: int
    mrr_at_10: float
    recall_at_5: float
    recall_at_10: float
    hit_at_10: int
    reach_at_all: int
    search_sec: float
    mean_tool_calls: float
    mean_returned_docs: float


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved[:TOP_K]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for doc_id in retrieved[:k] if doc_id in relevant) / len(relevant)


def _dedupe_extend(target: list[str], values: list[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _doc_ids_from_nodes(nodes: list[object]) -> list[str]:
    doc_ids: list[str] = []
    for item in nodes:
        node = getattr(item, "node", item)
        doc_id = (getattr(node, "properties", {}) or {}).get("doc_id", "")
        if doc_id and str(doc_id) not in doc_ids:
            doc_ids.append(str(doc_id))
    return doc_ids


def _doc_ids_from_tool_evidence(evidence: list[dict[str, Any]]) -> list[str]:
    doc_ids: list[str] = []
    for item in evidence:
        props = item.get("properties") or {}
        doc_id = item.get("document_id") or props.get("doc_id") or ""
        if doc_id and str(doc_id) not in doc_ids:
            doc_ids.append(str(doc_id))
    return doc_ids


async def _run_graph_search(
    graph: SynapticGraph,
    query: str,
    *,
    result_limit: int,
    fts_seed_limit: int | None,
) -> tuple[list[str], int]:
    result = await graph.search(query, limit=result_limit, fts_seed_limit=fts_seed_limit)
    return _doc_ids_from_nodes(result.nodes), 1


async def _run_agent_search(
    graph: SynapticGraph,
    query: str,
    *,
    result_limit: int,
    intent: str,
) -> tuple[list[str], int]:
    result = await graph.agent_search(query, intent=intent, limit=result_limit)
    return _doc_ids_from_nodes(result.nodes), 1


async def _run_deep_search(
    backend: object,
    query: str,
    *,
    tool_limit: int,
    read_top_k: int,
) -> tuple[list[str], int]:
    session = SearchSession(budget_tool_calls=max(6, 3 + read_top_k))
    result = await deep_search_tool(
        backend,
        session,
        query,
        limit=tool_limit,
        read_top_k=read_top_k,
    )
    evidence = result.data.get("evidence", []) if result.ok else []
    return _doc_ids_from_tool_evidence(evidence), session.tool_calls_used


async def _run_scripted_session(
    backend: object,
    query: str,
    *,
    tool_limit: int,
    read_top_k: int,
    turns: int,
) -> tuple[list[str], int]:
    session = SearchSession(budget_tool_calls=max(10, turns * 3 + read_top_k + 2))
    doc_ids: list[str] = []

    first = await deep_search_tool(
        backend,
        session,
        query,
        limit=tool_limit,
        read_top_k=read_top_k,
    )
    if first.ok:
        _dedupe_extend(doc_ids, _doc_ids_from_tool_evidence(first.data.get("evidence", [])))

    for _ in range(max(0, turns - 1)):
        page = await search_tool(
            backend,
            session,
            query,
            limit=tool_limit,
            exclude_seen=True,
        )
        if page.ok:
            _dedupe_extend(doc_ids, _doc_ids_from_tool_evidence(page.data.get("evidence", [])))

    return doc_ids, session.tool_calls_used


async def _run_mode(
    *,
    mode: str,
    backend: SqliteGraphBackend,
    graph: SynapticGraph,
    query_items: list[tuple[str, str]],
    qrels: dict[str, Any],
    n_docs: int,
    result_limit: int,
    tool_limit: int,
    read_top_k: int,
    scripted_turns: int,
    intent: str,
    fts_seed_limit: int | None,
) -> ModeReport:
    mrr_total = 0.0
    recall5_total = 0.0
    recall10_total = 0.0
    hit10 = 0
    reach_all = 0
    search_sec = 0.0
    tool_calls_total = 0
    returned_docs_total = 0

    for qid, query in query_items:
        rel = qrels.get(qid, {})
        relevant = set(map(str, rel.keys())) if isinstance(rel, dict) else set(map(str, rel))
        if not relevant:
            continue

        start = time.perf_counter()
        if mode == "graph_search":
            retrieved, calls = await _run_graph_search(
                graph,
                str(query),
                result_limit=result_limit,
                fts_seed_limit=fts_seed_limit,
            )
        elif mode == "agent_search":
            retrieved, calls = await _run_agent_search(
                graph,
                str(query),
                result_limit=result_limit,
                intent=intent,
            )
        elif mode == "deep_search":
            retrieved, calls = await _run_deep_search(
                backend,
                str(query),
                tool_limit=tool_limit,
                read_top_k=read_top_k,
            )
        elif mode == "scripted_session":
            retrieved, calls = await _run_scripted_session(
                backend,
                str(query),
                tool_limit=tool_limit,
                read_top_k=read_top_k,
                turns=scripted_turns,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")
        search_sec += time.perf_counter() - start
        tool_calls_total += calls
        returned_docs_total += len(retrieved)

        rr = _reciprocal_rank(retrieved, relevant)
        mrr_total += rr
        recall5_total += _recall_at_k(retrieved, relevant, 5)
        recall10_total += _recall_at_k(retrieved, relevant, TOP_K)
        if rr > 0:
            hit10 += 1
        if any(doc_id in relevant for doc_id in retrieved):
            reach_all += 1

    n = max(len(query_items), 1)
    return ModeReport(
        mode=mode,
        n_docs=n_docs,
        n_queries=len(query_items),
        mrr_at_10=mrr_total / n,
        recall_at_5=recall5_total / n,
        recall_at_10=recall10_total / n,
        hit_at_10=hit10,
        reach_at_all=reach_all,
        search_sec=search_sec,
        mean_tool_calls=tool_calls_total / n,
        mean_returned_docs=returned_docs_total / n,
    )


def _emit_markdown(
    reports: list[ModeReport],
    *,
    dataset_path: Path,
    sqlite_db_path: Path,
    subset: int,
    corpus_limit: int,
    result_limit: int,
    tool_limit: int,
    read_top_k: int,
    scripted_turns: int,
    intent: str,
    fts_seed_limit: int | None,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"agent_search_{stamp}.md"
    lines = [
        "# Agent Tool-Surface Retrieval Benchmark — Synaptic",
        "",
        f"- Run at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Dataset path: {_display_path(dataset_path)}",
        f"- SQLite DB path: {_display_path(sqlite_db_path)}",
        f"- Subset: {subset}",
        f"- Corpus limit: {corpus_limit}",
        f"- Result limit: {result_limit}",
        f"- Tool limit: {tool_limit}",
        f"- Deep-search read_top_k: {read_top_k}",
        f"- Scripted session turns: {scripted_turns}",
        f"- Agent search intent: {intent}",
        f"- FTS seed limit: {fts_seed_limit if fts_seed_limit else 'default'}",
        "- SQLite FTS AND-first threshold: "
        f"{os.environ.get('SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD', '').strip() or '0'}",
        "- SQLite FTS lexical rerank pool: "
        f"{os.environ.get('SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL', '').strip() or '0'}",
        "",
        "This is LLM-free. It measures the deterministic tool/session surfaces, "
        "not the full agent loop that changes follow-up queries based on earlier "
        "evidence. Reach@All counts queries where at least one relevant document "
        "appeared anywhere in the returned evidence for that mode. Retrieval Ops/Q "
        "counts the single `graph.search`/`agent_search` operation or the "
        "SearchSession tool calls used by agent-tool modes.",
        "",
        "| Mode | Docs | Queries | MRR@10 | R@5 | R@10 | Hit@10 | Reach@All | Search | Retrieval Ops/Q | Docs/Q |",
        "|------|-----:|--------:|-------:|----:|-----:|-------:|----------:|-------:|----------------:|-------:|",
    ]
    for report in reports:
        lines.append(
            f"| {report.mode} | {report.n_docs} | {report.n_queries} | "
            f"{report.mrr_at_10:.3f} | {report.recall_at_5:.3f} | "
            f"{report.recall_at_10:.3f} | {report.hit_at_10}/{report.n_queries} | "
            f"{report.reach_at_all}/{report.n_queries} | {report.search_sec:.1f}s | "
            f"{report.mean_tool_calls:.2f} | {report.mean_returned_docs:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco-path", type=Path, default=DEFAULT_MSMARCO_PATH)
    parser.add_argument("--sqlite-db-path", type=Path, required=True)
    parser.add_argument("--subset", type=int, default=20)
    parser.add_argument("--corpus-limit", type=int, default=0)
    parser.add_argument(
        "--modes",
        default="graph_search,deep_search,scripted_session",
        help="Comma-separated modes: graph_search, agent_search, deep_search, scripted_session",
    )
    parser.add_argument("--result-limit", type=int, default=20)
    parser.add_argument("--tool-limit", type=int, default=10)
    parser.add_argument("--read-top-k", type=int, default=0)
    parser.add_argument("--scripted-turns", type=int, default=2)
    parser.add_argument("--intent", default="context_explore")
    parser.add_argument("--fts-seed-limit", type=int, default=0)
    args = parser.parse_args(argv)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    valid_modes = {"graph_search", "agent_search", "deep_search", "scripted_session"}
    unknown = sorted(set(modes) - valid_modes)
    if unknown:
        raise SystemExit(f"unknown mode(s): {', '.join(unknown)}")
    if args.subset <= 0:
        raise SystemExit("--subset must be positive")
    if args.result_limit <= 0:
        raise SystemExit("--result-limit must be positive")
    if args.tool_limit <= 0:
        raise SystemExit("--tool-limit must be positive")
    if args.read_top_k < 0:
        raise SystemExit("--read-top-k cannot be negative")
    if args.scripted_turns <= 0:
        raise SystemExit("--scripted-turns must be positive")
    if not args.msmarco_path.exists():
        raise SystemExit(f"{args.msmarco_path} does not exist")
    if not args.sqlite_db_path.exists():
        raise SystemExit(f"{args.sqlite_db_path} does not exist")

    data = json.loads(args.msmarco_path.read_text(encoding="utf-8"))
    query_items = list(data["queries"].items())[: args.subset]
    qrels = data["qrels"]
    n_docs = args.corpus_limit or int(data.get("corpus_size") or 0)

    backend = SqliteGraphBackend(str(args.sqlite_db_path))
    await backend.connect()
    graph = SynapticGraph(backend)
    reports: list[ModeReport] = []
    try:
        for mode in modes:
            report = await _run_mode(
                mode=mode,
                backend=backend,
                graph=graph,
                query_items=query_items,
                qrels=qrels,
                n_docs=n_docs,
                result_limit=args.result_limit,
                tool_limit=args.tool_limit,
                read_top_k=args.read_top_k,
                scripted_turns=args.scripted_turns,
                intent=args.intent,
                fts_seed_limit=args.fts_seed_limit or None,
            )
            reports.append(report)
            print(
                f"{report.mode:18} {report.n_docs:8d} {report.n_queries:5d} "
                f"{report.mrr_at_10:7.3f} {report.recall_at_5:7.3f} "
                f"{report.recall_at_10:7.3f} {report.hit_at_10:4d}/{report.n_queries:<4d} "
                f"reach {report.reach_at_all:4d}/{report.n_queries:<4d} "
                f"{report.search_sec:7.1f}s calls/q {report.mean_tool_calls:.2f}",
                flush=True,
            )
    finally:
        await backend.close()

    report_path = _emit_markdown(
        reports,
        dataset_path=args.msmarco_path,
        sqlite_db_path=args.sqlite_db_path,
        subset=args.subset,
        corpus_limit=n_docs,
        result_limit=args.result_limit,
        tool_limit=args.tool_limit,
        read_top_k=args.read_top_k,
        scripted_turns=args.scripted_turns,
        intent=args.intent,
        fts_seed_limit=args.fts_seed_limit or None,
    )
    print(f"\nMarkdown report -> {_display_path(report_path)}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
