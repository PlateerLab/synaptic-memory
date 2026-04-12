"""Kuzu vs Memory backend parity check on the enterprise scenario.

Runs the 15-query enterprise benchmark against both backends, using the
exact same ingestion pipeline for each, and prints a side-by-side report.

Usage:
    uv run python eval/scripts/kuzu_parity.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import time

# Make sure repo-local modules are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.activity import ActivityTracker
from synaptic.backends.kuzu import KuzuBackend
from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind, NodeKind
from synaptic.protocols import StorageBackend
from tests.benchmark.metrics import BenchmarkResult

DATA_FILE = REPO_ROOT / "tests" / "benchmark" / "data" / "enterprise_scenario.json"
K = 5

_KIND_MAP: dict[str, NodeKind] = {
    "CONCEPT": NodeKind.CONCEPT,
    "ENTITY": NodeKind.ENTITY,
    "LESSON": NodeKind.LESSON,
    "DECISION": NodeKind.DECISION,
    "RULE": NodeKind.RULE,
    "ARTIFACT": NodeKind.ARTIFACT,
}

_EDGE_MAP: dict[str, EdgeKind] = {
    "RELATED": EdgeKind.RELATED,
    "DEPENDS_ON": EdgeKind.DEPENDS_ON,
    "LEARNED_FROM": EdgeKind.LEARNED_FROM,
    "CAUSED": EdgeKind.CAUSED,
    "PRODUCED": EdgeKind.PRODUCED,
}


@dataclass(slots=True)
class RunReport:
    backend: str
    summary: dict[str, float]
    per_query: list[dict[str, object]]


def _load_scenario() -> dict:
    with open(DATA_FILE) as f:
        return json.load(f)


async def _ingest_scenario(graph: SynapticGraph, scenario: dict) -> dict[str, str]:
    """Reproduces the enterprise_graph fixture ingestion."""
    tracker = ActivityTracker(graph)
    id_map: dict[str, str] = {}

    for doc in scenario["knowledge_sources"]:
        kind = _KIND_MAP.get(doc["kind"], NodeKind.CONCEPT)
        node = await graph.add(
            title=doc["title"],
            content=doc["content"],
            kind=kind,
            tags=doc.get("tags", []),
            source=doc.get("source", ""),
            properties=doc.get("properties"),
        )
        id_map[doc["id"]] = node.id

    for link in scenario["knowledge_links"]:
        src = id_map.get(link["source"])
        tgt = id_map.get(link["target"])
        if src and tgt:
            edge_kind = _EDGE_MAP.get(link["kind"], EdgeKind.RELATED)
            await graph.link(src, tgt, kind=edge_kind)

    for session_data in scenario["agent_sessions"]:
        session = await tracker.start_session(
            agent_id=session_data["agent_id"],
            description=session_data["description"],
        )
        for tc in session_data["tool_calls"]:
            await tracker.log_tool_call(
                session.id,
                tool_name=tc["tool"],
                parameters=tc.get("params"),
                result=tc.get("result", ""),
                success=tc.get("success", True),
                duration_ms=tc.get("duration_ms", 0.0),
            )
        for dec_data in session_data.get("decisions", []):
            decision = await tracker.record_decision(
                session.id,
                title=dec_data["title"],
                rationale=dec_data["rationale"],
                alternatives=dec_data.get("alternatives"),
            )
            if "outcome" in dec_data:
                out = dec_data["outcome"]
                await tracker.record_outcome(
                    decision.id,
                    title=out["title"],
                    content=out["content"],
                    success=out["success"],
                )
        accessed = session_data.get("knowledge_accessed", [])
        accessed_ids = [id_map[a] for a in accessed if a in id_map]
        if accessed_ids:
            await graph.reinforce(accessed_ids, success=True)
        await tracker.end_session(session.id)

    for session_data in scenario["agent_sessions"]:
        for dec_data in session_data.get("decisions", []):
            if "outcome" in dec_data and not dec_data["outcome"]["success"]:
                accessed = session_data.get("knowledge_accessed", [])
                accessed_ids = [id_map[a] for a in accessed if a in id_map]
                if accessed_ids:
                    await graph.reinforce(accessed_ids, success=False)

    return id_map


async def _run_queries(
    graph: SynapticGraph, scenario: dict, id_map: dict[str, str]
) -> BenchmarkResult:
    bench = BenchmarkResult()
    for q in scenario["evaluation_queries"]:
        relevant_ids = {id_map[rid] for rid in q["relevant_ids"] if rid in id_map}

        start = time()
        if q.get("intent", "auto") != "auto":
            result = await graph.agent_search(q["query"], intent=q["intent"], limit=K * 2)
        else:
            result = await graph.search(q["query"], limit=K * 2)
        elapsed = (time() - start) * 1000

        retrieved = [n.node.id for n in result.nodes]
        bench.add(
            query_id=q["id"],
            query=q["query"],
            retrieved=retrieved,
            relevant=relevant_ids,
            k=K,
            description=q.get("description", ""),
            search_time_ms=elapsed,
        )
    return bench


async def _run_backend(backend: StorageBackend, backend_name: str, scenario: dict) -> RunReport:
    await backend.connect()
    try:
        graph = SynapticGraph(backend)
        id_map = await _ingest_scenario(graph, scenario)
        bench = await _run_queries(graph, scenario, id_map)
        return RunReport(
            backend=backend_name,
            summary=bench.summary(),
            per_query=bench.queries,
        )
    finally:
        await backend.close()


def _print_side_by_side(memory: RunReport, kuzu: RunReport) -> None:
    m = memory.summary
    k = kuzu.summary
    metrics = [
        ("MRR", "mrr"),
        ("Mean P@5", "mean_precision@k"),
        ("Mean R@5", "mean_recall@k"),
        ("Mean nDCG@5", "mean_ndcg@k"),
        ("Avg latency (ms)", "mean_search_time_ms"),
    ]

    print()
    print("=" * 74)
    print(f"{'Kuzu vs Memory parity — enterprise scenario (15 queries)':^74}")
    print("=" * 74)
    print(f"{'Metric':<20} | {'Memory':>12} | {'Kuzu':>12} | {'Δ':>12} | {'Δ %':>8}")
    print("-" * 74)
    for label, key in metrics:
        mv = float(m.get(key, 0.0))
        kv = float(k.get(key, 0.0))
        delta = kv - mv
        pct = (delta / mv * 100) if mv else 0.0
        print(f"{label:<20} | {mv:>12.4f} | {kv:>12.4f} | {delta:>+12.4f} | {pct:>+7.2f}%")
    print("-" * 74)

    # Per-query agreement count
    m_hits = sum(1 for q in memory.per_query if float(q["mrr"]) > 0)  # type: ignore[arg-type]
    k_hits = sum(1 for q in kuzu.per_query if float(q["mrr"]) > 0)  # type: ignore[arg-type]
    print(f"Hit rate  ({'Memory':>12s}): {m_hits}/{len(memory.per_query)}")
    print(f"Hit rate  ({'Kuzu':>12s}): {k_hits}/{len(kuzu.per_query)}")
    print("=" * 74)

    # Verdict
    mrr_delta = float(k.get("mrr", 0.0)) - float(m.get("mrr", 0.0))
    print()
    if abs(mrr_delta) < 0.05:
        print(f"✓ PARITY: |ΔMRR| = {abs(mrr_delta):.4f} < 0.05 — Kuzu matches Memory.")
    elif mrr_delta >= 0.05:
        print(f"⬆ IMPROVEMENT: Kuzu MRR is +{mrr_delta:.4f} ahead of Memory.")
    else:
        print(f"⚠ REGRESSION: Kuzu MRR is {mrr_delta:.4f} below Memory. Investigate.")


async def main() -> int:
    scenario = _load_scenario()

    print("[1/2] Running MemoryBackend...")
    mem = MemoryBackend()
    mem_report = await _run_backend(mem, "memory", scenario)

    print("[2/2] Running KuzuBackend...")
    with tempfile.TemporaryDirectory(prefix="kuzu-parity-") as tmp:
        kuzu = KuzuBackend(str(Path(tmp) / "parity.kuzu"))
        kuzu_report = await _run_backend(kuzu, "kuzu", scenario)

    _print_side_by_side(mem_report, kuzu_report)

    # Save raw results
    out_dir = REPO_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "kuzu_parity.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "memory": {
                    "summary": mem_report.summary,
                    "per_query": mem_report.per_query,
                },
                "kuzu": {
                    "summary": kuzu_report.summary,
                    "per_query": kuzu_report.per_query,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nRaw results written to {out_file.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
