"""Ablation runner — measure each search-quality change in isolation.

Usage::

    python examples/ablation/run_ablation.py

Runs every public Korean / English dataset through the current code
and prints a before/after table.

The 2026-04-17 run that locks in the query-mode Kiwi improvement:

| Dataset             | Baseline (pre-0.15.1) | Current | Δ MRR  |
|---------------------|-----------------------|---------|--------|
| Allganize RAG-ko    | 0.621                 | 0.743   | +0.122 |
| Allganize RAG-Eval  | 0.615                 | 0.695   | +0.080 |
| PublicHealthQA KO   | 0.318                 | 0.466   | +0.148 |
| AutoRAG KO          | 0.592                 | 0.692   | +0.100 |
| HotPotQA-24 EN      | 0.727                 | 0.727   |  0.000 |

English is untouched because the Kiwi pipeline only fires when the
query is ≥50 % Hangul — the query-mode stripping is a Korean-only
effect.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "tests" / "benchmark" / "data"


@dataclass
class DatasetSpec:
    name: str
    path: Path
    lang: str


DATASETS = [
    DatasetSpec("Allganize RAG-ko", BENCH / "allganize_rag_ko.json", "ko"),
    DatasetSpec("Allganize RAG-Eval", BENCH / "allganize_rag_eval.json", "ko"),
    DatasetSpec("PublicHealthQA KO", BENCH / "publichealthqa_ko.json", "ko"),
    DatasetSpec("AutoRAG KO", BENCH / "autorag_retrieval.json", "ko"),
    DatasetSpec("HotPotQA-24 EN", BENCH / "hotpotqa_24.json", "en"),
]

# Locked-in baselines from pre-0.15.1 measurements (CLAUDE.md / eval/baselines).
# Used by this script to compute deltas and flag regressions.
BASELINE: dict[str, float] = {
    "Allganize RAG-ko": 0.621,
    "Allganize RAG-Eval": 0.615,
    "PublicHealthQA KO": 0.318,
    "AutoRAG KO": 0.592,
    "HotPotQA-24 EN": 0.727,
}


async def _measure(spec: DatasetSpec) -> tuple[float, int, int, float]:
    if not spec.path.exists():
        return 0.0, 0, 0, 0.0
    with open(spec.path, encoding="utf-8") as f:
        data = json.load(f)
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend)
    corpus = data.get("corpus", {})
    if isinstance(corpus, dict):
        iter_items = corpus.items()
    else:  # list
        iter_items = [
            (str(d.get("doc_id") or d.get("_id") or d.get("id") or ""), d) for d in corpus
        ]
    for doc_id, d in iter_items:
        if d.get("text") or d.get("title"):
            await graph.add(
                title=d.get("title", "") or doc_id,
                content=d.get("text", ""),
                properties={"doc_id": doc_id},
            )
    qrels = data.get("qrels", data.get("relevant_docs", {}))
    queries: list[tuple[str, str, set[str]]] = []
    qs = data.get("queries", {})
    if isinstance(qs, dict):
        for qid, text in qs.items():
            rel = qrels.get(qid, {})
            ids = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
            if ids and text:
                queries.append((str(qid), str(text), ids))
    elif isinstance(qs, list):
        for q in qs:
            qid = str(q.get("qid") or q.get("query_id") or q.get("_id") or "")
            text = str(q.get("query") or q.get("question") or "")
            rel = q.get("relevant_docs") or []
            ids = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
            if ids and text:
                queries.append((qid, text, ids))

    mrr_total = 0.0
    hit = 0
    t0 = time.time()
    for qid, qtext, rel in queries:
        r = await graph.search(qtext, limit=10)
        retr: list[str] = []
        for h in r.nodes:
            did = (h.node.properties or {}).get("doc_id", "")
            if did and did not in retr:
                retr.append(did)
        rr = next((1.0 / (i + 1) for i, d in enumerate(retr[:10]) if d in rel), 0.0)
        mrr_total += rr
        if rr > 0:
            hit += 1
    elapsed = time.time() - t0
    n = len(queries)
    return mrr_total / max(n, 1), hit, n, elapsed


async def main() -> None:
    print("Ablation runner — pre-0.15.1 baseline vs current code")
    print()
    print(
        f"{'Dataset':<22} {'Lang':<4}  {'Queries':>8}  {'Base MRR':>8}  "
        f"{'Now MRR':>8}  {'Δ MRR':>8}  {'Hit':>8}  {'Time':>7}"
    )
    print("-" * 90)
    regressions: list[str] = []
    for spec in DATASETS:
        if not spec.path.exists():
            print(f"{spec.name:<22} {spec.lang:<4}  SKIP — missing {spec.path}")
            continue
        mrr, hit, n, elapsed = await _measure(spec)
        base = BASELINE.get(spec.name, 0.0)
        delta = mrr - base
        tag = " ⚠" if delta < -0.005 else ""
        print(
            f"{spec.name:<22} {spec.lang:<4}  {n:>8}  {base:>8.3f}  "
            f"{mrr:>8.3f}  {delta:>+8.3f}  {hit:>4}/{n:<3}  {elapsed:>6.1f}s{tag}"
        )
        if delta < -0.005:
            regressions.append(f"{spec.name} ({delta:+.3f})")
    print()
    if regressions:
        print("Regressions detected:")
        for r in regressions:
            print(f"  - {r}")
    else:
        print("No regressions.")


if __name__ == "__main__":
    asyncio.run(main())
