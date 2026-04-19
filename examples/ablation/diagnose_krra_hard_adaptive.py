"""KRRA Hard query-level diff — fixed-blend (0.1) vs adaptive blend.

Round 4 showed KRRA Hard regressed 0.606 → 0.589 when adaptive blend
replaced the fixed 0.1 default. This script reruns each KRRA Hard query
under both configs, captures the cross-encoder rerank score distribution
for the top-20 candidates, and prints a per-query diff.

The goal is to see *which* queries lost and *why* — i.e. whether the
reranker score std actually correlates with reranker correctness on those
queries, or whether the adaptive heuristic is suppressing useful signal.

Output: a markdown table with per-query (rr_fixed, rr_adaptive, std,
chosen_blend, regression?) plus a short failure-case dump for any query
whose rr dropped under adaptive.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples" / "ablation"))

from local_bge import LocalBgeM3Embedder, LocalBgeRerankerV2  # noqa: E402

from synaptic.backends.sqlite_graph import SqliteGraphBackend  # noqa: E402

GRAPH_PATH = REPO_ROOT / "eval" / "data" / "krra_graph.sqlite"
QUERY_PATH = REPO_ROOT / "eval" / "data" / "queries" / "krra_hard.json"
TOP_K = 10
SEED_LIMIT = 30


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, did in enumerate(retrieved):
        if did in relevant:
            return 1.0 / (i + 1)
    return 0.0


async def _run_one(query, backend, embedder, reranker, *, blend_mode: str):
    """Run a single search with one of two blend modes; return (retrieved, rerank_std)."""
    from synaptic.extensions.evidence_search import EvidenceSearch

    if blend_mode == "fixed":
        searcher = EvidenceSearch(
            backend=backend, embedder=embedder, reranker=reranker, rerank_blend=0.1
        )
    else:
        # Adaptive blend is now the in-tree default — no kwarg needed.
        searcher = EvidenceSearch(
            backend=backend, embedder=embedder, reranker=reranker, rerank_blend=0.1
        )
        # Patch the reranker so we can capture the score distribution
    result = await searcher.search(
        query["query"], k=TOP_K, fts_seed_limit=SEED_LIMIT
    )
    retrieved: list[str] = []
    for ev in result.evidence:
        did = (ev.node.properties or {}).get("doc_id", "")
        if did and did not in retrieved:
            retrieved.append(did)
        elif ev.node.title and ev.node.title not in retrieved:
            retrieved.append(ev.node.title)
    return retrieved


async def _capture_rerank_stats(query: str, backend, embedder, reranker, *, top_n=20):
    """Run an FTS+vector seed pass identical to EvidenceSearch then ask the
    reranker for scores on the top-N. Returns (scores, std, top_titles)."""
    fts_nodes = await backend.search_fts(query, limit=SEED_LIMIT)
    docs = [
        f"{n.title}\n{n.content[:400]}"
        for n in fts_nodes[:top_n]
        if not (n.properties or {}).get("_table_name")
    ][:top_n]
    if not docs:
        return [], 0.0, []
    scores = await reranker.rerank(query, docs)
    titles = [n.title for n in fts_nodes[: len(scores)]]
    if len(scores) >= 2:
        std = statistics.pstdev(scores)
    else:
        std = 0.0
    return scores, std, titles


async def main() -> None:
    backend = SqliteGraphBackend(str(GRAPH_PATH))
    await backend.connect()

    print("Loading bge-m3 + bge-reranker-v2-m3 ...")
    embedder = LocalBgeM3Embedder(device="cuda:0")
    reranker = LocalBgeRerankerV2(device="cuda:0")

    queries = json.load(open(QUERY_PATH))["queries"]

    rows: list[dict] = []

    # Disable adaptive by wrapping the reranker so its scores have
    # std ≫ 3 (multiplies the deviation from the mean by 100) — the
    # discriminator clamps to 1.0 → effective blend = base 0.1.
    class _SaturatedReranker:
        def __init__(self, inner):
            self._inner = inner

        async def rerank(self, query: str, documents: list[str]) -> list[float]:
            scores = await self._inner.rerank(query, documents)
            if not scores:
                return scores
            mean = sum(scores) / len(scores)
            return [(s - mean) * 100 + mean for s in scores]

    saturated_reranker = _SaturatedReranker(reranker)

    print("\nFixed blend run (blend_eff = 0.1 always)...")
    fixed_rrs: dict[str, float] = {}
    for q in queries:
        ret = await _run_one(q, backend, embedder, saturated_reranker, blend_mode="fixed")
        rel = set(q.get("relevant_docs", []))
        fixed_rrs[q["qid"]] = _reciprocal_rank(ret, rel) if rel else 0.0

    print("Adaptive blend run (current default)...")
    adaptive_rrs: dict[str, float] = {}
    for q in queries:
        ret = await _run_one(q, backend, embedder, reranker, blend_mode="adaptive")
        rel = set(q.get("relevant_docs", []))
        adaptive_rrs[q["qid"]] = _reciprocal_rank(ret, rel) if rel else 0.0

    print("\nCapturing rerank score stats for each query...")
    for q in queries:
        scores, std, titles = await _capture_rerank_stats(
            q["query"], backend, embedder, reranker
        )
        rr_f = fixed_rrs.get(q["qid"], 0.0)
        rr_a = adaptive_rrs.get(q["qid"], 0.0)
        rows.append(
            {
                "qid": q["qid"],
                "query": q["query"][:60],
                "rr_fixed": rr_f,
                "rr_adaptive": rr_a,
                "delta": rr_a - rr_f,
                "rerank_std": std,
                "discriminator": min(1.0, std / 3.0),
            }
        )

    print()
    print(f"{'qid':<8} {'fixed':>6} {'adaptive':>9} {'Δ':>7} {'std':>6} {'disc':>6}  query")
    print("-" * 110)
    for r in rows:
        marker = "❌" if r["delta"] < 0 else (" " if r["delta"] == 0 else "✅")
        print(
            f"{r['qid']:<8} {r['rr_fixed']:>6.3f} {r['rr_adaptive']:>9.3f} "
            f"{r['delta']:>+7.3f} {r['rerank_std']:>6.2f} {r['discriminator']:>6.2f} "
            f"{marker} {r['query']}"
        )

    print()
    fixed_mrr = sum(r["rr_fixed"] for r in rows) / max(len(rows), 1)
    adaptive_mrr = sum(r["rr_adaptive"] for r in rows) / max(len(rows), 1)
    print(f"Fixed blend MRR: {fixed_mrr:.3f}")
    print(f"Adaptive MRR:    {adaptive_mrr:.3f}  (Δ {adaptive_mrr - fixed_mrr:+.3f})")

    regressions = [r for r in rows if r["delta"] < 0]
    print(f"\n{len(regressions)} queries regressed under adaptive:")
    for r in regressions:
        print(
            f"  {r['qid']}: rr {r['rr_fixed']:.3f} → {r['rr_adaptive']:.3f}, "
            f"std={r['rerank_std']:.2f} (disc={r['discriminator']:.2f}) "
            f"| {r['query']}"
        )

    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
