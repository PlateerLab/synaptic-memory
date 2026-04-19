"""AutoRAG regression diagnostic — isolate which component (bge-m3 vector
vs bge-reranker-v2-m3 cross-encoder) is responsible for the
0.906 → 0.642 MRR drop observed in v0.17.0 Round 1.

Runs four configs on the same 720-doc / 114-query AutoRAG corpus,
sharing one model load:

  A. FTS-only           (baseline reference, matches README 0.906)
  B. Embedder only      (bge-m3 vector seed + PRF; no reranker)
  C. Reranker only      (FTS seeds only; reranker re-scores them)
  D. Embedder + reranker (Round 1 config)

Output: one table showing which single component moves the score.

Not part of the suite — one-shot diagnostic, lives under examples/ablation/.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples" / "ablation"))

from local_bge import LocalBgeM3Embedder, LocalBgeRerankerV2

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.evidence_search import EvidenceSearch
from synaptic.graph import SynapticGraph

TOP_K = 10
DATA_PATH = REPO_ROOT / "tests" / "benchmark" / "data" / "autorag_retrieval.json"


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


async def _build_graph(corpus: list[tuple[str, str, str]], embedder: object | None):
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend, embedder=embedder)

    embeddings: list[list[float] | None] = [None] * len(corpus)
    if embedder is not None:
        inputs = [f"{title or doc_id}\n{(text or '')[:1500]}" for doc_id, title, text in corpus]
        BATCH = 64
        for i in range(0, len(inputs), BATCH):
            vecs = await embedder.embed_batch(inputs[i : i + BATCH])
            for j, v in enumerate(vecs):
                embeddings[i + j] = v if v else None

    for (doc_id, title, text), emb in zip(corpus, embeddings):
        if not text and not title:
            continue
        await graph.add(
            title=title or doc_id,
            content=text,
            properties={"doc_id": doc_id},
            embedding=emb,
        )
    return backend, graph


async def _score(
    corpus: list[tuple[str, str, str]],
    queries: list[tuple[str, str, set[str]]],
    *,
    embedder: object | None,
    reranker: object | None,
    label: str,
) -> tuple[float, int, float]:
    backend, graph = await _build_graph(corpus, embedder)
    searcher = EvidenceSearch(backend=backend, embedder=embedder, reranker=reranker)

    mrr_total = 0.0
    hit = 0
    t0 = time.time()
    for _qid, qtext, relevant in queries:
        result = await searcher.search(qtext, k=TOP_K, fts_seed_limit=30)
        retrieved: list[str] = []
        for ev in result.evidence:
            doc_id = (ev.node.properties or {}).get("doc_id", "")
            if doc_id and doc_id not in retrieved:
                retrieved.append(doc_id)
        rr = _reciprocal_rank(retrieved[:TOP_K], relevant)
        mrr_total += rr
        if rr > 0:
            hit += 1
    elapsed = time.time() - t0
    n = len(queries)
    return mrr_total / max(n, 1), hit, elapsed


async def main() -> None:
    data = json.loads(DATA_PATH.read_text())
    raw = data["corpus"]
    corpus: list[tuple[str, str, str]] = []
    if isinstance(raw, dict):
        for doc_id, d in raw.items():
            corpus.append((str(doc_id), str(d.get("title", "")), str(d.get("text", ""))))
    print(f"corpus: {len(corpus)} docs")

    qrels = data.get("qrels", data.get("relevant_docs", {}))
    queries: list[tuple[str, str, set[str]]] = []
    for qid, text in data["queries"].items():
        rel = qrels.get(qid, {})
        ids = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
        if ids and text:
            queries.append((str(qid), str(text), ids))
    print(f"queries: {len(queries)}")

    print("\nLoading bge-m3 + bge-reranker-v2-m3 on cuda:0 ...")
    embedder = LocalBgeM3Embedder(device="cuda:0")
    reranker = LocalBgeRerankerV2(device="cuda:0")

    print()
    print(f"{'Config':<32} {'MRR':>8} {'Hit':>10} {'Time':>8}")
    print("-" * 60)

    configs = [
        ("A. FTS-only", None, None),
        ("B. Embedder only (no reranker)", embedder, None),
        ("C. Reranker only (no embedder)", None, reranker),
        ("D. Embedder + reranker", embedder, reranker),
    ]
    for label, emb, rrk in configs:
        mrr, hit, elapsed = await _score(corpus, queries, embedder=emb, reranker=rrk, label=label)
        print(f"{label:<32} {mrr:>8.3f} {hit:>5}/{len(queries):<4} {elapsed:>6.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
