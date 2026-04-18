"""Sweep cross-encoder blend weight across all 5 public benchmarks.

Answers: does any single ``rerank_blend`` value both (a) keep the big
paraphrase wins (PublicHealthQA, Allganize) and (b) stop bleeding
AutoRAG's FTS-already-optimal ranking?

Blend values tested: 0.1, 0.2, 0.4 (current default).
Benchmarks: the five public-quick datasets — same corpora / queries /
metric code as ``eval/run_all.py --quick``.

Models load once, shared across all 15 (blend × bench) cells.
Output: one MRR table plus a per-blend mean for quick tuning.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples" / "ablation"))

from local_bge import LocalBgeM3Embedder, LocalBgeRerankerV2  # noqa: E402

from synaptic.backends.memory import MemoryBackend  # noqa: E402
from synaptic.extensions.evidence_search import EvidenceSearch  # noqa: E402
from synaptic.graph import SynapticGraph  # noqa: E402

TOP_K = 10
BLEND_VALUES = [0.1, 0.2, 0.4]

BENCH = REPO_ROOT / "tests" / "benchmark" / "data"
DATASETS = [
    ("HotPotQA-24", BENCH / "hotpotqa_24.json"),
    ("Allganize RAG-ko", BENCH / "allganize_rag_ko.json"),
    ("Allganize RAG-Eval", BENCH / "allganize_rag_eval.json"),
    ("PublicHealthQA", BENCH / "publichealthqa_ko.json"),
    ("AutoRAG", BENCH / "autorag_retrieval.json"),
]


@dataclass
class Cell:
    name: str
    blend: float
    mrr: float
    hit: int
    n: int
    elapsed: float


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _parse(data: dict) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, set[str]]]]:
    raw = data["corpus"]
    corpus: list[tuple[str, str, str]] = []
    if isinstance(raw, dict):
        for doc_id, d in raw.items():
            corpus.append(
                (
                    str(doc_id),
                    str(d.get("title", "")),
                    str(d.get("text", d.get("content", ""))),
                )
            )
    elif isinstance(raw, list):
        for d in raw:
            doc_id = str(d.get("doc_id") or d.get("_id") or d.get("id") or "")
            corpus.append(
                (
                    doc_id,
                    str(d.get("title", "")),
                    str(d.get("text", d.get("content", ""))),
                )
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
            rel = q.get("relevant_docs") or q.get("answer_ids") or []
            ids = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
            if ids and text:
                queries.append((qid, text, ids))
    return corpus, queries


async def _build(
    corpus: list[tuple[str, str, str]], embedder: LocalBgeM3Embedder
) -> MemoryBackend:
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend, embedder=embedder)

    inputs = [f"{title or doc_id}\n{(text or '')[:1500]}" for doc_id, title, text in corpus]
    embeddings: list[list[float] | None] = [None] * len(corpus)
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
    return backend


async def _score(
    backend: MemoryBackend,
    queries: list[tuple[str, str, set[str]]],
    *,
    embedder: LocalBgeM3Embedder,
    reranker: LocalBgeRerankerV2,
    blend: float,
) -> tuple[float, int, float]:
    searcher = EvidenceSearch(
        backend=backend, embedder=embedder, reranker=reranker, rerank_blend=blend
    )
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
    return mrr_total / max(len(queries), 1), hit, time.time() - t0


async def main() -> None:
    print("Loading bge-m3 + bge-reranker-v2-m3 on cuda:0 ...")
    embedder = LocalBgeM3Embedder(device="cuda:0")
    reranker = LocalBgeRerankerV2(device="cuda:0")

    # Build each graph once (heavy path), reuse for all blend values.
    # Blend is applied inside the reranker step, which runs per search()
    # call, so we don't need to rebuild per blend.
    print()
    print(f"{'Benchmark':<24} {'Queries':>8} {'b=0.1':>8} {'b=0.2':>8} {'b=0.4':>8}")
    print("-" * 70)

    per_blend: dict[float, list[float]] = {b: [] for b in BLEND_VALUES}

    for name, path in DATASETS:
        if not path.exists():
            print(f"{name:<24}  SKIP (missing)")
            continue
        data = json.loads(path.read_text())
        corpus, queries = _parse(data)
        if not queries:
            continue

        t_build = time.time()
        backend = await _build(corpus, embedder)
        build_sec = time.time() - t_build

        row = [f"{name:<24}", f"{len(queries):>8}"]
        for blend in BLEND_VALUES:
            mrr, hit, elapsed = await _score(
                backend, queries, embedder=embedder, reranker=reranker, blend=blend
            )
            row.append(f"{mrr:>8.3f}")
            per_blend[blend].append(mrr)
        await backend.close()
        print(" ".join(row) + f"  (build {build_sec:.1f}s)")

    print()
    print("Mean MRR across benchmarks:")
    for b in BLEND_VALUES:
        scores = per_blend[b]
        if scores:
            mean = sum(scores) / len(scores)
            print(f"  blend={b}: mean={mean:.3f} over {len(scores)} benches")


if __name__ == "__main__":
    asyncio.run(main())
