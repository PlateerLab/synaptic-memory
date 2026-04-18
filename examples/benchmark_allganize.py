"""Reproducible Allganize RAG-ko benchmark — embedder-free baseline.

This script evaluates Synaptic Memory on the public Allganize
RAG-Evaluation-Dataset-KO in **embedder-free mode** — no vector
index, no cross-encoder reranker, zero LLM calls. The retrieval
pipeline is the v0.16.0 default (EvidenceSearch: BM25 + PPR + MMR
+ graph expansion). It's the reproducible floor any reader can run
on a laptop in under 15 seconds.

Data source
-----------
https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO
(MIT-licensed, Korean enterprise RAG benchmark — finance, public
sector, healthcare, legal, commerce.)

The JSON snapshot shipped under ``tests/benchmark/data/`` was prepared
by the Synaptic Memory team from the Allganize Hugging Face release.
See ``eval/data/queries/_public_sources.json`` for attribution.

How to run
----------
::

    pip install "synaptic-memory[korean]"
    python examples/benchmark_allganize.py

What it prints
--------------
Per-dataset MRR, Hit rate, and Recall@10 over the full 200 / 300
query sets. Results are deterministic (no LLM, no sampling).

Adding an embedder + cross-encoder reranker will push the scores
further — that path is covered in ``eval/run_all.py``. This script
is the **minimum-dependency** baseline so the numbers can be
cross-checked independently of GPU infrastructure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.embedder import (
    EmbeddingProvider,
    OllamaEmbeddingProvider,
)
from synaptic.extensions.reranker_cross import TEIReranker
from synaptic.graph import SynapticGraph

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- Inlined IR metrics (mirror tests/benchmark/metrics.py) ---


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in relevant)
    return hits / len(relevant)

BENCH_DATA = REPO_ROOT / "tests" / "benchmark" / "data"
DATASETS = [
    ("Allganize RAG-ko", BENCH_DATA / "allganize_rag_ko.json"),
    ("Allganize RAG-Eval", BENCH_DATA / "allganize_rag_eval.json"),
]
TOP_K = 10


@dataclass
class Report:
    name: str
    corpus_size: int
    query_count: int
    mrr: float
    recall_at_k: float
    hit_count: int
    elapsed_sec: float


def _parse_queries(
    raw_queries, qrels
) -> list[tuple[str, str, set[str]]]:
    """Normalize BEIR-style qrels + either dict or list query formats."""
    out: list[tuple[str, str, set[str]]] = []
    if isinstance(raw_queries, dict):
        for qid, text in raw_queries.items():
            rel = qrels.get(qid, {})
            ids = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
            if ids and text:
                out.append((str(qid), str(text), ids))
    elif isinstance(raw_queries, list):
        for q in raw_queries:
            qid = str(q.get("qid") or q.get("query_id") or q.get("_id") or "")
            text = str(q.get("query") or q.get("question") or "")
            rel = q.get("relevant_docs") or q.get("answer_ids") or q.get("positive_doc_ids") or []
            ids = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
            if ids and text:
                out.append((qid, text, ids))
    return out


async def run_dataset(
    name: str,
    path: Path,
    *,
    embedder: EmbeddingProvider | None = None,
    reranker: object | None = None,
) -> Report:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # --- Load corpus ---
    corpus_raw = data.get("corpus", data.get("documents", []))
    corpus: list[tuple[str, str, str]] = []
    if isinstance(corpus_raw, dict):
        for doc_id, doc in corpus_raw.items():
            corpus.append((str(doc_id), str(doc.get("title", "")), str(doc.get("text", ""))))
    elif isinstance(corpus_raw, list):
        for doc in corpus_raw:
            doc_id = str(doc.get("doc_id") or doc.get("_id") or doc.get("id") or "")
            corpus.append(
                (doc_id, str(doc.get("title", "")), str(doc.get("text") or doc.get("content", "")))
            )

    # --- Load queries ---
    queries = _parse_queries(
        data.get("queries", []),
        data.get("relevant_docs", data.get("qrels", {})),
    )

    # --- Build graph (MemoryBackend — no disk I/O) ---
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend, embedder=embedder, reranker=reranker)
    for doc_id, title, text in corpus:
        if not text and not title:
            continue
        await graph.add(title=title or doc_id, content=text, properties={"doc_id": doc_id})

    # --- Run search + score ---
    mrr_total = 0.0
    recall_total = 0.0
    hit_count = 0
    t0 = time.time()
    for _qid, qtext, relevant in queries:
        result = await graph.search(qtext, limit=TOP_K * 2)
        retrieved: list[str] = []
        for hit in result.nodes:
            did = (hit.node.properties or {}).get("doc_id", "")
            if did and did not in retrieved:
                retrieved.append(did)
        rr = _reciprocal_rank(retrieved[:TOP_K], relevant)
        mrr_total += rr
        recall_total += _recall_at_k(retrieved, relevant, TOP_K)
        if rr > 0:
            hit_count += 1
    elapsed = time.time() - t0

    n = max(len(queries), 1)
    return Report(
        name=name,
        corpus_size=len(corpus),
        query_count=len(queries),
        mrr=mrr_total / n,
        recall_at_k=recall_total / n,
        hit_count=hit_count,
        elapsed_sec=elapsed,
    )


async def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder-url",
        default=None,
        help="Ollama base URL (e.g. http://localhost:11434). If unset, "
        "runs the embedder-free baseline (the v0.16.0 published numbers).",
    )
    p.add_argument(
        "--embedder-model",
        default="qwen3-embedding:4b",
        help="Ollama embedding model (default: qwen3-embedding:4b).",
    )
    p.add_argument(
        "--reranker-url",
        default=None,
        help="TEI reranker base URL (e.g. http://localhost:8180).",
    )
    p.add_argument(
        "--local-bge",
        action="store_true",
        help="Load BAAI/bge-m3 + bge-reranker-v2-m3 directly via "
        "transformers (no external endpoint). Requires torch + GPU.",
    )
    p.add_argument(
        "--local-bge-device",
        default="cuda:0",
        help="GPU device for --local-bge (default: cuda:0).",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    embedder: EmbeddingProvider | None = None
    embedder_label = "none (FTS-only baseline)"
    reranker: object | None = None
    reranker_label = "none"

    if args.local_bge:
        sys.path.insert(0, str(REPO_ROOT / "examples" / "ablation"))
        from local_bge import LocalBgeM3Embedder, LocalBgeRerankerV2

        print(f"Loading bge-m3 + bge-reranker-v2-m3 on {args.local_bge_device} ...")
        embedder = LocalBgeM3Embedder(device=args.local_bge_device)
        reranker = LocalBgeRerankerV2(device=args.local_bge_device)
        embedder_label = f"local BAAI/bge-m3 ({args.local_bge_device})"
        reranker_label = f"local BAAI/bge-reranker-v2-m3 ({args.local_bge_device})"
    else:
        if args.embedder_url:
            embedder = OllamaEmbeddingProvider(
                base_url=args.embedder_url, model=args.embedder_model
            )
            embedder_label = f"Ollama {args.embedder_model} @ {args.embedder_url}"
        if args.reranker_url:
            reranker = TEIReranker(base_url=args.reranker_url)
            reranker_label = f"TEI cross-encoder @ {args.reranker_url}"

    mode = "full pipeline" if (embedder or reranker) else "embedder-free baseline"
    print(f"Allganize RAG-ko benchmark — Synaptic Memory {mode}")
    print(f"  top-k = {TOP_K}, engine = evidence (BM25 + PPR + MMR + Kiwi)")
    print(f"  embedder: {embedder_label}")
    print(f"  reranker: {reranker_label}")
    print()
    print(f"{'Dataset':<22} {'Corpus':>8} {'Queries':>8} {'MRR':>8} {'R@10':>8} {'Hit':>10} {'Time':>8}")
    print("-" * 80)

    reports: list[Report] = []
    for name, path in DATASETS:
        if not path.exists():
            print(f"{name:<22} SKIP — file not found: {path}")
            continue
        report = await run_dataset(name, path, embedder=embedder, reranker=reranker)
        reports.append(report)
        print(
            f"{report.name:<22} "
            f"{report.corpus_size:>8} "
            f"{report.query_count:>8} "
            f"{report.mrr:>8.3f} "
            f"{report.recall_at_k:>8.3f} "
            f"{report.hit_count:>5}/{report.query_count:<4} "
            f"{report.elapsed_sec:>6.1f}s"
        )

    print()
    print("Notes:")
    if embedder or reranker:
        print(f"  * Mode: {mode}.")
        print("  * For the deterministic embedder-free baseline, re-run without flags.")
    else:
        print("  * Embedder-free — the EvidenceSearch default in v0.16.0+.")
        print("    Adding a GPU-backed embedder + cross-encoder pushes these")
        print("    further: pass --embedder-url / --reranker-url.")
        print("  * Deterministic — re-running produces identical numbers.")
    print("  * Corpus attribution: allganize/RAG-Evaluation-Dataset-KO (HuggingFace).")


if __name__ == "__main__":
    asyncio.run(main())
