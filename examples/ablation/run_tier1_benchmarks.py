"""Tier-1 English multi-hop benchmark runner.

Runs Synaptic's retrieval pipeline over three standard multi-hop
corpora: HotPotQA-dev (full), MuSiQue-Ans-dev, and 2WikiMultiHopQA-dev.
These are the datasets HippoRAG2, GraphRAG, and the broader KG-RAG
line use for head-to-head comparisons.

Two modes:

1. Default (no flags): embedder-free baseline (FTS + PPR only). Same
   pipeline as the v0.16.0 published numbers.
2. With ``--embedder-url`` and/or ``--reranker-url``: full pipeline with
   GPU-backed semantic signal. This is the configuration to compare
   against HippoRAG2 / NV-Embed-v2 head-to-head.

Prerequisite
------------
Download the datasets first::

    pip install datasets
    python examples/ablation/download_benchmarks.py

Usage
-----
::

    # Embedder-free baseline (current published numbers)
    python examples/ablation/run_tier1_benchmarks.py
    python examples/ablation/run_tier1_benchmarks.py --only hotpotqa
    python examples/ablation/run_tier1_benchmarks.py --subset 200

    # Full pipeline with Ollama embedder + TEI cross-encoder
    python examples/ablation/run_tier1_benchmarks.py --subset 500 \\
        --embedder-url http://localhost:11434 \\
        --embedder-model qwen3-embedding:4b \\
        --reranker-url http://localhost:8180

The JSON input files are gitignored (``tests/benchmark/data/*.json``);
the download script is the source of truth. This runner prints a
results table and writes ``examples/ablation/diagnostics/tier1_<ts>.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from synaptic.backends.memory import MemoryBackend
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.embedder import (
    EmbeddingProvider,
    OllamaEmbeddingProvider,
)
from synaptic.extensions.reranker_cross import TEIReranker
from synaptic.graph import SynapticGraph

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "tests" / "benchmark" / "data"
OUT_DIR = Path(__file__).parent / "diagnostics"

TOP_K = 10


@dataclass
class Dataset:
    name: str
    path: Path
    reference: str  # what prior published number to contextualise against


DATASETS = [
    Dataset(
        name="HotPotQA dev (full)",
        path=BENCH / "hotpotqa_full.json",
        reference="HippoRAG2: 56.7 % string accuracy",
    ),
    Dataset(
        name="MuSiQue-Ans dev",
        path=BENCH / "musique_dev.json",
        reference="HippoRAG2: F1 51.9, R@5 74.7 %",
    ),
    Dataset(
        name="2WikiMultihopQA dev",
        path=BENCH / "2wiki_dev.json",
        reference="HippoRAG2: R@5 90.4 %",
    ),
]


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, did in enumerate(retrieved):
        if did in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


@dataclass
class Report:
    name: str
    n_docs: int
    n_queries: int
    mrr: float
    recall_at_5: float
    recall_at_10: float
    hit_at_10: int
    build_sec: float
    search_sec: float
    reference: str


async def run_one(
    ds: Dataset,
    subset: int | None,
    *,
    embedder: EmbeddingProvider | None = None,
    reranker: object | None = None,
    use_sqlite_graph: bool = False,
    embed_batch: int = 256,
) -> Report:
    if not ds.path.exists():
        raise FileNotFoundError(
            f"{ds.path} missing. Run:  python examples/ablation/download_benchmarks.py"
        )
    with open(ds.path, encoding="utf-8") as f:
        data = json.load(f)

    corpus = data["corpus"]
    queries_all = data["queries"]
    qrels = data["qrels"]

    query_items = list(queries_all.items())
    if subset is not None and subset < len(query_items):
        query_items = query_items[:subset]

    # Build the graph once for the whole dataset.
    t_build = time.perf_counter()
    tmp_db_path: str | None = None
    if use_sqlite_graph:
        tmp_db = tempfile.NamedTemporaryFile(
            prefix=f"tier1_{ds.name.replace(' ', '_')}_",
            suffix=".db",
            delete=False,
        )
        tmp_db.close()
        tmp_db_path = tmp_db.name
        backend = SqliteGraphBackend(tmp_db_path)
    else:
        backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend, embedder=embedder, reranker=reranker)

    # Pre-compute embeddings in large batches (GPU-friendly).
    # ``graph.add()`` accepts an ``embedding`` arg; if we pass it we
    # avoid the per-node single embed call that bottlenecks at batch=1.
    items = [
        (
            doc_id,
            str(doc.get("title", "") or doc_id),
            str(doc.get("text", "")),
        )
        for doc_id, doc in corpus.items()
    ]
    items = [(d, t, x) for d, t, x in items if t or x]

    embeddings: list[list[float] | None] = [None] * len(items)
    if embedder is not None:
        embed_inputs = [
            f"{title}\n{(text or '')[:1500]}" for _doc_id, title, text in items
        ]
        for i in range(0, len(embed_inputs), embed_batch):
            chunk = embed_inputs[i : i + embed_batch]
            vecs = await embedder.embed_batch(chunk)
            for j, v in enumerate(vecs):
                embeddings[i + j] = v if v else None

    for (doc_id, title, text), emb in zip(items, embeddings):
        await graph.add(
            title=title,
            content=text,
            properties={"doc_id": doc_id},
            embedding=emb,
        )
    build_sec = time.perf_counter() - t_build

    mrr_total = 0.0
    r5_total = 0.0
    r10_total = 0.0
    hit10 = 0

    t_search = time.perf_counter()
    for qid, qtext in query_items:
        rel = qrels.get(qid, {})
        relevant = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
        if not relevant:
            continue
        result = await graph.search(str(qtext), limit=TOP_K * 2)
        retrieved: list[str] = []
        for hit in result.nodes:
            did = (hit.node.properties or {}).get("doc_id", "")
            if did and did not in retrieved:
                retrieved.append(did)
        rr = _reciprocal_rank(retrieved[:TOP_K], relevant)
        mrr_total += rr
        r5_total += _recall_at_k(retrieved, relevant, 5)
        r10_total += _recall_at_k(retrieved, relevant, TOP_K)
        if rr > 0:
            hit10 += 1
    search_sec = time.perf_counter() - t_search

    n = max(len(query_items), 1)
    return Report(
        name=ds.name,
        n_docs=len(corpus),
        n_queries=len(query_items),
        mrr=mrr_total / n,
        recall_at_5=r5_total / n,
        recall_at_10=r10_total / n,
        hit_at_10=hit10,
        build_sec=build_sec,
        search_sec=search_sec,
        reference=ds.reference,
    )


def _emit_markdown(
    reports: list[Report],
    subset: int | None,
    *,
    embedder_label: str,
    reranker_label: str,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"tier1_{stamp}.md"
    lines = [
        "# Tier-1 English multi-hop benchmark — Synaptic v0.16.0",
        "",
        f"- Run at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Subset: {subset if subset else 'full'}",
        f"- Embedder: {embedder_label}",
        f"- Reranker: {reranker_label}",
        "- Engine: `graph.search()` default (EvidenceSearch)",
        "",
        "| Dataset | Docs | Queries | MRR@10 | R@5 | R@10 | Hit@10 | Build | Search |",
        "|---------|-----:|--------:|-------:|----:|-----:|-------:|------:|-------:|",
    ]
    for r in reports:
        lines.append(
            f"| {r.name} | {r.n_docs} | {r.n_queries} | "
            f"{r.mrr:.3f} | {r.recall_at_5:.3f} | {r.recall_at_10:.3f} | "
            f"{r.hit_at_10}/{r.n_queries} | {r.build_sec:.1f}s | {r.search_sec:.1f}s |"
        )
    lines.append("")
    lines.append("## Context")
    lines.append("")
    for r in reports:
        lines.append(f"- **{r.name}** — published baseline: {r.reference}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


async def amain(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        default=",".join(["hotpotqa", "musique", "2wiki"]),
        help="comma-separated dataset keys (hotpotqa | musique | 2wiki)",
    )
    p.add_argument("--subset", type=int, default=None)
    p.add_argument(
        "--embedder-url",
        default=None,
        help="Ollama base URL (e.g. http://localhost:11434). If unset, "
        "runs FTS-only (embedder-free baseline).",
    )
    p.add_argument(
        "--embedder-model",
        default="qwen3-embedding:4b",
        help="Ollama embedding model name (default: qwen3-embedding:4b).",
    )
    p.add_argument(
        "--reranker-url",
        default=None,
        help="TEI reranker base URL (e.g. http://localhost:8180). "
        "If unset, no cross-encoder reranking.",
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
    p.add_argument(
        "--use-sqlite-graph",
        action="store_true",
        help="Use SqliteGraphBackend (usearch HNSW) instead of MemoryBackend. "
        "Required for fast vector search at corpus sizes > 5k.",
    )
    p.add_argument(
        "--embed-batch",
        type=int,
        default=64,
        help="Pre-compute corpus embeddings in batches of this size "
        "(default: 64 — safe under 6 GB free VRAM). Bump to 128–256 "
        "if more headroom.",
    )
    args = p.parse_args(argv)

    embedder: EmbeddingProvider | None = None
    embedder_label = "none (FTS-only baseline)"
    reranker: object | None = None
    reranker_label = "none"

    if args.local_bge:
        from local_bge import LocalBgeM3Embedder, LocalBgeRerankerV2

        print(f"Loading bge-m3 + bge-reranker-v2-m3 on {args.local_bge_device} ...")
        embedder = LocalBgeM3Embedder(device=args.local_bge_device)
        reranker = LocalBgeRerankerV2(device=args.local_bge_device)
        embedder_label = f"local BAAI/bge-m3 ({args.local_bge_device})"
        reranker_label = f"local BAAI/bge-reranker-v2-m3 ({args.local_bge_device})"
    else:
        if args.embedder_url:
            embedder = OllamaEmbeddingProvider(
                base_url=args.embedder_url,
                model=args.embedder_model,
            )
            embedder_label = f"Ollama {args.embedder_model} @ {args.embedder_url}"
        if args.reranker_url:
            reranker = TEIReranker(base_url=args.reranker_url)
            reranker_label = f"TEI cross-encoder @ {args.reranker_url}"

    by_key = {
        "hotpotqa": DATASETS[0],
        "musique": DATASETS[1],
        "2wiki": DATASETS[2],
    }
    selected = [by_key[k.strip()] for k in args.only.split(",") if k.strip()]

    mode = "full pipeline" if embedder or reranker else "embedder-free"
    backend_label = "SqliteGraphBackend (HNSW)" if args.use_sqlite_graph else "MemoryBackend"
    print(f"Tier-1 multi-hop English benchmarks — Synaptic v0.16.0 {mode}")
    print(f"  backend:  {backend_label}")
    print(f"  embedder: {embedder_label}")
    print(f"  reranker: {reranker_label}")
    if embedder is not None:
        print(f"  embed batch: {args.embed_batch}")
    print()
    header = f"{'Dataset':<24} {'Docs':>7} {'Qs':>6} {'MRR@10':>8} {'R@5':>7} {'R@10':>7} {'Hit':>10} {'Build':>7} {'Search':>8}"
    print(header)
    print("-" * len(header))

    reports: list[Report] = []
    for ds in selected:
        try:
            r = await run_one(
                ds,
                args.subset,
                embedder=embedder,
                reranker=reranker,
                use_sqlite_graph=args.use_sqlite_graph,
                embed_batch=args.embed_batch,
            )
        except FileNotFoundError as e:
            print(f"{ds.name:<24}  SKIP — {e}")
            continue
        reports.append(r)
        print(
            f"{r.name:<24} {r.n_docs:>7} {r.n_queries:>6} "
            f"{r.mrr:>8.3f} {r.recall_at_5:>7.3f} {r.recall_at_10:>7.3f} "
            f"{r.hit_at_10:>5}/{r.n_queries:<4} {r.build_sec:>6.1f}s {r.search_sec:>7.1f}s"
        )

    if reports:
        out = _emit_markdown(
            reports,
            args.subset,
            embedder_label=embedder_label,
            reranker_label=reranker_label,
        )
        print()
        print(f"Markdown report → {out.relative_to(REPO_ROOT)}")
    return 0


def main() -> None:
    import sys as _sys

    _sys.exit(asyncio.run(amain(_sys.argv[1:])))


if __name__ == "__main__":
    main()
