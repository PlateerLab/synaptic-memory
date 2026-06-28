"""The keystone proof — does synaptic's AGENT dominate naive RAG?

Every win so far is an internal A/B. This measures the product's actual claim:
on the SAME corpus, SAME LLM, SAME id-reach scoring, compare three retrievers:

  1. naive-vec   — top-k vector retrieval (what a typical RAG library does):
                   embed query → backend.search_vector → did top-k reach gold?
  2. synaptic-ss — synaptic's best SINGLE-SHOT (EvidenceSearch full pipeline:
                   FTS+vector+PRF+rerank+MMR), top-k. No agent iteration.
  3. agent       — synaptic's multi-turn agent loop (run_agent_loop). The moat.

Baselines 1-2 are DETERMINISTIC (no temp>0 LLM) → one run is exact. The agent is
noisy (project_v028_agent_bench_noise_floor) → run it ``--runs N`` times and
report the spread, so a win is claimed over noise, not under it.

Connectivity: pass ``--graph`` a COPY already bridged (graph.connect_components)
to measure the navigable-structure effect on the agent arm; compare two runs.

Usage:
  uv run python examples/ablation/rag_vs_agent.py \
    --dataset finreg_multihop --graph eval/data/finreg_graph.sqlite \
    --llm-base-url http://localhost:8012/v1 --model Qwen3.6-27B \
    --embed-url http://localhost:8013/v1 --embed-model Qwen3-Embedding-4B \
    -k 5 --runs 3 --concurrency 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time


def _reached(fids: set[str], relevant: set[str], multi_hop: bool) -> bool:
    return relevant.issubset(fids) if multi_hop else bool(fids & relevant)


async def _run(args) -> None:
    from openai import AsyncOpenAI

    from synaptic.agent_loop import run_agent_loop
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.extensions.embedder import OpenAIEmbeddingProvider
    from synaptic.extensions.evidence_search import EvidenceSearch

    backend = SqliteGraphBackend(args.graph)
    await backend.connect()
    embedder = OpenAIEmbeddingProvider(api_base=args.embed_url, model=args.embed_model)
    client = AsyncOpenAI(base_url=args.llm_base_url, api_key="ignored")

    # node id / title → {doc_id} resolution (mirrors gate_ab / overflow_check).
    docid_index: dict[str, set[str]] = {}
    for n in await backend.list_nodes(kind=None, limit=500_000):
        did = (n.properties or {}).get("doc_id")
        if not did:
            continue
        docid_index.setdefault(n.id, set()).add(did)
        if n.title:
            docid_index.setdefault(n.title, set()).add(did)

    def _resolve(ids: set[str]) -> set[str]:
        out = set(ids)
        for i in list(ids):
            out |= docid_index.get(i, set())
        return out

    with open(args.queries, encoding="utf-8") as f:
        queries = [q for q in json.load(f).get("queries", []) if q.get("query")]
    queries = [q for q in queries if q.get("relevant_docs")]
    if args.limit:
        queries = queries[: args.limit]

    sem = asyncio.Semaphore(args.concurrency)

    async def naive_vec(q: dict) -> bool:
        vec = await embedder.embed(q["query"])
        ids: set[str] = set()
        if vec:
            hits = await backend.search_vector(vec, limit=args.k)
            for h in hits:
                ids.add(h.id)
                if h.title:
                    ids.add(h.title)
                did = (h.properties or {}).get("doc_id")
                if did:
                    ids.add(did)
        fids = _resolve(ids)
        return _reached(fids, set(q["relevant_docs"]), q.get("type") == "multi_hop")

    async def synaptic_ss(q: dict) -> bool:
        searcher = EvidenceSearch(backend=backend, embedder=embedder)
        res = await searcher.search(q["query"], k=args.k)
        ids: set[str] = set()
        for ev in res.evidence[: args.k]:
            ids.add(ev.node.id)
            if ev.node.title:
                ids.add(ev.node.title)
            if ev.document_id:
                ids.add(ev.document_id)
        fids = _resolve(ids)
        return _reached(fids, set(q["relevant_docs"]), q.get("type") == "multi_hop")

    async def agent(q: dict) -> bool:
        res = await run_agent_loop(
            client=client,
            backend=backend,
            query=q["query"],
            model=args.model,
            max_turns=args.max_turns,
            embedder=embedder,
        )
        fids = _resolve(set(res.found_ids))
        return _reached(fids, set(q["relevant_docs"]), q.get("type") == "multi_hop")

    async def sweep(fn) -> int:
        async def one(q):
            async with sem:
                return await fn(q)

        return sum(1 for s in await asyncio.gather(*[one(q) for q in queries]) if s)

    n = len(queries)
    print(f"\n=== RAG vs agent ({n} queries, {args.dataset}, k={args.k}) ===", flush=True)

    # deterministic baselines — one pass each
    t = time.time()
    nv = await sweep(naive_vec)
    print(
        f"naive-vec   : {nv}/{n} ({nv / n:.3f})  [{time.time() - t:.0f}s, deterministic]",
        flush=True,
    )
    t = time.time()
    ss = await sweep(synaptic_ss)
    print(
        f"synaptic-ss : {ss}/{n} ({ss / n:.3f})  [{time.time() - t:.0f}s, deterministic]",
        flush=True,
    )

    # noisy agent — N runs
    ag: list[int] = []
    for r in range(args.runs):
        t = time.time()
        s = await sweep(agent)
        ag.append(s)
        print(f"agent run {r + 1}  : {s}/{n} ({s / n:.3f})  [{time.time() - t:.0f}s]", flush=True)
    lo, hi = min(ag), max(ag)
    mean = sum(ag) / len(ag)
    print(
        f"agent       : mean {mean:.1f}/{n} ({mean / n:.3f}), range {lo}-{hi} over {args.runs} runs",
        flush=True,
    )
    print(
        f"\n→ agent vs naive-vec: {mean - nv:+.1f}; agent vs synaptic-ss: {mean - ss:+.1f}",
        flush=True,
    )
    await backend.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="finreg_multihop")
    p.add_argument("--graph", default="eval/data/finreg_graph.sqlite")
    p.add_argument("--queries", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--llm-base-url", default="http://localhost:8012/v1")
    p.add_argument("--model", default="Qwen3.6-27B")
    p.add_argument("--embed-url", default="http://localhost:8013/v1")
    p.add_argument("--embed-model", default="Qwen3-Embedding-4B")
    p.add_argument("--max-turns", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=6)
    args = p.parse_args()
    if not args.queries:
        args.queries = f"eval/data/queries/{args.dataset}.json"
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
