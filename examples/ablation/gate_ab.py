"""Sufficiency-gate A/B via the REAL gated path (run_agent_loop).

The standard eval harness (eval/run_all.py) uses its own inline agent loop that
does NOT exercise the sufficiency gate, so it cannot measure it. This script
drives run_agent_loop directly (which has the gate) on a labelled dataset,
scoring id-reach the same way run_all.py does (found_ids resolved to doc_ids
via a title/id index; multi_hop scored strict).

Two comparison modes (--compare):
  gate   (default) — gate OFF vs gate ON (the L29 default-on validation).
  bridge           — gate ON plain vs gate ON + bridge-aware gap injection
                     (L29b multi-hop chaining). Use a multi-hop dataset.

Usage:
  SYNAPTIC_SUFFICIENCY_GATE= SYNAPTIC_GATE_BRIDGE= \
  uv run python examples/ablation/gate_ab.py \
    --dataset krra_hard --graph eval/data/krra_graph.sqlite \
    --compare bridge \
    --llm-base-url http://localhost:8012/v1 --model Qwen3.6-27B \
    --embed-url http://localhost:8013/v1 --embed-model Qwen3-Embedding-4B \
    --concurrency 6

Leave env SYNAPTIC_SUFFICIENCY_GATE / SYNAPTIC_GATE_BRIDGE UNSET so the per-arm
args control the gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time


async def _run(args) -> None:
    from openai import AsyncOpenAI

    from synaptic.agent_loop import run_agent_loop
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.extensions.embedder import OpenAIEmbeddingProvider

    backend = SqliteGraphBackend(args.graph)
    await backend.connect()
    embedder = OpenAIEmbeddingProvider(api_base=args.embed_url, model=args.embed_model)
    client = AsyncOpenAI(base_url=args.llm_base_url, api_key="ignored")

    # found_id (title or node id) -> {doc_id} resolution, mirroring run_all.py.
    docid_index: dict[str, set[str]] = {}
    for n in await backend.list_nodes(kind=None, limit=500_000):
        did = (n.properties or {}).get("doc_id")
        if not did:
            continue
        docid_index.setdefault(n.id, set()).add(did)
        if n.title:
            docid_index.setdefault(n.title, set()).add(did)

    with open(args.queries, encoding="utf-8") as f:
        queries = [q for q in json.load(f).get("queries", []) if q.get("query")]

    async def score_one(q: dict, *, gate: bool, bridge: bool, sem) -> bool | None:
        relevant = set(q.get("relevant_docs", []))
        if not relevant:
            return None
        async with sem:
            res = await run_agent_loop(
                client=client,
                backend=backend,
                query=q["query"],
                model=args.model,
                max_turns=args.max_turns,
                embedder=embedder,
                sufficiency_gate=gate,
                gate_bridge=bridge,
            )
        fids = set(res.found_ids)
        for fid in list(fids):
            fids |= docid_index.get(fid, set())
        if q.get("type") == "multi_hop":
            return relevant.issubset(fids)
        return bool(fids & relevant)

    # Arms: (label, sufficiency_gate, gate_bridge). Baseline first.
    if args.compare == "bridge":
        arms = [("gate ON         ", True, False), ("gate ON +bridge ", True, True)]
    else:
        arms = [("gate OFF        ", False, False), ("gate ON         ", True, False)]

    lines: list[str] = []
    for label, gate, bridge in arms:
        sem = asyncio.Semaphore(args.concurrency)
        t0 = time.time()
        scored = await asyncio.gather(
            *[score_one(q, gate=gate, bridge=bridge, sem=sem) for q in queries]
        )
        scored = [s for s in scored if s is not None]
        solved = sum(1 for s in scored if s)
        total = len(scored)
        line = f"{label}: {solved}/{total} ({solved / total:.3f}) in {time.time() - t0:.0f}s"
        print(line, flush=True)
        lines.append(line)

    print(f"\n=== gate A/B (compare={args.compare}) ===")
    for line in lines:
        print(line)
    await backend.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="krra_hard")
    p.add_argument("--graph", default="eval/data/krra_graph.sqlite")
    p.add_argument("--queries", default="")
    p.add_argument("--llm-base-url", default="http://localhost:8012/v1")
    p.add_argument("--model", default="Qwen3.6-27B")
    p.add_argument("--embed-url", default="http://localhost:8013/v1")
    p.add_argument("--embed-model", default="Qwen3-Embedding-4B")
    p.add_argument("--max-turns", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument(
        "--compare",
        choices=["gate", "bridge"],
        default="gate",
        help="gate=OFF vs ON (default); bridge=ON-plain vs ON+bridge (multi-hop).",
    )
    args = p.parse_args()
    if not args.queries:
        args.queries = f"eval/data/queries/{args.dataset}.json"
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
