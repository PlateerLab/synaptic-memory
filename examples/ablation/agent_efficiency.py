"""Profile agent efficiency — the DETERMINISTIC waste signals (no noise floor).

Accuracy deltas on the agent benches sit inside a ±8/120 run-to-run noise floor
(project_v028_agent_bench_noise_floor), so the productive lever now is COST, not
accuracy. This runs run_agent_loop over a dataset and aggregates the counts that
ARE deterministic per query:

  - turns_used / tool_calls_made (mean, distribution)
  - duplicate calls: same (tool, args) reissued in one query — pure waste
  - empty calls: tool returned 0 results — wasted round-trip
  - per-tool usage histogram

Pick a single-hop dataset (max_turns=5) for a quick read; enumeration queries
bump to 15 turns and dominate wall-clock.

Usage:
  uv run python examples/ablation/agent_efficiency.py \
    --dataset finreg --graph eval/data/finreg_graph.sqlite --limit 30 \
    --llm-base-url http://localhost:8012/v1 --model Qwen3.6-27B \
    --embed-url http://localhost:8013/v1 --embed-model Qwen3-Embedding-4B \
    --concurrency 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter


async def _run(args) -> None:
    from openai import AsyncOpenAI

    from synaptic.agent_loop import run_agent_loop
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.extensions.embedder import OpenAIEmbeddingProvider

    backend = SqliteGraphBackend(args.graph)
    await backend.connect()
    embedder = OpenAIEmbeddingProvider(api_base=args.embed_url, model=args.embed_model)
    client = AsyncOpenAI(base_url=args.llm_base_url, api_key="ignored")

    # doc-id resolution for the solve guard (mirrors gate_ab / overflow_check).
    docid_index: dict[str, set[str]] = {}
    for node in await backend.list_nodes(kind=None, limit=500_000):
        did = (node.properties or {}).get("doc_id")
        if not did:
            continue
        docid_index.setdefault(node.id, set()).add(did)
        if node.title:
            docid_index.setdefault(node.title, set()).add(did)

    with open(args.queries, encoding="utf-8") as f:
        queries = [q for q in json.load(f).get("queries", []) if q.get("query")]
    if args.limit:
        queries = queries[: args.limit]

    sem = asyncio.Semaphore(args.concurrency)

    async def run_one(q: dict):
        async with sem:
            res = await run_agent_loop(
                client=client,
                backend=backend,
                query=q["query"],
                model=args.model,
                max_turns=args.max_turns,
                embedder=embedder,
            )
        # solve guard: did the agent reach a relevant doc?
        solved = None
        relevant = set(q.get("relevant_docs", []))
        if relevant:
            fids = set(res.found_ids)
            for fid in list(fids):
                fids |= docid_index.get(fid, set())
            solved = (
                relevant.issubset(fids) if q.get("type") == "multi_hop" else bool(fids & relevant)
            )
        return res, solved

    t0 = time.time()
    pairs = await asyncio.gather(*[run_one(q) for q in queries])
    results = [r for r, _ in pairs]
    solves = [s for _, s in pairs if s is not None]
    wall = time.time() - t0

    n = len(results)
    turns = [r.turns_used for r in results]
    calls = [r.tool_calls_made for r in results]
    total_calls = sum(calls)
    dup = sum(1 for r in results for e in r.tool_log if e.get("duplicate"))
    empty = sum(1 for r in results for e in r.tool_log if e.get("n_results", 0) == 0)
    tool_hist: Counter = Counter(e["tool"] for r in results for e in r.tool_log)
    elapsed = [r.elapsed_ms for r in results]

    def pct(x: int) -> str:
        return f"{100 * x / total_calls:.1f}%" if total_calls else "—"

    print(f"\n=== agent efficiency ({n} queries, {args.dataset}) ===", flush=True)
    print(f"wall: {wall:.0f}s  mean per-query: {sum(elapsed) / n / 1000:.1f}s", flush=True)
    print(f"turns_used:      mean {sum(turns) / n:.2f}  max {max(turns)}", flush=True)
    print(
        f"tool_calls:      mean {sum(calls) / n:.2f}  total {total_calls}  max {max(calls)}",
        flush=True,
    )
    print(f"duplicate calls: {dup}  ({pct(dup)} of calls)  <- pure waste", flush=True)
    print(f"empty calls:     {empty}  ({pct(empty)} of calls)  <- wasted round-trip", flush=True)
    print(f"tool usage:      {dict(tool_hist.most_common())}", flush=True)
    if solves:
        solved = sum(1 for s in solves if s)
        print(f"solve (guard):   {solved}/{len(solves)} ({solved / len(solves):.3f})", flush=True)

    if args.dump:
        print(f"\n--- per-query tool sequences (first {args.dump}) ---", flush=True)
        for r in results[: args.dump]:
            print(f"\nQ: {r.query[:90]}  (turns={r.turns_used}, calls={r.tool_calls_made})")
            for e in r.tool_log:
                flag = " DUP" if e["duplicate"] else (" EMPTY" if e["n_results"] == 0 else "")
                print(
                    f"  t{e['turn']} {e['tool']}({e['key'].split(':', 1)[1][:60]}) -> {e['n_results']}{flag}"
                )
    await backend.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="finreg")
    p.add_argument("--graph", default="eval/data/finreg_graph.sqlite")
    p.add_argument("--queries", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--llm-base-url", default="http://localhost:8012/v1")
    p.add_argument("--model", default="Qwen3.6-27B")
    p.add_argument("--embed-url", default="http://localhost:8013/v1")
    p.add_argument("--embed-model", default="Qwen3-Embedding-4B")
    p.add_argument("--max-turns", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument(
        "--dump", type=int, default=0, help="print per-query tool sequences for the first N queries"
    )
    args = p.parse_args()
    if not args.queries:
        args.queries = f"eval/data/queries/{args.dataset}.json"
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
