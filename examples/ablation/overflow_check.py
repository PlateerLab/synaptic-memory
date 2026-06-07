"""Validate the agent context-overflow guard on the bench that triggered it.

KRRA Hard's enumeration queries (max_turns bumped to 15) used to hit the model's
32k context at turn 12-13, where run_agent_loop raised a 400 and BROKE — losing
all remaining retrieval (2 such hard failures observed in the bridge canary).

This drives run_agent_loop (which carries the fix; the eval harness's inline
loop does NOT) over the dataset with the default gate, counting three things
that are DETERMINISTIC counts (not subject to the solve-rate noise floor):
  - escaped: context-length 400s that still killed a turn (should be 0)
  - retried: reactive compaction fired and the turn recovered (>=0)
  - solved: id-reach, same scoring as gate_ab (a sanity check, noisy)

Usage:
  uv run python examples/ablation/overflow_check.py \
    --dataset krra_hard --graph eval/data/krra_graph.sqlite \
    --llm-base-url http://localhost:8012/v1 --model Qwen3.6-27B \
    --embed-url http://localhost:8013/v1 --embed-model Qwen3-Embedding-4B \
    --concurrency 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time


class _CountingHandler(logging.Handler):
    """Counts the loop's overflow-related log lines."""

    def __init__(self) -> None:
        super().__init__()
        self.escaped = 0  # "agent LLM call failed ... <context error>"
        self.retried = 0  # "hit context limit — compacted, retrying"

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage().lower()
        if "compacted, retrying" in msg:
            self.retried += 1
        elif "agent llm call failed" in msg and (
            "context length" in msg or "input_tokens" in msg
        ):
            self.escaped += 1


async def _run(args) -> None:
    from openai import AsyncOpenAI

    from synaptic.agent_loop import run_agent_loop
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.extensions.embedder import OpenAIEmbeddingProvider

    counter = _CountingHandler()
    lg = logging.getLogger("agent-loop")
    lg.setLevel(logging.INFO)
    lg.addHandler(counter)

    backend = SqliteGraphBackend(args.graph)
    await backend.connect()
    embedder = OpenAIEmbeddingProvider(api_base=args.embed_url, model=args.embed_model)
    client = AsyncOpenAI(base_url=args.llm_base_url, api_key="ignored")

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

    sem = asyncio.Semaphore(args.concurrency)

    async def score_one(q: dict) -> bool | None:
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
            )
        fids = set(res.found_ids)
        for fid in list(fids):
            fids |= docid_index.get(fid, set())
        if q.get("type") == "multi_hop":
            return relevant.issubset(fids)
        return bool(fids & relevant)

    t0 = time.time()
    scored = await asyncio.gather(*[score_one(q) for q in queries])
    scored = [s for s in scored if s is not None]
    solved = sum(1 for s in scored if s)

    print("\n=== overflow guard check ===", flush=True)
    print(f"escaped (hard 400s that broke a turn): {counter.escaped}", flush=True)
    print(f"retried (reactive compaction recovered): {counter.retried}", flush=True)
    print(f"solved: {solved}/{len(scored)} in {time.time()-t0:.0f}s", flush=True)
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
    args = p.parse_args()
    if not args.queries:
        args.queries = f"eval/data/queries/{args.dataset}.json"
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
