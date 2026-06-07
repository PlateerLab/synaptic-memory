"""Honest RAG-vs-agent proof — ANSWER QUALITY, not retrieval reach.

The retrieval-reach comparison is budget-unfair (the agent accumulates dozens of
ids; single-shot returns top-k), so it can't fairly answer "does the agent beat
RAG". The product's real output is an ANSWER, so compare that:

  - naive-RAG : best single-shot retrieval (EvidenceSearch top-k) → stuff the
                evidence into the LLM → one answer. The classic retrieve-then-read
                RAG. No iteration.
  - agent     : run_agent_loop → final_answer. The multi-turn moat.

Both use the SAME corpus, SAME LLM, SAME judge; correctness is judged against the
dataset's gold ``answer`` (so this needs a dataset that HAS gold answers — e.g.
finreg_multihop). The judge and the agent are non-deterministic, so run
``--runs N`` and report the spread (project_v028_agent_bench_noise_floor).

This is the fair question: given one retrieval shot vs iterative tool use, who
produces the correct answer more often?

Usage:
  uv run python examples/ablation/rag_vs_agent_answer.py \
    --dataset finreg_multihop --graph eval/data/finreg_graph.sqlite \
    --llm-base-url http://localhost:8012/v1 --model Qwen3.6-27B \
    --embed-url http://localhost:8013/v1 --embed-model Qwen3-Embedding-4B \
    -k 10 --runs 3 --concurrency 6
"""

from __future__ import annotations

import argparse
import asyncio
import time

_JUDGE_SYSTEM = (
    "You grade a candidate answer against a reference answer for the same "
    "question. Reply with ONLY 'YES' if the candidate is correct and consistent "
    "with the reference (same key facts), or 'NO' otherwise. Ignore wording / "
    "verbosity differences; judge the facts."
)

_RAG_SYSTEM = (
    "Answer the question using ONLY the provided context. Be concise and factual. "
    "If the context lacks the answer, say what you can from it."
)


async def _run(args) -> None:
    import json

    from openai import AsyncOpenAI

    from synaptic.agent_loop import run_agent_loop
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.extensions.embedder import OpenAIEmbeddingProvider
    from synaptic.extensions.evidence_search import EvidenceSearch

    backend = SqliteGraphBackend(args.graph)
    await backend.connect()
    embedder = OpenAIEmbeddingProvider(api_base=args.embed_url, model=args.embed_model)
    client = AsyncOpenAI(base_url=args.llm_base_url, api_key="ignored")

    with open(args.queries, encoding="utf-8") as f:
        queries = [q for q in json.load(f).get("queries", []) if q.get("query") and q.get("answer")]
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("no queries with a gold `answer` field — pick a dataset that has them.")
        await backend.close()
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def _chat(messages, max_tokens=512) -> str:
        resp = await client.chat.completions.create(
            model=args.model, messages=messages, max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""

    async def judge(answer: str, gold: str, query: str) -> bool:
        if not answer.strip():
            return False
        out = await _chat(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"QUESTION:\n{query}\n\nREFERENCE:\n{gold}\n\nCANDIDATE:\n{answer}",
                },
            ],
            max_tokens=8,
        )
        return out.strip().upper().startswith("YES")

    async def naive_rag(q: dict) -> str:
        res = await EvidenceSearch(backend=backend, embedder=embedder).search(q["query"], k=args.k)
        ctx = "\n---\n".join((ev.node.content or ev.node.title or "")[:700] for ev in res.evidence[: args.k])
        return await _chat(
            [
                {"role": "system", "content": _RAG_SYSTEM},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q['query']}\nAnswer:"},
            ]
        )

    empty_agent = [0]

    async def agent(q: dict) -> str:
        res = await run_agent_loop(
            client=client,
            backend=backend,
            query=q["query"],
            model=args.model,
            max_turns=args.max_turns,
            embedder=embedder,
        )
        if not (res.final_answer or "").strip():
            empty_agent[0] += 1
        return res.final_answer

    async def score_arm(answer_fn) -> int:
        async def one(q):
            async with sem:
                ans = await answer_fn(q)
                return await judge(ans, q["answer"], q["query"])

        return sum(1 for ok in await asyncio.gather(*[one(q) for q in queries]) if ok)

    n = len(queries)
    print(f"\n=== RAG vs agent — ANSWER quality ({n} queries, {args.dataset}, k={args.k}) ===", flush=True)

    rag: list[int] = []
    ag: list[int] = []
    for r in range(args.runs):
        t = time.time()
        rr = await score_arm(naive_rag)
        rag.append(rr)
        print(f"naive-RAG run {r + 1}: {rr}/{n} ({rr / n:.3f})  [{time.time() - t:.0f}s]", flush=True)
        t = time.time()
        empty_agent[0] = 0
        ar = await score_arm(agent)
        ag.append(ar)
        print(
            f"agent     run {r + 1}: {ar}/{n} ({ar / n:.3f})  [{time.time() - t:.0f}s, "
            f"{empty_agent[0]} empty answers]",
            flush=True,
        )

    rm, am = sum(rag) / len(rag), sum(ag) / len(ag)
    print(f"\nnaive-RAG : mean {rm:.1f}/{n} ({rm / n:.3f}), range {min(rag)}-{max(rag)}", flush=True)
    print(f"agent     : mean {am:.1f}/{n} ({am / n:.3f}), range {min(ag)}-{max(ag)}", flush=True)
    print(f"→ agent minus naive-RAG: {am - rm:+.1f} ({(am - rm) / n:+.3f})", flush=True)
    await backend.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="finreg_multihop")
    p.add_argument("--graph", default="eval/data/finreg_graph.sqlite")
    p.add_argument("--queries", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("-k", type=int, default=10)
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
