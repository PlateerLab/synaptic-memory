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

``--out-jsonl`` writes one record per (query, arm, run) — the per-query data
the v0.29 routing GT (eval/routing_gt.py) and McNemar analysis are built on:
qid, query, arm, run, judge_correct, prompt/completion tokens (answer-arm cost
only, judge excluded), answer, and agent extras (turns, tool_calls, empty).

Usage:
  uv run python examples/ablation/rag_vs_agent_answer.py \
    --dataset finreg_multihop --graph eval/data/finreg_graph.sqlite \
    --llm-base-url http://localhost:8012/v1 --model Qwen3.6-27B \
    --embed-url http://localhost:8013/v1 --embed-model Qwen3-Embedding-4B \
    -k 10 --runs 3 --concurrency 6 \
    --out-jsonl examples/ablation/diagnostics/rag_vs_agent_perquery.jsonl
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

    def _usage_of(resp) -> tuple[int, int]:
        u = getattr(resp, "usage", None)
        if u is None:
            return (0, 0)
        return (
            int(getattr(u, "prompt_tokens", 0) or 0),
            int(getattr(u, "completion_tokens", 0) or 0),
        )

    async def _chat(messages, max_tokens=512, temperature=None) -> tuple[str, tuple[int, int]]:
        kwargs = {} if temperature is None else {"temperature": temperature}
        resp = await client.chat.completions.create(
            model=args.model, messages=messages, max_tokens=max_tokens, **kwargs
        )
        return resp.choices[0].message.content or "", _usage_of(resp)

    # judge at temperature=0 — the verdict is a gate input (routing GT), so
    # squeeze its noise; the ANSWER arms keep the server default for
    # comparability with the 2026-06-10 multirun.
    async def judge(answer: str, gold: str, query: str) -> bool:
        if not answer.strip():
            return False
        out, _ = await _chat(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"QUESTION:\n{query}\n\nREFERENCE:\n{gold}\n\nCANDIDATE:\n{answer}",
                },
            ],
            max_tokens=8,
            temperature=0.0,
        )
        return out.strip().upper().startswith("YES")

    async def naive_rag(q: dict) -> dict:
        res = await EvidenceSearch(backend=backend, embedder=embedder).search(q["query"], k=args.k)
        ctx = "\n---\n".join((ev.node.content or ev.node.title or "")[:700] for ev in res.evidence[: args.k])
        ans, (pt, ct) = await _chat(
            [
                {"role": "system", "content": _RAG_SYSTEM},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q['query']}\nAnswer:"},
            ]
        )
        return {"answer": ans, "prompt_tokens": pt, "completion_tokens": ct}

    async def agent(q: dict) -> dict:
        res = await run_agent_loop(
            client=client,
            backend=backend,
            query=q["query"],
            model=args.model,
            max_turns=args.max_turns,
            embedder=embedder,
        )
        return {
            "answer": res.final_answer,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "turns": res.turns_used,
            "tool_calls": res.tool_calls_made,
        }

    records: list[dict] = []

    async def score_arm(answer_fn, arm: str, run_idx: int) -> int:
        async def one(qi: int, q: dict) -> bool:
            async with sem:
                t0 = time.time()
                rec = await answer_fn(q)
                ok = await judge(rec["answer"], q["answer"], q["query"])
                records.append(
                    {
                        "qid": q.get("id") or f"q{qi:03d}",
                        "query": q["query"],
                        "arm": arm,
                        "run": run_idx + 1,
                        "judge_correct": ok,
                        "empty": not rec["answer"].strip(),
                        "elapsed_s": round(time.time() - t0, 1),
                        **rec,
                    }
                )
                return ok

        results = await asyncio.gather(*[one(qi, q) for qi, q in enumerate(queries)])
        return sum(1 for ok in results if ok)

    def _flush_jsonl() -> None:
        # rewrite after every arm — a multi-hour run that dies mid-way must
        # not lose the completed runs
        if not args.out_jsonl:
            return
        from pathlib import Path

        out = Path(args.out_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(queries)
    print(f"\n=== RAG vs agent — ANSWER quality ({n} queries, {args.dataset}, k={args.k}) ===", flush=True)

    rag: list[int] = []
    ag: list[int] = []
    for r in range(args.runs):
        t = time.time()
        rr = await score_arm(naive_rag, "rag", r)
        rag.append(rr)
        _flush_jsonl()
        print(f"naive-RAG run {r + 1}: {rr}/{n} ({rr / n:.3f})  [{time.time() - t:.0f}s]", flush=True)
        t = time.time()
        ar = await score_arm(agent, "agent", r)
        ag.append(ar)
        _flush_jsonl()
        empty = sum(1 for x in records if x["arm"] == "agent" and x["run"] == r + 1 and x["empty"])
        print(
            f"agent     run {r + 1}: {ar}/{n} ({ar / n:.3f})  [{time.time() - t:.0f}s, "
            f"{empty} empty answers]",
            flush=True,
        )

    rm, am = sum(rag) / len(rag), sum(ag) / len(ag)
    print(f"\nnaive-RAG : mean {rm:.1f}/{n} ({rm / n:.3f}), range {min(rag)}-{max(rag)}", flush=True)
    print(f"agent     : mean {am:.1f}/{n} ({am / n:.3f}), range {min(ag)}-{max(ag)}", flush=True)
    print(f"→ agent minus naive-RAG: {am - rm:+.1f} ({(am - rm) / n:+.3f})", flush=True)

    if args.out_jsonl:
        from pathlib import Path

        out = Path(args.out_jsonl)
        rtok = sum(r["prompt_tokens"] + r["completion_tokens"] for r in records if r["arm"] == "rag")
        atok = sum(r["prompt_tokens"] + r["completion_tokens"] for r in records if r["arm"] == "agent")
        print(
            f"per-query JSONL → {out}  ({len(records)} records; "
            f"tokens/query rag {rtok / max(1, n * args.runs):,.0f} vs agent {atok / max(1, n * args.runs):,.0f})",
            flush=True,
        )
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
    p.add_argument("--out-jsonl", default="", help="write one record per (query, arm, run)")
    args = p.parse_args()
    if not args.queries:
        args.queries = f"eval/data/queries/{args.dataset}.json"
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
