"""Vanilla RAG baseline — retrieve top-k chunks + single LLM answer.

This is the head-to-head reference point for the multi-turn agent
benchmark (``run_all.py --agent``). It deliberately does *nothing* the
agent does: no graph expansion, no rerank, no PRF, no tool loop. Just
what a textbook RAG system does —

    embed/FTS query -> top-k chunks -> one LLM call -> answer

Numbers are directly comparable to ``run_agent_benchmark``: same
corpora, same query files, same ``_llm_judge``. Run both and the gap
is the value of the agent loop.

Usage
-----
    # FTS retrieval (matches an agent run launched without --embed-url)
    uv run python eval/rag_baseline.py \
        --dataset "KRRA Conv" \
        --llm-base-url http://localhost:8012/v1 --model Qwen3.6-27B

    # dense retrieval
    uv run python eval/rag_baseline.py --dataset "KRRA Conv" \
        --embed-url http://localhost:11434/v1 \
        --llm-base-url http://localhost:8012/v1 --model Qwen3.6-27B
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from openai import AsyncOpenAI
from run_all import CUSTOM_DATASETS, _llm_judge

from synaptic.backends.sqlite_graph import SqliteGraphBackend

_ANSWER_PROMPT = """아래 문서 발췌만 근거로 질문에 답하라. \
발췌에 답의 근거가 없으면 "모름"이라고만 답하라. 간결하게 답하라.

질문: {query}

문서 발췌:
{context}

답변:"""


async def _retrieve(backend, query: str, embedder, k: int):
    """Top-k retrievable document units — dense if an embedder is given, else FTS.

    A retrievable unit is any content-bearing node carrying a ``doc_id``
    property (KRRA: CHUNK nodes; finreg: per-article ENTITY nodes).
    Category / concept nodes are excluded.
    """
    if embedder is not None:
        vec = await embedder.embed(query)
        if vec:
            nodes = await backend.search_vector(vec, limit=k * 5)
        else:
            nodes = await backend.search_fts(query, limit=k * 5)
    else:
        nodes = await backend.search_fts(query, limit=k * 5)
    units = [n for n in nodes if (n.properties or {}).get("doc_id")]
    return units[:k]


async def _answer(client, model: str, query: str, chunks: list) -> str:
    if not chunks:
        return ""
    context = "\n\n".join(
        f"[{i + 1}] {c.title}\n{(c.content or '')[:800]}" for i, c in enumerate(chunks)
    )
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": _ANSWER_PROMPT.format(query=query, context=context),
                }
            ],
            max_tokens=512,
            temperature=0.0,
            seed=42,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        print(f"    ⚠ API error: {exc}")
        return ""


async def run_rag_baseline(
    cfg,
    *,
    client,
    model: str,
    embedder=None,
    k: int = 5,
    judge: bool = True,
) -> tuple[int, int, float]:
    """Run vanilla RAG on one dataset. Returns (solved, total, elapsed)."""
    if not cfg.query_path or not cfg.query_path.exists() or not cfg.path.exists():
        print(f"  {cfg.name}: graph/queries missing — skip")
        return 0, 0, 0.0

    backend = SqliteGraphBackend(str(cfg.path))
    await backend.connect()

    with open(cfg.query_path, encoding="utf-8") as f:  # noqa: ASYNC230
        gt = json.load(f)
    queries = gt.get("queries", [])

    solved = 0
    total = 0
    t0 = time.time()
    print(f"  {cfg.name} (vanilla RAG, k={k})...")

    for q in queries:
        query_text = q.get("query", "")
        relevant = set(q.get("relevant_docs", []))
        if not query_text or not relevant:
            continue
        total += 1

        # Multi-hop is scored strict: every GT article must be retrieved,
        # and the lenient LLM-judge fallback is disabled — the whole point
        # is whether the cross-referenced provision was actually reached.
        is_multi = q.get("type") == "multi_hop"

        chunks = await _retrieve(backend, query_text, embedder, k)
        found = {(c.properties or {}).get("doc_id", "") for c in chunks}
        found.discard("")
        id_hit = relevant.issubset(found) if is_multi else bool(found & relevant)

        answer = await _answer(client, model, query_text, chunks)
        hit = id_hit
        tag = "id" if id_hit else "miss"
        if (
            not id_hit
            and not is_multi
            and judge
            and answer
            and await _llm_judge(client, query_text, answer, list(relevant), model=model)
        ):
            hit, tag = True, "judge"
        if hit:
            solved += 1
        print(f"      [{q.get('qid', '')}] retrieved={len(chunks)} hit={hit} ({tag})")

    await backend.close()
    elapsed = time.time() - t0
    print(f"  solved={solved}/{total} ({elapsed:.1f}s)")
    return solved, total, elapsed


async def main() -> None:
    p = argparse.ArgumentParser(description="Vanilla RAG baseline")
    p.add_argument(
        "--dataset",
        action="append",
        help="Dataset name (repeatable). Default: all custom datasets.",
    )
    p.add_argument("--llm-base-url", default=None, help="OpenAI-compatible LLM endpoint")
    p.add_argument("--model", default="gpt-4o-mini", help="Answer + judge model")
    p.add_argument("--embed-url", default=None, help="Embedding endpoint (dense RAG)")
    p.add_argument("--embed-model", default="qwen3-embedding:4b")
    p.add_argument("--k", type=int, default=5, help="Chunks fed to the LLM")
    p.add_argument("--no-judge", action="store_true", help="ID-match only, no LLM judge")
    args = p.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    client = AsyncOpenAI(base_url=args.llm_base_url) if args.llm_base_url else AsyncOpenAI()

    embedder = None
    if args.embed_url:
        from synaptic.extensions.embedder import OpenAIEmbeddingProvider

        embedder = OpenAIEmbeddingProvider(api_base=args.embed_url, model=args.embed_model)

    wanted = set(args.dataset) if args.dataset else None
    datasets = [c for c in CUSTOM_DATASETS if wanted is None or c.name in wanted]
    if not datasets:
        print(f"No matching datasets. Available: {[c.name for c in CUSTOM_DATASETS]}")
        return

    print(f"Vanilla RAG baseline — {len(datasets)} dataset(s)")
    print(f"  retrieval: {'dense' if embedder else 'FTS'}  model: {args.model}\n")

    rows: list[tuple[str, int, int, float]] = []
    for cfg in datasets:
        solved, total, elapsed = await run_rag_baseline(
            cfg,
            client=client,
            model=args.model,
            embedder=embedder,
            k=args.k,
            judge=not args.no_judge,
        )
        rows.append((cfg.name, solved, total, elapsed))

    print(f"\n{'Dataset':<22}{'RAG solved':>14}{'rate':>9}{'time':>10}")
    print("-" * 55)
    for name, solved, total, elapsed in rows:
        rate = f"{solved / total:.0%}" if total else "—"
        print(f"{name:<22}{f'{solved}/{total}':>14}{rate:>9}{f'{elapsed:.1f}s':>10}")


if __name__ == "__main__":
    asyncio.run(main())
