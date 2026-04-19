"""Head-to-head RAG comparison driver.

Usage
-----
::

    # Synaptic only (no API keys, ~2s)
    python examples/benchmark_vs_competitors/run_comparison.py --only synaptic

    # Mem0 on a 10-query subset (sanity POC, ~$0.10 in OpenAI calls)
    export OPENAI_API_KEY=sk-...
    python examples/benchmark_vs_competitors/run_comparison.py \\
        --only synaptic,mem0 --subset 10

    # Everything installed, full 200 queries
    python examples/benchmark_vs_competitors/run_comparison.py

The results go to ``examples/benchmark_vs_competitors/results/``
as Markdown.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.benchmark_vs_competitors.protocol import (
    Corpus,
    RunResult,
    Timer,
    format_table,
    load_corpus,
    score_run,
)

RESULTS_DIR = Path(__file__).parent / "results"
BENCH_DATA = REPO_ROOT / "tests" / "benchmark" / "data"

DEFAULT_CORPORA = [
    ("Allganize RAG-ko", BENCH_DATA / "allganize_rag_ko.json"),
]


def _load_adapter(name: str):
    """Import adapters lazily — an uninstalled competitor shouldn't
    block running the rest."""
    if name == "synaptic":
        from examples.benchmark_vs_competitors.adapters.synaptic import SynapticAdapter

        return SynapticAdapter()
    if name == "mem0":
        from examples.benchmark_vs_competitors.adapters.mem0 import Mem0Adapter

        return Mem0Adapter()
    if name == "cognee":
        from examples.benchmark_vs_competitors.adapters.cognee import CogneeAdapter

        return CogneeAdapter()
    if name == "hipporag":
        from examples.benchmark_vs_competitors.adapters.hipporag import HippoRAG2Adapter

        return HippoRAG2Adapter()
    raise ValueError(f"Unknown adapter: {name}")


async def run_one(name: str, corpus: Corpus, k: int) -> RunResult:
    """Run a single adapter against a single corpus."""
    try:
        adapter = _load_adapter(name)
    except ImportError as exc:
        return RunResult(
            system=name,
            corpus=corpus.name,
            n_docs=len(corpus.docs),
            n_queries=len(corpus.queries),
            mrr=0.0,
            recall_at_k=0.0,
            precision_at_k=0.0,
            hit_count=0,
            build_sec=0.0,
            search_sec=0.0,
            k=k,
            error=f"not installed ({exc.msg if hasattr(exc, 'msg') else exc})",
        )

    try:
        with Timer() as build_t:
            await adapter.build(corpus)

        retrieved: list[list[str]] = []
        with Timer() as search_t:
            for query in corpus.queries:
                try:
                    hits = await adapter.search(query.text, k=k)
                except Exception as exc:
                    print(f"  [warn] {name} failed on qid={query.qid}: {exc}")
                    hits = []
                retrieved.append(hits)

        result = score_run(
            system=adapter.name,
            corpus=corpus,
            retrieved_per_query=retrieved,
            build_sec=build_t.elapsed,
            search_sec=search_t.elapsed,
            k=k,
        )
        return result
    except Exception as exc:
        traceback.print_exc()
        return RunResult(
            system=name,
            corpus=corpus.name,
            n_docs=len(corpus.docs),
            n_queries=len(corpus.queries),
            mrr=0.0,
            recall_at_k=0.0,
            precision_at_k=0.0,
            hit_count=0,
            build_sec=0.0,
            search_sec=0.0,
            k=k,
            error=str(exc),
        )
    finally:
        try:
            await adapter.close()
        except Exception:
            pass


def _write_markdown_report(results: list[RunResult], subset: int | None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"comparison_{stamp}.md"
    lines = [
        "# Synaptic head-to-head RAG comparison",
        "",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Subset: {subset if subset else 'full'} queries",
        "",
        "| System | Corpus | Docs | Queries | MRR | R@10 | Hit | Build | Search |",
        "|--------|--------|------|---------|-----|------|-----|-------|--------|",
    ]
    for r in results:
        if r.error:
            lines.append(f"| {r.system} | {r.corpus} | — | — | — | — | — | — | `{r.error}` |")
            continue
        lines.append(
            f"| {r.system} | {r.corpus} | {r.n_docs} | {r.n_queries} | "
            f"{r.mrr:.3f} | {r.recall_at_k:.3f} | {r.hit_count}/{r.n_queries} | "
            f"{r.build_sec:.1f}s | {r.search_sec:.1f}s |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Same corpus, same queries, same metrics. See "
        "`examples/benchmark_vs_competitors/README.md` for fairness caveats.",
        "- Build time includes any LLM-backed indexing (entity extraction, "
        "community summarization, etc.) — Synaptic performs none.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


async def amain(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        default="synaptic",
        help="comma-separated list of adapters to run (synaptic,mem0,cognee,hipporag)",
    )
    p.add_argument("--subset", type=int, default=None, help="first N queries only (for quick POC)")
    p.add_argument("--k", type=int, default=10, help="top-k for retrieval metrics")
    args = p.parse_args(argv)

    adapter_names = [n.strip() for n in args.only.split(",") if n.strip()]
    print(f"Running adapters: {adapter_names}")
    print(f"Subset: {args.subset or 'full'}, top-k: {args.k}")
    print()

    all_results: list[RunResult] = []
    for corpus_name, path in DEFAULT_CORPORA:
        if not path.exists():
            print(f"[skip] {corpus_name}: {path} not found")
            continue

        corpus = load_corpus(path, name=corpus_name)
        if args.subset is not None:
            corpus = Corpus(
                name=corpus.name,
                docs=corpus.docs,
                queries=corpus.queries[: args.subset],
            )
        print(f"Corpus: {corpus.name} — {len(corpus.docs)} docs, {len(corpus.queries)} queries")
        print()

        for name in adapter_names:
            print(f"▶ {name}")
            result = await run_one(name, corpus, args.k)
            all_results.append(result)
            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(
                    f"  MRR={result.mrr:.3f}  R@{args.k}={result.recall_at_k:.3f}  "
                    f"Hit={result.hit_count}/{result.n_queries}  "
                    f"build={result.build_sec:.1f}s  search={result.search_sec:.1f}s"
                )
            print()

    print(format_table(all_results))
    print()
    out = _write_markdown_report(all_results, args.subset)
    print(f"Markdown report → {out.relative_to(REPO_ROOT)}")
    return 0


def main() -> None:
    sys.exit(asyncio.run(amain(sys.argv[1:])))


if __name__ == "__main__":
    main()
