"""``synaptic-agent-batch`` CLI — run ``graph.chat()`` over a batch of queries.

Loads a queries file, runs the multi-turn agent loop concurrently against a
BYO OpenAI-compatible LLM endpoint (vLLM / Ollama / OpenAI / Anthropic shim),
and writes one JSONL line per query as each finishes — so a long batch keeps
partial output if it is interrupted.

Usage::

    synaptic-agent-batch graph.sqlite \\
        --queries questions.json \\
        --llm-base-url http://localhost:8012/v1 \\
        --model Qwen3.6-27B \\
        --output answers.jsonl --concurrency 4

The queries file may be any of:

- a JSON list of strings: ``["q1", "q2"]``
- a JSON list of objects with a ``query`` field (other keys, e.g. ``id``,
  are echoed back): ``[{"id": "a", "query": "..."}]``
- a ``{"queries": [...]}`` wrapper (the eval ground-truth shape)
- a ``.jsonl`` file, one query (string or object) per line

Each output line:
``{"id", "query", "answer", "found_ids", "turns", "tool_calls",
"elapsed_ms", "error"}``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from synaptic import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="synaptic-agent-batch",
        description=(
            "Run the multi-turn agent loop (graph.chat) over a batch of "
            "queries against a BYO OpenAI-compatible LLM endpoint and write "
            "JSONL answers + found node ids."
        ),
    )
    p.add_argument("db", help="Path to the SQLite graph file (or :memory: for ephemeral)")
    p.add_argument(
        "-q",
        "--queries",
        type=Path,
        required=True,
        help="Queries file: JSON list of strings/objects, {'queries': [...]}, or .jsonl",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSONL here. Default: stdout.",
    )
    p.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible base URL (e.g. http://localhost:8012/v1). "
        "Omit to use the real OpenAI API (needs OPENAI_API_KEY).",
    )
    p.add_argument("--model", default="gpt-4o-mini", help="Model name forwarded to the LLM.")
    p.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM endpoint. Default: $OPENAI_API_KEY or 'ollama'.",
    )
    p.add_argument(
        "-c", "--concurrency", type=int, default=4, help="Max queries in flight (default 4)."
    )
    p.add_argument("--max-turns", type=int, default=5, help="Max LLM turns per query (default 5).")
    p.add_argument(
        "--embed-url",
        default=None,
        help="Optional OpenAI-compatible embedder base URL to enable vector search.",
    )
    p.add_argument("--embed-model", default="text-embedding-3-small", help="Embedder model name.")
    p.add_argument(
        "--no-prime",
        action="store_true",
        help="Disable the graph-snapshot priming injected into the system prompt.",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Only run the first N queries (smoke tests)."
    )
    p.add_argument("--version", action="version", version=f"synaptic-agent-batch {__version__}")
    return p


def _load_queries(path: Path) -> list[dict[str, Any]]:
    """Normalise the supported query-file shapes to ``[{id, query, **extra}]``."""

    def _norm(item: Any, idx: int) -> dict[str, Any] | None:
        if isinstance(item, str):
            return {"id": idx, "query": item}
        if isinstance(item, dict):
            q = item.get("query") or item.get("question") or item.get("text")
            if not q:
                return None
            out = dict(item)
            out["query"] = q
            out.setdefault("id", item.get("id", idx))
            return out
        return None

    raw = path.read_text(encoding="utf-8").strip()
    items: list[Any]
    if path.suffix == ".jsonl":
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        items = data.get("queries", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("queries file must be a list, a {'queries': [...]}, or .jsonl")

    out = [n for i, item in enumerate(items) if (n := _norm(item, i)) is not None]
    if not out:
        raise ValueError(f"no usable queries parsed from {path}")
    return out


async def _run(args: argparse.Namespace, sink: TextIO) -> int:
    import os

    from openai import AsyncOpenAI

    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.graph import SynapticGraph

    queries = _load_queries(args.queries)
    if args.limit is not None:
        queries = queries[: args.limit]
    total = len(queries)

    os.environ.setdefault("OPENAI_API_KEY", args.api_key or os.environ.get("OPENAI_API_KEY", "ollama"))
    client = (
        AsyncOpenAI(base_url=args.llm_base_url, api_key=args.api_key)
        if args.api_key
        else (AsyncOpenAI(base_url=args.llm_base_url) if args.llm_base_url else AsyncOpenAI())
    )

    embedder = None
    if args.embed_url:
        from synaptic.extensions.embedder import OpenAIEmbeddingProvider

        embedder = OpenAIEmbeddingProvider(api_base=args.embed_url, model=args.embed_model)

    backend = SqliteGraphBackend(args.db)
    await backend.connect()
    graph = SynapticGraph(backend, embedder=embedder)

    sem = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()
    done = 0
    failed = 0

    async def _one(item: dict[str, Any]) -> None:
        nonlocal done, failed
        record: dict[str, Any] = {"id": item.get("id"), "query": item["query"], "error": None}
        async with sem:
            try:
                r = await graph.chat(
                    item["query"],
                    llm_client=client,
                    model=args.model,
                    max_turns=args.max_turns,
                    prime_with_snapshot=not args.no_prime,
                )
                record.update(
                    answer=r.final_answer,
                    found_ids=sorted(r.found_ids),
                    turns=r.turns_used,
                    tool_calls=r.tool_calls_made,
                    elapsed_ms=round(r.elapsed_ms, 1),
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"[:300]
        async with write_lock:
            done += 1
            if record["error"] is not None:
                failed += 1
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            status = "✓" if record["error"] is None else "✗"
            print(
                f"[{done}/{total}] {status} {item['query'][:60]}",
                file=sys.stderr,
                flush=True,
            )

    try:
        await asyncio.gather(*(_one(item) for item in queries))
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            try:
                await close()
            except Exception:
                pass

    # Per-query errors are surfaced via each line's "error" field; the
    # process only exits non-zero on a total wipeout (e.g. LLM unreachable).
    return 1 if total and failed == total else 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not Path(args.db).exists() and args.db != ":memory:":
        print(f"error: graph file not found: {args.db}", file=sys.stderr)
        return 2
    if not args.queries.exists():
        print(f"error: queries file not found: {args.queries}", file=sys.stderr)
        return 2

    if args.output is None:
        return asyncio.run(_run(args, sys.stdout))
    with open(args.output, "w", encoding="utf-8") as fh:
        rc = asyncio.run(_run(args, fh))
    print(f"wrote {args.output}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
