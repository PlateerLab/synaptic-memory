"""LLM-planned agent-loop retrieval benchmark.

This runner measures the existing ``run_agent_loop()`` path: the agent starts
from an initial search result, can change the follow-up query, switch tools, and
continue until it has enough evidence to answer.

Unlike ``run_tier1_benchmarks.py``, this is not a ranked top-10 retrieval test.
The agent loop returns an unordered set of evidence ids reached across turns, so
the primary metric is reach: did the loop find at least one relevant document,
and how many turns/tool calls/tokens did that take?
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synaptic.agent_loop import run_agent_loop
from synaptic.backends.sqlite_graph import SqliteGraphBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "diagnostics"
DEFAULT_MSMARCO_PATH = REPO_ROOT / "tests" / "benchmark" / "data" / "msmarco_passage.json"

_AGENT_LOOP_EXTRA_CONTEXT = """Benchmark context:
- You are evaluating retrieval, not general knowledge.
- Start from deep_search/search results, inspect evidence, and if the first
  evidence is insufficient, change the follow-up query or search target based on
  the snippets you saw.
- Prefer evidence found through tools over prior knowledge.
"""


@dataclass(slots=True)
class AgentLoopRow:
    qid: str
    query: str
    reached: bool
    relevant_docs: list[str]
    found_relevant_docs: list[str]
    found_ids_count: int
    turns: int
    tool_calls: int
    first_relevant_turn: int
    first_relevant_tool_calls: int
    duplicate_tool_calls: int
    empty_tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    elapsed_sec: float


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _found_relevant(found_ids: set[str], relevant: set[str]) -> list[str]:
    return sorted(str(item) for item in found_ids if str(item) in relevant)


def _first_relevant_trace_hit(
    trace: list[dict[str, Any]],
    relevant: set[str],
    *,
    final_found_ids: set[str] | None = None,
    final_turn: int = 0,
    final_tool_calls: int = 0,
) -> tuple[int, int]:
    for item in trace:
        found = {str(value) for value in item.get("found_ids", set())}
        if found & relevant:
            return int(item.get("turn") or 0), int(item.get("tool_calls") or 0)
    if final_found_ids and {str(value) for value in final_found_ids} & relevant:
        return int(final_turn), int(final_tool_calls)
    return 0, 0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def _summarize(rows: list[AgentLoopRow]) -> dict[str, object]:
    n = max(len(rows), 1)
    reached = sum(1 for row in rows if row.reached)
    elapsed = [row.elapsed_sec for row in rows]
    tool_calls = [row.tool_calls for row in rows]
    turns = [row.turns for row in rows]
    prompt_tokens = [row.prompt_tokens for row in rows]
    completion_tokens = [row.completion_tokens for row in rows]
    reached_turns = [row.first_relevant_turn for row in rows if row.first_relevant_turn > 0]
    reached_calls = [
        row.first_relevant_tool_calls for row in rows if row.first_relevant_tool_calls > 0
    ]
    return {
        "queries": len(rows),
        "reach": reached,
        "reach_rate": reached / n,
        "mean_turns": sum(turns) / n,
        "mean_tool_calls": sum(tool_calls) / n,
        "mean_elapsed_sec": sum(elapsed) / n,
        "p50_elapsed_sec": statistics.median(elapsed) if elapsed else 0.0,
        "p90_elapsed_sec": _percentile(elapsed, 0.90),
        "mean_prompt_tokens": sum(prompt_tokens) / n,
        "mean_completion_tokens": sum(completion_tokens) / n,
        "duplicate_tool_calls": sum(row.duplicate_tool_calls for row in rows),
        "empty_tool_calls": sum(row.empty_tool_calls for row in rows),
        "mean_first_relevant_turn": sum(reached_turns) / len(reached_turns)
        if reached_turns
        else 0.0,
        "mean_first_relevant_tool_calls": sum(reached_calls) / len(reached_calls)
        if reached_calls
        else 0.0,
    }


def _emit_markdown(
    rows: list[AgentLoopRow],
    *,
    dataset_path: Path,
    sqlite_db_path: Path,
    subset: int,
    corpus_limit: int,
    llm_base_url: str,
    model: str,
    max_turns: int,
    sufficiency_gate: bool,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"agent_loop_{stamp}.md"
    summary = _summarize(rows)
    lines = [
        "# Agent Loop Retrieval Benchmark — Synaptic",
        "",
        f"- Run at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Dataset path: {_display_path(dataset_path)}",
        f"- SQLite DB path: {_display_path(sqlite_db_path)}",
        f"- Subset: {subset}",
        f"- Corpus limit: {corpus_limit}",
        f"- LLM base URL: {llm_base_url}",
        f"- Model: {model}",
        f"- Max turns: {max_turns}",
        f"- Sufficiency gate: {'yes' if sufficiency_gate else 'no'}",
        "- SQLite FTS AND-first threshold: "
        f"{os.environ.get('SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD', '').strip() or '0'}",
        "- SQLite FTS lexical rerank pool: "
        f"{os.environ.get('SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL', '').strip() or '0'}",
        "",
        "This measures LLM-planned exploration. The agent can change follow-up "
        "queries and tool choices based on evidence from earlier turns. The main "
        "metric is document reach, not ranked MRR, because the agent loop returns "
        "a cumulative evidence set.",
        "",
        "## Summary",
        "",
        f"- Reach: {summary['reach']}/{summary['queries']} ({summary['reach_rate']:.3f})",
        f"- Mean turns: {summary['mean_turns']:.2f}",
        f"- Mean tool calls: {summary['mean_tool_calls']:.2f}",
        f"- Mean first relevant turn: {summary['mean_first_relevant_turn']:.2f}",
        f"- Mean first relevant tool calls: {summary['mean_first_relevant_tool_calls']:.2f}",
        f"- Mean elapsed: {summary['mean_elapsed_sec']:.1f}s",
        f"- P50/P90 elapsed: {summary['p50_elapsed_sec']:.1f}s / {summary['p90_elapsed_sec']:.1f}s",
        f"- Mean prompt tokens: {summary['mean_prompt_tokens']:.0f}",
        f"- Mean completion tokens: {summary['mean_completion_tokens']:.0f}",
        f"- Duplicate tool calls: {summary['duplicate_tool_calls']}",
        f"- Empty tool calls: {summary['empty_tool_calls']}",
        "",
        "## Per Query",
        "",
        "| QID | Reach | Turns | Calls | First Rel Turn | First Rel Calls | Found Relevant | Elapsed | Query |",
        "|-----|:-----:|------:|------:|---------------:|----------------:|----------------|--------:|-------|",
    ]
    for row in rows:
        found = ", ".join(row.found_relevant_docs) if row.found_relevant_docs else "-"
        query = row.query.replace("|", "\\|")[:90]
        lines.append(
            f"| {row.qid} | {'yes' if row.reached else 'no'} | {row.turns} | "
            f"{row.tool_calls} | {row.first_relevant_turn or '-'} | "
            f"{row.first_relevant_tool_calls or '-'} | {found} | "
            f"{row.elapsed_sec:.1f}s | {query} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def _run_one(
    *,
    client: object,
    backend: SqliteGraphBackend,
    qid: str,
    query: str,
    relevant: set[str],
    model: str,
    max_turns: int,
    sufficiency_gate: bool,
) -> AgentLoopRow:
    start = time.perf_counter()
    result = await run_agent_loop(
        client=client,
        backend=backend,
        query=query,
        model=model,
        max_turns=max_turns,
        extra_context=_AGENT_LOOP_EXTRA_CONTEXT,
        sufficiency_gate=sufficiency_gate,
        record_trace=True,
    )
    elapsed = time.perf_counter() - start
    found = {str(item) for item in result.found_ids}
    found_relevant = _found_relevant(found, relevant)
    first_turn, first_calls = _first_relevant_trace_hit(
        result.trace,
        relevant,
        final_found_ids=found,
        final_turn=result.turns_used,
        final_tool_calls=result.tool_calls_made,
    )
    duplicate_calls = sum(1 for item in result.tool_log if item.get("duplicate"))
    empty_calls = sum(1 for item in result.tool_log if int(item.get("n_results") or 0) == 0)
    return AgentLoopRow(
        qid=qid,
        query=query,
        reached=bool(found_relevant),
        relevant_docs=sorted(relevant),
        found_relevant_docs=found_relevant,
        found_ids_count=len(found),
        turns=result.turns_used,
        tool_calls=result.tool_calls_made,
        first_relevant_turn=first_turn,
        first_relevant_tool_calls=first_calls,
        duplicate_tool_calls=duplicate_calls,
        empty_tool_calls=empty_calls,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        elapsed_sec=elapsed,
    )


async def amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco-path", type=Path, default=DEFAULT_MSMARCO_PATH)
    parser.add_argument("--sqlite-db-path", type=Path, required=True)
    parser.add_argument("--subset", type=int, default=20)
    parser.add_argument("--corpus-limit", type=int, default=0)
    parser.add_argument("--llm-base-url", default="http://localhost:8012/v1")
    parser.add_argument("--model", default="Qwen3.6-27B")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--no-sufficiency-gate", action="store_true")
    args = parser.parse_args(argv)

    if args.subset <= 0:
        raise SystemExit("--subset must be positive")
    if args.max_turns <= 0:
        raise SystemExit("--max-turns must be positive")
    if not args.msmarco_path.exists():
        raise SystemExit(f"{args.msmarco_path} does not exist")
    if not args.sqlite_db_path.exists():
        raise SystemExit(f"{args.sqlite_db_path} does not exist")

    from openai import AsyncOpenAI

    data = json.loads(args.msmarco_path.read_text(encoding="utf-8"))
    query_items = list(data["queries"].items())[: args.subset]
    qrels = data["qrels"]
    n_docs = args.corpus_limit or int(data.get("corpus_size") or 0)
    api_key = os.environ.get(args.api_key_env) or ""
    local_endpoint = any(marker in args.llm_base_url for marker in ("localhost", "127.0.0.1"))
    if not api_key and not local_endpoint:
        raise SystemExit(f"{args.api_key_env} is not set; refusing to call remote LLM endpoint")
    if not api_key:
        api_key = "ignored"
    client = AsyncOpenAI(base_url=args.llm_base_url, api_key=api_key)

    backend = SqliteGraphBackend(str(args.sqlite_db_path))
    await backend.connect()
    rows: list[AgentLoopRow] = []
    try:
        for qid, query in query_items:
            rel = qrels.get(qid, {})
            relevant = set(map(str, rel.keys())) if isinstance(rel, dict) else set(map(str, rel))
            if not relevant:
                continue
            row = await _run_one(
                client=client,
                backend=backend,
                qid=str(qid),
                query=str(query),
                relevant=relevant,
                model=args.model,
                max_turns=args.max_turns,
                sufficiency_gate=not args.no_sufficiency_gate,
            )
            rows.append(row)
            found = ",".join(row.found_relevant_docs) if row.found_relevant_docs else "-"
            print(
                f"{row.qid:>8} reach={'yes' if row.reached else 'no ':>3} "
                f"turns={row.turns} calls={row.tool_calls} "
                f"first={row.first_relevant_turn or '-'}/{row.first_relevant_tool_calls or '-'} "
                f"found={found} elapsed={row.elapsed_sec:.1f}s",
                flush=True,
            )
    finally:
        await backend.close()

    report_path = _emit_markdown(
        rows,
        dataset_path=args.msmarco_path,
        sqlite_db_path=args.sqlite_db_path,
        subset=args.subset,
        corpus_limit=n_docs,
        llm_base_url=args.llm_base_url,
        model=args.model,
        max_turns=args.max_turns,
        sufficiency_gate=not args.no_sufficiency_gate,
    )
    print(f"\nMarkdown report -> {_display_path(report_path)}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
