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
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
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
- Preserve the original question's specific entities, attributes, and relation
  when rewriting. Do not retry with a vague one-word target if the original query
  contained more constraints.
"""


def _load_local_env(paths: list[Path] | None = None) -> None:
    """Load gitignored local env files without overriding shell env."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency is optional at runtime
        load_dotenv = None

    for path in paths or [REPO_ROOT / ".env", REPO_ROOT.parent / ".env"]:
        if load_dotenv is not None:
            load_dotenv(path, override=False)
            continue
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip("\"'")


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
    tool_sequence: list[str] = field(default_factory=list)
    search_targets: list[str] = field(default_factory=list)
    unique_tools: int = 0
    unique_search_targets: int = 0
    query_rewrites: int = 0


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


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        key = _normalize_text(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _tool_args_from_key(key: str) -> dict[str, Any]:
    _, sep, raw = str(key).partition(":")
    if not sep:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _search_target_from_call(tool: str, args: dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    if query:
        return query
    if tool == "get_document":
        return str(args.get("doc_id") or "").strip()
    if tool in {"expand", "follow"}:
        node_id = str(args.get("node_id") or "").strip()
        edge_kind = str(args.get("edge_kind") or "").strip()
        return " ".join(part for part in (node_id, edge_kind) if part)
    if tool == "filter_nodes":
        parts = [
            args.get("table"),
            args.get("property"),
            args.get("op"),
            args.get("value"),
        ]
        return " ".join(str(part) for part in parts if part)
    if tool == "aggregate_nodes":
        parts = [
            args.get("table"),
            args.get("group_by"),
            args.get("metric"),
            args.get("where_property"),
            args.get("where_op"),
            args.get("where_value"),
        ]
        return " ".join(str(part) for part in parts if part)
    if tool == "join_related":
        parts = [
            args.get("from_value"),
            args.get("fk_property"),
            args.get("target_table"),
        ]
        values = args.get("from_values")
        if isinstance(values, list):
            parts.insert(0, ",".join(str(item) for item in values[:5]))
        return " ".join(str(part) for part in parts if part)
    if tool == "top_nodes":
        parts = [
            args.get("table"),
            args.get("sort_by"),
            args.get("order"),
            args.get("where_property"),
            args.get("where_op"),
            args.get("where_value"),
        ]
        return " ".join(str(part) for part in parts if part)
    return ""


def _exploration_metrics(tool_log: list[dict[str, Any]], original_query: str) -> dict[str, object]:
    tool_sequence: list[str] = []
    targets: list[str] = []
    rewrite_queries: list[str] = []
    original = _normalize_text(original_query)
    for item in tool_log:
        tool = str(item.get("tool") or "").strip()
        if not tool:
            continue
        tool_sequence.append(tool)
        args = _tool_args_from_key(str(item.get("key") or ""))
        target = _search_target_from_call(tool, args)
        if target:
            targets.append(target)
        query = str(args.get("query") or "").strip()
        if query and _normalize_text(query) != original:
            rewrite_queries.append(query)
    unique_targets = _ordered_unique(targets)
    return {
        "tool_sequence": tool_sequence,
        "search_targets": unique_targets,
        "unique_tools": len(set(tool_sequence)),
        "unique_search_targets": len(unique_targets),
        "query_rewrites": len(_ordered_unique(rewrite_queries)),
    }


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
    unique_tools = [row.unique_tools for row in rows]
    unique_targets = [row.unique_search_targets for row in rows]
    query_rewrites = [row.query_rewrites for row in rows]
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
        "mean_unique_tools": sum(unique_tools) / n,
        "mean_unique_search_targets": sum(unique_targets) / n,
        "mean_query_rewrites": sum(query_rewrites) / n,
        "queries_with_multi_tool": sum(1 for row in rows if row.unique_tools > 1),
        "queries_with_query_rewrites": sum(1 for row in rows if row.query_rewrites > 0),
        "duplicate_tool_calls": sum(row.duplicate_tool_calls for row in rows),
        "empty_tool_calls": sum(row.empty_tool_calls for row in rows),
        "mean_first_relevant_turn": sum(reached_turns) / len(reached_turns)
        if reached_turns
        else 0.0,
        "mean_first_relevant_tool_calls": sum(reached_calls) / len(reached_calls)
        if reached_calls
        else 0.0,
    }


def _row_to_json(row: AgentLoopRow) -> str:
    return json.dumps(asdict(row), ensure_ascii=False, sort_keys=True)


def _row_from_dict(data: dict[str, Any]) -> AgentLoopRow:
    return AgentLoopRow(
        qid=str(data.get("qid") or ""),
        query=str(data.get("query") or ""),
        reached=bool(data.get("reached")),
        relevant_docs=[str(item) for item in data.get("relevant_docs", [])],
        found_relevant_docs=[str(item) for item in data.get("found_relevant_docs", [])],
        found_ids_count=int(data.get("found_ids_count") or 0),
        turns=int(data.get("turns") or 0),
        tool_calls=int(data.get("tool_calls") or 0),
        first_relevant_turn=int(data.get("first_relevant_turn") or 0),
        first_relevant_tool_calls=int(data.get("first_relevant_tool_calls") or 0),
        duplicate_tool_calls=int(data.get("duplicate_tool_calls") or 0),
        empty_tool_calls=int(data.get("empty_tool_calls") or 0),
        prompt_tokens=int(data.get("prompt_tokens") or 0),
        completion_tokens=int(data.get("completion_tokens") or 0),
        elapsed_sec=float(data.get("elapsed_sec") or 0.0),
        tool_sequence=[str(item) for item in data.get("tool_sequence", [])],
        search_targets=[str(item) for item in data.get("search_targets", [])],
        unique_tools=int(data.get("unique_tools") or 0),
        unique_search_targets=int(data.get("unique_search_targets") or 0),
        query_rewrites=int(data.get("query_rewrites") or 0),
    )


def _llm_preflight_error_message(base_url: str, model: str, exc: BaseException) -> str:
    return (
        "LLM endpoint preflight failed "
        f"(base_url={base_url!r}, model={model!r}): {type(exc).__name__}: {exc}. "
        "Check the tunnel/endpoint before running the benchmark, or pass "
        "--skip-preflight if the endpoint is reachable but does not implement /v1/models."
    )


async def _preflight_llm_endpoint(
    client: object,
    *,
    base_url: str,
    model: str,
    timeout_sec: float,
) -> None:
    try:
        await asyncio.wait_for(client.models.list(), timeout=timeout_sec)  # type: ignore[attr-defined]
    except Exception as exc:
        raise SystemExit(_llm_preflight_error_message(base_url, model, exc)) from exc


def _load_jsonl_rows(path: Path) -> list[AgentLoopRow]:
    if not path.exists():
        return []
    rows: list[AgentLoopRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(_row_from_dict(json.loads(line)))
    return rows


def _append_jsonl_row(path: Path, row: AgentLoopRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_row_to_json(row) + "\n")


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
    force_first_tool: bool,
    out_jsonl: Path | None,
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
        f"- Force first tool: {'yes' if force_first_tool else 'no'}",
        f"- Incremental JSONL: {_display_path(out_jsonl) if out_jsonl else 'disabled'}",
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
        f"- Mean unique tools: {summary['mean_unique_tools']:.2f}",
        f"- Mean unique search targets: {summary['mean_unique_search_targets']:.2f}",
        f"- Mean query rewrites: {summary['mean_query_rewrites']:.2f}",
        f"- Queries with >1 tool type: {summary['queries_with_multi_tool']}/{summary['queries']}",
        f"- Queries with query rewrites: {summary['queries_with_query_rewrites']}/{summary['queries']}",
        f"- Duplicate tool calls: {summary['duplicate_tool_calls']}",
        f"- Empty tool calls: {summary['empty_tool_calls']}",
        "",
        "## Per Query",
        "",
        "| QID | Reach | Turns | Calls | Tools | Targets | Rewrites | First Rel Turn | First Rel Calls | Found Relevant | Elapsed | Query |",
        "|-----|:-----:|------:|------:|------:|--------:|---------:|---------------:|----------------:|----------------|--------:|-------|",
    ]
    for row in rows:
        found = ", ".join(row.found_relevant_docs) if row.found_relevant_docs else "-"
        query = row.query.replace("|", "\\|")[:90]
        lines.append(
            f"| {row.qid} | {'yes' if row.reached else 'no'} | {row.turns} | "
            f"{row.tool_calls} | {row.unique_tools} | {row.unique_search_targets} | "
            f"{row.query_rewrites} | {row.first_relevant_turn or '-'} | "
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
    force_first_tool: bool,
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
        force_first_tool=force_first_tool,
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
    exploration = _exploration_metrics(result.tool_log, query)
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
        tool_sequence=list(exploration["tool_sequence"]),
        search_targets=list(exploration["search_targets"]),
        unique_tools=int(exploration["unique_tools"]),
        unique_search_targets=int(exploration["unique_search_targets"]),
        query_rewrites=int(exploration["query_rewrites"]),
    )


async def amain(argv: list[str] | None = None) -> int:
    _load_local_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco-path", type=Path, default=DEFAULT_MSMARCO_PATH)
    parser.add_argument("--sqlite-db-path", type=Path, required=True)
    parser.add_argument("--subset", type=int, default=20)
    parser.add_argument("--corpus-limit", type=int, default=0)
    parser.add_argument("--llm-base-url", default="http://localhost:8012/v1")
    parser.add_argument("--model", default="Qwen3.6-27B")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=60.0,
        help="Per-request LLM timeout in seconds.",
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=10.0,
        help="Timeout for the initial /v1/models endpoint check.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the initial /v1/models endpoint check.",
    )
    parser.add_argument("--no-sufficiency-gate", action="store_true")
    parser.add_argument(
        "--allow-zero-tool-answer",
        action="store_true",
        help="Allow the model to answer without using any retrieval tool first.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Append each completed query result to this JSONL file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load --out-jsonl and skip qids already present there.",
    )
    args = parser.parse_args(argv)

    if args.subset <= 0:
        raise SystemExit("--subset must be positive")
    if args.max_turns <= 0:
        raise SystemExit("--max-turns must be positive")
    if args.llm_timeout <= 0:
        raise SystemExit("--llm-timeout must be positive")
    if args.preflight_timeout <= 0:
        raise SystemExit("--preflight-timeout must be positive")
    if args.resume and args.out_jsonl is None:
        raise SystemExit("--resume requires --out-jsonl")
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
    client = AsyncOpenAI(base_url=args.llm_base_url, api_key=api_key, timeout=args.llm_timeout)
    if not args.skip_preflight:
        await _preflight_llm_endpoint(
            client,
            base_url=args.llm_base_url,
            model=args.model,
            timeout_sec=args.preflight_timeout,
        )

    backend = SqliteGraphBackend(str(args.sqlite_db_path))
    await backend.connect()
    rows: list[AgentLoopRow] = (
        _load_jsonl_rows(args.out_jsonl) if args.resume and args.out_jsonl else []
    )
    completed_qids = {row.qid for row in rows}
    try:
        for qid, query in query_items:
            if str(qid) in completed_qids:
                print(f"{qid:>8} skip=resume", flush=True)
                continue
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
                force_first_tool=not args.allow_zero_tool_answer,
            )
            rows.append(row)
            completed_qids.add(row.qid)
            if args.out_jsonl is not None:
                _append_jsonl_row(args.out_jsonl, row)
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
        force_first_tool=not args.allow_zero_tool_answer,
        out_jsonl=args.out_jsonl,
    )
    print(f"\nMarkdown report -> {_display_path(report_path)}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
