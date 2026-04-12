"""Multi-turn agent search demo — Claude Sonnet + synaptic agent tools.

Runs a real Claude Sonnet conversation where the LLM drives retrieval
through the v0.12 atomic tool layer. The agent is handed a natural-
language question and must call ``search`` / ``expand`` /
``get_document`` / ``count`` / ``list_categories`` / ``search_exact`` /
``follow`` tools as needed until it can answer.

Purpose:

- Prove that the tool layer works under a real LLM, not just unit tests.
- Show that "판단 로직은 코드가 아닌 LLM"이 실제로 가능하다.
- Log every tool call + final answer for 5 difficulty tiers so we can
  compare behaviour across query types later.

Usage::

    # ANTHROPIC_API_KEY must be set in the environment
    export ANTHROPIC_API_KEY=sk-ant-...

    uv run python examples/multi_turn_search.py

    # Run a single query by id
    uv run python examples/multi_turn_search.py --only absence

    # Use a different graph
    uv run python examples/multi_turn_search.py --graph my.sqlite

The demo loads the KRRA graph from ``eval/data/krra_graph.sqlite`` by
default. Build it first with::

    uv run python eval/scripts/ingest_krra.py --backend sqlite
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.agent_tools import (
    count_tool,
    expand_tool,
    follow_tool,
    get_document_tool,
    list_categories_tool,
    search_exact_tool,
    search_tool,
)
from synaptic.agent_tools_v2 import (
    compare_search_tool,
    deep_search_tool,
)
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.search_session import SearchSession, build_graph_context

# --- Anthropic tool schemas ---------------------------------------------------
#
# Mirror each synaptic agent tool to the Anthropic Messages API tool
# schema. Keep the descriptions imperative ("Use this when...") because
# that's what drives the LLM's tool-selection decisions.

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": (
            "Search the knowledge graph for relevant evidence about a query. "
            "This is the primary retrieval tool — use it first. Returns a list "
            "of evidence chunks with their parent document, category, and score. "
            "Repeat with refined queries or category filters when the first pass "
            "is incomplete."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query. Korean or English.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max evidence items (default 8).",
                    "default": 8,
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional category label filter, e.g. '규정 및 지침' or "
                        "'운영계획'. Use this to narrow a broad search."
                    ),
                },
                "kind": {
                    "type": "string",
                    "description": (
                        "Optional NodeKind filter, e.g. 'chunk', 'rule', 'decision', 'observation'."
                    ),
                },
                "exclude_seen": {
                    "type": "boolean",
                    "description": (
                        "When true (default), results already returned in "
                        "earlier turns are filtered out so you paginate."
                    ),
                    "default": True,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "expand",
        "description": (
            "Return 1-hop graph neighbours of a specific node. Use this after "
            "search finds a promising chunk or document to see surrounding "
            "chunks in the same document, sibling documents in the same "
            "category, and the next chunk in the sequence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Node id from a previous tool result.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max neighbours to return.",
                    "default": 10,
                },
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "get_document",
        "description": (
            "Fetch a full document and all of its chunks in reading order. "
            "Use this when you need to prove absence ('does this document "
            "really not contain X?') or when the top-k chunk view loses too "
            "much context. Essential for 'is there really no ...' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": (
                        "Document id — the 'document_id' field from a search "
                        "result, or the doc node's own id."
                    ),
                },
                "max_chunks": {
                    "type": "integer",
                    "description": "Safety fuse on chunk count (default 50).",
                    "default": 50,
                },
            },
            "required": ["doc_id"],
        },
    },
    {
        "name": "list_categories",
        "description": (
            "List all top-level categories in the knowledge graph with their "
            "document counts. Call this early to build a mental map of the "
            "corpus before searching, especially when the query is ambiguous "
            "or when you're doing a coverage check."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "count",
        "description": (
            "Count matching nodes without fetching them. Use this to decide "
            "whether an 'enumerate everything' question is even feasible — if "
            "the count is small you can iterate, if it's huge you need a "
            "narrower filter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": "Optional NodeKind filter.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category label filter.",
                },
                "year": {
                    "type": "integer",
                    "description": "Optional year filter (0 = no filter).",
                    "default": 0,
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_exact",
        "description": (
            "Literal substring match for codes, IDs, function names, Jira "
            "keys, section numbers. Use this when a normal search would "
            "dilute exact strings like 'E217' or 'SKU-1234'. Bypasses "
            "tokenisation entirely."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Exact string to search for.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max matches to return.",
                    "default": 20,
                },
            },
            "required": ["identifier"],
        },
    },
    {
        "name": "follow",
        "description": (
            "Walk one specific edge type from a starting node. Valid edge "
            "kinds: 'contains', 'part_of', 'next_chunk', 'mentions', "
            "'related', 'cites'. Use this as a surgical alternative to "
            "'expand' when you know the exact relation you need."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "edge_kind": {"type": "string"},
                "direction": {
                    "type": "string",
                    "description": "'outgoing', 'incoming', or 'both'.",
                    "default": "both",
                },
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["node_id", "edge_kind"],
        },
    },
    {
        "name": "deep_search",
        "description": (
            "RECOMMENDED for most questions. Searches, expands top hits, "
            "and reads relevant document chunks — all in ONE call. "
            "Returns evidence + expanded neighbours + document excerpts. "
            "Use this instead of calling search → expand → get_document "
            "separately. Set 'category' to narrow results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
                "category": {
                    "type": "string",
                    "description": "Category filter from graph metadata.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "compare_search",
        "description": (
            "For multi-topic questions like 'A와 B의 관계' or 'X 및 Y'. "
            "Automatically decomposes into sub-queries, searches each "
            "in parallel, and merges results. One call instead of 4-6."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
]


# --- Tool dispatcher ----------------------------------------------------------


async def dispatch_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    backend: SqliteGraphBackend,
    session: SearchSession,
) -> dict[str, Any]:
    """Map an Anthropic tool_use block to the matching synaptic tool call."""
    if tool_name == "search":
        result = await search_tool(
            backend,
            session,
            tool_input["query"],
            limit=int(tool_input.get("limit", 8)),
            category=tool_input.get("category") or None,
            kind=tool_input.get("kind") or None,
            exclude_seen=bool(tool_input.get("exclude_seen", True)),
        )
    elif tool_name == "expand":
        result = await expand_tool(
            backend,
            session,
            tool_input["node_id"],
            limit=int(tool_input.get("limit", 10)),
        )
    elif tool_name == "get_document":
        result = await get_document_tool(
            backend,
            session,
            tool_input["doc_id"],
            max_chunks=int(tool_input.get("max_chunks", 50)),
        )
    elif tool_name == "list_categories":
        result = await list_categories_tool(backend, session)
    elif tool_name == "count":
        year = int(tool_input.get("year", 0)) or None
        result = await count_tool(
            backend,
            session,
            kind=tool_input.get("kind") or None,
            category=tool_input.get("category") or None,
            year=year,
        )
    elif tool_name == "search_exact":
        result = await search_exact_tool(
            backend,
            session,
            tool_input["identifier"],
            limit=int(tool_input.get("limit", 20)),
        )
    elif tool_name == "follow":
        result = await follow_tool(
            backend,
            session,
            tool_input["node_id"],
            tool_input["edge_kind"],
            direction=tool_input.get("direction", "both"),
            limit=int(tool_input.get("limit", 20)),
        )
    elif tool_name == "deep_search":
        result = await deep_search_tool(
            backend,
            session,
            tool_input["query"],
            limit=int(tool_input.get("limit", 5)),
            category=tool_input.get("category") or None,
        )
    elif tool_name == "compare_search":
        result = await compare_search_tool(
            backend,
            session,
            tool_input["query"],
        )
    else:
        return {"ok": False, "error": f"unknown_tool: {tool_name}"}

    return result.to_dict()


# --- Agent loop ---------------------------------------------------------------


@dataclass
class AgentRun:
    """Full record of one agent conversation — used for the demo report."""

    query: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    final_answer: str = ""
    elapsed_seconds: float = 0.0
    error: str | None = None


SYSTEM_PROMPT = """\
You are a research agent with access to the Synaptic Memory knowledge graph.

The graph stores Korean public-sector documents organised as
Category → Document → Chunk. Your job is to answer the user's question
by iteratively calling the provided tools. You must not guess — every
claim in your final answer should be grounded in a tool result.

## Preferred tools (use these first)

- ``deep_search(query, category?)`` — BEST for most questions.
  Internally chains search → expand → get_document in ONE call.
  Returns evidence + neighbours + document excerpts.
- ``compare_search(query)`` — for multi-topic questions ("A와 B의 관계").
  Auto-decomposes and searches in parallel.

Only fall back to atomic tools (search, expand, get_document) when
deep_search doesn't return what you need.

## Strategy

1. Check the graph metadata below to pick the right category.
2. Call ``deep_search(query, category=...)`` with category filter.
3. If insufficient, try ``deep_search`` with rephrased query or
   different category.
4. For comparison/cross-document: use ``compare_search``.
5. Maximum 3 tool calls per question. Be efficient.

## Key principles
- ALWAYS try rephrasing before giving up. If "말 복지" returns nothing,
  try "승마", "힐링승마", "재활" — official document titles often use
  different terminology than casual questions.
- When searching in Korean, try both the full phrase AND individual
  keywords separately.
- Use `category` filter aggressively — it dramatically narrows the
  search space and often surfaces results that broad search misses.
- Stop after finding sufficient evidence. Don't over-explore.

## Answer format
- Give a direct answer first.
- Cite the evidence: which documents/chunks you used.
- If you couldn't find the answer after trying multiple approaches,
  say so plainly — do NOT hallucinate.
- Respond in Korean when the question is in Korean.
"""


async def run_agent(
    question: str,
    *,
    backend: SqliteGraphBackend,
    client,
    model: str,
    max_turns: int = 10,
    verbose: bool = True,
) -> AgentRun:
    """Run one multi-turn Claude conversation over the graph tools."""
    run = AgentRun(query=question)
    session = SearchSession(budget_tool_calls=max_turns * 3)

    # Inject graph metadata into system prompt so the agent doesn't
    # need to call list_categories (saves 1-2 turns per session)
    graph_ctx = await build_graph_context(backend)
    full_system = SYSTEM_PROMPT
    if graph_ctx:
        full_system += "\n\n" + graph_ctx

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": question},
    ]

    t0 = time.time()
    for turn in range(max_turns):
        run.turns = turn + 1
        # Retry on rate limit (429) and transient overload (529) with
        # exponential backoff. The Anthropic API enforces a per-minute
        # input-token quota; on the free tier that's 30K tokens/min,
        # so one tool-heavy call can saturate it. A 60-80s wait is
        # usually enough to unblock.
        response = None
        for attempt in range(4):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=full_system,
                    tools=TOOL_SCHEMAS,
                    messages=messages,
                )
                break
            except Exception as exc:
                msg = str(exc)
                is_rate = "429" in msg or "rate_limit" in msg
                is_overload = "529" in msg or "overloaded" in msg
                if (is_rate or is_overload) and attempt < 3:
                    wait = 20 * (attempt + 1)
                    if verbose:
                        print(f"    ⚠ {msg[:100]}… retry in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                run.error = f"api_error: {exc}"
                response = None
                break
        if response is None:
            break

        # Collect the content blocks for the assistant turn so we can
        # echo it back in the tool_result round trip.
        assistant_content = []
        tool_uses = []
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_uses.append(block)
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        # end_turn → assistant is done, extract answer
        if response.stop_reason == "end_turn":
            run.final_answer = "\n".join(text_parts)
            break

        # Append assistant turn
        messages.append({"role": "assistant", "content": assistant_content})

        # Execute every tool_use block and send results back
        if tool_uses:
            tool_results_content = []
            for tu in tool_uses:
                if verbose:
                    print(f"    → tool: {tu.name}({_compact(tu.input)})")
                result = await dispatch_tool(
                    tu.name,
                    dict(tu.input),
                    backend=backend,
                    session=session,
                )
                run.tool_calls.append(
                    {
                        "tool": tu.name,
                        "input": dict(tu.input),
                        "ok": result.get("ok", False),
                        "result_preview": _preview_result(result),
                    }
                )
                tool_results_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        # Cap result payload at 5K chars to stay under the
                        # per-minute input token quota on the free tier.
                        # Evidence previews already truncate to 240 chars so
                        # 5K is enough for ~15 hits with headroom.
                        "content": json.dumps(result, ensure_ascii=False)[:5000],
                    }
                )

            messages.append({"role": "user", "content": tool_results_content})
        else:
            # Assistant produced text without asking for a tool — treat as done
            run.final_answer = "\n".join(text_parts)
            break

    run.elapsed_seconds = time.time() - t0
    return run


def _compact(d: dict[str, Any]) -> str:
    """Short inline representation of tool input for console logging."""
    pieces = []
    for k, v in d.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + "…"
        pieces.append(f"{k}={v}")
    return ", ".join(pieces)


def _preview_result(result: dict[str, Any]) -> str:
    """One-line summary of a tool result for the run log."""
    if not result.get("ok"):
        return f"ERROR: {result.get('error', 'unknown')}"
    data = result.get("data", {})
    if "evidence" in data:
        return f"{len(data['evidence'])} evidence items"
    if "categories" in data:
        return f"{len(data['categories'])} categories"
    if "count" in data:
        return f"count={data['count']}"
    if "matches" in data:
        return f"{len(data['matches'])} exact matches"
    if "neighbours" in data:
        return f"{len(data['neighbours'])} neighbours"
    if "chunks" in data:
        return f"document with {len(data['chunks'])} chunks"
    return "ok"


# --- Scenarios ----------------------------------------------------------------


SCENARIOS = [
    {
        "id": "exact",
        "label": "정확 매칭",
        "question": "인권영향평가 결과는 어떻게 나왔어?",
    },
    {
        "id": "complex",
        "label": "복합 / 교차 문서",
        "question": (
            "경마산업 운영계획 수립에서 인권경영 지침이 어떻게 반영되는지 알려줘. 관련 근거도 같이."
        ),
    },
    {
        "id": "absence",
        "label": "부재 증명",
        "question": (
            "한국마사회 규정 중에 환불과 관련된 예외 조항이 있어? 없다면 없다고 단언해줘."
        ),
    },
    {
        "id": "exhaustive",
        "label": "전수 조회",
        "question": "한국마사회의 규정 및 지침 카테고리에 있는 문서가 총 몇 건인지 알려줘.",
    },
    {
        "id": "version",
        "label": "최신 버전",
        "question": "가장 최근 연도의 운영계획 문서 하나를 찾아서 요약해줘.",
    },
    # --- Hard queries (single-shot에서 실패하는 것들) ---
    {
        "id": "h001",
        "label": "패러프레이즈 (말 복지)",
        "question": "말 복지 향상을 위한 프로그램이 뭐가 있어?",
    },
    {
        "id": "h004",
        "label": "교차 문서 (인권+예산)",
        "question": "인권경영 지침이 예산 편성에 어떻게 반영되나?",
    },
    {
        "id": "h014",
        "label": "대화체 (승마 체험)",
        "question": "올해 승마 체험 행사를 기획하려는데 작년에 어떻게 했는지 참고할 자료 있나요?",
    },
    {
        "id": "h015",
        "label": "패러프레이즈 (윤리경영)",
        "question": "우리 회사 윤리경영 점수가 어떻게 되는지 보고서 좀 찾아줘",
    },
]


# --- Main ---------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "eval" / "data" / "krra_graph.sqlite",
        help="Path to the SQLite graph (default: eval/data/krra_graph.sqlite)",
    )
    p.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model id (default: claude-sonnet-4-6)",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Run a single scenario by id",
    )
    p.add_argument(
        "--skip",
        default=None,
        help="Comma-separated scenario ids to skip",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Max agent turns per scenario",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "eval" / "results" / "multi_turn_demo.json",
        help="Output JSON log",
    )
    return p.parse_args()


async def main() -> int:
    args = _parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment")
        return 1

    if not args.graph.exists():
        print(f"ERROR: graph not found: {args.graph}")
        print("Run: uv run python eval/scripts/ingest_krra.py --backend sqlite")
        return 1

    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        return 1

    scenarios = SCENARIOS
    if args.only:
        scenarios = [s for s in SCENARIOS if s["id"] == args.only]
        if not scenarios:
            print(f"ERROR: unknown scenario id '{args.only}'")
            return 1
    if args.skip:
        skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s["id"] not in skip_set]

    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)

    backend = SqliteGraphBackend(str(args.graph))
    await backend.connect()

    runs: list[AgentRun] = []
    try:
        for idx, scenario in enumerate(scenarios):
            if idx > 0:
                # Pause between scenarios so the per-minute token quota
                # can refill. 70s is slightly above the Anthropic
                # free-tier rate-limit window so we never race it.
                print("\n  … pausing 70s before next scenario to respect rate limit")
                await asyncio.sleep(70)

            print(f"\n{'=' * 70}")
            print(f"[{scenario['id']}] {scenario['label']}")
            print(f"Q: {scenario['question']}")
            print("-" * 70)
            run = await run_agent(
                scenario["question"],
                backend=backend,
                client=client,
                model=args.model,
                max_turns=args.max_turns,
            )
            runs.append(run)
            print("-" * 70)
            print(
                f"Turns: {run.turns}  |  Tool calls: {len(run.tool_calls)}  |  {run.elapsed_seconds:.1f}s"
            )
            if run.error:
                print(f"  ERROR: {run.error}")
            else:
                answer_preview = run.final_answer[:500]
                if len(run.final_answer) > 500:
                    answer_preview += "…"
                print(f"\nA: {answer_preview}")
    finally:
        await backend.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "scenario_id": s["id"],
                    "label": s["label"],
                    "question": s["question"],
                    "turns": r.turns,
                    "tool_calls": r.tool_calls,
                    "final_answer": r.final_answer,
                    "elapsed_seconds": r.elapsed_seconds,
                    "error": r.error,
                }
                for s, r in zip(scenarios, runs)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n✓ Run log → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
