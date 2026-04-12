"""Multi-turn agent search — OpenAI GPT-4o-mini version.

Same concept as multi_turn_search.py but uses OpenAI's chat completions
API with tool_use instead of Anthropic's Messages API.

Usage::
    OPENAI_API_KEY=sk-... uv run python examples/multi_turn_openai.py --only h001
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.agent_tools import (
    count_tool,
    expand_tool,
    get_document_tool,
    list_categories_tool,
    search_tool,
)
from synaptic.agent_tools_v2 import compare_search_tool, deep_search_tool
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.search_session import SearchSession, build_graph_context

# OpenAI tool schemas (function calling format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "deep_search",
            "description": "RECOMMENDED. Search + expand + read documents in ONE call. Use category to narrow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string", "description": "Category filter"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_search",
            "description": "For 'A와 B' multi-topic queries. Auto-decomposes and searches in parallel.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Basic FTS + vector search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Read a full document. Use for absence proof or detail questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_categories",
            "description": "List all categories with document counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand",
            "description": "Get 1-hop neighbours of a node.",
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count",
            "description": "Count nodes by kind/category/year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "category": {"type": "string"},
                },
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a research agent with access to a knowledge graph.

## Preferred tools
- deep_search(query, category?) — BEST for most questions. One call = search + expand + read.
- compare_search(query) — for "A와 B" multi-topic questions.

## Strategy
1. Check graph metadata below to pick the right category.
2. Call deep_search with category filter.
3. If insufficient, rephrase with synonyms or try different category.
4. Maximum 3-4 tool calls. Be efficient.

## Rules
- Respond in Korean when the question is in Korean.
- Cite evidence (document names/IDs).
- Don't hallucinate — say "찾지 못했습니다" if not found.
"""

SCENARIOS = [
    {
        "id": "h001",
        "label": "패러프레이즈",
        "question": "말 복지 향상을 위한 프로그램이 뭐가 있어?",
    },
    {
        "id": "h004",
        "label": "교차 문서",
        "question": "인권경영 지침이 예산 편성에 어떻게 반영되나?",
    },
    {
        "id": "h014",
        "label": "대화체",
        "question": "올해 승마 체험 행사를 기획하려는데 작년에 어떻게 했는지 참고할 자료 있나요?",
    },
    {
        "id": "h015",
        "label": "패러프레이즈",
        "question": "우리 회사 윤리경영 점수가 어떻게 되는지 보고서 좀 찾아줘",
    },
    {"id": "exact", "label": "정확 매칭", "question": "인권영향평가 결과는 어떻게 나왔어?"},
]


async def dispatch(name: str, args: dict, backend, session) -> dict:
    if name == "deep_search":
        r = await deep_search_tool(
            backend,
            session,
            args["query"],
            limit=args.get("limit", 5),
            category=args.get("category"),
        )
    elif name == "compare_search":
        r = await compare_search_tool(backend, session, args["query"])
    elif name == "search":
        r = await search_tool(
            backend,
            session,
            args["query"],
            limit=args.get("limit", 10),
            category=args.get("category"),
        )
    elif name == "get_document":
        r = await get_document_tool(backend, session, args["doc_id"], query=args.get("query", ""))
    elif name == "list_categories":
        r = await list_categories_tool(backend, session)
    elif name == "expand":
        r = await expand_tool(backend, session, args["node_id"])
    elif name == "count":
        r = await count_tool(backend, session, kind=args.get("kind"), category=args.get("category"))
    else:
        return {"error": f"unknown tool: {name}"}
    return r.to_dict()


async def run_agent(question, *, backend, model="gpt-4o-mini", max_turns=8):
    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    session = SearchSession(budget_tool_calls=max_turns * 3)
    graph_ctx = await build_graph_context(backend)
    system = SYSTEM_PROMPT + "\n\n" + graph_ctx

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    tool_calls_total = 0
    t0 = time.time()

    for turn in range(max_turns):
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            max_tokens=4096,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"    → {fn}({', '.join(f'{k}={v}' for k, v in args.items())})")
                result = await dispatch(fn, args, backend, session)
                tool_calls_total += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False)[:5000],
                    }
                )
        else:
            elapsed = time.time() - t0
            answer = msg.content or ""
            print("------")
            print(f"Turns: {turn + 1}  |  Tool calls: {tool_calls_total}  |  {elapsed:.1f}s")
            print(f"\nA: {answer[:500]}")
            return {
                "turns": turn + 1,
                "tool_calls": tool_calls_total,
                "elapsed": elapsed,
                "answer": answer,
            }

    elapsed = time.time() - t0
    return {
        "turns": max_turns,
        "tool_calls": tool_calls_total,
        "elapsed": elapsed,
        "answer": "max_turns_exceeded",
    }


async def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None)
    p.add_argument("--graph", default=str(REPO_ROOT / "eval/data/krra_graph.sqlite"))
    p.add_argument("--model", default="gpt-4o-mini")
    args = p.parse_args()

    backend = SqliteGraphBackend(args.graph)
    await backend.connect()

    scenarios = SCENARIOS
    if args.only:
        scenarios = [s for s in SCENARIOS if s["id"] == args.only]

    for s in scenarios:
        print(f"\n{'=' * 60}")
        print(f"[{s['id']}] {s['label']}")
        print(f"Q: {s['question']}")
        print("-" * 60)
        await run_agent(s["question"], backend=backend, model=args.model)

    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
