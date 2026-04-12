"""Multi-turn agent for assort (structured CSV data) — GPT-4o-mini.

Tests the structured data tools (filter_nodes, aggregate_nodes,
join_related) alongside text search tools on fashion e-commerce data.
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
    search_tool,
)
from synaptic.agent_tools_structured import (
    aggregate_nodes_tool,
    filter_nodes_tool,
    join_related_tool,
)
from synaptic.agent_tools_v2 import deep_search_tool
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.search_session import SearchSession

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "deep_search",
            "description": "Text search + expand + read in ONE call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_nodes",
            "description": "Filter by property value. Like SQL WHERE. Use for price ranges, dates, attribute values. Operators: >=, <=, >, <, ==, contains.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name: products, reviews, orders, broadcasts, product_variants, colors, sizes",
                    },
                    "property": {
                        "type": "string",
                        "description": "Property to filter on: selling_price, discount_rate, attribute_2_value, broadcast_date, season, etc.",
                    },
                    "op": {"type": "string", "description": "Operator: >=, <=, >, <, ==, contains"},
                    "value": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["property", "op", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_nodes",
            "description": "GROUP BY + COUNT/SUM/AVG. For questions like 'how many per category'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "group_by": {"type": "string"},
                    "metric": {
                        "type": "string",
                        "default": "count",
                        "description": "count, sum, avg, max, min",
                    },
                },
                "required": ["group_by"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "join_related",
            "description": "FK lookup. Find related records in another table. Like SQL JOIN.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_value": {"type": "string", "description": "The FK value to look up"},
                    "fk_property": {
                        "type": "string",
                        "description": "FK column name (e.g. product_code)",
                    },
                    "target_table": {"type": "string", "description": "Table to search in"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["from_value", "fk_property", "target_table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Basic text search. Use for product names, review text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a data analyst agent with access to a fashion e-commerce knowledge graph.

The graph contains these tables:
- products (80): product_code, product_name, season, selling_price, discount_rate, cumulative_sales
- product_variants (852): variant_code, product_code, color_id, size_id, assort_ratio
- reviews (640): review_id, product_code, attribute_2_value(핏), attribute_4_value(착용감), review_content
- orders (2472): order_id, product_code, delivery_date, age_group, member_gender
- broadcasts (290): broadcast_id, product_code, broadcast_date, pgm_time
- colors (10): color_id, color_name
- sizes (6): size_id, size_name
- sales_channels (6), sales_partners (5)

## Tool selection guide
- Price/date/attribute FILTER → use filter_nodes
- "how many per X" → use aggregate_nodes
- "reviews for product X" → use join_related
- Product name search → use deep_search or search

Respond in Korean. Be concise. Cite data.
"""

SCENARIOS = [
    {"id": "a003", "question": "가장 많이 팔린 상품의 리뷰를 보여줘"},
    {"id": "a005", "question": "9만원 이상 고가 상품 목록을 알려줘"},
    {"id": "a007", "question": "색상별 상품 변형 개수를 알려줘"},
    {"id": "a009", "question": "핏이 타이트하다는 불만이 있는 상품 리뷰를 찾아줘"},
    {"id": "a014", "question": "2024년 11월에 방송된 상품이 뭐야?"},
    {"id": "a011", "question": "엄마 생일 선물로 괜찮은 옷 추천해줘. 편하고 품질 좋은 걸로"},
]


async def dispatch(name, args, backend, session):
    if name == "deep_search":
        r = await deep_search_tool(backend, session, args["query"], limit=args.get("limit", 5))
    elif name == "filter_nodes":
        r = await filter_nodes_tool(
            backend,
            session,
            table=args.get("table", ""),
            property=args["property"],
            op=args["op"],
            value=args["value"],
            limit=args.get("limit", 20),
        )
    elif name == "aggregate_nodes":
        r = await aggregate_nodes_tool(
            backend,
            session,
            table=args.get("table", ""),
            group_by=args["group_by"],
            metric=args.get("metric", "count"),
        )
    elif name == "join_related":
        r = await join_related_tool(
            backend,
            session,
            from_value=args["from_value"],
            fk_property=args["fk_property"],
            target_table=args["target_table"],
            limit=args.get("limit", 10),
        )
    elif name == "search":
        r = await search_tool(backend, session, args["query"], limit=args.get("limit", 10))
    else:
        return {"error": f"unknown: {name}"}
    return r.to_dict()


async def run(question, backend, model="gpt-4o-mini"):
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    session = SearchSession(budget_tool_calls=30)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_calls_total = 0
    t0 = time.time()

    for turn in range(8):
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            max_tokens=2048,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"    → {fn}({', '.join(f'{k}={v}' for k, v in list(args.items())[:3])})")
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
            print(f"A: {answer[:400]}")
            return
    print(f"Turns: 8 (max)  |  Tool calls: {tool_calls_total}")


async def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None)
    p.add_argument("--graph", default=str(REPO_ROOT / "eval/data/assort_graph.sqlite"))
    args = p.parse_args()

    backend = SqliteGraphBackend(args.graph)
    await backend.connect()

    scenarios = SCENARIOS
    if args.only:
        scenarios = [s for s in SCENARIOS if s["id"] == args.only]

    for s in scenarios:
        print(f"\n{'=' * 60}")
        print(f"[{s['id']}] {s['question']}")
        print("-" * 60)
        await run(s["question"], backend)

    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
