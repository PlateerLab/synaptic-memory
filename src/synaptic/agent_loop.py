"""Multi-turn agent loop — Synaptic's measured-strongest retrieval mode.

Promoted from ``eval/run_all.py`` to a public module so applications can
use the same agent-loop machinery the benchmarks measure (custom Hard/
Conv: ``81 %`` mean solved across 6 benches with Qwen3.5-27B vLLM,
versus 0.30 mean MRR for single-shot retrieval on the same questions).

Why a separate module
---------------------
``graph.search()`` is a single-shot retrieval call that returns
candidates ranked by the EvidenceSearch pipeline. For Hard / Conv
queries — where a user asks several things at once, or the answer
requires combining facts from multiple nodes — single-shot caps out
at MRR ≈ 0.4. The agent loop turns retrieval into a *dialogue*: the
LLM picks a tool (search / expand / filter / etc.), reads the result,
decides what to do next. Measured outcome on the same corpora:
single-shot 0.000 → agent 91 % (assort Hard), 0.379 → 100 % (X2BEE
Hard).

This module exposes the loop as ``run_agent_loop`` and as a
:meth:`SynapticGraph.chat` convenience method, with the same
36 MCP tools the agent benchmark already uses (``deep_search``,
``compare_search``, ``filter_nodes``, ``aggregate_nodes``,
``join_related``, ``get_document``, ``expand``, ``follow``,
``search``).

Client requirements
-------------------
``client`` is any object exposing OpenAI-compatible
``chat.completions.create(model, messages, tools, max_tokens)``. Tested
with ``openai.AsyncOpenAI``; works with vLLM-OpenAI compatibility
endpoints (Qwen, Llama-style) and any other shim that implements the
same surface.

Usage::

    from openai import AsyncOpenAI
    from synaptic import SynapticGraph

    graph = await SynapticGraph.from_data("./my_corpus/")
    client = AsyncOpenAI(base_url="http://localhost:8012/v1", api_key="ollama")
    result = await graph.chat(
        "어떤 상품이 가장 인기인가?",
        llm_client=client,
        model="Qwen3.5-27b",
        max_turns=5,
    )
    print(result.final_answer)
    for n in result.nodes:
        print(n.title)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synaptic.extensions.embedder import EmbeddingProvider
    from synaptic.models import Node
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("agent-loop")


# --- System prompt + tool schema -----------------------------------


AGENT_SYSTEM = """\
You are a research agent answering a user question by calling tools
that explore a knowledge graph.

Strategy
========
1. Start with ``deep_search`` for natural-language questions (it does
   search + expand + read in one call).
2. For structured queries (filter / aggregate / join / count) use the
   structured tools — they're cheaper and exact.
3. For multi-hop questions, chain tool calls: search → expand → search
   again with the new entity name.
4. Emit independent probes in parallel within a single turn — the
   tool-call protocol supports it and one round-trip is cheaper than
   five. A compound question ("X의 Y 와 Z") should plan the 2-3 calls
   it needs and send them together.
5. Stop calling tools as soon as you have enough evidence. Output a
   short final answer in the user's language.

Available tools
===============
- ``deep_search(query, category?)`` — first-pass for any free-text question
- ``search(query)`` — basic FTS, returns top candidates
- ``expand(node_id)`` — neighbours of a node (1-hop)
- ``follow(node_id, edge_kind)`` — typed-edge traversal
- ``get_document(doc_id, query?)`` — full document chunks
- ``filter_nodes(table, property, op, value, ...)`` — exact attribute filter
- ``aggregate_nodes(table, group_by, metric, ...)`` — count / sum / avg
- ``top_nodes(table, sort_by, order="desc", limit=5, ...)`` — top/bottom rows by a column in a single call (use this for "가장 X한", "top N", "최대/최소", "최근" questions — no need to compose aggregate_nodes).
- ``join_related(from_value, fk_property, target_table, ...)`` — FK join

Tips
====
- If a search returns 0 results in Korean, try the English term — corpora
  often mix scripts (e.g. "치즈" → 0, "cheese" → many).
- For "top N" / "most" / "least" questions, use ``top_nodes(table,
  sort_by, order, limit)`` — one tool call returns the ranked rows
  complete with node_title and ID, ready to chain into ``join_related``
  or ``get_document``. ``aggregate_nodes`` is still the right answer
  for "per-X counts" / bucketed summaries.
- Prefer concrete IDs over titles when chaining tools.
- **Relative time references** ("올해" / "내년도" / "this year" / "next year"):
  do NOT inject a specific year number into the search query — the corpus
  may span multiple years, and a literal "2024" filter throws away all
  other matches. Search the topic without the year first; only narrow by
  year if the topic search returns too many candidates AND you have evidence
  the user wants a specific year.
- **"X 관련 자료 / 내용 / 정보" type questions** ask for *multiple* sources.
  After the first ``deep_search`` returns a few hits, do at least one more
  ``search`` with paraphrased keywords before concluding. A single document
  is rarely the complete answer to such a request.
- **When a tool returns 0 results, it also returns a ``hints`` array.**
  Each hint is a concrete corrective action (different operator, dropped
  WHERE, alternative column). Read the hints and follow the first one
  before reissuing a near-identical query — that is what wastes turns."""


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "deep_search",
            "description": "Search + expand + read in ONE call. First-pass for free-text questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Basic text search. Returns top candidate nodes.",
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
            "name": "expand",
            "description": "Get neighbours of a node (1-hop graph expand).",
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
            "name": "follow",
            "description": "Follow a typed edge from a source node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "edge_kind": {"type": "string"},
                },
                "required": ["node_id", "edge_kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Fetch the full chunks of a document.",
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
            "name": "filter_nodes",
            "description": "Filter rows by attribute. Supports multi-hop via from_ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "property": {"type": "string"},
                    "op": {
                        "type": "string",
                        "description": ">=, <=, >, <, ==, !=, contains, starts_with, date_range",
                    },
                    "value": {"type": "string"},
                    "limit": {"type": "integer"},
                    "from_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["property", "op", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_nodes",
            "description": "Group + metric (count/sum/avg/min/max) over a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "group_by": {"type": "string"},
                    "metric": {"type": "string"},
                    "metric_property": {"type": "string"},
                    "where_property": {"type": "string"},
                    "where_op": {"type": "string"},
                    "where_value": {"type": "string"},
                    "group_by_format": {
                        "type": "string",
                        "description": "Date bucket format: 'YYYY', 'YYYY-MM', 'YYYY-MM-DD'.",
                    },
                    "limit": {"type": "integer"},
                    "from_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["group_by"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "join_related",
            "description": "FK join across two tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_value": {"type": "string"},
                    "from_values": {"type": "array", "items": {"type": "string"}},
                    "fk_property": {"type": "string"},
                    "target_table": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["fk_property", "target_table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_nodes",
            "description": (
                "Top-N rows of a table ordered by a column. Single call for "
                "'가장 X한', 'top 5', '최대/최소', '최근' questions — no need to "
                "compose aggregate_nodes + post-sort."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "sort_by": {"type": "string"},
                    "order": {
                        "type": "string",
                        "description": "'desc' (default) or 'asc'",
                    },
                    "limit": {"type": "integer"},
                    "where_property": {"type": "string"},
                    "where_op": {"type": "string"},
                    "where_value": {"type": "string"},
                    "from_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional — restrict ranking to these node_titles (multi-hop chaining)",
                    },
                },
                "required": ["table", "sort_by"],
            },
        },
    },
]


# --- Result type ---------------------------------------------------


@dataclass(slots=True)
class AgentSearchResult:
    """Final output of one ``run_agent_loop`` call.

    Attributes:
        query: The user question we ran on.
        final_answer: The LLM's final text answer (empty if the loop
            exhausted ``max_turns`` without a non-tool message).
        found_ids: All node identifiers (doc_ids, titles) extracted
            from the agent's tool-call traces. Use this for ID-match
            evaluation against ground truth.
        nodes: Node objects fetched from ``found_ids`` (best-effort —
            populated only if the backend can resolve them).
        turns_used: How many LLM turns were consumed (≤ ``max_turns``).
        tool_calls_made: Total number of tool invocations across all turns.
        elapsed_ms: Wall-clock time of the whole loop.
    """

    query: str
    final_answer: str = ""
    found_ids: set[str] = field(default_factory=set)
    nodes: list[Node] = field(default_factory=list)
    turns_used: int = 0
    tool_calls_made: int = 0
    elapsed_ms: float = 0.0


# --- Internals -----------------------------------------------------


async def _dispatch_tool(
    name: str,
    args: dict,
    backend: StorageBackend,
    session,
    *,
    embedder: EmbeddingProvider | None = None,
) -> dict:
    """Route an LLM-emitted tool call to the actual Synaptic tool."""
    from synaptic.agent_tools import (
        expand_tool,
        follow_tool,
        get_document_tool,
        search_tool,
    )
    from synaptic.agent_tools_structured import (
        aggregate_nodes_tool,
        filter_nodes_tool,
        join_related_tool,
        top_nodes_tool,
    )
    from synaptic.agent_tools_v2 import deep_search_tool

    try:
        if name == "deep_search":
            r = await deep_search_tool(
                backend,
                session,
                args.get("query", ""),
                category=args.get("category"),
                embedder=embedder,
            )
        elif name == "search":
            r = await search_tool(backend, session, args.get("query", ""), embedder=embedder)
        elif name == "expand":
            r = await expand_tool(backend, session, args.get("node_id", ""))
        elif name == "follow":
            r = await follow_tool(
                backend,
                session,
                args.get("node_id", ""),
                args.get("edge_kind", "related"),
            )
        elif name == "get_document":
            r = await get_document_tool(
                backend,
                session,
                args.get("doc_id", ""),
                query=args.get("query", ""),
            )
        elif name == "filter_nodes":
            r = await filter_nodes_tool(
                backend,
                session,
                table=args.get("table", ""),
                property=args.get("property", ""),
                op=args.get("op", "contains"),
                value=args.get("value", ""),
                limit=int(args.get("limit", 20)),
                from_ids=args.get("from_ids") or None,
            )
        elif name == "aggregate_nodes":
            r = await aggregate_nodes_tool(
                backend,
                session,
                table=args.get("table", ""),
                group_by=args.get("group_by", ""),
                metric=args.get("metric", "count"),
                metric_property=args.get("metric_property", ""),
                where_property=args.get("where_property", ""),
                where_op=args.get("where_op", ""),
                where_value=args.get("where_value", ""),
                group_by_format=args.get("group_by_format", ""),
                limit=int(args.get("limit", 50)),
                from_ids=args.get("from_ids") or None,
            )
        elif name == "join_related":
            r = await join_related_tool(
                backend,
                session,
                from_value=args.get("from_value", ""),
                from_values=args.get("from_values") or None,
                fk_property=args.get("fk_property", ""),
                target_table=args.get("target_table", ""),
                limit=int(args.get("limit", 20)),
            )
        elif name == "top_nodes":
            r = await top_nodes_tool(
                backend,
                session,
                table=args.get("table", ""),
                sort_by=args.get("sort_by", ""),
                order=args.get("order", "desc"),
                limit=int(args.get("limit", 5)),
                where_property=args.get("where_property", ""),
                where_op=args.get("where_op", ""),
                where_value=args.get("where_value", ""),
                from_ids=args.get("from_ids") or None,
            )
        else:
            return {
                "tool": name,
                "ok": False,
                "data": {},
                "error": f"unknown_tool: {name!r}",
                "hints": [],
                "session": {},
            }
    except Exception as exc:
        logger.warning("tool dispatch %s failed: %s", name, exc)
        return {
            "tool": name,
            "ok": False,
            "data": {},
            "error": f"{type(exc).__name__}: {exc}",
            "hints": [],
            "session": {},
        }
    return r.to_dict()


def _extract_ids(data: dict, found_ids: set[str], known_tables: set[str] | None = None) -> None:
    """Collect every plausible doc identifier from a tool result.

    ``data`` is expected to be the *unwrapped* tool data (``result["data"]``),
    not the raw wrapper. Covers every tool's response shape:
      - evidence[].document_id / .properties.doc_id / .title
      - results[].properties.doc_id / .title (filter/join)
      - merged_evidence[].document_id (deep_search)
      - document_excerpts[].document.properties.doc_id (deep_search)
      - sub_results[].top_result.* (compare_search)
      - document.properties.doc_id (get_document)
      - groups[].group / .node_title + synthesised "table:value" composites
        (aggregate_nodes — needed for structured-only corpora like assort
        Hard where the answer IS the group key).
    """
    for key in (
        "evidence",
        "results",
        "merged_evidence",
        "matches",
        "expanded_neighbours",
        "neighbours",
    ):
        for item in data.get(key, []):
            if not isinstance(item, dict):
                continue
            did = item.get("document_id", "")
            if did:
                found_ids.add(did)
            props = item.get("properties", {})
            if isinstance(props, dict):
                d2 = props.get("doc_id", "")
                if d2:
                    found_ids.add(d2)
            title = item.get("title", "")
            if title:
                found_ids.add(title)

    for excerpt in data.get("document_excerpts", []):
        if not isinstance(excerpt, dict):
            continue
        doc = excerpt.get("document", {})
        if isinstance(doc, dict):
            did = (doc.get("properties", {}) or {}).get("doc_id", "")
            if did:
                found_ids.add(did)
            title = doc.get("title", "")
            if title:
                found_ids.add(title)

    for sub in data.get("sub_results", []):
        if not isinstance(sub, dict):
            continue
        top = sub.get("top_result")
        if isinstance(top, dict):
            did = top.get("document_id", "")
            if did:
                found_ids.add(did)
            d2 = (top.get("properties", {}) or {}).get("doc_id", "")
            if d2:
                found_ids.add(d2)

    doc_data = data.get("document", {})
    if isinstance(doc_data, dict):
        did = (doc_data.get("properties", {}) or {}).get("doc_id", "")
        if did:
            found_ids.add(did)

    # aggregate groups — group value may be a PK that *is* the answer
    agg_info = data.get("aggregation", {}) if isinstance(data.get("aggregation"), dict) else {}
    agg_table = agg_info.get("table", "")
    group_by = agg_info.get("group_by", "")
    for grp in data.get("groups", []):
        if not isinstance(grp, dict):
            continue
        g = grp.get("group", "")
        if not g:
            continue
        found_ids.add(g)
        nt = grp.get("node_title", "")
        if nt:
            found_ids.add(nt)

        looks_like_pk = (
            len(g) <= 30 and " " not in g and "-" not in g[:5] and not g.startswith("20")
        )
        if not looks_like_pk:
            continue

        if agg_table:
            found_ids.add(f"{agg_table}:{g}")

        if group_by:
            base = group_by.rsplit("_", 1)[0] if "_" in group_by else group_by
            if known_tables:
                for tbl in known_tables:
                    tbl_lower = tbl.lower()
                    if base in tbl_lower or tbl_lower.startswith(base):
                        found_ids.add(f"{tbl}:{g}")
            for candidate in (
                f"{base}:{g}",
                f"{base}s:{g}",
                f"{base}es:{g}",
                f"pr_{base}_base:{g}",
                f"pr_{base}:{g}",
            ):
                found_ids.add(candidate)


_TOOL_RESULT_BUDGET = 4000
"""Default char budget for a single projected tool-result message.

~1000 tokens. Five turns of tool results stay well under 16k vLLM
max_model_len even with system prompt + user query + assistant
responses overhead.
"""


def _truncate(s: str, cap: int) -> str:
    return s if len(s) <= cap else s[: cap - 1] + "…"


def _compact_properties(props: dict, *, max_entries: int = 8, max_value_chars: int = 80) -> dict:
    """Trim a node properties dict to a small scalar subset.

    Keeps only scalar values, drops verbose or nested entries.
    """
    out: dict[str, Any] = {}
    if not isinstance(props, dict):
        return out
    for k, v in props.items():
        if len(out) >= max_entries:
            break
        if isinstance(v, str | int | float | bool):
            sv = str(v)
            out[k] = _truncate(sv, max_value_chars)
    return out


def _project_result_item(r: dict) -> dict:
    """Compact a single filter/search/join/top result to id + title +
    brief preview.

    Preserves ``sort_value`` / ``score`` (both scalar, both load-
    bearing for the agent's next step — without them a ``top_nodes``
    result loses the ranking itself).
    """
    if not isinstance(r, dict):
        return r
    out: dict[str, Any] = {}
    if "id" in r:
        out["id"] = r["id"]
    if "title" in r:
        out["title"] = r["title"]
    for sig in ("sort_value", "score"):
        if sig in r:
            out[sig] = r[sig]
    preview = r.get("preview") or r.get("snippet") or ""
    if preview:
        out["preview"] = _truncate(str(preview), 120)
    props = _compact_properties(r.get("properties") or {})
    if props:
        out["properties"] = props
    return out


def _project_evidence_item(e: dict) -> dict:
    if not isinstance(e, dict):
        return e
    out: dict[str, Any] = {}
    for src in ("id", "node_id", "document_id"):
        if src in e:
            out["id"] = e[src]
            break
    if "title" in e:
        out["title"] = e["title"]
    snippet = e.get("snippet") or e.get("content") or e.get("preview") or ""
    if snippet:
        out["snippet"] = _truncate(str(snippet), 180)
    props = _compact_properties(e.get("properties") or {}, max_entries=3)
    if props:
        out["properties"] = props
    return out


def _project_document_excerpt(ex: dict) -> dict:
    if not isinstance(ex, dict):
        return ex
    out: dict[str, Any] = {}
    doc = ex.get("document") or {}
    if isinstance(doc, dict):
        compact_doc: dict[str, Any] = {}
        if "id" in doc:
            compact_doc["id"] = doc["id"]
        if "title" in doc:
            compact_doc["title"] = doc["title"]
        props = _compact_properties(doc.get("properties") or {}, max_entries=3)
        if props:
            compact_doc["properties"] = props
        if compact_doc:
            out["document"] = compact_doc
    chunks = ex.get("chunks") or ex.get("top_chunks") or []
    if isinstance(chunks, list) and chunks:
        out["chunks"] = [_project_chunk_item(c) for c in chunks[:3]]
    return out


def _project_chunk_item(c: dict) -> dict:
    if not isinstance(c, dict):
        return c
    out: dict[str, Any] = {}
    if "id" in c:
        out["id"] = c["id"]
    if "title" in c:
        out["title"] = c["title"]
    text = c.get("text") or c.get("content") or c.get("preview") or ""
    if text:
        out["text"] = _truncate(str(text), 300)
    return out


def _project_data(tool: str, data: dict) -> dict:
    """Route-table dispatch — tool-specific shape trimming."""
    if not isinstance(data, dict):
        return data
    out = dict(data)

    if "results" in out and isinstance(out["results"], list):
        out["results"] = [_project_result_item(r) for r in out["results"]]

    if "evidence" in out and isinstance(out["evidence"], list):
        out["evidence"] = [_project_evidence_item(e) for e in out["evidence"]]

    if "merged_evidence" in out and isinstance(out["merged_evidence"], list):
        out["merged_evidence"] = [_project_evidence_item(e) for e in out["merged_evidence"]]

    if "document_excerpts" in out and isinstance(out["document_excerpts"], list):
        out["document_excerpts"] = [_project_document_excerpt(x) for x in out["document_excerpts"]]

    if "expanded_neighbours" in out and isinstance(out["expanded_neighbours"], list):
        out["expanded_neighbours"] = [_project_result_item(n) for n in out["expanded_neighbours"]]

    if "neighbours" in out and isinstance(out["neighbours"], list):
        out["neighbours"] = [_project_result_item(n) for n in out["neighbours"]]

    # get_document returns {"document": {...}, "chunks": [...]}
    if isinstance(out.get("document"), dict):
        out["document"] = _project_result_item(out["document"])
    if isinstance(out.get("chunks"), list):
        out["chunks"] = [_project_chunk_item(c) for c in out["chunks"]]

    # aggregate_nodes groups are already compact; no-op unless huge.
    return out


def _overflow_stub(tool: str, ok: bool) -> str:
    """Last-resort envelope used when shrinking can't fit the budget.

    Always valid JSON, always stable size. Losing the data is better
    than silently corrupting a conversation with chopped-mid-value
    JSON (the pre-v0.18 failure mode that caused agent retry loops).
    """
    return json.dumps(
        {
            "tool": tool,
            "ok": ok,
            "data": {"_overflow": True},
            "error": "tool_result_exceeded_context_budget",
        },
        ensure_ascii=False,
    )


def _shrink_lists(data: dict, max_chars: int, envelope: dict) -> str:
    """Iteratively halve list lengths inside ``data`` until envelope fits."""
    list_keys = (
        "results",
        "evidence",
        "merged_evidence",
        "document_excerpts",
        "expanded_neighbours",
        "neighbours",
        "chunks",
        "groups",
        "sub_results",
    )
    for _ in range(10):
        serialized = json.dumps(envelope, ensure_ascii=False)
        if len(serialized) <= max_chars:
            return serialized
        shrunk = False
        for key in list_keys:
            v = data.get(key)
            if isinstance(v, list) and len(v) > 1:
                # Halve, floor at 1 once we're under the mercy point.
                new_len = max(1, len(v) // 2)
                data[key] = v[:new_len]
                data["_trimmed_for_context"] = True
                shrunk = True
        if not shrunk:
            break
    serialized = json.dumps(envelope, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return serialized
    return _overflow_stub(envelope.get("tool", ""), envelope.get("ok", False))


def project_tool_result(result: dict | Any, *, max_chars: int = _TOOL_RESULT_BUDGET) -> str:
    """Project a tool result to a compact JSON string within ``max_chars``.

    Replaces naive ``json.dumps(result)[:5000]`` truncation. Goals:

    - Keep JSON structurally valid — never chop mid-value.
    - Preserve identifiers (``id``, ``title``, ``doc_id``) so the agent
      can chain tool calls.
    - Drop high-volume low-signal fields (``tags``, nested properties,
      raw chunk bodies beyond a short preview).
    - Cap list lengths iteratively if the projected shape still exceeds
      the budget.

    Downstream note: ``_extract_ids`` runs *before* this projection, so
    ID collection is unaffected.
    """
    if not isinstance(result, dict):
        # Fallback for unexpected shapes — ensure output is valid JSON.
        try:
            encoded = json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            encoded = json.dumps(str(result), ensure_ascii=False)
        if len(encoded) <= max_chars:
            return encoded
        # Truncate the underlying *value*, then re-encode — preserves JSON validity.
        headroom = max(0, max_chars - 10)
        if isinstance(result, str):
            return json.dumps(_truncate(result, headroom), ensure_ascii=False)
        return json.dumps({"_overflow": True}, ensure_ascii=False)

    tool = result.get("tool", "")
    data = _project_data(tool, result.get("data") or {})
    envelope: dict[str, Any] = {"tool": tool, "ok": result.get("ok", True), "data": data}
    err = result.get("error")
    if err:
        envelope["error"] = err
    hints = result.get("hints")
    if hints:
        envelope["hints"] = hints[:3]

    serialized = json.dumps(envelope, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return serialized
    return _shrink_lists(data, max_chars, envelope)


# --- Public API ----------------------------------------------------


async def run_agent_loop(
    *,
    client: Any,
    backend: StorageBackend,
    query: str,
    model: str = "gpt-4o-mini",
    max_turns: int = 5,
    embedder: EmbeddingProvider | None = None,
    system_prompt: str | None = None,
    extra_context: str | None = None,
) -> AgentSearchResult:
    """Run one multi-turn agent search.

    Args:
        client: OpenAI-compatible async client (must implement
            ``chat.completions.create(model, messages, tools, max_tokens)``).
        backend: Storage backend to query against.
        query: User question.
        model: Model name passed to ``client.chat.completions.create``.
        max_turns: Max LLM turns. Each turn may contain multiple tool
            calls. The loop stops early if the LLM emits a final text
            answer (no tool calls).
        embedder: Optional embedder for tools that benefit from vector
            cascade (currently ``deep_search`` and ``search``).
        system_prompt: Override the default ``AGENT_SYSTEM``. The
            graph context (categories / table summary) is always
            appended.
        extra_context: Additional context appended to the system
            prompt — useful for injecting per-corpus instructions.

    Returns:
        :class:`AgentSearchResult` containing the final answer, the
        union of all retrieved doc identifiers, and timing metadata.
    """
    from synaptic.search_session import SearchSession, build_graph_context

    t0 = time.time()
    graph_ctx = await build_graph_context(backend)
    base_prompt = system_prompt or AGENT_SYSTEM
    parts = [base_prompt, graph_ctx]
    if extra_context:
        parts.append(extra_context)
    system = "\n\n".join(p for p in parts if p)

    # Sniff known table names so _extract_ids can synthesise composite
    # IDs from aggregate-tool group values (e.g. "G00007" → "pr_goods_base:G00007").
    from synaptic.models import NodeKind as _NK

    sample = await backend.list_nodes(kind=_NK.ENTITY, limit=50_000)
    known_tables: set[str] = set()
    for n in sample:
        t = (n.properties or {}).get("_table_name")
        if t:
            known_tables.add(t)

    session = SearchSession(budget_tool_calls=max_turns * 3)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    found_ids: set[str] = set()
    final_answer = ""
    turns_used = 0
    tool_calls = 0

    for turn in range(max_turns):
        turns_used = turn + 1
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=AGENT_TOOLS,
                max_tokens=2048,
            )
        except Exception as exc:
            logger.warning("agent LLM call failed at turn %d: %s", turn, exc)
            break

        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                fn = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}
                result = await _dispatch_tool(fn, fn_args, backend, session, embedder=embedder)
                tool_calls += 1
                # Tool wrappers return {"tool": ..., "data": {...}}; unwrap
                # before extraction so _extract_ids sees evidence/groups
                # at top level.
                payload = result.get("data", {}) if isinstance(result, dict) else {}
                _extract_ids(payload, found_ids, known_tables=known_tables)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": project_tool_result(result),
                    }
                )
        else:
            final_answer = msg.content or ""
            break

    # Best-effort fetch of node objects for the final result
    nodes: list = []
    for did in list(found_ids)[:50]:  # cap to avoid runaway expansion
        try:
            n = await backend.get_node(did)
        except Exception:
            n = None
        if n is not None:
            nodes.append(n)

    return AgentSearchResult(
        query=query,
        final_answer=final_answer,
        found_ids=found_ids,
        nodes=nodes,
        turns_used=turns_used,
        tool_calls_made=tool_calls,
        elapsed_ms=(time.time() - t0) * 1000.0,
    )
