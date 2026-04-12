"""Atomic search tools for LLM agents — the actual tool layer.

This module is the *contract* between synaptic-memory and an LLM
agent that wants to explore a knowledge graph. The agent doesn't run
a hand-coded retrieval pipeline. It calls these tools one at a time,
reads the structured results, decides what to do next, and loops.

Every tool takes a ``SearchSession`` so state accumulates across
turns, and every tool returns a ``ToolResult`` dataclass with three
parts:

1. ``data`` — the actual payload (evidence list, document content,
   categories, counts, whatever the tool produced).
2. ``session`` — a small snapshot of the session state so the LLM
   can see "how many calls used, what have I looked at already".
3. ``hints`` — optional "you might try this next" suggestions. Pure
   rule-based; the LLM is free to ignore them.

Tools:

- :func:`search_tool` — FTS-seeded hybrid search, returns evidence.
- :func:`expand_tool` — 1-hop graph expansion around a specific node.
- :func:`get_document_tool` — fetch a full document by id.
- :func:`list_categories_tool` — enumerate category nodes.
- :func:`count_tool` — structural count without fetching nodes.
- :func:`search_exact_tool` — literal substring match for IDs/codes.
- :func:`follow_tool` — walk one edge type from a starting node.

All tools are async and backend-agnostic. They work with any object
implementing the ``StorageBackend`` protocol (Memory, SQLite, Kuzu,
Composite). The only shared state is the ``SearchSession`` — tools
never talk to each other directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from synaptic.extensions.evidence_search import EvidenceSearch
from synaptic.extensions.graph_expander import ExpansionBudget, GraphExpander
from synaptic.models import EdgeKind, Node, NodeKind
from synaptic.search_session import SearchSession

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("agent-tools")


# --- Shared result shape ---


@dataclass(slots=True)
class Hint:
    """One actionable suggestion for the agent.

    Attributes:
        action: Tool name the agent could call next, e.g. ``"search"``
            or ``"get_document"``.
        args: Suggested arguments. The LLM is free to adapt them — the
            hint is advisory, not prescriptive.
        reason: One-sentence explanation of *why* this might help.
            The LLM uses this to decide whether to follow the hint.
    """

    action: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(slots=True)
class ToolResult:
    """Common envelope for all tool responses.

    The shape is identical across tools so the MCP server can encode
    every response with a single serialiser. The ``data`` dict holds
    the tool-specific payload; the other fields are always present.

    Attributes:
        tool: Name of the tool that produced this result.
        ok: ``True`` on success, ``False`` when the session was out
            of budget or an expected invariant failed.
        data: Tool-specific payload dict. Structure varies per tool
            but is always JSON-friendly.
        hints: Optional list of :class:`Hint` objects suggesting
            follow-up actions. May be empty.
        session: Snapshot of :meth:`SearchSession.summary` taken at
            the end of the tool call.
        error: Short error string when ``ok`` is ``False``. ``None``
            when the call succeeded.
    """

    tool: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    hints: list[Hint] = field(default_factory=list)
    session: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict representation for MCP / API layers."""
        return {
            "tool": self.tool,
            "ok": self.ok,
            "data": self.data,
            "hints": [{"action": h.action, "args": h.args, "reason": h.reason} for h in self.hints],
            "session": self.session,
            "error": self.error,
        }


# --- Internals ---


def _budget_check(session: SearchSession, tool: str) -> ToolResult | None:
    """Short-circuit if the session has no budget left.

    Every tool calls this first. Keeps the budget-exceeded response
    consistent across tools and makes budget enforcement visible at
    the entry point of every function.
    """
    if session.is_exhausted():
        return ToolResult(
            tool=tool,
            ok=False,
            data={},
            hints=[],
            session=session.summary(),
            error="budget_exceeded",
        )
    session.record_call()
    return None


def _node_to_summary(
    node: Node,
    *,
    content_preview_chars: int = 240,
    query: str = "",
) -> dict[str, Any]:
    """Compact JSON projection of a Node with query-aware snippets.

    When ``query`` is provided, extracts the most relevant fragment
    from the content (the sentence containing the most query terms)
    instead of a blind prefix. This gives the LLM better signal per
    token — a 200-char snippet that actually matches the query is
    worth more than 200 chars of document preamble.
    """
    content = node.content or ""
    if query and content:
        preview = _extract_snippet(content, query, max_chars=content_preview_chars)
    else:
        preview = content[:content_preview_chars]
        if len(content) > content_preview_chars:
            preview += "…"
    return {
        "id": node.id,
        "kind": str(node.kind),
        "title": node.title,
        "preview": preview,
        "tags": list(node.tags or []),
        "properties": dict(node.properties or {}),
    }


def _extract_snippet(content: str, query: str, *, max_chars: int = 240) -> str:
    """Extract the most query-relevant fragment from content.

    Splits content into sentences (by period/newline), scores each by
    query term overlap, and returns the best window up to max_chars.
    Falls back to prefix if no good match found.
    """
    import re

    q_terms = set(query.lower().split())
    if not q_terms:
        return content[:max_chars]

    # Split into rough sentences
    sentences = re.split(r"[.。\n]+", content)
    if not sentences:
        return content[:max_chars]

    # Score each sentence
    scored = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s_lower = s.lower()
        overlap = sum(1 for t in q_terms if t in s_lower)
        scored.append((s, overlap))

    if not scored:
        return content[:max_chars]

    scored.sort(key=lambda x: -x[1])
    best = scored[0]

    if best[1] == 0:
        # No term overlap — fall back to prefix
        return content[:max_chars] + ("…" if len(content) > max_chars else "")

    # Build snippet from best sentence + neighbors
    snippet = best[0][:max_chars]
    if len(best[0]) > max_chars:
        snippet += "…"
    return snippet


# --- Tool 1: search ---


async def search_tool(
    backend: StorageBackend,
    session: SearchSession,
    query: str,
    *,
    limit: int = 10,
    category: str | None = None,
    kind: NodeKind | str | None = None,
    exclude_seen: bool = True,
    embedder: object | None = None,
) -> ToolResult:
    """Run the 3rd-gen evidence pipeline for ``query``.

    This is the agent's main entry point. The tool drives the full
    anchor → expand → rerank → aggregate flow via
    :class:`EvidenceSearch`, filters out already-seen nodes, and
    hands back a compact evidence list.

    Args:
        backend: Storage backend to search.
        session: Active search session — used for dedup + history.
        query: User query string.
        limit: Max evidence items to return. The aggregator's MMR
            and per-document cap still apply, so fewer items may come
            back when the corpus is small or homogeneous.
        category: Optional category label filter. The filter is
            applied post-retrieval against each hit's
            ``properties["category"]`` so the agent can narrow a
            broad search without re-running the full pipeline.
        kind: Optional ``NodeKind`` filter. Same shape as ``category``.
        exclude_seen: When ``True`` (default), any node already in
            ``session.seen_node_ids`` is filtered out before the
            aggregator runs. Lets the agent paginate through a topic.

    Returns:
        :class:`ToolResult` with ``data.evidence`` — list of chunk
        summaries — plus ``data.anchors`` from the extractor so the
        agent can see which categories / entities the query touched.
    """
    budget = _budget_check(session, "search")
    if budget is not None:
        return budget

    session.record_query(query)

    searcher = EvidenceSearch(backend=backend, embedder=embedder)
    result = await searcher.search(
        query,
        k=limit * 2,  # over-fetch, then apply filters
        fts_seed_limit=max(20, limit * 3),
    )

    # Apply category / kind filter if the caller asked for one
    evidence = list(result.evidence)
    if category:
        cat_lower = category.lower()
        evidence = [e for e in evidence if cat_lower in (e.category or "").lower()]
    if kind is not None:
        kind_str = str(kind).lower() if not isinstance(kind, NodeKind) else str(kind)
        evidence = [e for e in evidence if str(e.node.kind).lower() == kind_str]
    if exclude_seen:
        evidence = [e for e in evidence if not session.has_seen(e.node.id)]

    evidence = evidence[:limit]
    session.mark_seen(e.node.id for e in evidence)
    session.mark_categories(result.anchors.categories)
    session.set_fact(
        "last_query_anchors",
        {
            "categories": list(result.anchors.categories),
            "entities": list(result.anchors.entities),
        },
    )
    session.set_fact("last_evidence_ids", [e.node.id for e in evidence])

    hints: list[Hint] = []

    if not evidence:
        hints.append(
            Hint(
                action="search",
                args={"query": query, "exclude_seen": False},
                reason="no new results — retry without the seen filter to revisit prior hits",
            )
        )
        if result.anchors.categories:
            first_cat = result.anchors.categories[0]
            hints.append(
                Hint(
                    action="list_categories",
                    args={},
                    reason=f"query touched '{first_cat}' — inspect the full category list to pick a different angle",
                )
            )
    else:
        top = evidence[0]
        hints.append(
            Hint(
                action="get_document",
                args={"doc_id": top.document_id},
                reason="fetch the full parent document of the top evidence to verify absence/completeness",
            )
        )
        if len(result.anchors.categories) > 1:
            for cat in result.anchors.categories[1:3]:
                hints.append(
                    Hint(
                        action="search",
                        args={"query": query, "category": cat},
                        reason=f"query also touched '{cat}' — narrow search to that category",
                    )
                )

    return ToolResult(
        tool="search",
        ok=True,
        data={
            "evidence": [
                {
                    **_node_to_summary(e.node, query=query),
                    "score": round(e.score, 4),
                    "category": e.category,
                    "document_id": e.document_id,
                    "reason": e.reason,
                }
                for e in evidence
            ],
            "anchors": {
                "categories": list(result.anchors.categories),
                "entities": list(result.anchors.entities),
                "keywords": list(result.anchors.keywords),
            },
        },
        hints=hints,
        session=session.summary(),
    )


# --- Tool 2: expand ---


async def expand_tool(
    backend: StorageBackend,
    session: SearchSession,
    node_id: str,
    *,
    limit: int = 10,
    exclude_seen: bool = True,
) -> ToolResult:
    """Return 1-hop graph neighbours of ``node_id``.

    Reuses :class:`GraphExpander` so the semantics match the rest
    of the pipeline — category siblings, document-scoped chunks,
    NEXT_CHUNK walks, and entity mentions. The agent typically calls
    this after seeing a promising result from ``search`` to look at
    surrounding context.
    """
    budget = _budget_check(session, "expand")
    if budget is not None:
        return budget

    seed = await backend.get_node(node_id)
    if seed is None:
        return ToolResult(
            tool="expand",
            ok=False,
            data={},
            hints=[],
            session=session.summary(),
            error=f"node_not_found: {node_id}",
        )

    expander = GraphExpander(backend=backend)
    expanded = await expander.expand(
        anchors=_anchors_from_seed(seed),
        seed_nodes=[seed],
        budget=ExpansionBudget(max_total_expanded=limit * 2),
    )

    out_nodes = [e for e in expanded if e.node.id != node_id]
    if exclude_seen:
        out_nodes = [e for e in out_nodes if not session.has_seen(e.node.id)]
    out_nodes = out_nodes[:limit]
    session.mark_seen(e.node.id for e in out_nodes)
    session.expanded_nodes.add(node_id)

    hints: list[Hint] = []
    if not out_nodes:
        hints.append(
            Hint(
                action="get_document",
                args={"doc_id": _doc_id_of(seed)},
                reason="no new neighbours — fall back to the full document",
            )
        )

    return ToolResult(
        tool="expand",
        ok=True,
        data={
            "seed": _node_to_summary(seed),
            "neighbours": [
                {
                    **_node_to_summary(e.node),
                    "reason": e.reason,
                    "anchor_hit": e.anchor_hit,
                }
                for e in out_nodes
            ],
        },
        hints=hints,
        session=session.summary(),
    )


def _anchors_from_seed(seed: Node):
    """Build a minimal QueryAnchors object for a single seed node.

    Used by ``expand_tool`` so the expander's category-sibling path
    still fires when the seed is itself a document/chunk inside a
    known category. Imported lazily to avoid a circular import
    against ``query_anchor``.
    """
    from synaptic.extensions.query_anchor import QueryAnchors

    cat = (seed.properties or {}).get("category") or ""
    return QueryAnchors(
        query=seed.title or seed.id,
        keywords=[],
        entities=[],
        categories=[cat] if cat else [],
        category_node_ids=[],  # we don't know the cat node id without a lookup
    )


def _doc_id_of(node: Node) -> str:
    return (node.properties or {}).get("doc_id", "")


# --- Tool 3: get_document ---


async def get_document_tool(
    backend: StorageBackend,
    session: SearchSession,
    doc_id: str,
    *,
    query: str = "",
    max_chunks: int = 50,
    max_full_chunks: int = 5,
) -> ToolResult:
    """Fetch a document node and its chunks — smart context control.

    When ``query`` is provided, chunks are scored by keyword overlap
    and only the top ``max_full_chunks`` get full text. The rest are
    returned as one-line summaries. This keeps context under ~2K tokens
    instead of ~5K+ for a typical document.

    Without ``query``, all chunks are returned in full (backward compat).

    Args:
        doc_id: Document id or node id.
        query: Optional query for chunk relevance scoring.
        max_chunks: Total chunks to fetch.
        max_full_chunks: How many chunks get full text (rest = title only).
    """
    budget = _budget_check(session, "get_document")
    if budget is not None:
        return budget

    # The agent may hand us either the doc_node_id (e.g. "doc_abc") or
    # the raw doc_id from properties. Try the direct lookup first.
    doc_node: Node | None = await backend.get_node(doc_id)
    if doc_node is None:
        # Fall back: use search_fuzzy to find by doc_id string instead
        # of loading all nodes into memory. Much cheaper on large corpora.
        candidates = await backend.search_fuzzy(doc_id, limit=50)
        for n in candidates:
            props = n.properties or {}
            if props.get("doc_id") == doc_id and "document" in (n.tags or []):
                doc_node = n
                break

    if doc_node is None:
        return ToolResult(
            tool="get_document",
            ok=False,
            data={},
            session=session.summary(),
            error=f"document_not_found: {doc_id}",
        )

    # Walk CONTAINS edges to assemble chunks in index order.
    # Uses get_nodes_batch (single SQL WHERE IN) instead of N+1 get_node calls.
    edges = await backend.get_edges(doc_node.id, direction="outgoing")
    chunk_ids = [e.target_id for e in edges if e.kind == EdgeKind.CONTAINS][:max_chunks]

    chunks = await backend.get_nodes_batch(chunk_ids)

    chunks.sort(key=lambda c: int((c.properties or {}).get("chunk_index", "0") or "0"))

    session.mark_seen([doc_node.id, *[c.id for c in chunks]])

    # Smart context: when query is provided, score chunks and return
    # full text only for top-N most relevant. Rest get title-only.
    chunk_data: list[dict] = []
    if query and query.strip():
        q_terms = set(query.lower().split())
        scored_chunks = []
        for c in chunks:
            text_lower = (c.content or "").lower()
            overlap = sum(1 for t in q_terms if t in text_lower)
            scored_chunks.append((c, overlap))
        scored_chunks.sort(key=lambda x: -x[1])

        full_ids = {sc[0].id for sc in scored_chunks[:max_full_chunks]}
        for c in chunks:  # preserve reading order
            idx = (c.properties or {}).get("chunk_index", "")
            if c.id in full_ids:
                chunk_data.append(
                    {
                        "id": c.id,
                        "index": idx,
                        "content": c.content,
                        "relevant": True,
                    }
                )
            else:
                # Title-only summary — saves ~90% context
                chunk_data.append(
                    {
                        "id": c.id,
                        "index": idx,
                        "summary": (c.content or "")[:80] + "…",
                    }
                )
    else:
        for c in chunks:
            chunk_data.append(
                {
                    "id": c.id,
                    "index": (c.properties or {}).get("chunk_index", ""),
                    "content": c.content,
                }
            )

    return ToolResult(
        tool="get_document",
        ok=True,
        data={
            "document": _node_to_summary(doc_node, content_preview_chars=400),
            "chunk_count": len(chunks),
            "full_chunks": sum(1 for c in chunk_data if "content" in c),
            "chunks": chunk_data,
        },
        hints=[],
        session=session.summary(),
    )


# --- Tool 4: list_categories ---


async def list_categories_tool(
    backend: StorageBackend,
    session: SearchSession,
    *,
    limit: int = 100,
) -> ToolResult:
    """List top-level category nodes in the graph.

    Used by the agent to build a mental map of the corpus before
    searching. Returns label + count of documents per category so
    the LLM can judge coverage at a glance.
    """
    budget = _budget_check(session, "list_categories")
    if budget is not None:
        return budget

    cats = await backend.list_nodes(kind=NodeKind.CONCEPT, limit=limit)
    categories = [c for c in cats if "category" in (c.tags or [])]

    # For each category, count outgoing PART_OF edges (documents)
    category_entries = []
    for cat in categories:
        try:
            edges = await backend.get_edges(cat.id, direction="incoming")
            doc_count = sum(1 for e in edges if e.kind == EdgeKind.PART_OF)
        except Exception:
            doc_count = 0
        category_entries.append(
            {
                "id": cat.id,
                "label": cat.title,
                "document_count": doc_count,
            }
        )

    category_entries.sort(key=lambda c: -c["document_count"])

    return ToolResult(
        tool="list_categories",
        ok=True,
        data={
            "categories": category_entries,
            "total": len(category_entries),
        },
        hints=[],
        session=session.summary(),
    )


# --- Tool 5: count ---


async def count_tool(
    backend: StorageBackend,
    session: SearchSession,
    *,
    kind: NodeKind | str | None = None,
    category: str | None = None,
    year: int | None = None,
) -> ToolResult:
    """Count matching nodes without fetching them.

    The agent uses this to decide whether a "for all / enumerate"
    question is even feasible — if count returns 5, the agent can
    iterate; if count returns 50,000 it needs a different strategy.
    """
    budget = _budget_check(session, "count")
    if budget is not None:
        return budget

    node_kind: NodeKind | None = None
    if isinstance(kind, NodeKind):
        node_kind = kind
    elif isinstance(kind, str) and kind:
        try:
            node_kind = NodeKind(kind.lower())
        except ValueError:
            pass

    matched = await backend.count_nodes(kind=node_kind, category=category, year=year)

    return ToolResult(
        tool="count",
        ok=True,
        data={
            "count": matched,
            "filters": {
                "kind": str(node_kind) if node_kind else None,
                "category": category,
                "year": year,
            },
        },
        hints=[],
        session=session.summary(),
    )


# --- Tool 6: search_exact ---


async def search_exact_tool(
    backend: StorageBackend,
    session: SearchSession,
    identifier: str,
    *,
    limit: int = 20,
) -> ToolResult:
    """Literal substring match for IDs, codes, function names.

    Bypasses FTS tokenisation so exact strings like ``E217``,
    ``SKU-1234``, or ``api/v1/users`` land where BM25 would dilute
    them. Implementation walks ``list_nodes`` and checks content
    substring — fine for the KRRA-scale corpora this library targets.
    """
    budget = _budget_check(session, "search_exact")
    if budget is not None:
        return budget

    identifier = (identifier or "").strip()
    if not identifier:
        return ToolResult(
            tool="search_exact",
            ok=False,
            data={},
            session=session.summary(),
            error="empty_identifier",
        )

    # Use search_fuzzy (LIKE '%identifier%') to push the scan into SQL
    # instead of loading all nodes into Python memory.
    candidates = await backend.search_fuzzy(identifier, limit=limit * 5)
    matches: list[Node] = []
    for n in candidates:
        haystack = f"{n.title or ''}\n{n.content or ''}"
        if identifier in haystack:
            matches.append(n)
            if len(matches) >= limit:
                break

    session.mark_seen(n.id for n in matches)

    return ToolResult(
        tool="search_exact",
        ok=True,
        data={
            "identifier": identifier,
            "count": len(matches),
            "matches": [_node_to_summary(n) for n in matches],
        },
        hints=[],
        session=session.summary(),
    )


# --- Tool 7: follow ---


async def follow_tool(
    backend: StorageBackend,
    session: SearchSession,
    node_id: str,
    edge_kind: str | EdgeKind,
    *,
    direction: str = "both",
    limit: int = 20,
) -> ToolResult:
    """Walk one specific edge type from a starting node.

    Gives the agent a surgical alternative to ``expand`` when it
    knows exactly which relation it wants to follow (e.g. only
    ``CONTAINS`` to get a document's chunks, or only ``MENTIONS``
    to get entity sources).
    """
    budget = _budget_check(session, "follow")
    if budget is not None:
        return budget

    try:
        kind_enum = (
            edge_kind if isinstance(edge_kind, EdgeKind) else EdgeKind(str(edge_kind).lower())
        )
    except ValueError:
        return ToolResult(
            tool="follow",
            ok=False,
            data={},
            session=session.summary(),
            error=f"unknown_edge_kind: {edge_kind}",
        )

    edges = await backend.get_edges(node_id, direction=direction)
    matching = [e for e in edges if e.kind == kind_enum][:limit]

    neighbour_nodes: list[Node] = []
    for e in matching:
        other_id = e.target_id if e.source_id == node_id else e.source_id
        other = await backend.get_node(other_id)
        if other is not None:
            neighbour_nodes.append(other)

    session.mark_seen(n.id for n in neighbour_nodes)

    return ToolResult(
        tool="follow",
        ok=True,
        data={
            "source_id": node_id,
            "edge_kind": str(kind_enum),
            "count": len(neighbour_nodes),
            "neighbours": [_node_to_summary(n) for n in neighbour_nodes],
        },
        hints=[],
        session=session.summary(),
    )
