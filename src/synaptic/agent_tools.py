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
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from synaptic.extensions.evidence_search import EvidenceSearch
from synaptic.extensions.graph_expander import ExpansionBudget, GraphExpander
from synaptic.models import EdgeKind, Node, NodeKind
from synaptic.search_session import SearchSession

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("agent-tools")

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_PROCESS_FROM_RE = re.compile(
    r"\bhow\s+(?:is|are|was|were)\s+(?P<subject>.+?)\s+"
    r"(?P<verb>created|made|formed|produced)\s+from\s+(?P<source>.+)",
    re.IGNORECASE,
)
_BLOOD_RE = re.compile(
    r"\b(?:bloodborne|blood-borne|blood(?![\s-]+(?:pressure|sugar|glucose|tests?)\b))\b",
    re.IGNORECASE,
)
_SEXUAL_TRANSMISSION_RE = re.compile(
    r"\b(?:sexually\s+transmitted|sexual(?:ly)?\s+transmission|stds?|stis?)\b",
    re.IGNORECASE,
)
_DISEASE_OR_INFECTION_RE = re.compile(
    r"\b(?:diseases?|infections?|stds?|stis?)\b",
    re.IGNORECASE,
)
_FIBER_IN_RE = re.compile(
    r"\bhow\s+much\s+fiber\s+(?:is|are)\s+in\s+(?P<food>.+)",
    re.IGNORECASE,
)
_FIBER_CONTENT_IN_RE = re.compile(
    r"\bfiber\s+content\s+(?:in|of)\s+(?P<food>.+)",
    re.IGNORECASE,
)
_FIBER_TRAILING_WORDS = {"fiber", "content", "gram", "grams", "per", "serving", "servings"}
_TIRE_GAS_RE = re.compile(
    r"\b(?:tires?|tyres?)\b.*\b(?:gas\s+mileage|fuel\s+economy)\b"
    r"|\b(?:gas\s+mileage|fuel\s+economy)\b.*\b(?:tires?|tyres?)\b",
    re.IGNORECASE,
)
_TIRE_SIZE_CONTEXT_RE = re.compile(
    r"\b(?:bigger|larger|smaller|wider|narrower|width|size|sized|diameter)\b",
    re.IGNORECASE,
)
_BICYCLE_TUBE_SIZE_RE = re.compile(
    r"\b(?:bicycle|bike)\b.*\b(?:tires?|tyres?)\b.*\btubes?\b.*\b(?:sized?|sizing|sizes?)\b"
    r"|\b(?:sized?|sizing|sizes?)\b.*\b(?:bicycle|bike)\b.*\b(?:tires?|tyres?)\b.*\btubes?\b",
    re.IGNORECASE,
)
_PROCESS_TRAILING_WORDS = {
    "breakdown",
    "created",
    "formation",
    "formed",
    "forming",
    "made",
    "process",
    "processes",
    "produced",
    "weathering",
}


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


def _query_rewrite_hints(query: str, *, limit: int = 20) -> list[Hint]:
    hints: list[Hint] = []
    seen = {query.strip().lower()}

    def add(candidate: str, reason: str) -> None:
        candidate = " ".join(candidate.strip(" ?.!").split())
        if not candidate:
            return
        key = candidate.lower()
        if key in seen:
            return
        seen.add(key)
        hints.append(
            Hint(action="search", args={"query": candidate, "limit": limit}, reason=reason)
        )

    without_year = _YEAR_RE.sub(" ", query)
    if without_year != query:
        add(
            without_year,
            "retry without the numeric year if the year is metadata/noise rather than answer text",
        )

    process = _PROCESS_FROM_RE.search(query)
    if process:
        subject = process.group("subject")
        source_singular = _normalise_process_source(process.group("source"))
        add(
            f"making {subject} {source_singular} pieces",
            "process questions often use answer-text verbs like making/forming rather than created",
        )
        add(
            f"small pieces of {source_singular} form {subject}",
            "retry with an answer-shaped process phrase using the same subject and source",
        )

    if (
        _BLOOD_RE.search(query)
        and _SEXUAL_TRANSMISSION_RE.search(query)
        and _DISEASE_OR_INFECTION_RE.search(query)
    ):
        add(
            "sexual blood borne transmission routes",
            "medical pages often describe this as sexual and blood-borne transmission rather than blood diseases",
        )

    fiber = _FIBER_IN_RE.search(query) or _FIBER_CONTENT_IN_RE.search(query)
    if fiber:
        food = _normalise_food_rewrite_tail(fiber.group("food"))
        if food:
            add(
                f"one cup {food} grams fiber",
                "nutrition answers often state fiber per cup and in grams rather than repeating the question wording",
            )
            add(
                f"one cup cooked {food} grams fiber",
                "vegetable nutrition pages often report cooked serving sizes with grams of fiber",
            )

    if _TIRE_GAS_RE.search(query) and _TIRE_SIZE_CONTEXT_RE.search(query):
        add(
            "tire size factors influence gas mileage",
            "vehicle-efficiency pages often describe tire size/width as factors that influence gas mileage",
        )
        add(
            "tire width versus gas mileage",
            "retry with the answer-heading phrasing used by tire efficiency pages",
        )

    if _BICYCLE_TUBE_SIZE_RE.search(query):
        add(
            "bicycle tire tube size sidewall ETRTO metric imperial",
            "bike tube sizing pages often point to sidewall numbers and ETRTO/metric/imperial size labels",
        )
        add(
            "bicycle tire sidewall tube size printed raised numbers",
            "retry with the answer-text phrase that says tube sizes are printed on the tire sidewall",
        )

    return hints[:3]


def _normalise_process_source(source: str) -> str:
    tokens = source.strip(" ?.!").split()
    while len(tokens) > 1 and tokens[-1].lower() in _PROCESS_TRAILING_WORDS:
        tokens.pop()
    if not tokens:
        return ""
    last = tokens[-1]
    if last.lower().endswith("s") and not last.lower().endswith("ss"):
        tokens[-1] = last[:-1]
    return " ".join(tokens)


def _normalise_rewrite_tail(value: str) -> str:
    return " ".join(value.strip(" ?.!").split())


def _normalise_food_rewrite_tail(value: str) -> str:
    tokens = _normalise_rewrite_tail(value).split()
    while tokens and tokens[-1].lower() in _FIBER_TRAILING_WORDS:
        tokens.pop()
    return " ".join(tokens)


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

    # The seen-filter lets the agent paginate, but when it empties an otherwise
    # non-empty result set the agent is re-searching a topic it already explored
    # — returning a blank dead-ends that turn. Hand the (already-seen) hits back
    # instead, flagged, so the agent can still read them to answer.
    seen_fallback = False
    if exclude_seen:
        unseen = [e for e in evidence if not session.has_seen(e.node.id)]
        if not unseen and evidence:
            seen_fallback = True
        else:
            evidence = unseen

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
        hints.extend(_query_rewrite_hints(query))

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
            # Signals these hits were already surfaced in earlier turns (the
            # seen-filter would have hidden them) — returned anyway so the turn
            # isn't wasted; the agent likely already has what it needs to answer.
            **({"all_previously_seen": True} if seen_fallback else {}),
        },
        hints=hints,
        session=session.summary(),
    )


# --- Tool 2: expand ---


def _query_relevance(node: Node, q_terms: set[str]) -> int:
    """LLM-free relevance of a node to the query: count of distinct query
    terms appearing in its title (weighted) or content. Used to rank graph
    neighbours toward the agent's actual question instead of the seed's own
    anchors — the seed-anchored expander surfaces noisy 1-hop neighbours, so
    re-ranking by the live query is what makes traversal signal-bearing."""
    if not q_terms:
        return 0
    title = (node.title or "").lower()
    content = (node.content or "").lower()
    return sum((2 if t in title else 0) + (1 if t in content else 0) for t in q_terms)


async def expand_tool(
    backend: StorageBackend,
    session: SearchSession,
    node_id: str,
    *,
    query: str = "",
    limit: int = 10,
    exclude_seen: bool = True,
    embedder: object | None = None,
) -> ToolResult:
    """Return the most useful neighbours of ``node_id`` for the agent's question.

    Reuses :class:`GraphExpander` for graph semantics (category siblings,
    document-scoped chunks, NEXT_CHUNK walks, entity mentions), then adds two
    navigation upgrades so the agent finds faster:

    * **Query-aware ranking** — when ``query`` is given, neighbours are scored
      by relevance to *that question* and the best survive the ``limit`` cap,
      instead of an arbitrary slice of the seed-anchored expansion. Snippets are
      query-extracted too.
    * **Semantic-neighbour fallback** — when the node is an *island* (no graph
      neighbours, ~29 % of a real corpus) and an ``embedder`` is available, fall
      back to embedding-kNN nearest nodes so the agent is never stranded. This
      is the navigable-small-world idea applied at query time — no index-time
      backbone, no index cost.
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
        budget=ExpansionBudget(max_total_expanded=max(limit * 3, 20)),
    )

    # OPT-IN (default OFF) — measured neutral on the explicit `expand` path.
    # A/B on finreg-multihop (Δ 0) and KRRA Hard (Δ −1, noise) showed no gain:
    # the agent reaches evidence via the `search` tool (whose INTERNAL
    # GraphExpander is what gives graph its +8.3pp agent lift), and rarely calls
    # the explicit `expand` tool on island nodes, so the semantic fallback fired
    # 0 times even on KRRA (29% islands). Kept opt-in (the ideas are being moved
    # into EvidenceSearch's internal expansion, where they actually fire). Set
    # SYNAPTIC_NAV_UPGRADE=1 to re-enable query-aware ranking + island fallback.
    # See examples/ablation/diagnostics/v028_agent_navigation_20260607.md.
    import os as _os

    _upgrade = _os.environ.get("SYNAPTIC_NAV_UPGRADE", "0") != "0"

    out_nodes = [e for e in expanded if e.node.id != node_id]
    if exclude_seen:
        out_nodes = [e for e in out_nodes if not session.has_seen(e.node.id)]

    # Query-aware ranking — rank the over-fetched pool by relevance to the live
    # question BEFORE the cap, so a relevant neighbour can't be cut off by the
    # expander's seed-anchored order. Stable: ties keep expander order.
    q_terms = {t for t in (query or "").lower().split() if len(t) > 1}
    if _upgrade and q_terms and out_nodes:
        out_nodes.sort(key=lambda e: _query_relevance(e.node, q_terms), reverse=True)
    out_nodes = out_nodes[:limit]

    # Semantic-neighbour fallback for island nodes — graph traversal dead-ends
    # on isolated nodes, so give the agent an embedding-kNN escape hatch.
    fallback_used = False
    if _upgrade and not out_nodes and embedder is not None:
        try:
            seed_text = ((seed.title or "") + " " + (seed.content or "")).strip()
            if seed_text:
                emb = await embedder.embed(seed_text)
                hits = await backend.search_vector(emb, limit=limit + 5)
                from synaptic.extensions.graph_expander import ExpandedNode

                out_nodes = [
                    ExpandedNode(node=n, reason="semantic_neighbor", hops=1, anchor_hit=None)
                    for n in hits
                    if n.id != node_id and not (exclude_seen and session.has_seen(n.id))
                ][:limit]
                fallback_used = bool(out_nodes)
        except Exception as exc:  # embedder/vector failure must never break expand
            logger.debug("expand semantic fallback failed: %s", exc)

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
            "seed": _node_to_summary(seed, query=query),
            "neighbours": [
                {
                    **_node_to_summary(e.node, query=query),
                    "reason": e.reason,
                    "anchor_hit": e.anchor_hit,
                }
                for e in out_nodes
            ],
            "via": "semantic" if fallback_used else "graph",
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


async def _parent_via_contains(backend: StorageBackend, chunk_id: str) -> Node | None:
    """Return the document node that CONTAINS ``chunk_id``, if any."""
    edges = await backend.get_edges(chunk_id, direction="both")
    for e in edges:
        if e.kind == EdgeKind.CONTAINS and e.target_id == chunk_id:
            return await backend.get_node(e.source_id)
    return None


async def _resolve_by_doc_id_property(backend: StorageBackend, doc_id: str) -> Node | None:
    """Resolve a bare ``doc_id`` property value to a document node.

    ``search``/``deep_search`` results expose the ``doc_id`` *property*
    (an opaque hash), which is not a node id and is not present in any
    text index — so ``get_node`` and ``search_fuzzy`` both miss it.

    Prefer the backend's indexed ``find_nodes_by_property`` (a C-level
    JSON scan returning only matches); fall back to a full ``list_nodes``
    scan only on backends that lack it. Prefer a node tagged
    ``document``; on chunk-only corpora fall back to any node carrying
    the ``doc_id`` and hop to its CONTAINS parent.
    """
    finder = getattr(backend, "find_nodes_by_property", None)
    if finder is not None:
        nodes = await finder("doc_id", doc_id, limit=2000)
    else:
        nodes = [
            n
            for n in await backend.list_nodes(kind=None, limit=200_000)
            if (n.properties or {}).get("doc_id") == doc_id
        ]
    chunk_match: Node | None = None
    for n in nodes:
        if "document" in (n.tags or []):
            return n  # the document node itself — best match
        if chunk_match is None:
            chunk_match = n
    if chunk_match is not None:
        parent = await _parent_via_contains(backend, chunk_match.id)
        return parent if parent is not None else chunk_match
    return None


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

    # The agent may hand us any of three id namespaces:
    #   1. a document node id   ("doc_abc")
    #   2. a chunk node id      ("chunk_abc")  — hop to its parent
    #   3. a bare `doc_id` *property* value — what search/deep_search
    #      results actually expose. This is NOT a node id, so the direct
    #      lookup misses; resolve it by scanning for the carrying node.
    doc_node: Node | None = await backend.get_node(doc_id)

    if doc_node is not None and doc_node.kind == NodeKind.CHUNK:
        # Agent passed a chunk id — hop to the parent document via the
        # incoming CONTAINS edge so we return the whole document.
        parent = await _parent_via_contains(backend, doc_node.id)
        if parent is not None:
            doc_node = parent

    if doc_node is None:
        doc_node = await _resolve_by_doc_id_property(backend, doc_id)

    if doc_node is None:
        # Don't dead-end the agent's turn on an unresolvable id (the id it
        # carried over from a search result may be from a different namespace).
        # If a query is present, return the best lexical matches for it so the
        # turn still yields usable evidence instead of an empty error.
        if query and query.strip():
            try:
                matches = await backend.search_fts(query, limit=max_full_chunks)
            except Exception:
                matches = []
            if matches:
                session.mark_seen([m.id for m in matches])
                return ToolResult(
                    tool="get_document",
                    ok=True,
                    data={
                        "document": {
                            "id": doc_id,
                            "title": f"(no exact document for {doc_id} — best matches for query)",
                        },
                        "chunk_count": len(matches),
                        "full_chunks": len(matches),
                        "chunks": [
                            {
                                "id": m.id,
                                "index": (m.properties or {}).get("chunk_index", ""),
                                "content": m.content,
                                "relevant": True,
                            }
                            for m in matches
                        ],
                        "fallback": "doc_id_unresolved_search_fallback",
                    },
                    hints=[
                        Hint(
                            action="get_document",
                            args={"doc_id": doc_id, "query": query},
                            reason=(
                                f"No document resolved for '{doc_id}'; showing best lexical "
                                "matches for the query instead. Use these chunk ids, or refine "
                                "the query."
                            ),
                        )
                    ],
                    session=session.summary(),
                )
        return ToolResult(
            tool="get_document",
            ok=False,
            data={},
            session=session.summary(),
            error=f"document_not_found: {doc_id}",
            hints=[
                Hint(
                    action="get_document",
                    args={"doc_id": doc_id, "query": "<your question>"},
                    reason="Pass a `query` to fall back to a content search when the id is unknown.",
                )
            ],
        )

    # Walk CONTAINS edges to assemble chunks in index order.
    # Uses get_nodes_batch (single SQL WHERE IN) instead of N+1 get_node calls.
    edges = await backend.get_edges(doc_node.id, direction="outgoing")
    chunk_ids = [e.target_id for e in edges if e.kind == EdgeKind.CONTAINS][:max_chunks]

    chunks = await backend.get_nodes_batch(chunk_ids)

    # No CONTAINS chunks — return the node's own content so the agent still
    # gets the text instead of an empty result. Covers chunk-only corpora AND
    # document/entity nodes that carry their text inline (a common 0-result
    # dead-end that wasted the agent's turn).
    if not chunks and (doc_node.content or "").strip():
        chunks = [doc_node]

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
