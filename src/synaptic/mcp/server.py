"""Synaptic Memory MCP Server — expose knowledge graph as MCP tools.

Usage:
    synaptic-mcp                          # stdio (default, for Claude Code)
    synaptic-mcp --db ./knowledge.db      # custom DB path
    synaptic-mcp --dsn postgresql://...   # PostgreSQL backend
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from synaptic.mcp import __version__

logger = logging.getLogger("synaptic.mcp")

server = FastMCP(
    "Synaptic Memory",
    dependencies=["aiosqlite"],
)

# Module-level state (initialized on first tool call)
_graph: Any = None
_backend: Any = None
_embedder: Any = None
_tracker: Any = None
_db_path: str = "knowledge.db"
_dsn: str = ""
_embed_url: str = ""
_embed_model: str = "default"


async def _ensure_graph() -> Any:
    """Lazy-initialize the SynapticGraph on first use."""
    global _graph, _backend, _embedder

    if _graph is not None:
        return _graph

    from synaptic.extensions.tagger_regex import RegexTagExtractor
    from synaptic.graph import SynapticGraph
    from synaptic.ontology import build_agent_ontology

    if _dsn:
        from synaptic.backends.postgresql import PostgreSQLBackend

        _backend = PostgreSQLBackend(_dsn)
    else:
        from synaptic.backends.sqlite import SQLiteBackend

        _backend = SQLiteBackend(_db_path)

    await _backend.connect()

    # Auto-embedding: connect to any OpenAI-compatible endpoint
    if _embed_url:
        from synaptic.extensions.embedder import OpenAIEmbeddingProvider

        _embedder = OpenAIEmbeddingProvider(api_base=_embed_url, model=_embed_model)
        logger.info("Embedder configured: %s (model=%s)", _embed_url, _embed_model)

    _graph = SynapticGraph(
        _backend,
        tag_extractor=RegexTagExtractor(),
        ontology=build_agent_ontology(),
        embedder=_embedder,
    )
    logger.info("Knowledge graph initialized (backend=%s)", type(_backend).__name__)
    return _graph


async def _ensure_tracker() -> Any:
    """Lazy-initialize the ActivityTracker."""
    global _tracker

    if _tracker is not None:
        return _tracker

    from synaptic.activity import ActivityTracker

    graph = await _ensure_graph()
    _tracker = ActivityTracker(graph)
    return _tracker


# --- Tools ---


@server.tool()
async def knowledge_search(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Search the knowledge graph for lessons, decisions, patterns, and past outcomes.

    Use this to find relevant company knowledge before starting a task.
    Supports Korean and English queries with synonym expansion.

    Args:
        query: Search query (Korean or English)
        limit: Maximum number of results to return
    """
    graph = await _ensure_graph()
    result = await graph.search(query, limit=limit)

    if not result.nodes:
        return {"success": True, "message": "No knowledge found for this query.", "results": []}

    results = []
    for activated in result.nodes:
        node = activated.node
        results.append(
            {
                "id": node.id,
                "kind": str(node.kind),
                "title": node.title,
                "content": node.content[:500],
                "tags": node.tags,
                "level": str(node.level),
                "score": round(activated.resonance, 3),
            }
        )

    return {
        "success": True,
        "results": results,
        "total_candidates": result.total_candidates,
        "search_time_ms": round(result.search_time_ms, 1),
        "stages_used": result.stages_used,
    }


@server.tool()
async def knowledge_add(
    title: str,
    content: str,
    kind: str = "concept",
    tags: str = "",
    source: str = "",
) -> dict[str, Any]:
    """Add a new knowledge node to the graph.

    Args:
        title: Node title (concise summary)
        content: Full content/description
        kind: Node type — concept, entity, lesson, decision, rule, artifact, agent, task, sprint
        tags: Comma-separated tags (e.g. "deploy,ci/cd,automation")
        source: Origin of this knowledge (e.g. "sprint:123", "manual")
    """
    from synaptic.models import NodeKind

    graph = await _ensure_graph()

    try:
        node_kind = NodeKind(kind)
    except ValueError:
        return {"success": False, "message": f"Invalid kind: {kind}. Use: {', '.join(NodeKind)}"}

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    node = await graph.add(
        title=title,
        content=content,
        kind=node_kind,
        tags=tag_list,
        source=source,
    )

    return {
        "success": True,
        "node_id": node.id,
        "title": node.title,
        "kind": str(node.kind),
        "tags": node.tags,
    }


@server.tool()
async def knowledge_link(
    source_id: str,
    target_id: str,
    kind: str = "related",
    weight: float = 1.0,
) -> dict[str, Any]:
    """Create a link between two knowledge nodes.

    Args:
        source_id: Source node ID
        target_id: Target node ID
        kind: Edge type (related/caused/learned_from/depends_on/produced/contradicts/supersedes)
        weight: Connection strength (0.0 to 5.0)
    """
    from synaptic.models import EdgeKind

    graph = await _ensure_graph()

    try:
        edge_kind = EdgeKind(kind)
    except ValueError:
        return {"success": False, "message": f"Invalid kind: {kind}. Use: {', '.join(EdgeKind)}"}

    edge = await graph.link(source_id, target_id, kind=edge_kind, weight=weight)

    return {
        "success": True,
        "edge_id": edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "kind": str(edge.kind),
        "weight": edge.weight,
    }


@server.tool()
async def knowledge_reinforce(
    node_ids: str,
    success: bool = True,
) -> dict[str, Any]:
    """Reinforce knowledge nodes after use (Hebbian learning).

    Strengthens connections between co-activated nodes on success,
    weakens them on failure.

    Args:
        node_ids: Comma-separated node IDs to reinforce
        success: True if the knowledge was useful, False if not
    """
    graph = await _ensure_graph()
    ids = [nid.strip() for nid in node_ids.split(",") if nid.strip()]
    if not ids:
        return {"success": False, "message": "No node IDs provided"}

    await graph.reinforce(ids, success=success)
    return {
        "success": True,
        "reinforced": len(ids),
        "outcome": "success" if success else "failure",
    }


@server.tool()
async def knowledge_stats() -> dict[str, Any]:
    """Get knowledge graph statistics — node counts by kind and level, cache stats."""
    graph = await _ensure_graph()
    stats = await graph.stats()
    return {"success": True, **{k: v for k, v in stats.items()}}


@server.tool()
async def knowledge_export(
    output_format: str = "markdown",
) -> dict[str, Any]:
    """Export the knowledge graph.

    Args:
        output_format: Export format — "markdown" or "json"
    """
    graph = await _ensure_graph()

    if output_format == "json":
        content = await graph.export_json()
    else:
        content = await graph.export_markdown()

    return {"success": True, "format": output_format, "content": content}


@server.tool()
async def knowledge_consolidate() -> dict[str, Any]:
    """Run memory consolidation — expire old L0 nodes, promote accessed ones.

    L0 (72h TTL) → L1 (accessed 3+) → L2 (accessed 10+) → L3 (permanent, 80%+ success rate).
    Also runs vitality decay and edge pruning.
    """
    graph = await _ensure_graph()
    result = await graph.consolidate()
    decayed = await graph.decay()
    pruned = await graph.prune()

    return {
        "success": True,
        "nodes_promoted": len(result.nodes_updated),
        "nodes_created": len(result.nodes_created),
        "vitality_decayed": decayed,
        "edges_pruned": pruned,
    }


# --- Agent Workflow Tools ---


@server.tool()
async def agent_start_session(
    agent_id: str = "",
    description: str = "",
) -> dict[str, Any]:
    """Start an agent work session. All subsequent actions can be linked to this session.

    Args:
        agent_id: Identifier for the agent (e.g. "claude-code", "deploy-bot")
        description: What this session is about
    """
    tracker = await _ensure_tracker()
    session = await tracker.start_session(agent_id=agent_id, description=description)
    return {
        "success": True,
        "session_id": session.id,
        "agent_id": agent_id,
    }


@server.tool()
async def agent_log_action(
    session_id: str,
    tool_name: str,
    result: str = "",
    parameters: str = "",
    success: bool = True,
    duration_ms: float = 0.0,
) -> dict[str, Any]:
    """Log a tool call or action within an agent session.

    Args:
        session_id: Session ID from agent_start_session
        tool_name: Name of the tool that was called
        result: Summary of the tool's output
        parameters: JSON string of parameters passed to the tool
        success: Whether the tool call succeeded
        duration_ms: How long the tool call took in milliseconds
    """
    import json as _json

    tracker = await _ensure_tracker()
    params = _json.loads(parameters) if parameters else None
    node = await tracker.log_tool_call(
        session_id,
        tool_name=tool_name,
        parameters=params,
        result=result,
        success=success,
        duration_ms=duration_ms,
    )
    return {
        "success": True,
        "node_id": node.id,
        "tool_name": tool_name,
    }


@server.tool()
async def agent_record_decision(
    session_id: str,
    title: str,
    rationale: str,
    alternatives: str = "",
    context_node_ids: str = "",
) -> dict[str, Any]:
    """Record a decision made by the agent with rationale and considered alternatives.

    Args:
        session_id: Session ID from agent_start_session
        title: What was decided
        rationale: Why this choice was made
        alternatives: Comma-separated list of alternatives that were considered
        context_node_ids: Comma-separated IDs of related knowledge nodes
    """
    tracker = await _ensure_tracker()
    alt_list = [a.strip() for a in alternatives.split(",") if a.strip()] if alternatives else None
    ctx_ids = (
        [c.strip() for c in context_node_ids.split(",") if c.strip()] if context_node_ids else None
    )

    node = await tracker.record_decision(
        session_id,
        title=title,
        rationale=rationale,
        alternatives=alt_list,
        context_node_ids=ctx_ids,
    )
    return {
        "success": True,
        "decision_id": node.id,
        "title": title,
    }


@server.tool()
async def agent_record_outcome(
    decision_id: str,
    title: str,
    content: str,
    success: bool = True,
) -> dict[str, Any]:
    """Record the outcome of a previous decision. Triggers Hebbian learning.

    Args:
        decision_id: ID of the decision this outcome relates to
        title: Short summary of the outcome
        content: Detailed description of what happened
        success: Whether the outcome was positive
    """
    tracker = await _ensure_tracker()
    node = await tracker.record_outcome(
        decision_id,
        title=title,
        content=content,
        success=success,
    )
    return {
        "success": True,
        "outcome_id": node.id,
        "decision_id": decision_id,
        "outcome": "success" if success else "failure",
    }


# --- Semantic Search Tools ---


@server.tool()
async def agent_find_similar(
    query: str,
    intent: str = "general",
    context_tags: str = "",
    limit: int = 10,
) -> dict[str, Any]:
    """Search knowledge with agent-aware intent for smarter results.

    Intents:
    - similar_decisions: find past decisions on similar problems
    - past_failures: find what went wrong before
    - related_rules: find governing rules and constraints
    - reasoning_chain: follow decision → outcome → lesson paths
    - context_explore: explore neighborhood of a topic
    - general: standard hybrid search

    Args:
        query: Search query (Korean or English)
        intent: Search intent (see above)
        context_tags: Comma-separated tags for context-aware ranking
        limit: Maximum results
    """
    graph = await _ensure_graph()
    tags = [t.strip() for t in context_tags.split(",") if t.strip()] if context_tags else None

    try:
        result = await graph.agent_search(
            query,
            intent=intent,
            context_tags=tags,
            limit=limit,
        )
    except ValueError as e:
        return {"success": False, "message": str(e)}

    results = []
    for activated in result.nodes:
        node = activated.node
        results.append(
            {
                "id": node.id,
                "kind": str(node.kind),
                "title": node.title,
                "content": node.content[:500],
                "tags": node.tags,
                "score": round(activated.resonance, 3),
                "properties": node.properties,
            }
        )

    return {
        "success": True,
        "intent": intent,
        "results": results,
        "total_candidates": result.total_candidates,
        "search_time_ms": round(result.search_time_ms, 1),
        "stages_used": result.stages_used,
    }


@server.tool()
async def agent_get_reasoning_chain(
    decision_id: str,
) -> dict[str, Any]:
    """Get the full reasoning chain for a decision: decision → outcome → lessons learned.

    Args:
        decision_id: ID of the decision node to trace
    """
    tracker = await _ensure_tracker()
    graph = await _ensure_graph()

    decision = await graph.backend.get_node(decision_id)
    if decision is None:
        return {"success": False, "message": f"Decision {decision_id} not found"}

    chain = await tracker.get_decision_chain(decision_id)
    result = {
        "success": True,
        "decision": {
            "id": decision.id,
            "title": decision.title,
            "properties": decision.properties,
        },
        "chain": [
            {
                "id": node.id,
                "kind": str(node.kind),
                "title": node.title,
                "edge_kind": str(edge.kind),
                "properties": node.properties,
            }
            for node, edge in chain
        ],
    }
    return result


@server.tool()
async def agent_explore_context(
    node_id: str,
    depth: int = 2,
) -> dict[str, Any]:
    """Explore the knowledge graph around a specific node, following semantic relationships.

    Args:
        node_id: ID of the center node to explore from
        depth: How many hops to traverse (1-3)
    """
    graph = await _ensure_graph()
    node = await graph.backend.get_node(node_id)
    if node is None:
        return {"success": False, "message": f"Node {node_id} not found"}

    depth = max(1, min(3, depth))
    neighbors = await graph.backend.get_neighbors(node_id, depth=depth)

    return {
        "success": True,
        "center": {"id": node.id, "title": node.title, "kind": str(node.kind)},
        "neighbors": [
            {
                "id": n.id,
                "kind": str(n.kind),
                "title": n.title,
                "edge_kind": str(e.kind),
                "edge_weight": e.weight,
            }
            for n, e in neighbors
        ],
        "total": len(neighbors),
    }


# --- Ontology Tools ---


@server.tool()
async def ontology_define_type(
    name: str,
    parent: str = "",
    description: str = "",
    properties: str = "",
) -> dict[str, Any]:
    """Define or update a custom node/edge type in the ontology.

    Args:
        name: Type name (e.g. "incident", "api_endpoint")
        parent: Parent type for inheritance (e.g. "knowledge", "agent_activity")
        description: What this type represents
        properties: JSON array of property defs, e.g. [{"name":"severity","required":true}]
    """
    import json as _json

    from synaptic.ontology import PropertyDef, TypeDef

    graph = await _ensure_graph()
    ontology = graph.ontology
    if ontology is None:
        return {"success": False, "message": "Ontology not initialized"}

    props: list[PropertyDef] = []
    if properties:
        try:
            raw = _json.loads(properties)
            if isinstance(raw, list):
                for p in raw:
                    if isinstance(p, dict):
                        props.append(
                            PropertyDef(
                                name=str(p.get("name", "")),
                                value_type=str(p.get("value_type", "str")),
                                required=bool(p.get("required", False)),
                                default=str(p.get("default", "")),
                            )
                        )
        except _json.JSONDecodeError:
            return {"success": False, "message": "Invalid JSON in properties parameter"}

    try:
        ontology.register_type(
            TypeDef(
                name=name,
                parent=parent,
                properties=props,
                description=description,
            )
        )
    except ValueError as e:
        return {"success": False, "message": str(e)}

    return {
        "success": True,
        "type": name,
        "parent": parent,
        "properties_count": len(props),
    }


@server.tool()
async def ontology_query_schema(
    type_name: str = "",
) -> dict[str, Any]:
    """Query the ontology schema. Returns type definitions including inherited properties.

    Args:
        type_name: Specific type to query. If empty, returns all types.
    """
    graph = await _ensure_graph()
    ontology = graph.ontology
    if ontology is None:
        return {"success": False, "message": "Ontology not initialized"}

    if type_name:
        td = ontology.get_type(type_name)
        if td is None:
            return {"success": False, "message": f"Type '{type_name}' not found"}
        all_props = ontology.infer_properties(type_name)
        return {
            "success": True,
            "type": {
                "name": td.name,
                "parent": td.parent,
                "description": td.description,
                "ancestors": ontology.get_ancestors(type_name),
                "subtypes": ontology.subtypes_of(type_name),
                "properties": [
                    {"name": p.name, "type": p.value_type, "required": p.required}
                    for p in all_props
                ],
            },
        }

    # Return all types
    return {
        "success": True,
        "types": [
            {
                "name": td.name,
                "parent": td.parent,
                "description": td.description,
            }
            for td in ontology.all_types()
        ],
        "total": len(ontology.all_types()),
    }


# --- Agent tool layer (v0.12) -----------------------------------------------
#
# The knowledge_* / agent_* tools above are the v0.5+ single-shot API. The
# tools below are the v0.12 multi-turn agent layer: they share a
# SearchSession so the LLM can explore the graph iteratively, paginate
# through results, and ask structural questions (count, list, expand) the
# old tools didn't support. Both APIs coexist — old tools for backward
# compatibility, new tools for agentic use cases.

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
from synaptic.agent_tools_structured import (
    aggregate_nodes_tool,
    filter_nodes_tool,
    join_related_tool,
)
from synaptic.search_session import SessionStore

_session_store = SessionStore()


async def _ensure_backend() -> Any:
    """Ensure the graph is initialized and return the backend.

    Wraps ``_ensure_graph()`` and raises a clear error if the backend
    is still ``None`` after initialization (e.g. because the connect
    failed). Every agent tool calls this instead of ``_ensure_graph()``
    directly so we never pass a ``None`` backend into the tool layer.
    """
    await _ensure_graph()
    if _backend is None:
        msg = "Backend not initialized — check the --db / --dsn configuration"
        raise RuntimeError(msg)
    return _backend


async def _session(session_id: str | None) -> Any:
    """Resolve (or create) a SearchSession from an agent-supplied id."""
    return _session_store.get_or_create(session_id=session_id or None)


@server.tool()
async def agent_search(
    query: str,
    session_id: str = "",
    limit: int = 10,
    category: str = "",
    kind: str = "",
    exclude_seen: bool = True,
) -> dict[str, Any]:
    """Multi-turn search — finds evidence for a natural-language query.

    This is the primary agent entry point. It runs the 3rd-generation
    retrieval pipeline (anchor extraction → graph expansion → hybrid
    rerank → evidence aggregation) and returns a compact evidence list
    plus follow-up hints.

    The agent is expected to call this tool iteratively: read the
    evidence, decide whether to refine the query, narrow by category,
    or dive into a specific document with ``agent_get_document``, and
    call again. The ``session_id`` threads state across calls so
    already-seen nodes are filtered out automatically.

    Args:
        query: Natural-language query. Korean or English both work.
        session_id: Session to continue. Omit to start a fresh session
            (a new id will be returned in the response).
        limit: Max evidence items per call. 5-12 is the typical range.
        category: Optional category label filter ("규정 및 지침" etc.).
        kind: Optional NodeKind filter ("rule", "chunk", ...).
        exclude_seen: When True, results already returned in this
            session are filtered out so the agent paginates.
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await search_tool(
        backend,
        session,
        query,
        limit=limit,
        category=category or None,
        kind=kind or None,
        exclude_seen=exclude_seen,
        embedder=_embedder,
    )
    return result.to_dict()


@server.tool()
async def agent_expand(
    node_id: str,
    session_id: str = "",
    limit: int = 10,
    exclude_seen: bool = True,
) -> dict[str, Any]:
    """Walk one graph hop out from a specific node.

    Use this when ``agent_search`` returned a promising result and you
    want to see its neighbours — sibling chunks in the same document,
    other documents in the same category, or the next chunk in a
    sequence.

    Args:
        node_id: Id of the node to expand (from a previous tool result).
        session_id: Session to continue.
        limit: Max neighbours to return.
        exclude_seen: Skip neighbours the agent has already seen.
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await expand_tool(
        backend,
        session,
        node_id,
        limit=limit,
        exclude_seen=exclude_seen,
    )
    return result.to_dict()


@server.tool()
async def agent_get_document(
    doc_id: str,
    session_id: str = "",
    query: str = "",
    max_chunks: int = 50,
) -> dict[str, Any]:
    """Fetch a document with smart context control.

    When ``query`` is provided, only the most relevant chunks get full
    text (default 5). The rest are returned as one-line summaries. This
    keeps context under ~2K tokens instead of ~5K+ for a typical doc.

    Args:
        doc_id: Document id or node id.
        session_id: Session to continue.
        query: Optional query for chunk relevance scoring.
        max_chunks: Total chunks to fetch.
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await get_document_tool(
        backend, session, doc_id, query=query, max_chunks=max_chunks
    )
    return result.to_dict()


@server.tool()
async def agent_list_categories(
    session_id: str = "",
    limit: int = 100,
) -> dict[str, Any]:
    """List all top-level categories in the knowledge graph.

    Use this early in a session to build a mental map of what the
    graph contains before searching. Each category comes with a
    document count so the agent can judge where to look first.
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await list_categories_tool(backend, session, limit=limit)
    return result.to_dict()


@server.tool()
async def agent_count(
    session_id: str = "",
    kind: str = "",
    category: str = "",
    year: int = 0,
) -> dict[str, Any]:
    """Count how many nodes match a filter without fetching them.

    Use this to decide whether an "enumerate everything" question is
    even feasible. If the count is small the agent can iterate; if it's
    huge the agent needs a narrower filter first.

    Args:
        session_id: Session to continue.
        kind: Optional NodeKind filter.
        category: Optional category label filter.
        year: Optional year filter. ``0`` means "no year filter".
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await count_tool(
        backend,
        session,
        kind=kind or None,
        category=category or None,
        year=year if year > 0 else None,
    )
    return result.to_dict()


@server.tool()
async def agent_search_exact(
    identifier: str,
    session_id: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """Literal substring match for codes, IDs, or exact strings.

    Use this when BM25 / FTS would dilute the search — e.g. error
    codes ("E217"), SKUs ("SKU-1234"), function names, Jira keys,
    or section numbers ("4.3.2"). Bypasses tokenisation entirely.

    Args:
        identifier: The exact string to search for.
        session_id: Session to continue.
        limit: Max matches to return.
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await search_exact_tool(backend, session, identifier, limit=limit)
    return result.to_dict()


@server.tool()
async def agent_follow(
    node_id: str,
    edge_kind: str,
    session_id: str = "",
    direction: str = "both",
    limit: int = 20,
) -> dict[str, Any]:
    """Walk a specific edge type from a starting node.

    Surgical alternative to ``agent_expand`` when you know exactly
    which relation you want to follow. Valid edge kinds include
    ``contains``, ``part_of``, ``next_chunk``, ``mentions``,
    ``related``, ``cites``.

    Args:
        node_id: Source node.
        edge_kind: Edge kind to follow.
        session_id: Session to continue.
        direction: ``"outgoing"``, ``"incoming"``, or ``"both"``.
        limit: Max neighbours to return.
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await follow_tool(
        backend,
        session,
        node_id,
        edge_kind,
        direction=direction,
        limit=limit,
    )
    return result.to_dict()


@server.tool()
async def agent_session_info(session_id: str = "") -> dict[str, Any]:
    """Inspect the current state of an agent session.

    Returns how many tool calls have been used, budget remaining,
    which categories have been explored, and the last few queries.
    Use this when you want to reason about coverage or when you're
    deciding whether to stop exploring and answer.
    """
    session = await _session(session_id)
    return {
        "tool": "session_info",
        "ok": True,
        "session": session.summary(),
    }


@server.tool()
async def agent_deep_search(
    query: str,
    session_id: str = "",
    limit: int = 5,
    category: str = "",
) -> dict[str, Any]:
    """Deep search — search + expand + read documents in ONE call.

    This is the recommended tool for most questions. It internally
    chains search → expand → get_document so you get evidence,
    neighbours, AND document excerpts in a single turn instead of 3-5.

    Use ``category`` to narrow the search when you know the topic area.
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await deep_search_tool(
        backend, session, query,
        limit=limit,
        category=category or None,
        embedder=_embedder,
    )
    return result.to_dict()


@server.tool()
async def agent_compare_search(
    query: str,
    session_id: str = "",
) -> dict[str, Any]:
    """Compare search — decompose multi-topic query and search in parallel.

    For questions like "A와 B의 관계" or "X 및 Y 비교", this tool
    automatically splits into sub-queries, searches each independently
    (possibly with different category filters), and merges results.

    One turn instead of 4-6 for cross-document questions.
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await compare_search_tool(
        backend, session, query, embedder=_embedder
    )
    return result.to_dict()


@server.tool()
async def agent_filter_nodes(
    property: str,
    op: str,
    value: str,
    table: str = "",
    session_id: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """Filter nodes by property value — for structured/tabular data.

    Queries typed properties stored in the graph. Supports numeric
    comparison (>=, <=, >, <, ==) and text containment.

    Examples:
      filter_nodes(property="selling_price", op=">=", value="90000")
      filter_nodes(table="reviews", property="attribute_2_value", op="contains", value="타이트")
      filter_nodes(property="broadcast_date", op="contains", value="2024-11")
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await filter_nodes_tool(
        backend, session, table=table or "",
        property=property, op=op, value=value, limit=limit,
    )
    return result.to_dict()


@server.tool()
async def agent_aggregate_nodes(
    group_by: str,
    table: str = "",
    metric: str = "count",
    session_id: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """Aggregate nodes by property — GROUP BY + COUNT/SUM/AVG.

    For questions like "색상별 상품 수" or "시즌별 매출 합계".

    Examples:
      aggregate_nodes(table="products", group_by="season", metric="count")
      aggregate_nodes(table="product_variants", group_by="color_id", metric="count")
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await aggregate_nodes_tool(
        backend, session, table=table or "",
        group_by=group_by, metric=metric, limit=limit,
    )
    return result.to_dict()


@server.tool()
async def agent_join_related(
    from_value: str,
    fk_property: str,
    target_table: str,
    session_id: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """Follow a foreign key to find related records.

    Like SQL JOIN: finds all nodes in target_table where
    fk_property equals from_value.

    Examples:
      join_related(from_value="12800000", fk_property="product_code", target_table="reviews")
      → all reviews for product 12800000
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await join_related_tool(
        backend, session, from_value=from_value,
        fk_property=fk_property, target_table=target_table, limit=limit,
    )
    return result.to_dict()


def main() -> None:
    """Entry point for synaptic-mcp command."""
    global _db_path, _dsn, _embed_url, _embed_model

    if "--version" in sys.argv:
        print(f"synaptic-mcp {__version__}")
        return

    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Usage: synaptic-mcp [OPTIONS]\n"
            "\n"
            "Options:\n"
            "  --db PATH          SQLite database path (default: knowledge.db)\n"
            "  --dsn DSN          PostgreSQL connection string\n"
            "  --embed-url URL    Embedding API base URL (OpenAI-compatible)\n"
            "                     Examples: http://localhost:8080/v1 (vLLM/llama.cpp)\n"
            "                              http://localhost:11434/v1 (Ollama)\n"
            "  --embed-model NAME Embedding model name (default: 'default')\n"
            "  --version          Show version\n"
        )
        return

    # Parse args
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--db" and i + 1 < len(args):
            _db_path = args[i + 1]
        elif arg == "--dsn" and i + 1 < len(args):
            _dsn = args[i + 1]
        elif arg == "--embed-url" and i + 1 < len(args):
            _embed_url = args[i + 1]
        elif arg == "--embed-model" and i + 1 < len(args):
            _embed_model = args[i + 1]

    # Configure logging to stderr (stdout is MCP protocol)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    logger.info("Starting Synaptic Memory MCP server (db=%s, dsn=%s)", _db_path, _dsn or "none")
    if _embed_url:
        logger.info("Embedding: %s (model=%s)", _embed_url, _embed_model)
    server.run()


if __name__ == "__main__":
    main()
