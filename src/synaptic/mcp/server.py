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
_tracker: Any = None
_db_path: str = "knowledge.db"
_dsn: str = ""
_embed_url: str = ""
_embed_model: str = "default"


async def _ensure_graph() -> Any:
    """Lazy-initialize the SynapticGraph on first use."""
    global _graph, _backend

    if _graph is not None:
        return _graph

    from synaptic.extensions.tagger_regex import RegexTagExtractor  # noqa: PLC0415
    from synaptic.graph import SynapticGraph  # noqa: PLC0415
    from synaptic.ontology import build_agent_ontology  # noqa: PLC0415

    if _dsn:
        from synaptic.backends.postgresql import PostgreSQLBackend  # noqa: PLC0415

        _backend = PostgreSQLBackend(_dsn)
    else:
        from synaptic.backends.sqlite import SQLiteBackend  # noqa: PLC0415

        _backend = SQLiteBackend(_db_path)

    await _backend.connect()

    # Auto-embedding: connect to any OpenAI-compatible endpoint
    embedder = None
    if _embed_url:
        from synaptic.extensions.embedder import OpenAIEmbeddingProvider  # noqa: PLC0415

        embedder = OpenAIEmbeddingProvider(api_base=_embed_url, model=_embed_model)
        logger.info("Embedder configured: %s (model=%s)", _embed_url, _embed_model)

    _graph = SynapticGraph(
        _backend,
        tag_extractor=RegexTagExtractor(),
        ontology=build_agent_ontology(),
        embedder=embedder,
    )
    logger.info("Knowledge graph initialized (backend=%s)", type(_backend).__name__)
    return _graph


async def _ensure_tracker() -> Any:
    """Lazy-initialize the ActivityTracker."""
    global _tracker

    if _tracker is not None:
        return _tracker

    from synaptic.activity import ActivityTracker  # noqa: PLC0415

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
    from synaptic.models import NodeKind  # noqa: PLC0415

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
    from synaptic.models import EdgeKind  # noqa: PLC0415

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
    import json as _json  # noqa: PLC0415

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
        [c.strip() for c in context_node_ids.split(",") if c.strip()]
        if context_node_ids else None
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
            query, intent=intent, context_tags=tags, limit=limit,
        )
    except ValueError as e:
        return {"success": False, "message": str(e)}

    results = []
    for activated in result.nodes:
        node = activated.node
        results.append({
            "id": node.id,
            "kind": str(node.kind),
            "title": node.title,
            "content": node.content[:500],
            "tags": node.tags,
            "score": round(activated.resonance, 3),
            "properties": node.properties,
        })

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
    import json as _json  # noqa: PLC0415

    from synaptic.ontology import PropertyDef, TypeDef  # noqa: PLC0415

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
                        props.append(PropertyDef(
                            name=str(p.get("name", "")),
                            value_type=str(p.get("value_type", "str")),
                            required=bool(p.get("required", False)),
                            default=str(p.get("default", "")),
                        ))
        except _json.JSONDecodeError:
            return {"success": False, "message": "Invalid JSON in properties parameter"}

    try:
        ontology.register_type(TypeDef(
            name=name,
            parent=parent,
            properties=props,
            description=description,
        ))
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
