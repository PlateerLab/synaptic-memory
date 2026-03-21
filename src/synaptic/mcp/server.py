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
_db_path: str = "knowledge.db"
_dsn: str = ""


async def _ensure_graph() -> Any:
    """Lazy-initialize the SynapticGraph on first use."""
    global _graph, _backend

    if _graph is not None:
        return _graph

    from synaptic.extensions.tagger_regex import RegexTagExtractor  # noqa: PLC0415
    from synaptic.graph import SynapticGraph  # noqa: PLC0415

    if _dsn:
        from synaptic.backends.postgresql import PostgreSQLBackend  # noqa: PLC0415

        _backend = PostgreSQLBackend(_dsn)
    else:
        from synaptic.backends.sqlite import SQLiteBackend  # noqa: PLC0415

        _backend = SQLiteBackend(_db_path)

    await _backend.connect()
    _graph = SynapticGraph(_backend, tag_extractor=RegexTagExtractor())
    logger.info("Knowledge graph initialized (backend=%s)", type(_backend).__name__)
    return _graph


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


def main() -> None:
    """Entry point for synaptic-mcp command."""
    global _db_path, _dsn

    if "--version" in sys.argv:
        print(f"synaptic-mcp {__version__}")
        return

    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Usage: synaptic-mcp [--db PATH] [--dsn DSN]\n"
            "\n"
            "Options:\n"
            "  --db PATH    SQLite database path (default: knowledge.db)\n"
            "  --dsn DSN    PostgreSQL connection string\n"
            "  --version    Show version\n"
        )
        return

    # Parse --db and --dsn args
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--db" and i + 1 < len(args):
            _db_path = args[i + 1]
        elif arg == "--dsn" and i + 1 < len(args):
            _dsn = args[i + 1]

    # Configure logging to stderr (stdout is MCP protocol)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    logger.info("Starting Synaptic Memory MCP server (db=%s, dsn=%s)", _db_path, _dsn or "none")
    server.run()


if __name__ == "__main__":
    main()
