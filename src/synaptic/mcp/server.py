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
_source_dsn: str = ""  # Default source DB for CDC sync tools (optional)
_embed_url: str = ""
_embed_model: str = "default"
# Vector cascade tuning — see synaptic.search.HybridSearch docstring
# for the per-embedder cosine distribution guide. None means "use
# the package default" (DEFAULT_VECTOR_MIN_COSINE / RELATIVE_DROP),
# which is also overridable via the SYNAPTIC_VECTOR_* env vars.
_vector_min_cosine: float | None = None
_vector_relative_drop: float | None = None


async def _ensure_graph() -> Any:
    """Lazy-initialize the SynapticGraph on first use."""
    global _graph, _backend, _embedder

    if _graph is not None:
        return _graph

    from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
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

    # A ChunkEntityIndex is required for `add_document` to produce
    # nodes of NodeKind.CHUNK — without it chunks default to
    # NodeKind.CONCEPT and the PART_OF validation constraint
    # rejects the inter-chunk edges.
    _graph = SynapticGraph(
        _backend,
        tag_extractor=RegexTagExtractor(),
        ontology=build_agent_ontology(),
        embedder=_embedder,
        chunk_entity_index=ChunkEntityIndex(),
        vector_min_cosine=_vector_min_cosine,
        vector_relative_drop=_vector_relative_drop,
    )
    logger.info(
        "Knowledge graph initialized (backend=%s, vector_min_cos=%s, vector_rel_drop=%s)",
        type(_backend).__name__,
        _vector_min_cosine if _vector_min_cosine is not None else "default",
        _vector_relative_drop if _vector_relative_drop is not None else "default",
    )
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


# --- Ingest Tools ---


@server.tool()
async def knowledge_add_document(
    title: str,
    content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    tags: str = "",
    source: str = "",
) -> dict[str, Any]:
    """Add a long document to the graph with automatic chunking.

    Short documents become a single node; long documents are split at
    sentence boundaries and connected with NEXT_CHUNK edges so the
    search pipeline can surface context around a hit.

    Args:
        title: Document title (becomes the node title and a chunk prefix).
        content: Full document text.
        chunk_size: Max characters per chunk (default 1000).
        chunk_overlap: Overlapping characters between adjacent chunks (default 200).
        tags: Comma-separated tags.
        source: Origin identifier (e.g. "manual:admin-guide", "url:https://...").
    """
    graph = await _ensure_graph()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    nodes = await graph.add_document(
        title=title,
        content=content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tags=tag_list,
        source=source,
    )
    return {
        "success": True,
        "title": title,
        "chunks": len(nodes),
        "first_node_id": nodes[0].id if nodes else None,
    }


@server.tool()
async def knowledge_add_table(
    table_name: str,
    columns: list[dict[str, str]],
    rows: list[dict[str, Any]],
    primary_key: str = "id",
    foreign_keys: dict[str, list[str]] | None = None,
    tags: str = "",
    source: str = "",
) -> dict[str, Any]:
    """Ingest a structured table into the graph.

    Each row becomes an ENTITY node and foreign keys become RELATED
    edges to the referenced table's rows. The table schema is
    auto-registered in the ontology so downstream filter / aggregate /
    join tools work immediately.

    Args:
        table_name: Logical table name (used for ontology type + node titles).
        columns: Column definitions, e.g. ``[{"name": "id", "type": "int"}, ...]``.
        rows: Row data, e.g. ``[{"id": 1, "name": "..."}, ...]``.
        primary_key: Primary key column (default "id").
        foreign_keys: Mapping ``{"col": ["target_table", "target_col"]}``.
            JSON-friendly shape — ``["target_table", "target_col"]`` is
            converted to a tuple internally.
        tags: Comma-separated tags.
        source: Origin identifier.
    """
    graph = await _ensure_graph()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    fk_map: dict[str, tuple[str, str]] | None = None
    if foreign_keys:
        fk_map = {
            col: (target[0], target[1]) if len(target) >= 2 else (target[0], "id")
            for col, target in foreign_keys.items()
        }
    nodes = await graph.add_table(
        table_name,
        columns,
        rows,
        primary_key=primary_key,
        foreign_keys=fk_map,
        tags=tag_list,
        source=source,
    )
    return {
        "success": True,
        "table_name": table_name,
        "rows_ingested": len(nodes),
        "fk_count": len(fk_map) if fk_map else 0,
    }


@server.tool()
async def knowledge_add_chunks(
    chunks: list[dict[str, Any]],
    default_source: str = "",
) -> dict[str, Any]:
    """Ingest pre-chunked content (BYO-chunker workflow).

    Use when you have already split a document with your own parser
    (LangChain, Unstructured, custom OCR, ...) and want to hand the
    chunks directly to the graph. Each chunk dict should contain:

    - ``title`` (required): Node title for the chunk.
    - ``content`` (required): Chunk text.
    - ``tags`` (optional): List of tag strings.
    - ``source`` (optional): Per-chunk source identifier. Falls back
      to ``default_source`` when omitted.
    - ``properties`` (optional): Extra string→string metadata.
    """
    graph = await _ensure_graph()
    added = 0
    errors: list[str] = []
    first_id: str | None = None
    for i, chunk in enumerate(chunks):
        title = chunk.get("title")
        content = chunk.get("content")
        if not title or not content:
            errors.append(f"chunk[{i}]: missing title or content")
            continue
        node = await graph.add(
            title=title,
            content=content,
            tags=chunk.get("tags"),
            source=chunk.get("source") or default_source,
            properties=chunk.get("properties"),
        )
        if first_id is None:
            first_id = node.id
        added += 1
    return {
        "success": True,
        "chunks_added": added,
        "errors": errors,
        "first_node_id": first_id,
    }


def _inspect_path(path: str) -> dict[str, Any]:
    """Sync helper: classify a filesystem path for ingest routing.

    Keeping the file-system touch in a sync function lets the async
    tool body stay blocking-I/O-free (ruff ASYNC230/ASYNC240).
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return {"exists": False}
    return {
        "exists": True,
        "is_file": p.is_file(),
        "suffix": p.suffix.lower(),
        "stem": p.stem,
        "path": str(p),
    }


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    """Sync helper: read a CSV file into a list of dict rows."""
    import csv
    from pathlib import Path

    with Path(path).open(encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _read_jsonl_records(path: str) -> list[dict[str, Any]]:
    """Sync helper: read a JSONL file into a list of record dicts."""
    import json
    from pathlib import Path

    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return records


def _read_text_file(path: str) -> str:
    """Sync helper: read a text file with error-tolerant decoding."""
    from pathlib import Path

    return Path(path).read_text(encoding="utf-8", errors="replace")


@server.tool()
async def knowledge_ingest_path(
    path: str,
    source: str = "",
) -> dict[str, Any]:
    """Ingest a file from the local filesystem into the *current* graph.

    Handles CSV, JSONL, and plain text files. For directories or
    office files (PDF/DOCX/...), use ``SynapticGraph.from_data()``
    from a CLI script and point synaptic-mcp at the resulting ``.db``.

    The MCP server and the MCP client must share a filesystem for
    this tool to be useful.

    Args:
        path: Absolute filesystem path.
        source: Source identifier attached to every new node.
    """
    graph = await _ensure_graph()
    info = _inspect_path(path)
    if not info["exists"]:
        return {"success": False, "error": f"path not found: {path}"}

    if info["is_file"] and info["suffix"] == ".csv":
        from synaptic.extensions.table_ingester import TableIngester

        rows = _read_csv_rows(info["path"])
        if not rows:
            return {"success": True, "format": "csv", "rows": 0}
        columns = [{"name": k, "type": "str"} for k in rows[0]]
        ingester = TableIngester()
        nodes = await ingester.ingest(
            graph,
            info["stem"],
            columns,
            rows,
            source=source or info["path"],
        )
        return {
            "success": True,
            "format": "csv",
            "table_name": info["stem"],
            "rows": len(nodes),
        }

    if info["is_file"] and info["suffix"] == ".jsonl":
        records = _read_jsonl_records(info["path"])
        count = 0
        for obj in records:
            title = obj.get("title") or obj.get("id") or f"doc-{count}"
            content = obj.get("content") or obj.get("text") or ""
            if not content:
                continue
            await graph.add_document(
                title=str(title),
                content=str(content),
                source=source or info["path"],
            )
            count += 1
        return {"success": True, "format": "jsonl", "documents": count}

    if info["is_file"]:
        try:
            text = _read_text_file(info["path"])
        except (OSError, UnicodeDecodeError) as exc:
            return {"success": False, "error": f"cannot read {path}: {exc}"}
        if not text.strip():
            return {"success": True, "format": "text", "documents": 0}
        nodes = await graph.add_document(
            title=info["stem"],
            content=text,
            source=source or info["path"],
        )
        return {
            "success": True,
            "format": "text",
            "title": info["stem"],
            "chunks": len(nodes),
        }

    return {
        "success": False,
        "error": (
            "directory ingest not supported from MCP yet — "
            "run a CLI job with SynapticGraph.from_data() and point "
            "synaptic-mcp at the resulting .db file"
        ),
    }


@server.tool()
async def knowledge_remove(node_id: str) -> dict[str, Any]:
    """Delete a single node and cascade-remove its edges.

    Use when a node was ingested incorrectly or is stale. Bulk
    deletion is intentionally not exposed — for large cleanups
    drop the graph file and re-ingest.
    """
    graph = await _ensure_graph()
    removed = await graph.remove(node_id)
    return {"success": removed, "node_id": node_id}


@server.tool()
async def knowledge_sync_from_database(
    connection_string: str = "",
    tables: list[str] | None = None,
) -> dict[str, Any]:
    """Incrementally sync the graph with a live database (CDC).

    First call on a fresh graph seeds the sync state and does a
    deterministic full load; subsequent calls read only rows whose
    change column advanced past the last watermark (or whose row
    hash changed, for tables without an ``updated_at``-style
    column). Tables without a primary key in the source schema are
    skipped with a clear error entry.

    Args:
        connection_string: Source database DSN. Falls back to
            ``--source-dsn`` passed on the command line when omitted.
            Supports ``sqlite://``, ``postgresql://``, ``mysql://``.
        tables: Optional allow-list of table names. Empty / null
            means sync every table in the source schema.
    """
    graph = await _ensure_graph()
    dsn = connection_string or _source_dsn
    if not dsn:
        return {
            "success": False,
            "error": (
                "no source DSN — either pass connection_string or start "
                "synaptic-mcp with --source-dsn"
            ),
        }
    result = await graph.sync_from_database(dsn, tables=tables)
    return {
        "success": True,
        "added": result.added,
        "updated": result.updated,
        "deleted": result.deleted,
        "elapsed_ms": round(result.elapsed_ms, 1),
        "tables": [
            {
                "table": t.table,
                "strategy": t.strategy,
                "added": t.added,
                "updated": t.updated,
                "deleted": t.deleted,
                "fk_edges_added": t.fk_edges_added,
                "fk_edges_removed": t.fk_edges_removed,
                "error": t.error,
            }
            for t in result.tables
        ],
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
from synaptic.agent_tools_structured import (
    aggregate_nodes_tool,
    filter_nodes_tool,
    join_related_tool,
)
from synaptic.agent_tools_v2 import (
    compare_search_tool,
    deep_search_tool,
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
    result = await get_document_tool(backend, session, doc_id, query=query, max_chunks=max_chunks)
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
        backend,
        session,
        query,
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
    result = await compare_search_tool(backend, session, query, embedder=_embedder)
    return result.to_dict()


@server.tool()
async def agent_filter_nodes(
    property: str,
    op: str,
    value: str,
    table: str = "",
    session_id: str = "",
    limit: int = 20,
    from_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Filter nodes by property value — for structured/tabular data.

    Supports numeric comparison (>=, <=, >, <, ==), text containment,
    prefix matching (starts_with), and date ranges (date_range with
    ``YYYY-MM-DD..YYYY-MM-DD``). Pass ``from_ids`` with node titles
    from a previous step to chain multi-hop queries.

    Examples:
      filter_nodes(property="selling_price", op=">=", value="90000")
      filter_nodes(property="sold_dtm", op="starts_with", value="2023-12")
      filter_nodes(property="sold_dtm", op="date_range", value="2023-06-01..2023-08-31")
      filter_nodes(from_ids=["products:12800000"], property="discount_rate", op=">", value="10")
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await filter_nodes_tool(
        backend,
        session,
        table=table or "",
        property=property,
        op=op,
        value=value,
        limit=limit,
        from_ids=from_ids,
    )
    return result.to_dict()


@server.tool()
async def agent_aggregate_nodes(
    group_by: str,
    table: str = "",
    metric: str = "count",
    metric_property: str = "",
    where_property: str = "",
    where_op: str = "",
    where_value: str = "",
    group_by_format: str = "",
    session_id: str = "",
    limit: int = 50,
    from_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate nodes by property — GROUP BY + COUNT/SUM/AVG.

    Supports optional WHERE pre-filter for conditional aggregation,
    date bucketing via ``group_by_format``, and multi-hop chaining
    via ``from_ids``.

    Examples:
      aggregate_nodes(table="products", group_by="season", metric="count")
      aggregate_nodes(table="feedback", group_by="goods_no", metric="count",
                      where_property="score", where_op="==", where_value="5")
      aggregate_nodes(table="sold_hist", group_by="sold_dtm",
                      group_by_format="YYYY-MM", metric="count")
      aggregate_nodes(from_ids=top_products, group_by="category", metric="count")
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await aggregate_nodes_tool(
        backend,
        session,
        table=table or "",
        group_by=group_by,
        metric=metric,
        metric_property=metric_property,
        where_property=where_property,
        where_op=where_op,
        where_value=where_value,
        group_by_format=group_by_format,
        limit=limit,
        from_ids=from_ids,
    )
    return result.to_dict()


@server.tool()
async def agent_join_related(
    fk_property: str,
    target_table: str,
    from_value: str = "",
    from_values: list[str] | None = None,
    session_id: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """Follow a foreign key to find related records.

    Accepts either ``from_value`` (single) or ``from_values`` (batch)
    for multi-hop chaining. The IN-clause JOIN variant makes it easy
    to pass aggregate top-K results directly.

    Examples:
      join_related(from_value="12800000", fk_property="product_code", target_table="reviews")
      join_related(from_values=["G00001","G00007"], fk_property="goods_no", target_table="pr_goods_sold_hist")
    """
    backend = await _ensure_backend()
    session = await _session(session_id)
    result = await join_related_tool(
        backend,
        session,
        from_value=from_value,
        from_values=from_values,
        fk_property=fk_property,
        target_table=target_table,
        limit=limit,
    )
    return result.to_dict()


def main() -> None:
    """Entry point for synaptic-mcp command."""
    global _db_path, _dsn, _source_dsn, _embed_url, _embed_model
    global _vector_min_cosine, _vector_relative_drop

    if "--version" in sys.argv:
        print(f"synaptic-mcp {__version__}")
        return

    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Usage: synaptic-mcp [OPTIONS]\n"
            "\n"
            "Options:\n"
            "  --db PATH                  SQLite database path for the graph (default: knowledge.db)\n"
            "  --dsn DSN                  PostgreSQL backend for the graph itself\n"
            "  --source-dsn DSN           Default source database for CDC sync tools\n"
            "                             (sqlite://, postgresql://, mysql://). Optional —\n"
            "                             the knowledge_sync_from_database tool accepts a\n"
            "                             per-call connection_string too.\n"
            "  --embed-url URL            Embedding API base URL (OpenAI-compatible)\n"
            "                             Examples: http://localhost:8080/v1 (vLLM/llama.cpp)\n"
            "                                       http://localhost:11434/v1 (Ollama)\n"
            "  --embed-model NAME         Embedding model name (default: 'default')\n"
            "  --vector-min-cosine FLOAT  Absolute noise floor for vector cascade (default 0.10)\n"
            "  --vector-relative-drop FLOAT\n"
            "                             Fraction below the top vector hit that is still\n"
            "                             accepted (default 0.30 → keep cosines within top*0.70).\n"
            "                             Lower = stricter, higher = looser. The relative\n"
            "                             cutoff makes the search embedder-agnostic.\n"
            "  --version                  Show version\n"
            "\n"
            "Vector tuning can also come from env vars:\n"
            "  SYNAPTIC_VECTOR_MIN_COSINE, SYNAPTIC_VECTOR_RELATIVE_DROP\n"
        )
        return

    # Parse args
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--db" and i + 1 < len(args):
            _db_path = args[i + 1]
        elif arg == "--dsn" and i + 1 < len(args):
            _dsn = args[i + 1]
        elif arg == "--source-dsn" and i + 1 < len(args):
            _source_dsn = args[i + 1]
        elif arg == "--embed-url" and i + 1 < len(args):
            _embed_url = args[i + 1]
        elif arg == "--embed-model" and i + 1 < len(args):
            _embed_model = args[i + 1]
        elif arg == "--vector-min-cosine" and i + 1 < len(args):
            try:
                _vector_min_cosine = float(args[i + 1])
            except ValueError:
                print(f"--vector-min-cosine must be a float, got {args[i + 1]!r}", file=sys.stderr)
                sys.exit(2)
        elif arg == "--vector-relative-drop" and i + 1 < len(args):
            try:
                _vector_relative_drop = float(args[i + 1])
            except ValueError:
                print(
                    f"--vector-relative-drop must be a float, got {args[i + 1]!r}",
                    file=sys.stderr,
                )
                sys.exit(2)

    # Configure logging to stderr (stdout is MCP protocol)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    logger.info(
        "Starting Synaptic Memory MCP server (db=%s, dsn=%s, source_dsn=%s)",
        _db_path,
        _dsn or "none",
        _source_dsn or "none",
    )
    if _embed_url:
        logger.info("Embedding: %s (model=%s)", _embed_url, _embed_model)
    server.run()


if __name__ == "__main__":
    main()
