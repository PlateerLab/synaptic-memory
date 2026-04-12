"""Kuzu storage backend — embedded property graph DB with native Cypher.

Kuzu is an embedded graph database (like SQLite for graphs) with:
- Property graph model (nodes + typed relationships with properties)
- openCypher query language
- Built-in FTS extension (Okapi BM25)
- Built-in vector extension (HNSW)
- Built-in algo extension (PageRank, shortest path)
- Async Python API (`kuzu.AsyncConnection`)
- Single-file storage, zero-config deployment
- MIT license

This backend uses a single `Node` table and a single `Edge` REL table,
with `kind` columns to discriminate between node/edge types. This mirrors
the schemaless model used by Neo4jBackend while staying within Kuzu's
statically-typed schema requirements.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from synaptic.backends._scoring import bm25_hybrid_score, fuzzy_score
from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
)

try:
    import kuzu
except ImportError as e:
    msg = "Install synaptic-memory[kuzu] for Kuzu backend: pip install synaptic-memory[kuzu]"
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)


_NODE_TABLE_DDL = """
CREATE NODE TABLE IF NOT EXISTS Node (
    id STRING,
    kind STRING,
    title STRING,
    content STRING,
    tags_json STRING,
    level STRING,
    vitality DOUBLE,
    access_count INT64,
    success_count INT64,
    failure_count INT64,
    source STRING,
    properties_json STRING,
    created_at DOUBLE,
    updated_at DOUBLE,
    PRIMARY KEY (id)
)
"""

_EDGE_TABLE_DDL = """
CREATE REL TABLE IF NOT EXISTS Edge (
    FROM Node TO Node,
    id STRING,
    kind STRING,
    weight DOUBLE,
    created_at DOUBLE
)
"""


class KuzuBackend:
    """Embedded Kuzu graph database backend.

    Implements StorageBackend protocol + GraphTraversal extensions.

    Storage model:
      - Single ``Node`` table with all properties (kind discriminates type)
      - Single ``Edge`` REL table (kind discriminates relationship type)
      - Optional FTS index on (title, content) via Kuzu fts extension

    Concurrency:
      - Kuzu allows multiple readers and a single writer per database
      - This backend uses a single AsyncConnection; serialize writes from caller
        side if you fan out across multiple coroutines
    """

    __slots__ = ("_async_conn", "_conn", "_db", "_path")

    def __init__(self, path: str | Path = ":memory:") -> None:
        # Kuzu requires an explicit directory; use a tmp path for in-memory mode
        if str(path) == ":memory:":
            import tempfile

            self._path = tempfile.mkdtemp(prefix="kuzu-mem-")
        else:
            self._path = str(path)
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None
        self._async_conn: kuzu.AsyncConnection | None = None

    async def connect(self) -> None:
        # Database / Connection construction is sync but cheap
        self._db = kuzu.Database(self._path)
        self._conn = kuzu.Connection(self._db)
        # AsyncConnection wraps a thread pool internally
        self._async_conn = kuzu.AsyncConnection(self._db)

        # Schema bootstrap (sync, runs once)
        self._conn.execute(_NODE_TABLE_DDL)
        self._conn.execute(_EDGE_TABLE_DDL)

        logger.info("Kuzu connected: %s", self._path)

    async def close(self) -> None:
        # Kuzu connections close on GC; explicit close not strictly needed
        self._async_conn = None
        self._conn = None
        self._db = None

    def _get_async(self) -> kuzu.AsyncConnection:
        if self._async_conn is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._async_conn

    def _get_sync(self) -> kuzu.Connection:
        if self._conn is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._conn

    async def _execute(self, query: str, params: dict[str, Any] | None = None) -> kuzu.QueryResult:
        """Run a Cypher query asynchronously and return the QueryResult."""
        conn = self._get_async()
        if params is None:
            return await conn.execute(query)
        return await conn.execute(query, parameters=params)

    # --- Node CRUD ---

    async def save_node(self, node: Node) -> None:
        # Upsert: check existence first (MERGE in Kuzu is limited)
        existing = await self._execute(
            "MATCH (n:Node {id: $id}) RETURN n.id AS id",
            {"id": node.id},
        )
        if existing.has_next():
            await self._execute(
                """MATCH (n:Node {id: $id})
                SET n.kind = $kind,
                    n.title = $title,
                    n.content = $content,
                    n.tags_json = $tags_json,
                    n.level = $level,
                    n.vitality = $vitality,
                    n.access_count = $access_count,
                    n.success_count = $success_count,
                    n.failure_count = $failure_count,
                    n.source = $source,
                    n.properties_json = $properties_json,
                    n.updated_at = $updated_at""",
                _node_update_params(node),
            )
        else:
            await self._execute(
                """CREATE (n:Node {
                    id: $id,
                    kind: $kind,
                    title: $title,
                    content: $content,
                    tags_json: $tags_json,
                    level: $level,
                    vitality: $vitality,
                    access_count: $access_count,
                    success_count: $success_count,
                    failure_count: $failure_count,
                    source: $source,
                    properties_json: $properties_json,
                    created_at: $created_at,
                    updated_at: $updated_at
                })""",
                _node_create_params(node),
            )

    async def get_node(self, node_id: str) -> Node | None:
        result = await self._execute(
            "MATCH (n:Node {id: $id}) RETURN n.*",
            {"id": node_id},
        )
        rows = _result_rows(result)
        if not rows:
            return None
        return _row_to_node(rows[0], result.get_column_names())

    async def update_node(self, node: Node) -> None:
        await self._execute(
            """MATCH (n:Node {id: $id})
            SET n.kind = $kind,
                n.title = $title,
                n.content = $content,
                n.tags_json = $tags_json,
                n.level = $level,
                n.vitality = $vitality,
                n.access_count = $access_count,
                n.success_count = $success_count,
                n.failure_count = $failure_count,
                n.source = $source,
                n.properties_json = $properties_json,
                n.updated_at = $updated_at""",
            _node_update_params(node),
        )

    async def delete_node(self, node_id: str) -> None:
        # Kuzu requires directed DELETE — remove outgoing and incoming edges separately
        await self._execute(
            "MATCH (n:Node {id: $id})-[r:Edge]->() DELETE r",
            {"id": node_id},
        )
        await self._execute(
            "MATCH ()-[r:Edge]->(n:Node {id: $id}) DELETE r",
            {"id": node_id},
        )
        await self._execute(
            "MATCH (n:Node {id: $id}) DELETE n",
            {"id": node_id},
        )

    async def list_nodes(
        self,
        *,
        kind: str | NodeKind | None = None,
        level: ConsolidationLevel | None = None,
        limit: int = 100,
    ) -> list[Node]:
        conditions: list[str] = []
        params: dict[str, Any] = {"limit": int(limit)}
        if kind is not None:
            conditions.append("n.kind = $kind")
            params["kind"] = str(kind)
        if level is not None:
            conditions.append("n.level = $level")
            params["level"] = str(level)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"MATCH (n:Node){where} RETURN n.* ORDER BY n.updated_at DESC LIMIT $limit"
        result = await self._execute(query, params)
        return _rows_to_nodes(result)

    # --- Edge CRUD ---

    async def save_edge(self, edge: Edge) -> None:
        # Check if an edge with this id already exists
        existing = await self._execute(
            "MATCH ()-[r:Edge {id: $id}]->() RETURN r.id AS id",
            {"id": edge.id},
        )
        if existing.has_next():
            await self._execute(
                """MATCH ()-[r:Edge {id: $id}]->()
                SET r.kind = $kind,
                    r.weight = $weight""",
                {
                    "id": edge.id,
                    "kind": str(edge.kind),
                    "weight": float(edge.weight),
                },
            )
            return

        await self._execute(
            """MATCH (a:Node {id: $src}), (b:Node {id: $tgt})
            CREATE (a)-[r:Edge {
                id: $id,
                kind: $kind,
                weight: $weight,
                created_at: $created_at
            }]->(b)""",
            {
                "src": edge.source_id,
                "tgt": edge.target_id,
                "id": edge.id,
                "kind": str(edge.kind),
                "weight": float(edge.weight),
                "created_at": float(edge.created_at),
            },
        )

    async def get_edges(self, node_id: str, *, direction: str = "both") -> list[Edge]:
        if direction == "outgoing":
            result = await self._execute(
                "MATCH (a:Node {id: $id})-[r:Edge]->(b:Node) "
                "RETURN r.id AS id, r.kind AS kind, r.weight AS weight, "
                "r.created_at AS created_at, a.id AS src, b.id AS tgt",
                {"id": node_id},
            )
            return _rows_to_edges(result)
        if direction == "incoming":
            result = await self._execute(
                "MATCH (a:Node)-[r:Edge]->(b:Node {id: $id}) "
                "RETURN r.id AS id, r.kind AS kind, r.weight AS weight, "
                "r.created_at AS created_at, a.id AS src, b.id AS tgt",
                {"id": node_id},
            )
            return _rows_to_edges(result)
        # both: outgoing + incoming, dedupe by edge id
        outgoing = await self.get_edges(node_id, direction="outgoing")
        incoming = await self.get_edges(node_id, direction="incoming")
        seen: set[str] = set()
        merged: list[Edge] = []
        for edge in outgoing + incoming:
            if edge.id in seen:
                continue
            seen.add(edge.id)
            merged.append(edge)
        return merged

    async def update_edge(self, edge: Edge) -> None:
        await self._execute(
            """MATCH ()-[r:Edge {id: $id}]->()
            SET r.weight = $weight, r.kind = $kind""",
            {"id": edge.id, "weight": float(edge.weight), "kind": str(edge.kind)},
        )

    async def delete_edge(self, edge_id: str) -> None:
        await self._execute(
            "MATCH ()-[r:Edge {id: $id}]->() DELETE r",
            {"id": edge_id},
        )

    # --- Search ---

    async def _fetch_all_nodes(self, *, cap: int = 1_000_000) -> list[Node]:
        """Return every node in the graph (used by shared Python scoring).

        The cap exists purely as a safety fuse for pathological corpora;
        below it we want EVERY node scored so search recall is not silently
        truncated. Dropping even a single candidate here becomes invisible
        at the search layer — the missed node simply never appears in any
        result. The previous 10,000 cap silently hid ~52% of nodes on
        corpora above that threshold.
        """
        result = await self._execute(
            "MATCH (n:Node) RETURN n.* LIMIT $limit",
            {"limit": int(cap)},
        )
        return _rows_to_nodes(result)

    async def search_fts(self, query: str, *, limit: int = 20) -> list[Node]:
        """Hybrid BM25 + substring scoring via the shared Python ranker.

        Uses the same scoring as ``MemoryBackend`` for IR parity. Kuzu's
        built-in FTS extension is not used here because its plain Okapi
        BM25 diverges from the library's tuned hybrid scoring; we'll wire
        it in as a candidate prefilter once the scoring path has been
        benchmarked at scale.
        """
        if not query.strip():
            return []
        nodes = await self._fetch_all_nodes()
        return bm25_hybrid_score(nodes, query, limit=limit)

    async def search_fuzzy(
        self, query: str, *, limit: int = 20, threshold: float = 0.4
    ) -> list[Node]:
        if not query.strip():
            return []
        nodes = await self._fetch_all_nodes()
        return fuzzy_score(nodes, query, limit=limit, threshold=threshold)

    async def search_vector(self, embedding: list[float], *, limit: int = 20) -> list[Node]:
        # Vector search via Kuzu vector extension is not yet wired up here.
        # Use CompositeBackend with Qdrant for vector search at scale.
        return []

    # --- Graph traversal ---

    @staticmethod
    def _extract_neighbor_rows(
        result: kuzu.QueryResult,
        node_prefix: str,
        seen_edges: set[str],
    ) -> list[tuple[Node, Edge]]:
        cols = result.get_column_names()
        out: list[tuple[Node, Edge]] = []
        for row in _result_rows(result):
            row_dict = dict(zip(cols, row))
            edge_id = str(row_dict.get("edge_id", ""))
            if edge_id and edge_id in seen_edges:
                continue
            if edge_id:
                seen_edges.add(edge_id)
            node = _row_to_node(row, cols, prefix=node_prefix)
            edge = Edge(
                id=edge_id,
                source_id=str(row_dict.get("edge_src", "")),
                target_id=str(row_dict.get("edge_tgt", "")),
                kind=_safe_edge_kind(str(row_dict.get("edge_kind", "related"))),
                weight=float(row_dict.get("edge_weight", 1.0) or 1.0),
                created_at=float(row_dict.get("edge_created_at", 0.0) or 0.0),
            )
            out.append((node, edge))
        return out

    async def _one_hop_neighbors(self, node_id: str) -> list[tuple[Node, Edge]]:
        """Single-hop both-direction neighbors via two directed queries."""
        seen_edges: set[str] = set()
        results: list[tuple[Node, Edge]] = []

        out_result = await self._execute(
            "MATCH (a:Node {id: $id})-[r:Edge]->(b:Node) "
            "WHERE b.id <> $id "
            "RETURN b.*, "
            "r.id AS edge_id, r.kind AS edge_kind, "
            "r.weight AS edge_weight, r.created_at AS edge_created_at, "
            "a.id AS edge_src, b.id AS edge_tgt",
            {"id": node_id},
        )
        results.extend(self._extract_neighbor_rows(out_result, "b.", seen_edges))

        in_result = await self._execute(
            "MATCH (a:Node)-[r:Edge]->(b:Node {id: $id}) "
            "WHERE a.id <> $id "
            "RETURN a.*, "
            "r.id AS edge_id, r.kind AS edge_kind, "
            "r.weight AS edge_weight, r.created_at AS edge_created_at, "
            "a.id AS edge_src, b.id AS edge_tgt",
            {"id": node_id},
        )
        results.extend(self._extract_neighbor_rows(in_result, "a.", seen_edges))
        return results

    async def get_neighbors(self, node_id: str, *, depth: int = 1) -> list[tuple[Node, Edge]]:
        depth_int = max(1, int(depth))
        seen_nodes: set[str] = {node_id}
        results: list[tuple[Node, Edge]] = []

        frontier: set[str] = {node_id}
        for _ in range(depth_int):
            next_frontier: set[str] = set()
            for current in frontier:
                hops = await self._one_hop_neighbors(current)
                for node, edge in hops:
                    if node.id in seen_nodes:
                        continue
                    seen_nodes.add(node.id)
                    next_frontier.add(node.id)
                    results.append((node, edge))
            frontier = next_frontier
            if not frontier:
                break
        return results

    # --- Batch ---

    async def save_nodes_batch(self, nodes: Sequence[Node]) -> None:
        for node in nodes:
            await self.save_node(node)

    async def save_edges_batch(self, edges: Sequence[Edge]) -> None:
        for edge in edges:
            await self.save_edge(edge)

    # --- Maintenance ---

    async def prune_edges(self, *, weight_below: float = 0.1) -> int:
        count_result = await self._execute(
            "MATCH ()-[r:Edge]->() WHERE r.weight < $threshold RETURN count(r) AS cnt",
            {"threshold": float(weight_below)},
        )
        rows = _result_rows(count_result)
        count = int(rows[0][0]) if rows else 0
        if count > 0:
            await self._execute(
                "MATCH ()-[r:Edge]->() WHERE r.weight < $threshold DELETE r",
                {"threshold": float(weight_below)},
            )
        return count

    async def decay_vitality(self, *, factor: float = 0.95) -> int:
        count_result = await self._execute("MATCH (n:Node) RETURN count(n) AS cnt")
        rows = _result_rows(count_result)
        count = int(rows[0][0]) if rows else 0
        if count > 0:
            await self._execute(
                "MATCH (n:Node) SET n.vitality = n.vitality * $factor",
                {"factor": float(factor)},
            )
        return count

    # --- GraphTraversal extensions ---

    async def shortest_path(
        self, from_id: str, to_id: str, *, max_depth: int = 5
    ) -> list[tuple[Node, Edge]]:
        """BFS shortest path between two nodes (1-hop expansion via two-direction queries).

        Returns the sequence of (node, edge) tuples representing the path,
        excluding the start node. Empty list if no path within max_depth.
        """
        if from_id == to_id:
            return []
        depth_int = max(1, int(max_depth))
        visited: set[str] = {from_id}
        # queue entry: (current_node_id, path_so_far)
        queue: list[tuple[str, list[tuple[Node, Edge]]]] = [(from_id, [])]
        for _ in range(depth_int):
            next_queue: list[tuple[str, list[tuple[Node, Edge]]]] = []
            for current, path in queue:
                hops = await self._one_hop_neighbors(current)
                for node, edge in hops:
                    if node.id in visited:
                        continue
                    new_path = [*path, (node, edge)]
                    if node.id == to_id:
                        return new_path
                    visited.add(node.id)
                    next_queue.append((node.id, new_path))
            queue = next_queue
            if not queue:
                break
        return []

    async def pattern_match(self, pattern: str, *, limit: int = 20) -> list[dict[str, object]]:
        """Execute a Cypher pattern match query.

        Example pattern: "(:Node {kind: 'decision'})-[:Edge {kind: 'resulted_in'}]->(:Node)"
        """
        query = f"MATCH {pattern} RETURN * LIMIT $limit"
        result = await self._execute(query, {"limit": int(limit)})
        cols = result.get_column_names()
        return [dict(zip(cols, row)) for row in _result_rows(result)]

    async def find_by_type_hierarchy(self, type_name: str, *, limit: int = 50) -> list[Node]:
        """Find all nodes whose kind matches type_name (hierarchy expansion TBD)."""
        result = await self._execute(
            "MATCH (n:Node) WHERE n.kind = $kind "
            "RETURN n.* ORDER BY n.updated_at DESC LIMIT $limit",
            {"kind": type_name, "limit": int(limit)},
        )
        return _rows_to_nodes(result)

    # --- Admin ---

    async def clear_all(self) -> None:
        """Delete all nodes and edges. For testing only."""
        await self._execute("MATCH ()-[r:Edge]->() DELETE r")
        await self._execute("MATCH (n:Node) DELETE n")


# --- Helper functions ---


def _node_create_params(node: Node) -> dict[str, Any]:
    """Parameter dict for CREATE — includes created_at."""
    return {
        "id": node.id,
        "kind": str(node.kind),
        "title": node.title,
        "content": node.content,
        "tags_json": json.dumps(node.tags),
        "level": str(node.level),
        "vitality": float(node.vitality),
        "access_count": int(node.access_count),
        "success_count": int(node.success_count),
        "failure_count": int(node.failure_count),
        "source": node.source,
        "properties_json": json.dumps(node.properties),
        "created_at": float(node.created_at),
        "updated_at": float(node.updated_at),
    }


def _node_update_params(node: Node) -> dict[str, Any]:
    """Parameter dict for UPDATE — excludes created_at (immutable)."""
    return {
        "id": node.id,
        "kind": str(node.kind),
        "title": node.title,
        "content": node.content,
        "tags_json": json.dumps(node.tags),
        "level": str(node.level),
        "vitality": float(node.vitality),
        "access_count": int(node.access_count),
        "success_count": int(node.success_count),
        "failure_count": int(node.failure_count),
        "source": node.source,
        "properties_json": json.dumps(node.properties),
        "updated_at": float(node.updated_at),
    }


def _safe_node_kind(value: str) -> str | NodeKind:
    """Convert to NodeKind if known, otherwise keep as raw string."""
    try:
        return NodeKind(value)
    except ValueError:
        return value


def _safe_edge_kind(value: str) -> EdgeKind:
    """Convert to EdgeKind, defaulting to RELATED on unknown values."""
    try:
        return EdgeKind(value)
    except ValueError:
        return EdgeKind.RELATED


def _result_rows(result: kuzu.QueryResult) -> list[list[Any]]:
    """Drain a QueryResult into a list of row lists."""
    rows: list[list[Any]] = []
    while result.has_next():
        rows.append(result.get_next())
    return rows


def _row_to_node(row: list[Any], columns: list[str], *, prefix: str = "n.") -> Node:
    """Build a Node from a Cypher row using ``RETURN n.*`` style columns.

    Column names are expected as ``{prefix}id``, ``{prefix}kind``, ...
    """
    # Map columns → indices, handling both "n.id" and bare "id"
    by_name: dict[str, Any] = {}
    for col, val in zip(columns, row):
        # Normalize "<prefix>id" → "id"
        if col.startswith(prefix):
            by_name[col[len(prefix) :]] = val
        else:
            by_name[col] = val

    props_raw = by_name.get("properties_json") or "{}"
    try:
        properties = json.loads(props_raw) if isinstance(props_raw, str) else {}
    except json.JSONDecodeError:
        properties = {}

    tags_raw = by_name.get("tags_json") or "[]"
    try:
        tags = json.loads(tags_raw) if isinstance(tags_raw, str) else []
    except json.JSONDecodeError:
        tags = []

    return Node(
        id=str(by_name.get("id", "")),
        kind=_safe_node_kind(str(by_name.get("kind", "concept"))),
        title=str(by_name.get("title", "")),
        content=str(by_name.get("content", "")),
        tags=tags,
        level=ConsolidationLevel(str(by_name.get("level", "L0"))),
        vitality=float(by_name.get("vitality", 1.0) or 1.0),
        access_count=int(by_name.get("access_count", 0) or 0),
        success_count=int(by_name.get("success_count", 0) or 0),
        failure_count=int(by_name.get("failure_count", 0) or 0),
        properties=properties,
        source=str(by_name.get("source", "")),
        created_at=float(by_name.get("created_at", 0.0) or 0.0),
        updated_at=float(by_name.get("updated_at", 0.0) or 0.0),
    )


def _rows_to_nodes(result: kuzu.QueryResult, *, prefix: str = "n.") -> list[Node]:
    cols = result.get_column_names()
    return [_row_to_node(row, cols, prefix=prefix) for row in _result_rows(result)]


def _rows_to_edges(result: kuzu.QueryResult) -> list[Edge]:
    cols = result.get_column_names()
    edges: list[Edge] = []
    for row in _result_rows(result):
        d = dict(zip(cols, row))
        edges.append(
            Edge(
                id=str(d.get("id", "")),
                source_id=str(d.get("src", "")),
                target_id=str(d.get("tgt", "")),
                kind=_safe_edge_kind(str(d.get("kind", "related"))),
                weight=float(d.get("weight", 1.0) or 1.0),
                created_at=float(d.get("created_at", 0.0) or 0.0),
            )
        )
    return edges
