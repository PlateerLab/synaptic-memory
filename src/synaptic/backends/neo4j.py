"""Neo4j storage backend — native graph traversal with typed relationships."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
)

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase
except ImportError as e:
    msg = "Install synaptic-memory[neo4j] for Neo4j backend: pip install synaptic-memory[neo4j]"
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)

# Cypher label for edge kind (uppercase, e.g. "RESULTED_IN")
_EDGE_LABEL_MAP: dict[str, str] = {}


def _edge_label(kind: str) -> str:
    """Convert edge kind to Neo4j relationship type label."""
    if kind not in _EDGE_LABEL_MAP:
        _EDGE_LABEL_MAP[kind] = kind.upper()
    return _EDGE_LABEL_MAP[kind]


def _node_label(kind: str) -> str:
    """Convert node kind to Neo4j secondary label (PascalCase)."""
    return kind.replace("_", " ").title().replace(" ", "")


class Neo4jBackend:
    """Neo4j backend with dual labels, typed relationships, and native graph traversal.

    Implements StorageBackend protocol + GraphTraversal extensions.

    Each node gets:
      - :Node label (universal, for generic queries)
      - :Kind label (e.g. :Decision, :ToolCall, for type-specific queries)

    Each relationship uses the EdgeKind as its type (e.g. -[:RESULTED_IN]->).
    """

    __slots__ = ("_auth", "_database", "_driver", "_uri")

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        *,
        auth: tuple[str, str] = ("neo4j", "password"),
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._auth = auth
        self._database = database
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(self._uri, auth=self._auth)
        await self._driver.verify_connectivity()
        async with self._driver.session(database=self._database) as session:
            # Create indexes
            await session.run("CREATE INDEX node_id IF NOT EXISTS FOR (n:Node) ON (n.id)")
            await session.run("CREATE INDEX node_kind IF NOT EXISTS FOR (n:Node) ON (n.kind)")
            await session.run("CREATE INDEX node_level IF NOT EXISTS FOR (n:Node) ON (n.level)")
            # Fulltext index for FTS search
            try:
                await session.run(
                    "CREATE FULLTEXT INDEX node_fts IF NOT EXISTS "
                    "FOR (n:Node) ON EACH [n.title, n.content]"
                )
            except Exception:
                logger.debug("Fulltext index creation skipped (may already exist)")
        logger.info("Neo4j connected: %s", self._uri)

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._driver

    # --- Node CRUD ---

    async def save_node(self, node: Node) -> None:
        driver = self._get_driver()
        kind_label = _node_label(str(node.kind))
        async with driver.session(database=self._database) as session:
            await session.run(
                f"MERGE (n:Node {{id: $id}}) "
                f"ON CREATE SET n:{kind_label}, n += $props "
                f"ON MATCH SET n += $props",
                id=node.id,
                props=_node_to_props(node),
            )

    async def get_node(self, node_id: str) -> Node | None:
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            result = await session.run("MATCH (n:Node {id: $id}) RETURN n", id=node_id)
            record = await result.single()
        if record is None:
            return None
        return _record_to_node(record["n"])

    async def update_node(self, node: Node) -> None:
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            await session.run(
                "MATCH (n:Node {id: $id}) SET n += $props",
                id=node.id,
                props=_node_to_props(node),
            )

    async def delete_node(self, node_id: str) -> None:
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            await session.run("MATCH (n:Node {id: $id}) DETACH DELETE n", id=node_id)

    async def list_nodes(
        self,
        *,
        kind: str | NodeKind | None = None,
        level: ConsolidationLevel | None = None,
        limit: int = 100,
    ) -> list[Node]:
        driver = self._get_driver()
        conditions: list[str] = []
        params: dict[str, object] = {"limit": limit}
        if kind is not None:
            conditions.append("n.kind = $kind")
            params["kind"] = str(kind)
        if level is not None:
            conditions.append("n.level = $level")
            params["level"] = str(level)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"MATCH (n:Node){where} RETURN n ORDER BY n.updated_at DESC LIMIT $limit"
        async with driver.session(database=self._database) as session:
            result = await session.run(query, **params)
            records = await result.data()
        return [_record_to_node(r["n"]) for r in records]

    # --- Edge CRUD ---

    async def save_edge(self, edge: Edge) -> None:
        driver = self._get_driver()
        rel_type = _edge_label(str(edge.kind))
        async with driver.session(database=self._database) as session:
            await session.run(
                f"MATCH (a:Node {{id: $src}}), (b:Node {{id: $tgt}}) "
                f"MERGE (a)-[r:{rel_type} {{id: $id}}]->(b) "
                f"SET r.kind = $kind, r.weight = $weight, r.created_at = $created_at",
                src=edge.source_id,
                tgt=edge.target_id,
                id=edge.id,
                kind=str(edge.kind),
                weight=edge.weight,
                created_at=edge.created_at,
            )

    async def get_edges(self, node_id: str, *, direction: str = "both") -> list[Edge]:
        driver = self._get_driver()
        match direction:
            case "outgoing":
                query = (
                    "MATCH (a:Node {id: $id})-[r]->(b:Node) "
                    "RETURN properties(r) AS rp, type(r) AS rtype, a.id AS src, b.id AS tgt"
                )
            case "incoming":
                query = (
                    "MATCH (a:Node)-[r]->(b:Node {id: $id}) "
                    "RETURN properties(r) AS rp, type(r) AS rtype, a.id AS src, b.id AS tgt"
                )
            case _:
                query = (
                    "MATCH (a:Node {id: $id})-[r]->(b:Node) "
                    "RETURN properties(r) AS rp, type(r) AS rtype, a.id AS src, b.id AS tgt "
                    "UNION "
                    "MATCH (a:Node)-[r]->(b:Node {id: $id}) "
                    "RETURN properties(r) AS rp, type(r) AS rtype, a.id AS src, b.id AS tgt"
                )
        async with driver.session(database=self._database) as session:
            result = await session.run(query, id=node_id)
            records = await result.data()
        return [_record_to_edge(rec) for rec in records]

    async def update_edge(self, edge: Edge) -> None:
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            await session.run(
                "MATCH ()-[r {id: $id}]->() SET r.weight = $weight, r.kind = $kind",
                id=edge.id,
                weight=edge.weight,
                kind=str(edge.kind),
            )

    async def delete_edge(self, edge_id: str) -> None:
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            await session.run("MATCH ()-[r {id: $id}]->() DELETE r", id=edge_id)

    # --- Search ---

    async def search_fts(self, query: str, *, limit: int = 20) -> list[Node]:
        driver = self._get_driver()
        terms = query.strip().split()
        if not terms:
            return []
        # Lucene query syntax for Neo4j fulltext
        fts_query = " OR ".join(terms)
        async with driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    "CALL db.index.fulltext.queryNodes('node_fts', $query) "
                    "YIELD node, score "
                    "RETURN node ORDER BY score DESC LIMIT $limit",
                    query=fts_query,
                    limit=limit,
                )
                records = await result.data()
                return [_record_to_node(r["node"]) for r in records]
            except Exception:
                logger.debug("FTS search failed, falling back to CONTAINS")
                return await self.search_fuzzy(query, limit=limit)

    async def search_fuzzy(
        self, query: str, *, limit: int = 20, threshold: float = 0.3
    ) -> list[Node]:
        driver = self._get_driver()
        terms = query.strip().split()
        if not terms:
            return []
        # CONTAINS-based fuzzy matching
        conditions = " OR ".join(
            f"(toLower(n.title) CONTAINS toLower($t{i}) OR "
            f"toLower(n.content) CONTAINS toLower($t{i}))"
            for i in range(len(terms))
        )
        params: dict[str, object] = {"limit": limit}
        for i, t in enumerate(terms):
            params[f"t{i}"] = t
        query_str = (
            f"MATCH (n:Node) WHERE {conditions} RETURN n ORDER BY n.updated_at DESC LIMIT $limit"
        )
        async with driver.session(database=self._database) as session:
            result = await session.run(query_str, **params)
            records = await result.data()
        return [_record_to_node(r["n"]) for r in records]

    async def search_vector(self, embedding: list[float], *, limit: int = 20) -> list[Node]:
        # Vector search not supported in Neo4j Community — delegate to Qdrant via CompositeBackend
        return []

    # --- Graph traversal (native Cypher) ---

    async def get_neighbors(self, node_id: str, *, depth: int = 1) -> list[tuple[Node, Edge]]:
        driver = self._get_driver()
        # Neo4j doesn't allow parameters in variable-length range, so we use string formatting
        # (depth is always an int from our code, not user input)
        query = (
            f"MATCH (start:Node {{id: $id}})-[rels*1..{depth}]-(neighbor:Node) "
            f"WHERE neighbor.id <> $id "
            f"WITH DISTINCT neighbor, head(rels) AS first_rel "
            f"RETURN neighbor, properties(first_rel) AS rp, "
            f"startNode(first_rel).id AS src_id, endNode(first_rel).id AS tgt_id"
        )
        async with driver.session(database=self._database) as session:
            result = await session.run(query, id=node_id)
            records = await result.data()

        results: list[tuple[Node, Edge]] = []
        for rec in records:
            node = _record_to_node(rec["neighbor"])
            rp = rec.get("rp", {})
            d: dict[str, object] = rp if isinstance(rp, dict) else {}
            edge = Edge(
                id=str(d.get("id", "")),
                source_id=str(rec.get("src_id", "")),
                target_id=str(rec.get("tgt_id", "")),
                kind=EdgeKind(str(d.get("kind", "related"))),
                weight=float(d.get("weight", 1.0)),  # type: ignore[arg-type]
                created_at=float(d.get("created_at", 0.0)),  # type: ignore[arg-type]
            )
            results.append((node, edge))
        return results

    # --- Batch ---

    async def save_nodes_batch(self, nodes: Sequence[Node]) -> None:
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            async with await session.begin_transaction() as tx:
                for node in nodes:
                    kind_label = _node_label(str(node.kind))
                    await tx.run(
                        f"MERGE (n:Node {{id: $id}}) "
                        f"ON CREATE SET n:{kind_label}, n += $props "
                        f"ON MATCH SET n += $props",
                        id=node.id,
                        props=_node_to_props(node),
                    )
                await tx.commit()

    async def save_edges_batch(self, edges: Sequence[Edge]) -> None:
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            async with await session.begin_transaction() as tx:
                for edge in edges:
                    rel_type = _edge_label(str(edge.kind))
                    await tx.run(
                        f"MATCH (a:Node {{id: $src}}), (b:Node {{id: $tgt}}) "
                        f"MERGE (a)-[r:{rel_type} {{id: $id}}]->(b) "
                        f"SET r.kind = $kind, r.weight = $weight, r.created_at = $created_at",
                        src=edge.source_id,
                        tgt=edge.target_id,
                        id=edge.id,
                        kind=str(edge.kind),
                        weight=edge.weight,
                        created_at=edge.created_at,
                    )
                await tx.commit()

    # --- Maintenance ---

    async def prune_edges(self, *, weight_below: float = 0.1) -> int:
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH ()-[r]->() WHERE r.weight < $threshold "
                "WITH r, count(r) AS cnt DELETE r RETURN cnt",
                threshold=weight_below,
            )
            record = await result.single()
            return int(record["cnt"]) if record else 0

    async def decay_vitality(self, *, factor: float = 0.95) -> int:
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (n:Node) SET n.vitality = n.vitality * $factor RETURN count(n) AS cnt",
                factor=factor,
            )
            record = await result.single()
            return int(record["cnt"]) if record else 0

    # --- GraphTraversal extensions ---

    async def shortest_path(
        self, from_id: str, to_id: str, *, max_depth: int = 5
    ) -> list[tuple[Node, Edge]]:
        """Find shortest path between two nodes."""
        driver = self._get_driver()
        # Neo4j doesn't allow parameters in shortestPath range
        query = (
            f"MATCH p = shortestPath("
            f"(a:Node {{id: $from_id}})-[*1..{max_depth}]-(b:Node {{id: $to_id}})"
            f") "
            f"UNWIND nodes(p) AS node "
            f"UNWIND relationships(p) AS rel "
            f"WITH DISTINCT node, properties(rel) AS rp, "
            f"startNode(rel).id AS src_id, endNode(rel).id AS tgt_id "
            f"WHERE node.id <> $from_id "
            f"RETURN node, rp, src_id, tgt_id"
        )
        async with driver.session(database=self._database) as session:
            result = await session.run(query, from_id=from_id, to_id=to_id)
            records = await result.data()

        results: list[tuple[Node, Edge]] = []
        seen_edges: set[str] = set()
        for rec in records:
            node = _record_to_node(rec["node"])
            rp = rec.get("rp", {})
            d: dict[str, object] = rp if isinstance(rp, dict) else {}
            edge_id = str(d.get("id", ""))
            if edge_id in seen_edges:
                continue
            seen_edges.add(edge_id)
            edge = Edge(
                id=edge_id,
                source_id=str(rec["src_id"]),
                target_id=str(rec["tgt_id"]),
                kind=EdgeKind(str(d.get("kind", "related"))),
                weight=float(d.get("weight", 1.0)),  # type: ignore[arg-type]
                created_at=float(d.get("created_at", 0.0)),  # type: ignore[arg-type]
            )
            results.append((node, edge))
        return results

    async def pattern_match(self, pattern: str, *, limit: int = 20) -> list[dict[str, object]]:
        """Execute a Cypher pattern match query.

        Example pattern: "(:Decision)-[:RESULTED_IN]->(:Outcome {success: 'true'})"
        """
        driver = self._get_driver()
        query = f"MATCH {pattern} RETURN * LIMIT $limit"
        async with driver.session(database=self._database) as session:
            result = await session.run(query, limit=limit)
            return await result.data()  # type: ignore[return-value]

    async def find_by_type_hierarchy(self, type_name: str, *, limit: int = 50) -> list[Node]:
        """Find all nodes of a type, using IS_A edges for hierarchy traversal."""
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            # Direct kind match + IS_A hierarchy traversal
            result = await session.run(
                "MATCH (n:Node) WHERE n.kind = $kind "
                "RETURN n "
                "UNION "
                "MATCH (type_node:Node {kind: 'type_def', title: $kind})"
                "<-[:IS_A*1..5]-(child:Node {kind: 'type_def'})"
                "-[:IS_A]->(n:Node) "
                "WHERE n.kind = child.title "
                "RETURN n "
                "LIMIT $limit",
                kind=type_name,
                limit=limit,
            )
            records = await result.data()
        return [_record_to_node(r["n"]) for r in records]

    # --- Admin ---

    async def clear_all(self) -> None:
        """Delete all nodes and edges. For testing only."""
        driver = self._get_driver()
        async with driver.session(database=self._database) as session:
            await session.run("MATCH (n) DETACH DELETE n")


# --- Helpers ---


def _node_to_props(node: Node) -> dict[str, object]:
    """Convert Node dataclass to Neo4j property dict."""
    return {
        "kind": str(node.kind),
        "title": node.title,
        "content": node.content,
        "tags_json": json.dumps(node.tags),
        "level": str(node.level),
        "vitality": node.vitality,
        "access_count": node.access_count,
        "success_count": node.success_count,
        "failure_count": node.failure_count,
        "source": node.source,
        "properties_json": json.dumps(node.properties),
        "created_at": node.created_at,
        "updated_at": node.updated_at,
    }


def _safe_node_kind(value: str) -> str | NodeKind:
    """Convert to NodeKind if known, otherwise keep as raw string."""
    try:
        return NodeKind(value)
    except ValueError:
        return value


def _record_to_node(data: object) -> Node:
    """Convert Neo4j node record to Node dataclass."""
    # neo4j driver returns dict-like objects
    d: dict[str, object] = dict(data) if not isinstance(data, dict) else data  # type: ignore[arg-type]
    props_raw = d.get("properties_json", "{}")
    return Node(
        id=str(d.get("id", "")),
        kind=_safe_node_kind(str(d.get("kind", "concept"))),
        title=str(d.get("title", "")),
        content=str(d.get("content", "")),
        tags=json.loads(str(d.get("tags_json", "[]"))),
        level=ConsolidationLevel(str(d.get("level", "L0"))),
        vitality=float(d.get("vitality", 1.0)),  # type: ignore[arg-type]
        access_count=int(d.get("access_count", 0)),  # type: ignore[arg-type]
        success_count=int(d.get("success_count", 0)),  # type: ignore[arg-type]
        failure_count=int(d.get("failure_count", 0)),  # type: ignore[arg-type]
        properties=json.loads(str(props_raw)) if props_raw else {},
        source=str(d.get("source", "")),
        created_at=float(d.get("created_at", 0.0)),  # type: ignore[arg-type]
        updated_at=float(d.get("updated_at", 0.0)),  # type: ignore[arg-type]
    )


def _record_to_edge(rec: dict[str, object]) -> Edge:
    """Convert Neo4j relationship record to Edge dataclass.

    Expects keys: rp (properties dict), rtype (relationship type), src, tgt.
    """
    rp = rec.get("rp", {})
    d: dict[str, object] = rp if isinstance(rp, dict) else {}
    return Edge(
        id=str(d.get("id", "")),
        source_id=str(rec.get("src", "")),
        target_id=str(rec.get("tgt", "")),
        kind=EdgeKind(str(d.get("kind", "related"))),
        weight=float(d.get("weight", 1.0)),  # type: ignore[arg-type]
        created_at=float(d.get("created_at", 0.0)),  # type: ignore[arg-type]
    )
