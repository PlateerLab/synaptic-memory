"""PostgreSQL backend — pgvector + pg_trgm + recursive CTE."""

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
    import asyncpg
except ImportError as e:
    msg = "Install synaptic-memory[postgresql]: pip install synaptic-memory[postgresql]"
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS syn_nodes (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'concept',
    title       TEXT NOT NULL DEFAULT '',
    content     TEXT NOT NULL DEFAULT '',
    tags        TEXT[] NOT NULL DEFAULT '{}',
    level       TEXT NOT NULL DEFAULT 'L0',
    embedding   vector(1536),
    vitality    REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    source      TEXT NOT NULL DEFAULT '',
    properties_json TEXT NOT NULL DEFAULT '{}',
    created_at  DOUBLE PRECISION NOT NULL,
    updated_at  DOUBLE PRECISION NOT NULL
);

-- FTS (simple tokenizer for Korean + English)
DO $$ BEGIN
    ALTER TABLE syn_nodes ADD COLUMN IF NOT EXISTS tsv tsvector
        GENERATED ALWAYS AS (
            setweight(to_tsvector('simple', title), 'A') ||
            setweight(to_tsvector('simple', content), 'B')
        ) STORED;
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;

CREATE INDEX IF NOT EXISTS idx_syn_nodes_tsv ON syn_nodes USING GIN(tsv);
CREATE INDEX IF NOT EXISTS idx_syn_nodes_kind_level ON syn_nodes(kind, level);
CREATE INDEX IF NOT EXISTS idx_syn_nodes_tags ON syn_nodes USING GIN(tags);

CREATE TABLE IF NOT EXISTS syn_edges (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES syn_nodes(id) ON DELETE CASCADE,
    target_id   TEXT NOT NULL REFERENCES syn_nodes(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL DEFAULT 'related',
    weight      REAL NOT NULL DEFAULT 1.0,
    created_at  DOUBLE PRECISION NOT NULL,
    UNIQUE(source_id, target_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_syn_edges_source ON syn_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_syn_edges_target ON syn_edges(target_id);
"""

# pg_trgm indexes — created only if extension is available
_TRGM_SCHEMA = """\
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_syn_nodes_trgm_title
    ON syn_nodes USING GIN(title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_syn_nodes_trgm_content
    ON syn_nodes USING GIN(content gin_trgm_ops);
"""

# HNSW index is created separately (needs data for optimal tuning)
_HNSW_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_syn_nodes_embedding ON syn_nodes
    USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
"""


class PostgreSQLBackend:
    """PostgreSQL backend with pgvector (HNSW) + pg_trgm (fuzzy) + recursive CTE.

    Requires: PostgreSQL 15+ with pgvector and pg_trgm extensions.
    """

    __slots__ = ("_dsn", "_embedding_dim", "_has_trgm", "_pool")

    def __init__(self, dsn: str, *, embedding_dim: int = 1536) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._embedding_dim = embedding_dim
        self._has_trgm = False

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            schema = _SCHEMA.replace("vector(1536)", f"vector({self._embedding_dim})")
            await conn.execute(schema)
            # Migrate: add properties_json column if missing (v0.4 → v0.5)
            try:
                await conn.execute(
                    "ALTER TABLE syn_nodes ADD COLUMN properties_json TEXT NOT NULL DEFAULT '{}'"
                )
            except Exception:
                pass  # Column already exists
            # pg_trgm (optional — graceful fallback to LIKE)
            try:
                await conn.execute(_TRGM_SCHEMA)
                self._has_trgm = True
                logger.info("pg_trgm enabled — trigram fuzzy search active")
            except Exception:
                logger.info("pg_trgm unavailable — using LIKE fallback for fuzzy search")
            # HNSW index (idempotent)
            try:
                await conn.execute(_HNSW_INDEX)
            except Exception:
                logger.debug("HNSW index creation skipped (may need data first)")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def execute_raw(self, sql: str) -> None:
        """Execute raw SQL. For admin/testing purposes."""
        pool = self._get_pool()
        await pool.execute(sql)

    def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._pool

    # --- Node CRUD ---

    async def save_node(self, node: Node) -> None:
        pool = self._get_pool()
        embedding = node.embedding if node.embedding else None
        embedding_str = _vec_to_str(embedding) if embedding else None
        await pool.execute(
            """INSERT INTO syn_nodes
            (id, kind, title, content, tags, level, embedding, vitality,
             access_count, success_count, failure_count, source, properties_json,
             created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9, $10, $11, $12, $13, $14, $15)
            ON CONFLICT(id) DO UPDATE SET
                title=EXCLUDED.title, content=EXCLUDED.content, tags=EXCLUDED.tags,
                level=EXCLUDED.level, embedding=EXCLUDED.embedding,
                vitality=EXCLUDED.vitality, properties_json=EXCLUDED.properties_json,
                updated_at=EXCLUDED.updated_at""",
            node.id,
            str(node.kind),
            node.title,
            node.content,
            node.tags,
            str(node.level),
            embedding_str,
            node.vitality,
            node.access_count,
            node.success_count,
            node.failure_count,
            node.source,
            json.dumps(node.properties),
            node.created_at,
            node.updated_at,
        )

    async def get_node(self, node_id: str) -> Node | None:
        pool = self._get_pool()
        row = await pool.fetchrow("SELECT * FROM syn_nodes WHERE id = $1", node_id)
        if row is None:
            return None
        return _row_to_node(row)

    async def update_node(self, node: Node) -> None:
        pool = self._get_pool()
        embedding_str = _vec_to_str(node.embedding) if node.embedding else None
        await pool.execute(
            """UPDATE syn_nodes SET kind=$1, title=$2, content=$3, tags=$4, level=$5,
            embedding=$6::vector, vitality=$7, access_count=$8, success_count=$9,
            failure_count=$10, source=$11, properties_json=$12, updated_at=$13 WHERE id=$14""",
            str(node.kind),
            node.title,
            node.content,
            node.tags,
            str(node.level),
            embedding_str,
            node.vitality,
            node.access_count,
            node.success_count,
            node.failure_count,
            node.source,
            json.dumps(node.properties),
            node.updated_at,
            node.id,
        )

    async def delete_node(self, node_id: str) -> None:
        pool = self._get_pool()
        await pool.execute("DELETE FROM syn_nodes WHERE id = $1", node_id)

    async def list_nodes(
        self,
        *,
        kind: str | NodeKind | None = None,
        level: ConsolidationLevel | None = None,
        limit: int = 100,
    ) -> list[Node]:
        pool = self._get_pool()
        conditions: list[str] = []
        params: list[str | int] = []
        idx = 1
        if kind is not None:
            conditions.append(f"kind = ${idx}")
            params.append(str(kind))
            idx += 1
        if level is not None:
            conditions.append(f"level = ${idx}")
            params.append(str(level))
            idx += 1
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        sql = f"SELECT * FROM syn_nodes{where} ORDER BY updated_at DESC LIMIT ${idx}"  # noqa: S608
        rows = await pool.fetch(sql, *params)
        return [_row_to_node(r) for r in rows]

    # --- Edge CRUD ---

    async def save_edge(self, edge: Edge) -> None:
        pool = self._get_pool()
        await pool.execute(
            """INSERT INTO syn_edges (id, source_id, target_id, kind, weight, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT(source_id, target_id, kind) DO UPDATE SET weight=EXCLUDED.weight""",
            edge.id,
            edge.source_id,
            edge.target_id,
            str(edge.kind),
            edge.weight,
            edge.created_at,
        )

    async def get_edges(self, node_id: str, *, direction: str = "both") -> list[Edge]:
        pool = self._get_pool()
        match direction:
            case "outgoing":
                sql = "SELECT * FROM syn_edges WHERE source_id = $1"
            case "incoming":
                sql = "SELECT * FROM syn_edges WHERE target_id = $1"
            case _:
                sql = "SELECT * FROM syn_edges WHERE source_id = $1 OR target_id = $1"
        rows = await pool.fetch(sql, node_id)
        return [_row_to_edge(r) for r in rows]

    async def update_edge(self, edge: Edge) -> None:
        pool = self._get_pool()
        await pool.execute(
            "UPDATE syn_edges SET weight=$1, kind=$2 WHERE id=$3",
            edge.weight,
            str(edge.kind),
            edge.id,
        )

    async def delete_edge(self, edge_id: str) -> None:
        pool = self._get_pool()
        await pool.execute("DELETE FROM syn_edges WHERE id = $1", edge_id)

    # --- Search ---

    async def search_fts(self, query: str, *, limit: int = 20) -> list[Node]:
        pool = self._get_pool()
        terms = query.strip().split()
        if not terms:
            return []
        tsquery = " | ".join(terms)
        rows = await pool.fetch(
            """SELECT *, ts_rank(tsv, plainto_tsquery('simple', $1)) AS score
            FROM syn_nodes WHERE tsv @@ plainto_tsquery('simple', $1)
            ORDER BY score DESC LIMIT $2""",
            tsquery,
            limit,
        )
        return [_row_to_node(r) for r in rows]

    async def search_fuzzy(
        self, query: str, *, limit: int = 20, threshold: float = 0.3
    ) -> list[Node]:
        pool = self._get_pool()
        if not query.strip():
            return []

        if self._has_trgm:
            rows = await pool.fetch(
                """SELECT *, similarity(title || ' ' || content, $1) AS sim
                FROM syn_nodes
                WHERE similarity(title || ' ' || content, $1) >= $2
                ORDER BY sim DESC LIMIT $3""",
                query,
                threshold,
                limit,
            )
            return [_row_to_node(r) for r in rows]

        # LIKE fallback when pg_trgm is not available
        terms = query.strip().split()
        conditions = " OR ".join(
            f"(title ILIKE ${i * 2 + 1} OR content ILIKE ${i * 2 + 2})" for i in range(len(terms))
        )
        params: list[object] = []
        for t in terms:
            like = f"%{t}%"
            params.extend([like, like])
        limit_idx = len(params) + 1
        sql = (
            f"SELECT * FROM syn_nodes WHERE {conditions}"  # noqa: S608
            f" ORDER BY updated_at DESC LIMIT ${limit_idx}"
        )
        rows = await pool.fetch(sql, *params, limit)
        return [_row_to_node(r) for r in rows]

    async def search_vector(self, embedding: list[float], *, limit: int = 20) -> list[Node]:
        pool = self._get_pool()
        if not embedding:
            return []
        vec_str = _vec_to_str(embedding)
        rows = await pool.fetch(
            """SELECT *, 1 - (embedding <=> $1::vector) AS score
            FROM syn_nodes WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector LIMIT $2""",
            vec_str,
            limit,
        )
        return [_row_to_node(r) for r in rows]

    # --- Graph traversal (recursive CTE) ---

    async def get_neighbors(self, node_id: str, *, depth: int = 1) -> list[tuple[Node, Edge]]:
        pool = self._get_pool()
        rows = await pool.fetch(
            """WITH RECURSIVE neighbors(node_id, edge_id, depth) AS (
                SELECT CASE WHEN source_id = $1 THEN target_id ELSE source_id END,
                       id, 1
                FROM syn_edges
                WHERE source_id = $1 OR target_id = $1
                UNION
                SELECT CASE WHEN e.source_id = nb.node_id THEN e.target_id ELSE e.source_id END,
                       e.id, nb.depth + 1
                FROM syn_edges e
                JOIN neighbors nb ON e.source_id = nb.node_id OR e.target_id = nb.node_id
                WHERE nb.depth < $2
                  AND CASE WHEN e.source_id = nb.node_id THEN e.target_id
                      ELSE e.source_id END != $1
            )
            SELECT DISTINCT node_id, edge_id FROM neighbors""",
            node_id,
            depth,
        )

        result: list[tuple[Node, Edge]] = []
        for row in rows:
            nid = row["node_id"]
            eid = row["edge_id"]
            node = await self.get_node(nid)
            erow = await pool.fetchrow("SELECT * FROM syn_edges WHERE id = $1", eid)
            if node is not None and erow is not None:
                result.append((node, _row_to_edge(erow)))
        return result

    # --- Hybrid search ---

    async def search_hybrid(
        self,
        query: str,
        *,
        embedding: list[float] | None = None,
        limit: int = 20,
    ) -> list[tuple[Node, float]]:
        """Combined FTS + fuzzy + vector search with merged results."""
        fts_nodes = await self.search_fts(query, limit=limit)
        fuzzy_nodes = await self.search_fuzzy(query, limit=limit, threshold=0.2)

        merged: dict[str, tuple[Node, float]] = {}
        for node in fts_nodes:
            merged[node.id] = (node, 0.8)
        for node in fuzzy_nodes:
            if node.id in merged:
                old = merged[node.id]
                merged[node.id] = (old[0], min(1.0, old[1] + 0.2))
            else:
                merged[node.id] = (node, 0.6)

        if embedding:
            vec_nodes = await self.search_vector(embedding, limit=limit)
            for node in vec_nodes:
                if node.id in merged:
                    old = merged[node.id]
                    merged[node.id] = (old[0], min(1.0, old[1] + 0.2))
                else:
                    merged[node.id] = (node, 0.7)

        results = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        return results[:limit]

    # --- Batch ---

    async def save_nodes_batch(self, nodes: Sequence[Node]) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for node in nodes:
                    await self.save_node(node)

    async def save_edges_batch(self, edges: Sequence[Edge]) -> None:
        pool = self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for edge in edges:
                    await self.save_edge(edge)

    # --- Maintenance ---

    async def prune_edges(self, *, weight_below: float = 0.1) -> int:
        pool = self._get_pool()
        result = await pool.execute("DELETE FROM syn_edges WHERE weight < $1", weight_below)
        return int(result.split()[-1]) if result else 0

    async def decay_vitality(self, *, factor: float = 0.95) -> int:
        pool = self._get_pool()
        result = await pool.execute("UPDATE syn_nodes SET vitality = vitality * $1", factor)
        return int(result.split()[-1]) if result else 0


def _safe_node_kind(value: str) -> str | NodeKind:
    """Convert to NodeKind if known, otherwise keep as raw string."""
    try:
        return NodeKind(value)
    except ValueError:
        return value


def _row_to_node(row: asyncpg.Record) -> Node:
    tags = list(row["tags"]) if row["tags"] else []
    props_raw = row.get("properties_json", "{}")
    return Node(
        id=row["id"],
        kind=_safe_node_kind(row["kind"]),
        title=row["title"],
        content=row["content"],
        tags=tags,
        level=ConsolidationLevel(row["level"]),
        vitality=row["vitality"],
        access_count=row["access_count"],
        success_count=row["success_count"],
        failure_count=row["failure_count"],
        properties=json.loads(props_raw) if props_raw else {},
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_edge(row: asyncpg.Record) -> Edge:
    return Edge(
        id=row["id"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        kind=EdgeKind(row["kind"]),
        weight=row["weight"],
        created_at=row["created_at"],
    )


def _vec_to_str(embedding: list[float] | None) -> str | None:
    if not embedding:
        return None
    return "[" + ",".join(str(x) for x in embedding) + "]"
