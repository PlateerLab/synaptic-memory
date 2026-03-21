"""SQLite storage backend with FTS5 and recursive CTE."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
)

try:
    import aiosqlite
except ImportError as e:
    msg = "Install synaptic-memory[sqlite] for SQLite backend: pip install synaptic-memory[sqlite]"
    raise ImportError(msg) from e


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS syn_nodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'concept',
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    level TEXT NOT NULL DEFAULT 'L0',
    vitality REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT '',
    properties_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS syn_nodes_fts USING fts5(
    node_id, title, content, tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS syn_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES syn_nodes(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES syn_nodes(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'related',
    weight REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    UNIQUE(source_id, target_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_syn_edges_source ON syn_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_syn_edges_target ON syn_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_syn_nodes_kind_level ON syn_nodes(kind, level);
"""


class SQLiteBackend:
    """SQLite backend with FTS5 full-text search and CTE graph traversal."""

    __slots__ = ("_conn", "_path")

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(_SCHEMA)
        # Migrate: add properties_json column if missing (v0.4 → v0.5)
        async with self._conn.execute("PRAGMA table_info(syn_nodes)") as cur:
            columns = {row[1] for row in await cur.fetchall()}
        if "properties_json" not in columns:
            await self._conn.execute(
                "ALTER TABLE syn_nodes ADD COLUMN properties_json TEXT NOT NULL DEFAULT '{}'"
            )
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._conn

    # --- Node CRUD ---

    async def save_node(self, node: Node) -> None:
        db = self._db()
        await db.execute(
            """INSERT INTO syn_nodes
            (id, kind, title, content, tags_json, level, vitality,
             access_count, success_count, failure_count, source, properties_json,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, content=excluded.content, tags_json=excluded.tags_json,
                level=excluded.level, vitality=excluded.vitality,
                properties_json=excluded.properties_json, updated_at=excluded.updated_at""",
            (
                node.id,
                str(node.kind),
                node.title,
                node.content,
                json.dumps(node.tags),
                str(node.level),
                node.vitality,
                node.access_count,
                node.success_count,
                node.failure_count,
                node.source,
                json.dumps(node.properties),
                node.created_at,
                node.updated_at,
            ),
        )
        # FTS sync (virtual tables don't support UPSERT)
        await db.execute("DELETE FROM syn_nodes_fts WHERE node_id = ?", (node.id,))
        await db.execute(
            "INSERT INTO syn_nodes_fts(node_id, title, content) VALUES (?, ?, ?)",
            (node.id, node.title, node.content),
        )
        await db.commit()

    async def get_node(self, node_id: str) -> Node | None:
        db = self._db()
        async with db.execute("SELECT * FROM syn_nodes WHERE id = ?", (node_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_node(row)

    async def update_node(self, node: Node) -> None:
        db = self._db()
        await db.execute(
            """UPDATE syn_nodes SET kind=?, title=?, content=?, tags_json=?, level=?,
            vitality=?, access_count=?, success_count=?, failure_count=?,
            source=?, properties_json=?, updated_at=? WHERE id=?""",
            (
                str(node.kind),
                node.title,
                node.content,
                json.dumps(node.tags),
                str(node.level),
                node.vitality,
                node.access_count,
                node.success_count,
                node.failure_count,
                node.source,
                json.dumps(node.properties),
                node.updated_at,
                node.id,
            ),
        )
        # FTS sync
        await db.execute("DELETE FROM syn_nodes_fts WHERE node_id = ?", (node.id,))
        await db.execute(
            "INSERT INTO syn_nodes_fts(node_id, title, content) VALUES (?, ?, ?)",
            (node.id, node.title, node.content),
        )
        await db.commit()

    async def delete_node(self, node_id: str) -> None:
        db = self._db()
        await db.execute("DELETE FROM syn_nodes WHERE id = ?", (node_id,))
        await db.execute("DELETE FROM syn_nodes_fts WHERE node_id = ?", (node_id,))
        await db.commit()

    async def list_nodes(
        self,
        *,
        kind: NodeKind | None = None,
        level: ConsolidationLevel | None = None,
        limit: int = 100,
    ) -> list[Node]:
        db = self._db()
        conditions: list[str] = []
        params: list[str | int] = []
        if kind is not None:
            conditions.append("kind = ?")
            params.append(str(kind))
        if level is not None:
            conditions.append("level = ?")
            params.append(str(level))
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        sql = f"SELECT * FROM syn_nodes{where} ORDER BY updated_at DESC LIMIT ?"  # noqa: S608
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_node(r) for r in rows]

    # --- Edge CRUD ---

    async def save_edge(self, edge: Edge) -> None:
        db = self._db()
        await db.execute(
            """INSERT INTO syn_edges (id, source_id, target_id, kind, weight, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, kind) DO UPDATE SET weight=excluded.weight""",
            (edge.id, edge.source_id, edge.target_id, str(edge.kind), edge.weight, edge.created_at),
        )
        await db.commit()

    async def get_edges(self, node_id: str, *, direction: str = "both") -> list[Edge]:
        db = self._db()
        if direction == "outgoing":
            sql = "SELECT * FROM syn_edges WHERE source_id = ?"
        elif direction == "incoming":
            sql = "SELECT * FROM syn_edges WHERE target_id = ?"
        else:
            sql = "SELECT * FROM syn_edges WHERE source_id = ? OR target_id = ?"
        params: tuple[str, ...] = (node_id,) if direction != "both" else (node_id, node_id)
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_edge(r) for r in rows]

    async def update_edge(self, edge: Edge) -> None:
        db = self._db()
        await db.execute(
            "UPDATE syn_edges SET weight=?, kind=? WHERE id=?",
            (edge.weight, str(edge.kind), edge.id),
        )
        await db.commit()

    async def delete_edge(self, edge_id: str) -> None:
        db = self._db()
        await db.execute("DELETE FROM syn_edges WHERE id = ?", (edge_id,))
        await db.commit()

    # --- Search ---

    async def search_fts(self, query: str, *, limit: int = 20) -> list[Node]:
        db = self._db()
        # Sanitize FTS query: wrap each term in quotes
        terms = query.strip().split()
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        sql = """
            SELECT n.* FROM syn_nodes_fts f
            JOIN syn_nodes n ON n.id = f.node_id
            WHERE syn_nodes_fts MATCH ?
            ORDER BY rank LIMIT ?
        """
        try:
            async with db.execute(sql, (fts_query, limit)) as cur:
                rows = await cur.fetchall()
            return [_row_to_node(r) for r in rows]
        except Exception:
            return []

    async def search_fuzzy(
        self, query: str, *, limit: int = 20, threshold: float = 0.3
    ) -> list[Node]:
        # SQLite doesn't have native trigram — use LIKE fallback
        db = self._db()
        terms = query.strip().split()
        if not terms:
            return []
        conditions = " OR ".join("(title LIKE ? OR content LIKE ?)" for _ in terms)
        params: list[str | int] = []
        for t in terms:
            like = f"%{t}%"
            params.extend([like, like])
        params.append(limit)
        sql = f"SELECT * FROM syn_nodes WHERE {conditions} ORDER BY updated_at DESC LIMIT ?"  # noqa: S608
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_node(r) for r in rows]

    async def search_vector(self, embedding: list[float], *, limit: int = 20) -> list[Node]:
        # Vector search not available in SQLite — use PostgreSQL backend for vector support
        return []

    # --- Graph traversal (recursive CTE) ---

    async def get_neighbors(self, node_id: str, *, depth: int = 1) -> list[tuple[Node, Edge]]:
        db = self._db()
        sql = """
            WITH RECURSIVE neighbors(node_id, edge_id, depth) AS (
                SELECT CASE WHEN source_id = ? THEN target_id ELSE source_id END,
                       id, 1
                FROM syn_edges
                WHERE source_id = ? OR target_id = ?
                UNION
                SELECT CASE WHEN e.source_id = nb.node_id THEN e.target_id ELSE e.source_id END,
                       e.id, nb.depth + 1
                FROM syn_edges e
                JOIN neighbors nb ON e.source_id = nb.node_id OR e.target_id = nb.node_id
                WHERE nb.depth < ?
                  AND CASE WHEN e.source_id = nb.node_id THEN e.target_id ELSE e.source_id END != ?
            )
            SELECT DISTINCT nb.node_id, nb.edge_id FROM neighbors nb
        """
        async with db.execute(sql, (node_id, node_id, node_id, depth, node_id)) as cur:
            rows = await cur.fetchall()

        result: list[tuple[Node, Edge]] = []
        for row in rows:
            nid, eid = row["node_id"], row["edge_id"]
            node = await self.get_node(nid)
            async with db.execute("SELECT * FROM syn_edges WHERE id = ?", (eid,)) as ecur:
                erow = await ecur.fetchone()
            if node is not None and erow is not None:
                result.append((node, _row_to_edge(erow)))
        return result

    # --- Batch ---

    async def save_nodes_batch(self, nodes: Sequence[Node]) -> None:
        db = self._db()
        try:
            for node in nodes:
                await self.save_node(node)
        except Exception:
            await db.rollback()
            raise

    async def save_edges_batch(self, edges: Sequence[Edge]) -> None:
        db = self._db()
        try:
            for edge in edges:
                await self.save_edge(edge)
        except Exception:
            await db.rollback()
            raise

    # --- Maintenance ---

    async def prune_edges(self, *, weight_below: float = 0.1) -> int:
        db = self._db()
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM syn_edges WHERE weight < ?", (weight_below,)
        ) as cur:
            row = await cur.fetchone()
            count = row["cnt"] if row else 0
        await db.execute("DELETE FROM syn_edges WHERE weight < ?", (weight_below,))
        await db.commit()
        return int(count)

    async def decay_vitality(self, *, factor: float = 0.95) -> int:
        db = self._db()
        async with db.execute("SELECT COUNT(*) as cnt FROM syn_nodes") as cur:
            row = await cur.fetchone()
            count = row["cnt"] if row else 0
        await db.execute("UPDATE syn_nodes SET vitality = vitality * ?", (factor,))
        await db.commit()
        return int(count)


def _row_to_node(row: aiosqlite.Row) -> Node:
    props_raw = row["properties_json"] if "properties_json" in row.keys() else "{}"
    return Node(
        id=row["id"],
        kind=NodeKind(row["kind"]),
        title=row["title"],
        content=row["content"],
        tags=json.loads(row["tags_json"]),
        level=ConsolidationLevel(row["level"]),
        vitality=row["vitality"],
        access_count=row["access_count"],
        success_count=row["success_count"],
        failure_count=row["failure_count"],
        properties=json.loads(props_raw),
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_edge(row: aiosqlite.Row) -> Edge:
    return Edge(
        id=row["id"],
        source_id=row["source_id"],
        target_id=row["target_id"],
        kind=EdgeKind(row["kind"]),
        weight=row["weight"],
        created_at=row["created_at"],
    )
