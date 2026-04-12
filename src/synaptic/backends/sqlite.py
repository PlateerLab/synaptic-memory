"""SQLite storage backend with FTS5 and recursive CTE."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path


# --- Korean FTS normalization ---
#
# Two tiers:
# 1. kiwipiepy (optional) — morphological analysis. Splits compound nouns,
#    extracts noun/verb stems, handles irregular verbs. Best quality.
# 2. Regex fallback — strips common postposition particles. Good enough
#    for the main case (정보화기기를 → 정보화기기) when Kiwi isn't installed.
#
# Both tiers are applied at index time (save_node FTS sync) AND query
# time (search_fts) so the tokenization is consistent on both sides.

_kiwi_instance = None
_kiwi_available: bool | None = None  # None = not yet checked


def _get_kiwi():
    """Lazy-load Kiwi. Returns the instance or None if not installed."""
    global _kiwi_instance, _kiwi_available
    if _kiwi_available is None:
        try:
            from kiwipiepy import Kiwi
            _kiwi_instance = Kiwi()
            _kiwi_available = True
        except ImportError:
            _kiwi_available = False
    return _kiwi_instance


# Regex fallback for when Kiwi is not installed
_KO_PARTICLE = re.compile(
    r"([가-힣]{2,}?)(에서|부터|까지|으로|에게|에는|에도|에서는|에서도"
    r"|으로서|으로써|이라|이며|이고|이나|이든|처럼|만큼"
    r"|의|을|를|에|은|는|이|가|와|로|서|며|고|나)(?=[^가-힣]|$)"
)


def _normalize_korean(text: str) -> str:
    """Normalize Korean text for FTS indexing/querying.

    With Kiwi: full morphological analysis → noun/verb stems + numbers.
    "경마산업관리 규정" → "경마 산업 관리 규정"
    "정보화기기를 교체하고" → "정보 기기 교체"

    Without Kiwi: regex particle stripping only.
    "정보화기기를" → "정보화기기"
    """
    if not text:
        return text

    kiwi = _get_kiwi()
    if kiwi is not None:
        try:
            tokens = kiwi.tokenize(text)
            # Keep nouns (NN*), verbs (VV), adjectives (VA),
            # foreign words (SL), numbers (SN)
            stems = [
                tk.form for tk in tokens
                if tk.tag.startswith(("NN", "VV", "VA", "SL", "SN"))
            ]
            if stems:
                return " ".join(stems)
        except Exception:
            pass  # fall through to regex

    return _KO_PARTICLE.sub(r"\1", text)

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
    embedding_json TEXT NOT NULL DEFAULT '[]',
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
        if "embedding_json" not in columns:
            await self._conn.execute(
                "ALTER TABLE syn_nodes ADD COLUMN embedding_json TEXT NOT NULL DEFAULT '[]'"
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
        title = unicodedata.normalize("NFC", node.title) if node.title else node.title
        content = unicodedata.normalize("NFC", node.content) if node.content else node.content
        embedding_json = json.dumps(node.embedding) if node.embedding else "[]"
        await db.execute(
            """INSERT INTO syn_nodes
            (id, kind, title, content, tags_json, level, vitality,
             access_count, success_count, failure_count, source, properties_json,
             embedding_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, content=excluded.content, tags_json=excluded.tags_json,
                level=excluded.level, vitality=excluded.vitality,
                properties_json=excluded.properties_json,
                embedding_json=excluded.embedding_json, updated_at=excluded.updated_at""",
            (
                node.id,
                str(node.kind),
                title,
                content,
                json.dumps(node.tags),
                str(node.level),
                node.vitality,
                node.access_count,
                node.success_count,
                node.failure_count,
                node.source,
                json.dumps(node.properties),
                embedding_json,
                node.created_at,
                node.updated_at,
            ),
        )
        # FTS sync — strip Korean particles so "정보화기기를" indexes as
        # "정보화기기", matching queries for the bare stem.
        fts_title = _normalize_korean(title)
        fts_content = _normalize_korean(content)
        await db.execute("DELETE FROM syn_nodes_fts WHERE node_id = ?", (node.id,))
        await db.execute(
            "INSERT INTO syn_nodes_fts(node_id, title, content) VALUES (?, ?, ?)",
            (node.id, fts_title, fts_content),
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
        title = unicodedata.normalize("NFC", node.title) if node.title else node.title
        content = unicodedata.normalize("NFC", node.content) if node.content else node.content
        await db.execute(
            """UPDATE syn_nodes SET kind=?, title=?, content=?, tags_json=?, level=?,
            vitality=?, access_count=?, success_count=?, failure_count=?,
            source=?, properties_json=?, updated_at=? WHERE id=?""",
            (
                str(node.kind),
                title,
                content,
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
        # FTS sync — particle-stripped for Korean stem matching
        fts_title = _normalize_korean(title)
        fts_content = _normalize_korean(content)
        await db.execute("DELETE FROM syn_nodes_fts WHERE node_id = ?", (node.id,))
        await db.execute(
            "INSERT INTO syn_nodes_fts(node_id, title, content) VALUES (?, ?, ?)",
            (node.id, fts_title, fts_content),
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
        kind: str | NodeKind | None = None,
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
        # Strip Korean particles from query terms so "정보화기기를"
        # matches the particle-stripped FTS index ("정보화기기").
        query = _normalize_korean(query)
        terms = query.strip().split()
        if not terms:
            return []

        seen: dict[str, tuple[Node, float]] = {}

        # Pass 1: FTS5 with title 3x boost.
        # bm25(table, w0, w1, w2) — col0=node_id(0), col1=title(3.0), col2=content(1.0).
        # FTS5 bm25 returns negative values; more negative = better match.
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        fts_sql = """
            SELECT n.*, bm25(syn_nodes_fts, 0, 3.0, 1.0) AS _bm25
            FROM syn_nodes_fts
            JOIN syn_nodes n ON n.id = syn_nodes_fts.node_id
            WHERE syn_nodes_fts MATCH ?
            ORDER BY bm25(syn_nodes_fts, 0, 3.0, 1.0)
            LIMIT ?
        """
        try:
            async with db.execute(fts_sql, (fts_query, limit * 2)) as cur:
                rows = await cur.fetchall()
            for r in rows:
                node = _row_to_node(r)
                bm25_val = r["_bm25"] or 0.0  # negative; lower = better
                seen[node.id] = (node, bm25_val)
        except Exception:
            pass

        # Pass 2: LIKE-based substring scan for terms FTS5 missed.
        # Handles Korean compound words where tokenisation may not align.
        if len(seen) < limit:
            like_parts = " OR ".join(
                "(title LIKE ? OR content LIKE ?)" for _ in terms
            )
            params: list[str | int] = []
            for t in terms:
                like = f"%{t}%"
                params.extend([like, like])
            params.append(limit * 2)
            like_sql = (  # noqa: S608
                f"SELECT * FROM syn_nodes WHERE {like_parts} LIMIT ?"
            )
            async with db.execute(like_sql, params) as cur:
                rows2 = await cur.fetchall()
            for r in rows2:
                node = _row_to_node(r)
                if node.id in seen:
                    continue
                title_lower = node.title.lower()
                content_lower = node.content.lower()
                sub = sum(
                    3.0 if t.lower() in title_lower else 1.0
                    for t in terms
                    if t.lower() in title_lower or t.lower() in content_lower
                )
                if sub > 0:
                    # Use large positive offset so substring hits sort after FTS5
                    seen[node.id] = (node, 10000.0 - sub)

        # Sort: FTS5 negatives first (ascending), then substring positives
        ranked = sorted(seen.values(), key=lambda x: x[1])
        return [n for n, _ in ranked[:limit]]

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
        """Brute-force cosine similarity scan over stored embeddings.

        SQLite has no native vector index, so we load all non-empty
        embeddings, compute cosine in Python, and return the top-k.
        Scales to ~50K nodes comfortably (~100ms on Apple Silicon);
        beyond that, switch to Qdrant or PostgreSQL + pgvector.
        """
        if not embedding:
            return []
        db = self._db()
        async with db.execute(
            "SELECT * FROM syn_nodes WHERE embedding_json != '[]'"
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return []

        scored: list[tuple[Node, float]] = []
        for r in rows:
            node = _row_to_node(r)
            if not node.embedding:
                continue
            sim = _cosine_sim(embedding, node.embedding)
            if sim > 0:
                scored.append((node, sim))

        scored.sort(key=lambda x: -x[1])
        return [n for n, _ in scored[:limit]]

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


def _safe_node_kind(value: str) -> str | NodeKind:
    """Convert to NodeKind if known, otherwise keep as raw string."""
    try:
        return NodeKind(value)
    except ValueError:
        return value


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _row_to_node(row: aiosqlite.Row) -> Node:
    keys = row.keys()
    props_raw = row["properties_json"] if "properties_json" in keys else "{}"
    emb_raw = row["embedding_json"] if "embedding_json" in keys else "[]"
    emb = json.loads(emb_raw) if emb_raw and emb_raw != "[]" else []
    return Node(
        id=row["id"],
        kind=_safe_node_kind(row["kind"]),
        title=row["title"],
        content=row["content"],
        tags=json.loads(row["tags_json"]),
        level=ConsolidationLevel(row["level"]),
        embedding=emb,
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
