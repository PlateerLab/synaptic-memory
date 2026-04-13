"""SQLite storage backend with FTS5 and recursive CTE."""

from __future__ import annotations

import json
import logging
import re
import unicodedata

logger = logging.getLogger("sqlite-backend")
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

    Guard: only apply Kiwi when the text has significant Korean content
    (≥30% Hangul characters). For structured/tabular data (CSV rows,
    code identifiers, English-heavy text) Kiwi over-segments tokens
    like "25SS" → "25 SS" or "product_code" → "product code", breaking
    exact matches. The regex fallback handles particle stripping without
    damaging non-Korean tokens.
    """
    if not text:
        return text

    # Check Korean content ratio — skip Kiwi for non-Korean-dominant text
    hangul_count = sum(1 for c in text if "가" <= c <= "힣")
    total_chars = sum(1 for c in text if not c.isspace())
    korean_ratio = hangul_count / total_chars if total_chars > 0 else 0

    if korean_ratio >= 0.5:
        kiwi = _get_kiwi()
        if kiwi is not None:
            try:
                tokens = kiwi.tokenize(text)
                stems = [
                    tk.form for tk in tokens if tk.tag.startswith(("NN", "VV", "VA", "SL", "SN"))
                ]
                if stems:
                    return " ".join(stems)
            except Exception:
                pass

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

    __slots__ = ("_conn", "_hnsw_id_map", "_hnsw_index", "_hnsw_meta", "_path")

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn: aiosqlite.Connection | None = None
        # in-memory state — None=not loaded, False=skip (too few), Index=ready
        self._hnsw_index: object | None = None
        self._hnsw_id_map: dict[int, str] = {}  # hnsw key → node_id
        # cached signature so we can detect stale on-disk artifacts:
        # {"count": int, "max_updated_at": float, "ndim": int}
        self._hnsw_meta: dict[str, object] = {}

    # --- HNSW disk persistence helpers ---

    @property
    def _hnsw_index_path(self) -> str:
        """Sidecar file storing the usearch binary index."""
        return f"{self._path}.hnsw"

    @property
    def _hnsw_meta_path(self) -> str:
        """Sidecar file storing the id-map and validity signature."""
        return f"{self._path}.hnsw.meta.json"

    async def _hnsw_signature(self) -> tuple[int, float]:
        """Return ``(node_count, max_updated_at)`` for embedding nodes.

        Used to validate the on-disk HNSW cache. If either changes,
        the cache is stale and must be rebuilt — node was added,
        removed, or re-embedded.
        """
        db = self._db()
        async with db.execute(
            """
            SELECT COUNT(*) AS cnt, COALESCE(MAX(updated_at), 0) AS mu
              FROM syn_nodes
             WHERE embedding_json != '[]'
            """
        ) as cur:
            row = await cur.fetchone()
        return (int(row["cnt"]), float(row["mu"]))

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
        # Embedding may have changed → in-memory HNSW is now stale. The
        # on-disk sidecar will be revalidated by signature on next search,
        # so we don't delete it here.
        if node.embedding:
            self._hnsw_index = None

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
        # Embedding node may have been deleted — drop in-memory HNSW cache.
        # Disk sidecar will be revalidated by signature on next search.
        self._hnsw_index = None

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
        sql = f"SELECT * FROM syn_nodes{where} ORDER BY updated_at DESC LIMIT ?"
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

    # --- Batch read ---

    async def get_nodes_batch(self, node_ids: list[str]) -> list[Node]:
        """Fetch multiple nodes in one SQL query (WHERE id IN (...))."""
        if not node_ids:
            return []
        db = self._db()
        placeholders = ",".join("?" for _ in node_ids)
        sql = f"SELECT * FROM syn_nodes WHERE id IN ({placeholders})"
        async with db.execute(sql, node_ids) as cur:
            rows = await cur.fetchall()
        return [_row_to_node(r) for r in rows]

    # --- Count ---

    async def count_nodes(
        self,
        *,
        kind: str | NodeKind | None = None,
        category: str | None = None,
        year: int | None = None,
    ) -> int:
        """SQL COUNT — no full scan, no Python loop."""
        db = self._db()
        clauses = []
        params: list[str] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(str(kind).lower() if isinstance(kind, NodeKind) else str(kind).lower())
        if category:
            clauses.append("properties_json LIKE ?")
            params.append(f'%"category": "{category}"%')
        if year is not None:
            clauses.append("properties_json LIKE ?")
            params.append(f'%"year": "{year}"%')
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT COUNT(*) FROM syn_nodes WHERE {where}"
        async with db.execute(sql, params) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    # --- Search ---

    async def search_fts(self, query: str, *, limit: int = 20) -> list[Node]:
        db = self._db()
        # Normalize query the same way content was indexed.
        # Also try the original query as a fallback — Kiwi over-
        # segmentation on the query side can miss exact matches that
        # the regex path (used for indexed structured data) would hit.
        normalized = _normalize_korean(query)
        original_terms = query.strip().split()
        norm_terms = normalized.strip().split()
        # Merge: normalized terms first, then any original terms not
        # already present (handles Kiwi-split vs unsplit mismatch)
        term_seen = set(norm_terms)
        terms = list(norm_terms)
        for t in original_terms:
            if t not in term_seen:
                terms.append(t)
                term_seen.add(t)
        if not terms:
            return []

        scored_nodes: dict[str, tuple[Node, float]] = {}

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
                scored_nodes[node.id] = (node, bm25_val)
        except Exception:
            pass

        # Pass 2: LIKE-based substring scan for terms FTS5 missed.
        # Handles Korean compound words where tokenisation may not align.
        if len(scored_nodes) < limit:
            like_parts = " OR ".join("(title LIKE ? OR content LIKE ?)" for _ in terms)
            params: list[str | int] = []
            for t in terms:
                like = f"%{t}%"
                params.extend([like, like])
            params.append(limit * 2)
            like_sql = f"SELECT * FROM syn_nodes WHERE {like_parts} LIMIT ?"
            async with db.execute(like_sql, params) as cur:
                rows2 = await cur.fetchall()
            for r in rows2:
                node = _row_to_node(r)
                if node.id in scored_nodes:
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
                    scored_nodes[node.id] = (node, 10000.0 - sub)

        # Sort: FTS5 negatives first (ascending), then substring positives
        ranked = sorted(scored_nodes.values(), key=lambda x: x[1])
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
        sql = f"SELECT * FROM syn_nodes WHERE {conditions} ORDER BY updated_at DESC LIMIT ?"
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_node(r) for r in rows]

    async def search_vector(self, embedding: list[float], *, limit: int = 20) -> list[Node]:
        """Vector search with optional HNSW acceleration.

        When ``usearch`` is installed (``pip install usearch``), an
        in-memory HNSW index is built lazily on the first call and
        reused for subsequent queries. Search latency drops from
        ~11s (brute-force on 90K nodes) to ~1ms.

        Without usearch, falls back to brute-force cosine scan.
        """
        if not embedding:
            return []

        # Try HNSW index first
        results = await self._search_vector_hnsw(embedding, limit)
        if results is not None:
            return results

        # Fallback: brute-force
        db = self._db()
        async with db.execute("SELECT * FROM syn_nodes WHERE embedding_json != '[]'") as cur:
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

    async def _search_vector_hnsw(self, embedding: list[float], limit: int) -> list[Node] | None:
        """HNSW search via usearch.

        Resolution order (each step is O(ms) when warm):

        1. **In-memory cache** — index already loaded this process.
        2. **Disk sidecar** — load ``{db}.hnsw`` + ``{db}.hnsw.meta.json``
           and validate its signature against the current node table.
        3. **Build from DB** — read every embedding, build a fresh
           index, save it to disk for next time.

        Returns ``None`` when usearch isn't installed or there are too
        few vectors for HNSW to beat a brute-force scan.
        """
        try:
            import numpy as np
            from usearch.index import Index
        except ImportError:
            return None

        if self._hnsw_index is False:
            return None

        # 1. In-memory cache
        if self._hnsw_index is None:
            # 2. Try disk sidecar
            loaded = await self._try_load_hnsw_from_disk(Index)
            if loaded:
                logger.info(
                    "sqlite: loaded HNSW from disk (%s, %d vectors)",
                    self._hnsw_index_path,
                    len(self._hnsw_id_map),
                )
            else:
                # 3. Build from scratch + persist
                built = await self._build_and_persist_hnsw(Index, np)
                if not built:
                    return None

        # Search
        if self._hnsw_index is False or self._hnsw_index is None:
            return None
        q = np.array(embedding, dtype=np.float32)
        results = self._hnsw_index.search(q, limit)  # type: ignore[union-attr]
        node_ids = [self._hnsw_id_map[int(k)] for k in results.keys if int(k) in self._hnsw_id_map]
        return await self.get_nodes_batch(node_ids)

    async def _try_load_hnsw_from_disk(self, index_cls: type) -> bool:
        """Attempt to load index + meta from disk. Returns True on success."""
        from pathlib import Path as _P

        idx_path = _P(self._hnsw_index_path)
        meta_path = _P(self._hnsw_meta_path)
        if not idx_path.exists() or not meta_path.exists():
            return False

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            disk_count = int(meta.get("count", 0))
            disk_mu = float(meta.get("max_updated_at", 0))
            disk_ndim = int(meta.get("ndim", 0))
            disk_id_map = {int(k): str(v) for k, v in meta.get("id_map", {}).items()}

            # Validate signature against current DB state
            cur_count, cur_mu = await self._hnsw_signature()
            if disk_count != cur_count or abs(disk_mu - cur_mu) > 1e-6:
                logger.info(
                    "sqlite: HNSW disk cache stale (count %d→%d, mu %f→%f) — rebuilding",
                    disk_count,
                    cur_count,
                    disk_mu,
                    cur_mu,
                )
                return False

            # Load the binary usearch index
            idx = index_cls(ndim=disk_ndim, metric="cos")
            idx.load(str(idx_path))
            self._hnsw_index = idx
            self._hnsw_id_map = disk_id_map
            self._hnsw_meta = {
                "count": cur_count,
                "max_updated_at": cur_mu,
                "ndim": disk_ndim,
            }
            return True
        except Exception as exc:
            logger.warning("sqlite: failed to load HNSW disk cache: %s", exc)
            return False

    async def _build_and_persist_hnsw(self, index_cls: type, np_module) -> bool:
        """Build HNSW from DB and persist to disk. Returns True on success."""
        db = self._db()
        async with db.execute(
            "SELECT id, embedding_json, updated_at FROM syn_nodes WHERE embedding_json != '[]'"
        ) as cur:
            rows = await cur.fetchall()
        if not rows or len(rows) < 100:
            # Too few vectors for HNSW to be worthwhile
            self._hnsw_index = False
            return False

        # Detect dimension from first non-empty embedding
        first_emb = json.loads(rows[0]["embedding_json"])
        if not first_emb:
            self._hnsw_index = False
            return False
        ndim = len(first_emb)

        idx = index_cls(ndim=ndim, metric="cos")
        vectors: list[list[float]] = []
        keys: list[int] = []
        id_map: dict[int, str] = {}
        max_mu = 0.0
        for i, r in enumerate(rows):
            emb = json.loads(r["embedding_json"])
            if len(emb) == ndim:
                vectors.append(emb)
                keys.append(i)
                id_map[i] = r["id"]
                mu = float(r["updated_at"] or 0)
                if mu > max_mu:
                    max_mu = mu

        if vectors:
            arr = np_module.array(vectors, dtype=np_module.float32)
            karr = np_module.array(keys, dtype=np_module.int64)
            idx.add(karr, arr)

        self._hnsw_index = idx
        self._hnsw_id_map = id_map
        self._hnsw_meta = {
            "count": len(vectors),
            "max_updated_at": max_mu,
            "ndim": ndim,
        }
        logger.info(
            "sqlite: built HNSW index with %d vectors (dim=%d)",
            len(vectors),
            ndim,
        )

        # Persist to disk so the next process starts warm.
        try:
            idx.save(self._hnsw_index_path)
            with open(self._hnsw_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "count": len(vectors),
                        "max_updated_at": max_mu,
                        "ndim": ndim,
                        "id_map": {str(k): v for k, v in id_map.items()},
                    },
                    f,
                )
            logger.info("sqlite: saved HNSW disk cache → %s", self._hnsw_index_path)
        except Exception as exc:
            logger.warning("sqlite: failed to persist HNSW disk cache: %s", exc)

        return True

    def invalidate_vector_index(self) -> None:
        """Drop the in-memory HNSW cache.

        The on-disk sidecar is left alone; it will be revalidated by
        signature on the next search. Call this after bulk embedding
        updates to force the next search to rebuild from scratch.
        """
        self._hnsw_index = None
        self._hnsw_id_map = {}
        self._hnsw_meta = {}

    def delete_hnsw_disk_cache(self) -> None:
        """Remove the on-disk HNSW sidecar files.

        Use after migrating embedding models or when you know the
        cached vectors are no longer compatible with the current index.
        """
        from pathlib import Path as _P

        for path in (self._hnsw_index_path, self._hnsw_meta_path):
            try:
                _P(path).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("sqlite: failed to delete %s: %s", path, exc)

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
        """Batch insert/upsert nodes with a single commit.

        Previous implementation called ``save_node`` per item, issuing
        one fsync per node. This version batches the SQL and FTS writes
        then commits once — ~10-50x faster on large ingests.
        """
        if not nodes:
            return
        db = self._db()
        node_rows = []
        fts_rows = []
        for node in nodes:
            title = unicodedata.normalize("NFC", node.title) if node.title else node.title
            content = unicodedata.normalize("NFC", node.content) if node.content else node.content
            embedding_json = json.dumps(node.embedding) if node.embedding else "[]"
            node_rows.append(
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
                )
            )
            fts_title = _normalize_korean(title) if title else ""
            fts_content = _normalize_korean(content) if content else ""
            fts_rows.append((node.id, fts_title, fts_content))

        try:
            await db.executemany(
                """INSERT INTO syn_nodes
                (id, kind, title, content, tags_json, level, vitality,
                 access_count, success_count, failure_count, source,
                 properties_json, embedding_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title, content=excluded.content,
                    tags_json=excluded.tags_json, level=excluded.level,
                    vitality=excluded.vitality,
                    properties_json=excluded.properties_json,
                    embedding_json=excluded.embedding_json,
                    updated_at=excluded.updated_at""",
                node_rows,
            )
            # FTS sync: delete then re-insert
            await db.executemany(
                "DELETE FROM syn_nodes_fts WHERE node_id = ?",
                [(n.id,) for n in nodes],
            )
            await db.executemany(
                "INSERT INTO syn_nodes_fts(node_id, title, content) VALUES (?, ?, ?)",
                fts_rows,
            )
            await db.commit()
            # Invalidate HNSW cache — new embeddings need re-indexing
            if any(n.embedding for n in nodes):
                self.invalidate_vector_index()
        except Exception:
            await db.rollback()
            raise

    async def save_edges_batch(self, edges: Sequence[Edge]) -> None:
        """Batch insert edges with a single commit."""
        if not edges:
            return
        db = self._db()
        rows = [
            (e.id, e.source_id, e.target_id, str(e.kind), e.weight, e.created_at) for e in edges
        ]
        try:
            await db.executemany(
                """INSERT INTO syn_edges (id, source_id, target_id, kind, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET weight=excluded.weight""",
                rows,
            )
            await db.commit()
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
