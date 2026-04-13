"""Persistent CDC sync state — watermark, PK index, FK snapshots.

Two tables stored inside the graph SQLite database itself (so a graph
file is fully self-contained):

``syn_cdc_state`` — one row per ``(source_url, table)`` pair.

    Holds the strategy ('timestamp' / 'hash' / 'full'), the change
    column name, the last watermark value (max change_col seen), and a
    schema fingerprint we compare on every sync to detect drift.

``syn_cdc_pk_index`` — one row per source-database row.

    Maps ``(source_url, table, pk) → node_id`` so the next sync can
    resolve "which graph node belongs to this row" without scanning
    the entire ``syn_nodes`` table. Also stores ``row_hash`` for the
    hash strategy and ``fk_edges`` for FK re-computation in Phase 4.

The store talks to ``aiosqlite`` directly. We piggy-back on the
existing connection from ``SQLiteBackend`` rather than opening a
second one — that keeps everything inside a single transaction and
avoids the dreaded "database is locked" between sync writes and
``save_node`` writes.

This module is foundation code for Phase 1. Sync logic lives in
``cdc/sync.py`` (Phase 2+). Phase 1 only ships the schema, CRUD
helpers, and unit tests for the persistence layer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from synaptic.extensions.cdc.ids import canonical_pk

if TYPE_CHECKING:
    import aiosqlite


_CDC_STATE_SQL = """
CREATE TABLE IF NOT EXISTS syn_cdc_state (
    source_url         TEXT NOT NULL,
    table_name         TEXT NOT NULL,
    strategy           TEXT NOT NULL DEFAULT 'timestamp',
    change_col         TEXT,
    last_sync_at       REAL NOT NULL DEFAULT 0.0,
    last_watermark     TEXT,
    primary_key_col    TEXT NOT NULL DEFAULT 'id',
    row_count          INTEGER NOT NULL DEFAULT 0,
    schema_fingerprint TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (source_url, table_name)
);
"""

_CDC_PK_INDEX_SQL = """
CREATE TABLE IF NOT EXISTS syn_cdc_pk_index (
    source_url  TEXT NOT NULL,
    table_name  TEXT NOT NULL,
    pk          TEXT NOT NULL,
    node_id     TEXT NOT NULL,
    row_hash    TEXT,
    fk_edges    TEXT,
    PRIMARY KEY (source_url, table_name, pk)
);
"""

_CDC_PK_INDEX_BY_NODE_SQL = """
CREATE INDEX IF NOT EXISTS idx_syn_cdc_pk_node
    ON syn_cdc_pk_index(node_id);
"""


@dataclass(slots=True)
class TableSyncState:
    """Snapshot of one ``syn_cdc_state`` row.

    Returned by :meth:`SyncStateStore.load_state`. ``last_watermark``
    is intentionally a string — different DBs return timestamps in
    different shapes (datetime, ISO string, epoch float) and we
    serialise them via ``str()`` so the comparison is always
    lexicographic-safe for ISO formats and numerically-safe for
    integer / float watermarks.
    """

    source_url: str
    table_name: str
    strategy: str
    change_col: str | None
    last_sync_at: float
    last_watermark: str | None
    primary_key_col: str
    row_count: int
    schema_fingerprint: str


class SyncStateStore:
    """CRUD facade over the two CDC bookkeeping tables.

    Constructed by ``SQLiteBackend.ensure_cdc_tables()`` and reused by
    the sync orchestrator. All methods are coroutine-based because
    they share the backend's ``aiosqlite`` connection — opening a
    second one would risk deadlocks against in-flight ``save_node``
    transactions.
    """

    __slots__ = ("_conn",)

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @staticmethod
    async def install_schema(conn: aiosqlite.Connection) -> None:
        """Create the two CDC tables if they don't exist.

        Idempotent. Safe to call on every backend ``connect()`` —
        ``IF NOT EXISTS`` on every statement guarantees no migration
        is needed for graphs created before CDC support landed.
        """
        await conn.execute(_CDC_STATE_SQL)
        await conn.execute(_CDC_PK_INDEX_SQL)
        await conn.execute(_CDC_PK_INDEX_BY_NODE_SQL)
        await conn.commit()

    # --- syn_cdc_state CRUD ---

    async def load_state(self, source_url: str, table: str) -> TableSyncState | None:
        """Return the stored sync state for a ``(source_url, table)`` pair.

        Returns ``None`` when no prior sync has run — callers should
        then perform an initial full load.
        """
        async with self._conn.execute(
            """
            SELECT source_url, table_name, strategy, change_col, last_sync_at,
                   last_watermark, primary_key_col, row_count, schema_fingerprint
              FROM syn_cdc_state
             WHERE source_url = ? AND table_name = ?
            """,
            (source_url, table),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return TableSyncState(
            source_url=row[0],
            table_name=row[1],
            strategy=row[2],
            change_col=row[3],
            last_sync_at=float(row[4] or 0.0),
            last_watermark=row[5],
            primary_key_col=row[6] or "id",
            row_count=int(row[7] or 0),
            schema_fingerprint=row[8] or "",
        )

    async def save_state(self, state: TableSyncState) -> None:
        """Upsert a sync state row.

        Used at the end of every successful sync to advance the
        watermark and update the row count + fingerprint.
        """
        await self._conn.execute(
            """
            INSERT INTO syn_cdc_state
                (source_url, table_name, strategy, change_col, last_sync_at,
                 last_watermark, primary_key_col, row_count, schema_fingerprint)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_url, table_name) DO UPDATE SET
                strategy = excluded.strategy,
                change_col = excluded.change_col,
                last_sync_at = excluded.last_sync_at,
                last_watermark = excluded.last_watermark,
                primary_key_col = excluded.primary_key_col,
                row_count = excluded.row_count,
                schema_fingerprint = excluded.schema_fingerprint
            """,
            (
                state.source_url,
                state.table_name,
                state.strategy,
                state.change_col,
                state.last_sync_at or time.time(),
                state.last_watermark,
                state.primary_key_col,
                state.row_count,
                state.schema_fingerprint,
            ),
        )
        await self._conn.commit()

    async def delete_state(self, source_url: str, table: str) -> None:
        """Forget the sync state for a ``(source_url, table)`` pair.

        Use to force the next sync to behave as a fresh full load.
        """
        await self._conn.execute(
            "DELETE FROM syn_cdc_state WHERE source_url = ? AND table_name = ?",
            (source_url, table),
        )
        await self._conn.commit()

    # --- syn_cdc_pk_index CRUD ---

    async def upsert_pk(
        self,
        source_url: str,
        table: str,
        pk: str,
        node_id: str,
        *,
        row_hash: str | None = None,
        fk_edges: dict[str, str] | None = None,
    ) -> None:
        """Record (or replace) the mapping for a single source row."""
        await self._conn.execute(
            """
            INSERT INTO syn_cdc_pk_index (source_url, table_name, pk, node_id, row_hash, fk_edges)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_url, table_name, pk) DO UPDATE SET
                node_id = excluded.node_id,
                row_hash = excluded.row_hash,
                fk_edges = excluded.fk_edges
            """,
            (
                source_url,
                table,
                canonical_pk(pk),
                node_id,
                row_hash,
                json.dumps(fk_edges, ensure_ascii=False) if fk_edges else None,
            ),
        )

    async def upsert_pk_batch(
        self,
        source_url: str,
        table: str,
        rows: Iterable[tuple[str, str, str | None, dict[str, str] | None]],
    ) -> None:
        """Bulk version of :meth:`upsert_pk`.

        Each ``rows`` tuple is ``(pk, node_id, row_hash, fk_edges)``.
        Use this on initial load and large incremental syncs — the
        per-row variant burns a Python round-trip per row, which adds
        up at 100k+ rows.
        """
        payload = [
            (
                source_url,
                table,
                canonical_pk(pk),
                node_id,
                row_hash,
                json.dumps(fk_edges, ensure_ascii=False) if fk_edges else None,
            )
            for (pk, node_id, row_hash, fk_edges) in rows
        ]
        if not payload:
            return
        await self._conn.executemany(
            """
            INSERT INTO syn_cdc_pk_index (source_url, table_name, pk, node_id, row_hash, fk_edges)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_url, table_name, pk) DO UPDATE SET
                node_id = excluded.node_id,
                row_hash = excluded.row_hash,
                fk_edges = excluded.fk_edges
            """,
            payload,
        )

    async def get_node_id(
        self,
        source_url: str,
        table: str,
        pk: str,
    ) -> str | None:
        """Return the graph ``node_id`` for a known source row, or ``None``."""
        async with self._conn.execute(
            """
            SELECT node_id FROM syn_cdc_pk_index
             WHERE source_url = ? AND table_name = ? AND pk = ?
            """,
            (source_url, table, canonical_pk(pk)),
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    async def get_fk_edges(
        self,
        source_url: str,
        table: str,
        pk: str,
    ) -> dict[str, str] | None:
        """Return the previously-stored FK snapshot for a row, or ``None``."""
        async with self._conn.execute(
            """
            SELECT fk_edges FROM syn_cdc_pk_index
             WHERE source_url = ? AND table_name = ? AND pk = ?
            """,
            (source_url, table, canonical_pk(pk)),
        ) as cur:
            row = await cur.fetchone()
        if not row or not row[0]:
            return None
        try:
            return dict(json.loads(row[0]))
        except (ValueError, TypeError):
            return None

    async def get_row_hash(
        self,
        source_url: str,
        table: str,
        pk: str,
    ) -> str | None:
        """Return the previously-stored row hash for a row, or ``None``."""
        async with self._conn.execute(
            """
            SELECT row_hash FROM syn_cdc_pk_index
             WHERE source_url = ? AND table_name = ? AND pk = ?
            """,
            (source_url, table, canonical_pk(pk)),
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    async def delete_pk(
        self,
        source_url: str,
        table: str,
        pk: str,
    ) -> None:
        """Remove a single PK from the index (after deleting its node)."""
        await self._conn.execute(
            """
            DELETE FROM syn_cdc_pk_index
             WHERE source_url = ? AND table_name = ? AND pk = ?
            """,
            (source_url, table, canonical_pk(pk)),
        )

    async def delete_pk_batch(
        self,
        source_url: str,
        table: str,
        pks: Iterable[str],
    ) -> int:
        """Bulk version of :meth:`delete_pk`. Returns count deleted."""
        items = [(source_url, table, canonical_pk(p)) for p in pks]
        if not items:
            return 0
        await self._conn.executemany(
            """
            DELETE FROM syn_cdc_pk_index
             WHERE source_url = ? AND table_name = ? AND pk = ?
            """,
            items,
        )
        return len(items)

    async def list_pks(
        self,
        source_url: str,
        table: str,
    ) -> list[str]:
        """Return every PK currently tracked for ``(source_url, table)``.

        Memory-cheap on small/medium tables. For very large tables,
        prefer the temp-table delete-detection path added in Phase 3.
        """
        async with self._conn.execute(
            """
            SELECT pk FROM syn_cdc_pk_index
             WHERE source_url = ? AND table_name = ?
            """,
            (source_url, table),
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def count_pks(
        self,
        source_url: str,
        table: str,
    ) -> int:
        """Return ``COUNT(*)`` of tracked PKs for ``(source_url, table)``."""
        async with self._conn.execute(
            """
            SELECT COUNT(*) FROM syn_cdc_pk_index
             WHERE source_url = ? AND table_name = ?
            """,
            (source_url, table),
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def find_deleted_pks(
        self,
        source_url: str,
        table: str,
        live_pks: Iterable[str],
    ) -> list[tuple[str, str]]:
        """Return ``(pk, node_id)`` rows that exist in the index but not in ``live_pks``.

        Loads ``live_pks`` into a transient ``TEMP TABLE`` and runs a
        ``LEFT JOIN`` so the diff stays inside SQLite — Python only
        sees the result set, not the full PK universe. That keeps
        memory flat for very large tables (the alternative would be
        materialising both sides as Python sets).
        """
        # Use a deterministic temp-table name so concurrent calls on
        # the same connection cannot collide. We DROP and recreate to
        # force schema-fresh state — `IF NOT EXISTS` would let stale
        # rows from a prior run leak through.
        await self._conn.execute("DROP TABLE IF EXISTS cdc_current_pks")
        await self._conn.execute(
            "CREATE TEMP TABLE cdc_current_pks (pk TEXT PRIMARY KEY)"
        )

        payload = [(canonical_pk(p),) for p in live_pks]
        if payload:
            await self._conn.executemany(
                "INSERT OR IGNORE INTO cdc_current_pks (pk) VALUES (?)",
                payload,
            )

        async with self._conn.execute(
            """
            SELECT idx.pk, idx.node_id
              FROM syn_cdc_pk_index AS idx
              LEFT JOIN cdc_current_pks AS cur ON cur.pk = idx.pk
             WHERE idx.source_url = ?
               AND idx.table_name = ?
               AND cur.pk IS NULL
            """,
            (source_url, table),
        ) as cur:
            rows = await cur.fetchall()

        await self._conn.execute("DROP TABLE IF EXISTS cdc_current_pks")
        return [(r[0], r[1]) for r in rows]
