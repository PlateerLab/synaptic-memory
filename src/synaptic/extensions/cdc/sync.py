"""Incremental sync orchestration for CDC.

Phase 2 ships the ``timestamp`` strategy: tables that expose a
monotonically-increasing column (``updated_at``, ``modified_at``, ...)
are read with a ``WHERE change_col >= last_watermark`` filter, and the
watermark is advanced after the batch commits.

Delete detection (Phase 3), FK edge re-computation (Phase 4), the hash
fallback (Phase 5), and the non-SQLite row readers (Phase 6) extend
this module incrementally.

The sync layer never talks to source databases directly — it consumes
an already-parsed :class:`TableSchema` plus a ``row_reader`` callable
that the dispatcher binds per DB dialect. That keeps dialect code
isolated inside ``db_ingester.py``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from synaptic.extensions.cdc.ids import canonical_pk, deterministic_row_id
from synaptic.extensions.cdc.state import SyncStateStore, TableSyncState

if TYPE_CHECKING:
    from synaptic.extensions.db_ingester import TableSchema
    from synaptic.extensions.table_ingester import TableIngester
    from synaptic.graph import SynapticGraph

logger = logging.getLogger("cdc-sync")


# Column-name candidates scanned by :func:`detect_change_column`. Ordered
# by preference — the first column whose name matches one of these
# substrings (case-insensitive) wins. Kept intentionally short: broader
# pattern matching tends to pick up ``created_at`` which is monotonic
# per-row but does *not* advance on UPDATE, defeating the whole point.
_CHANGE_COL_CANDIDATES: tuple[str, ...] = (
    "updated_at",
    "modified_at",
    "last_modified",
    "update_time",
    "modified_time",
    "mtime",
    "last_update",
    "updatedat",
    "updated",
    "modified",
)


def detect_change_column(columns: list[dict[str, str]]) -> str | None:
    """Return the best ``updated_at``-style column name, or ``None``.

    Matches against lower-cased column names. Used by
    :class:`DbSyncer` to auto-pick a change column when the caller
    does not supply one explicitly.
    """
    lowered = {c.get("name", "").lower(): c.get("name", "") for c in columns}
    for cand in _CHANGE_COL_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    # Second pass: substring match for camelCase / snake_case variants
    for cand in _CHANGE_COL_CANDIDATES:
        for low, orig in lowered.items():
            if cand in low:
                return orig
    return None


@dataclass(slots=True)
class TableSyncStats:
    """Per-table result from a single sync run."""

    table: str
    added: int = 0
    updated: int = 0
    deleted: int = 0
    fk_edges_added: int = 0
    fk_edges_removed: int = 0
    strategy: str = ""
    error: str | None = None
    schema_changed: bool = False
    """True when a source-schema drift was detected at the start of this sync
    and the table was force-reloaded. Callers that surface sync results to
    ops dashboards should highlight this — a drift means the *previous*
    graph state for this table was stale."""


@dataclass(slots=True)
class SyncResult:
    """Aggregate result returned by :meth:`SynapticGraph.sync_from_database`."""

    added: int = 0
    updated: int = 0
    deleted: int = 0
    elapsed_ms: float = 0.0
    tables: list[TableSyncStats] = field(default_factory=list)
    source_url: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SyncResult(added={self.added}, updated={self.updated}, "
            f"deleted={self.deleted}, elapsed_ms={self.elapsed_ms:.0f}, "
            f"tables={len(self.tables)})"
        )


RowReader = Callable[..., Awaitable[list[dict[str, Any]]] | list[dict[str, Any]]]
PkReader = Callable[[str, str], Awaitable[list[str]] | list[str]]


class TimestampTableSyncer:
    """Sync one table using the ``WHERE change_col >= watermark`` strategy.

    The syncer is stateful only for the duration of a single
    ``sync_table`` call — it reads the prior watermark from the
    :class:`SyncStateStore`, diffs against the fresh rows, calls
    ``TableIngester.ingest`` (which upserts on deterministic node IDs),
    then writes the advanced watermark back.

    Delete detection lives in Phase 3 and will run *before* this
    method. FK edge re-computation is Phase 4.
    """

    __slots__ = ("_ingester", "_source_url", "_store")

    def __init__(
        self,
        ingester: TableIngester,
        store: SyncStateStore,
        source_url: str,
    ) -> None:
        self._ingester = ingester
        self._store = store
        self._source_url = source_url

    async def detect_deletes(
        self,
        graph: SynapticGraph,
        schema: TableSchema,
        pk_reader: PkReader,
    ) -> int:
        """Find PKs missing from the source DB and delete their nodes.

        Streams every live PK from the source via ``pk_reader`` into
        a SQLite ``TEMP TABLE``, then ``LEFT JOIN`` against
        ``syn_cdc_pk_index`` returns the deleted rows. Each deleted
        node is removed from the graph (which cascades to edges
        thanks to the ``ON DELETE CASCADE`` on ``syn_edges``) and the
        PK index entry is dropped.

        Skipped on the very first sync — there is nothing to compare
        against until the initial load has populated the PK index.
        """
        prior_state = await self._store.load_state(self._source_url, schema.name)
        if prior_state is None:
            return 0

        live_result = pk_reader(schema.name, schema.primary_key)
        if hasattr(live_result, "__await__"):
            live_pks = await live_result  # type: ignore[misc]
        else:
            live_pks = live_result  # type: ignore[assignment]

        deleted_rows = await self._store.find_deleted_pks(self._source_url, schema.name, live_pks)
        if not deleted_rows:
            return 0

        for pk, node_id in deleted_rows:
            try:
                await graph.remove(node_id)
            except Exception:  # pragma: no cover - best-effort node delete
                logger.exception(
                    "delete_node failed for %s pk=%s node_id=%s",
                    schema.name,
                    pk,
                    node_id,
                )
            await self._store.delete_pk(self._source_url, schema.name, pk)

        # Decrement row_count so save_state reflects reality on the
        # next pass. We don't write state here — the timestamp
        # sync_table call that follows will overwrite it anyway.
        prior_state.row_count = max(0, prior_state.row_count - len(deleted_rows))
        await self._store.save_state(prior_state)

        logger.info(
            "detect_deletes(%s): -%d nodes",
            schema.name,
            len(deleted_rows),
        )
        return len(deleted_rows)

    async def sync_table(
        self,
        graph: SynapticGraph,
        schema: TableSchema,
        row_reader: RowReader,
        *,
        change_col: str | None = None,
    ) -> TableSyncStats:
        """Sync one table. Returns stats for the caller to aggregate."""
        stats = TableSyncStats(table=schema.name, strategy="timestamp")

        col_defs = [{"name": c.name, "type": c.type} for c in schema.columns]
        prior_state = await self._store.load_state(self._source_url, schema.name)
        current_fp = _schema_fingerprint(schema)
        prior_state, stats.schema_changed = await _detect_and_reset_on_schema_drift(
            self._store, self._source_url, prior_state, current_fp
        )
        is_initial = prior_state is None

        # Change column: prefer the stored one (schema is authoritative
        # across runs — if the user renamed a column, a fresh detect
        # call would diverge silently and break the WHERE clause).
        if prior_state and prior_state.change_col:
            change_col = prior_state.change_col
        elif change_col is None:
            change_col = detect_change_column(col_defs)

        # Without a change column we cannot do a timestamp sync. The
        # caller is expected to fall back to HashTableSyncer (Phase 5)
        # or full reload.
        if change_col is None:
            stats.error = "no change column detected"
            return stats

        # Initial load reads everything, incremental reads the delta.
        prior_watermark = prior_state.last_watermark if prior_state else None
        where_clause: str | None = None
        where_params: tuple[Any, ...] = ()
        if not is_initial and prior_watermark is not None:
            # `>=` on purpose — a sub-second resolution change column
            # (e.g. integer epoch seconds) can have multiple rows
            # sharing the last watermark. Re-ingesting them is a
            # no-op thanks to deterministic IDs + `ON CONFLICT DO
            # UPDATE`, so duplicates are safe.
            where_clause = f'"{change_col}" >= ?'
            where_params = (prior_watermark,)

        reader_result = row_reader(
            schema.name,
            where_clause=where_clause,
            where_params=where_params,
        )
        if hasattr(reader_result, "__await__"):
            rows = await reader_result  # type: ignore[misc]
        else:
            rows = reader_result  # type: ignore[assignment]

        if not rows:
            # Nothing to do — still advance last_sync_at so the
            # monitoring view shows the sync ran.
            if prior_state:
                prior_state.last_sync_at = time.time()
                await self._store.save_state(prior_state)
            return stats

        fk_map = {fk.column: (fk.ref_table, fk.ref_column) for fk in schema.foreign_keys}

        # Track which PKs were already known so we can bucket
        # add/update correctly without a second SELECT.
        new_watermark: Any = prior_watermark
        pk_batch: list[tuple[str, str, str | None, dict[str, str] | None]] = []

        # Bucket add/update *before* ingest — after upsert_pk is
        # called, every PK looks 'known'. We fetch every prior PK
        # index entry in **one batch SELECT** rather than paying N
        # sequential round-trips on a table with N changed rows.
        pk_strs: list[str] = []
        for row in rows:
            pk_val = row.get(schema.primary_key)
            if pk_val is not None:
                pk_strs.append(canonical_pk(pk_val))
        prior_index = await self._store.get_pk_index_batch(self._source_url, schema.name, pk_strs)

        row_is_new: list[bool] = []
        prior_fks: dict[str, dict[str, str]] = {}
        for row in rows:
            pk_val = row.get(schema.primary_key)
            if pk_val is None:
                row_is_new.append(False)
                continue
            pk_str = canonical_pk(pk_val)
            existing, _prior_hash, prior_fk_json = prior_index.get(pk_str, (None, None, None))
            row_is_new.append(existing is None)
            if existing is not None and fk_map and prior_fk_json:
                try:
                    prior_fks[pk_str] = dict(json.loads(prior_fk_json))
                except (ValueError, TypeError):
                    pass

        # Ingest via TableIngester with deterministic IDs. New FK
        # edges are created here; stale edges (from a previous FK
        # value) are pruned in the diff loop below.
        await self._ingester.ingest(
            graph,
            schema.name,
            col_defs,
            rows,
            primary_key=schema.primary_key,
            foreign_keys=fk_map if fk_map else None,
            source_url=self._source_url,
        )

        # Phase 4 — FK diff: any (col, target_pk) that was in the
        # prior snapshot but is gone or repointed in the new row is
        # an edge we need to delete. New edges are already in place
        # thanks to the ingest call above.
        if fk_map and prior_fks:
            stats.fk_edges_removed += await self._prune_stale_fk_edges(
                graph, schema, fk_map, rows, prior_fks
            )

        # Count newly-added FK edges for observability — anything
        # that wasn't in the prior snapshot but is in the new row.
        if fk_map:
            for row in rows:
                pk_val = row.get(schema.primary_key)
                if pk_val is None:
                    continue
                prior = prior_fks.get(canonical_pk(pk_val), {})
                for col in fk_map:
                    new_target = row.get(col)
                    if new_target is None:
                        continue
                    if prior.get(col) != str(new_target):
                        stats.fk_edges_added += 1

        # Record the new PK index entries.
        for row, is_new in zip(rows, row_is_new):
            pk_val = row.get(schema.primary_key)
            if pk_val is None:
                continue
            if is_new:
                stats.added += 1
            else:
                stats.updated += 1

            node_id = deterministic_row_id(self._source_url, schema.name, pk_val)
            fk_snapshot: dict[str, str] | None = None
            if fk_map:
                fk_snapshot = {col: str(row[col]) for col in fk_map if row.get(col) is not None}
            pk_batch.append((canonical_pk(pk_val), node_id, None, fk_snapshot))

            change_val = row.get(change_col)
            if change_val is not None:
                candidate = str(change_val)
                if new_watermark is None or candidate > str(new_watermark):
                    new_watermark = candidate

        await self._store.upsert_pk_batch(self._source_url, schema.name, pk_batch)

        # Advance state only after the batch has committed upstream
        # (aiosqlite writes are already inside the same connection).
        row_count = (prior_state.row_count if prior_state else 0) + stats.added
        new_state = TableSyncState(
            source_url=self._source_url,
            table_name=schema.name,
            strategy="timestamp",
            change_col=change_col,
            last_sync_at=time.time(),
            last_watermark=str(new_watermark) if new_watermark is not None else None,
            primary_key_col=schema.primary_key,
            row_count=row_count,
            schema_fingerprint=_schema_fingerprint(schema),
        )
        await self._store.save_state(new_state)

        logger.info(
            "sync_table(%s): +%d ~%d strategy=%s watermark=%s",
            schema.name,
            stats.added,
            stats.updated,
            stats.strategy,
            new_watermark,
        )
        return stats

    async def _prune_stale_fk_edges(
        self,
        graph: SynapticGraph,
        schema: TableSchema,
        fk_map: dict[str, tuple[str, str]],
        rows: list[dict[str, Any]],
        prior_fks: dict[str, dict[str, str]],
    ) -> int:
        return await _prune_stale_fk_edges(graph, schema, fk_map, rows, prior_fks, self._source_url)


async def _prune_stale_fk_edges(
    graph: SynapticGraph,
    schema: TableSchema,
    fk_map: dict[str, tuple[str, str]],
    rows: list[dict[str, Any]],
    prior_fks: dict[str, dict[str, str]],
    source_url: str,
) -> int:
    """Delete FK edges whose target moved or was removed.

    Module-level so both timestamp and hash syncers can share the
    diffing logic. Caller supplies ``source_url`` since edge target
    derivation needs the same canonicalisation as
    :func:`deterministic_row_id`.
    """
    from synaptic.models import EdgeKind

    removed = 0
    edge_cache: dict[str, list[Any]] = {}

    for row in rows:
        pk_val = row.get(schema.primary_key)
        if pk_val is None:
            continue
        pk_str = str(pk_val)
        prior = prior_fks.get(pk_str)
        if not prior:
            continue

        source_node = deterministic_row_id(source_url, schema.name, pk_val)

        for col, old_target_pk in prior.items():
            target_table = fk_map[col][0] if col in fk_map else None

            new_val = row.get(col)
            if new_val is not None and str(new_val) == old_target_pk:
                continue
            if target_table is None:
                continue

            old_target_node = deterministic_row_id(source_url, target_table, old_target_pk)

            if source_node not in edge_cache:
                edge_cache[source_node] = await graph.backend.get_edges(
                    source_node, direction="outgoing"
                )
            for edge in edge_cache[source_node]:
                if edge.target_id == old_target_node and edge.kind == EdgeKind.RELATED:
                    await graph.backend.delete_edge(edge.id)
                    removed += 1
                    edge_cache[source_node] = [
                        e for e in edge_cache[source_node] if e.id != edge.id
                    ]
                    break

    return removed


class HashTableSyncer:
    """Sync one table by content-hashing every row.

    Used as a fallback for tables that lack an ``updated_at``-style
    column. Reads every live row each sync, computes
    :func:`row_hash`, and skips ingestion for rows whose hash
    matches the prior snapshot in ``syn_cdc_pk_index.row_hash``.

    Strictly more expensive than :class:`TimestampTableSyncer`
    because it must always do a full table scan, but it is the
    only correct strategy when the source schema offers no
    monotonic change marker.
    """

    __slots__ = ("_ingester", "_source_url", "_store")

    def __init__(
        self,
        ingester: TableIngester,
        store: SyncStateStore,
        source_url: str,
    ) -> None:
        self._ingester = ingester
        self._store = store
        self._source_url = source_url

    async def detect_deletes(
        self,
        graph: SynapticGraph,
        schema: TableSchema,
        pk_reader: PkReader,
    ) -> int:
        # Hash strategy reuses the timestamp syncer's delete logic
        # by routing through the same store call. Behaviour is
        # identical: full PK diff via TEMP TABLE + LEFT JOIN.
        return await TimestampTableSyncer(
            self._ingester, self._store, self._source_url
        ).detect_deletes(graph, schema, pk_reader)

    async def sync_table(
        self,
        graph: SynapticGraph,
        schema: TableSchema,
        row_reader: RowReader,
    ) -> TableSyncStats:
        """Sync one table via per-row content hashing."""
        from synaptic.extensions.cdc.hashing import row_hash

        stats = TableSyncStats(table=schema.name, strategy="hash")
        col_defs = [{"name": c.name, "type": c.type} for c in schema.columns]

        prior_state_for_drift = await self._store.load_state(self._source_url, schema.name)
        current_fp = _schema_fingerprint(schema)
        _, stats.schema_changed = await _detect_and_reset_on_schema_drift(
            self._store, self._source_url, prior_state_for_drift, current_fp
        )

        # Hash mode always does a full read — there is no watermark
        # to filter on. Pass `where_clause=None` so the SQLite
        # reader still applies the LIMIT but skips the WHERE.
        reader_result = row_reader(schema.name, where_clause=None, where_params=())
        if hasattr(reader_result, "__await__"):
            rows = await reader_result  # type: ignore[misc]
        else:
            rows = reader_result  # type: ignore[assignment]

        if not rows:
            return stats

        fk_map = {fk.column: (fk.ref_table, fk.ref_column) for fk in schema.foreign_keys}

        # Bucket rows: which need ingest, which can be skipped. We do
        # **one batch SELECT** instead of the previous 3 × N per-row
        # calls (get_row_hash + get_node_id + get_fk_edges). For a
        # 100-row table the cost drops from 300 sequential awaits to
        # one.
        pk_list: list[str] = []
        pk_to_row: dict[str, dict[str, Any]] = {}
        new_hashes: dict[str, str] = {}
        for row in rows:
            pk_val = row.get(schema.primary_key)
            if pk_val is None:
                continue
            pk_str = canonical_pk(pk_val)
            pk_list.append(pk_str)
            pk_to_row[pk_str] = row
            new_hashes[pk_str] = row_hash(row)

        prior_index = await self._store.get_pk_index_batch(self._source_url, schema.name, pk_list)

        to_ingest: list[dict[str, Any]] = []
        prior_fks: dict[str, dict[str, str]] = {}
        for pk_str, row in pk_to_row.items():
            new_hash = new_hashes[pk_str]
            existing_node, prior_hash, prior_fk_json = prior_index.get(pk_str, (None, None, None))

            if existing_node is None:
                stats.added += 1
                to_ingest.append(row)
            elif prior_hash != new_hash:
                stats.updated += 1
                to_ingest.append(row)
                if fk_map and prior_fk_json:
                    try:
                        prior_fks[pk_str] = dict(json.loads(prior_fk_json))
                    except (ValueError, TypeError):
                        pass
            # else: unchanged — skip

        if not to_ingest:
            # Nothing changed; no state advance needed beyond the
            # last_sync_at heartbeat for monitoring.
            prior_state = await self._store.load_state(self._source_url, schema.name)
            new_state = TableSyncState(
                source_url=self._source_url,
                table_name=schema.name,
                strategy="hash",
                change_col=None,
                last_sync_at=time.time(),
                last_watermark=None,
                primary_key_col=schema.primary_key,
                row_count=prior_state.row_count if prior_state else len(rows),
                schema_fingerprint=_schema_fingerprint(schema),
            )
            await self._store.save_state(new_state)
            return stats

        await self._ingester.ingest(
            graph,
            schema.name,
            col_defs,
            to_ingest,
            primary_key=schema.primary_key,
            foreign_keys=fk_map if fk_map else None,
            source_url=self._source_url,
        )

        if fk_map and prior_fks:
            stats.fk_edges_removed += await _prune_stale_fk_edges(
                graph, schema, fk_map, to_ingest, prior_fks, self._source_url
            )

        if fk_map:
            for row in to_ingest:
                pk_val = row.get(schema.primary_key)
                if pk_val is None:
                    continue
                prior = prior_fks.get(canonical_pk(pk_val), {})
                for col in fk_map:
                    new_target = row.get(col)
                    if new_target is None:
                        continue
                    if prior.get(col) != str(new_target):
                        stats.fk_edges_added += 1

        # Persist new hashes + FK snapshots into the PK index.
        pk_batch: list[tuple[str, str, str | None, dict[str, str] | None]] = []
        for row in to_ingest:
            pk_val = row.get(schema.primary_key)
            if pk_val is None:
                continue
            pk_str = canonical_pk(pk_val)
            node_id = deterministic_row_id(self._source_url, schema.name, pk_val)
            fk_snapshot: dict[str, str] | None = None
            if fk_map:
                fk_snapshot = {col: str(row[col]) for col in fk_map if row.get(col) is not None}
            pk_batch.append((pk_str, node_id, new_hashes.get(pk_str), fk_snapshot))

        await self._store.upsert_pk_batch(self._source_url, schema.name, pk_batch)

        prior_state = await self._store.load_state(self._source_url, schema.name)
        row_count = (prior_state.row_count if prior_state else 0) + stats.added
        new_state = TableSyncState(
            source_url=self._source_url,
            table_name=schema.name,
            strategy="hash",
            change_col=None,
            last_sync_at=time.time(),
            last_watermark=None,
            primary_key_col=schema.primary_key,
            row_count=row_count,
            schema_fingerprint=_schema_fingerprint(schema),
        )
        await self._store.save_state(new_state)

        logger.info(
            "sync_table_hash(%s): +%d ~%d (scanned %d)",
            schema.name,
            stats.added,
            stats.updated,
            len(rows),
        )
        return stats


def _schema_fingerprint(schema: TableSchema) -> str:
    """Cheap fingerprint used to detect drift between sync runs.

    Not a cryptographic hash — just a deterministic string of the
    column names and types. If the shape changes we invalidate the
    watermark and force a full reload for that table.
    """
    parts = [f"{c.name}:{c.type}" for c in schema.columns]
    parts.sort()
    return "|".join(parts)


async def _detect_and_reset_on_schema_drift(
    store: SyncStateStore,
    source_url: str,
    prior_state: TableSyncState | None,
    current_fingerprint: str,
) -> tuple[TableSyncState | None, bool]:
    """Compare prior vs current fingerprint; on mismatch, wipe state.

    Returns ``(effective_prior_state, schema_changed)``. When drift is
    detected, the state row and every PK-index entry for the table are
    deleted so the downstream code sees an initial load and re-ingests
    every row under the new schema.

    Legacy rows pre-v0.14.1 carried an empty fingerprint; we treat
    that as "unknown, don't force a reload" so upgrading Synaptic
    without changing the source schema is a no-op.
    """
    if prior_state is None:
        return None, False
    prior_fp = prior_state.schema_fingerprint or ""
    if not prior_fp:
        # Legacy / missing fingerprint — skip comparison, do NOT force
        # a reload. Next sync will write the new fingerprint.
        return prior_state, False
    if prior_fp == current_fingerprint:
        return prior_state, False

    logger.warning(
        "schema drift detected on %s: fingerprint changed, forcing full "
        "reload (prior=%r current=%r)",
        prior_state.table_name,
        prior_fp,
        current_fingerprint,
    )
    # Wipe the PK index (so stale hashes / FK snapshots don't bleed
    # into the new load) and the state row (so the downstream code
    # takes the `is_initial = prior_state is None` branch).
    prior_pks = await store.list_pks(source_url, prior_state.table_name)
    if prior_pks:
        await store.delete_pk_batch(source_url, prior_state.table_name, prior_pks)
    await store.delete_state(source_url, prior_state.table_name)
    return None, True
