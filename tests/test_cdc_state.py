"""Tests for SyncStateStore — the persistence layer for CDC bookkeeping.

These tests use the real ``SQLiteBackend`` so the schema, transactions,
and ``ON CONFLICT`` semantics are exercised end-to-end. The aim is to
verify the contract that downstream sync logic (Phase 2+) will rely on.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from synaptic.backends.sqlite import SQLiteBackend
from synaptic.extensions.cdc.state import SyncStateStore, TableSyncState

aiosqlite = pytest.importorskip("aiosqlite")


@pytest.fixture
async def backend():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "cdc.db")
        b = SQLiteBackend(path)
        await b.connect()
        await b.ensure_cdc_tables()
        yield b
        await b.close()


@pytest.fixture
async def store(backend):
    return backend.cdc_state_store()


class TestSyncStateCRUD:
    async def test_load_state_returns_none_when_missing(self, store):
        assert await store.load_state("postgres://h/d", "products") is None

    async def test_save_then_load_roundtrip(self, store: SyncStateStore):
        state = TableSyncState(
            source_url="postgres://h/d",
            table_name="products",
            strategy="timestamp",
            change_col="updated_at",
            last_sync_at=1700000000.0,
            last_watermark="2024-01-01T00:00:00",
            primary_key_col="id",
            row_count=42,
            schema_fingerprint="abc123",
        )
        await store.save_state(state)
        loaded = await store.load_state("postgres://h/d", "products")
        assert loaded is not None
        assert loaded.strategy == "timestamp"
        assert loaded.change_col == "updated_at"
        assert loaded.last_watermark == "2024-01-01T00:00:00"
        assert loaded.row_count == 42
        assert loaded.schema_fingerprint == "abc123"

    async def test_save_state_upserts(self, store: SyncStateStore):
        state = TableSyncState(
            source_url="postgres://h/d",
            table_name="products",
            strategy="timestamp",
            change_col="updated_at",
            last_sync_at=1700000000.0,
            last_watermark="t1",
            primary_key_col="id",
            row_count=10,
            schema_fingerprint="f1",
        )
        await store.save_state(state)

        state.last_watermark = "t2"
        state.row_count = 20
        await store.save_state(state)

        loaded = await store.load_state("postgres://h/d", "products")
        assert loaded is not None
        assert loaded.last_watermark == "t2"
        assert loaded.row_count == 20

    async def test_delete_state(self, store: SyncStateStore):
        state = TableSyncState(
            source_url="postgres://h/d",
            table_name="products",
            strategy="timestamp",
            change_col="updated_at",
            last_sync_at=1700000000.0,
            last_watermark="t1",
            primary_key_col="id",
            row_count=10,
            schema_fingerprint="f1",
        )
        await store.save_state(state)
        await store.delete_state("postgres://h/d", "products")
        assert await store.load_state("postgres://h/d", "products") is None


class TestPKIndexCRUD:
    async def test_upsert_and_get_node_id(self, store: SyncStateStore):
        await store.upsert_pk("postgres://h/d", "products", "P001", "node-abc")
        assert await store.get_node_id("postgres://h/d", "products", "P001") == "node-abc"

    async def test_upsert_replaces(self, store: SyncStateStore):
        await store.upsert_pk("postgres://h/d", "products", "P001", "node-abc")
        await store.upsert_pk("postgres://h/d", "products", "P001", "node-xyz")
        assert await store.get_node_id("postgres://h/d", "products", "P001") == "node-xyz"

    async def test_upsert_pk_batch(self, store: SyncStateStore):
        rows = [
            ("P001", "n001", "h001", {"category_id": "C1"}),
            ("P002", "n002", "h002", {"category_id": "C2"}),
            ("P003", "n003", None, None),
        ]
        await store.upsert_pk_batch("postgres://h/d", "products", rows)
        assert await store.get_node_id("postgres://h/d", "products", "P001") == "n001"
        assert await store.get_node_id("postgres://h/d", "products", "P002") == "n002"
        assert await store.get_node_id("postgres://h/d", "products", "P003") == "n003"

    async def test_get_fk_edges(self, store: SyncStateStore):
        await store.upsert_pk(
            "postgres://h/d",
            "products",
            "P001",
            "n001",
            fk_edges={"category_id": "C1", "vendor_id": "V42"},
        )
        fk = await store.get_fk_edges("postgres://h/d", "products", "P001")
        assert fk == {"category_id": "C1", "vendor_id": "V42"}

    async def test_get_row_hash(self, store: SyncStateStore):
        await store.upsert_pk(
            "postgres://h/d",
            "products",
            "P001",
            "n001",
            row_hash="deadbeef",
        )
        assert await store.get_row_hash("postgres://h/d", "products", "P001") == "deadbeef"

    async def test_delete_pk(self, store: SyncStateStore):
        await store.upsert_pk("postgres://h/d", "products", "P001", "n001")
        await store.delete_pk("postgres://h/d", "products", "P001")
        assert await store.get_node_id("postgres://h/d", "products", "P001") is None

    async def test_delete_pk_batch(self, store: SyncStateStore):
        for i in range(5):
            await store.upsert_pk("postgres://h/d", "products", f"P{i}", f"n{i}")
        deleted = await store.delete_pk_batch("postgres://h/d", "products", ["P1", "P2", "P3"])
        assert deleted == 3
        assert await store.count_pks("postgres://h/d", "products") == 2

    async def test_list_and_count_pks(self, store: SyncStateStore):
        for i in range(7):
            await store.upsert_pk("postgres://h/d", "products", f"P{i}", f"n{i}")
        for i in range(3):
            await store.upsert_pk("postgres://h/d", "reviews", f"R{i}", f"r{i}")

        product_pks = await store.list_pks("postgres://h/d", "products")
        review_pks = await store.list_pks("postgres://h/d", "reviews")
        assert len(product_pks) == 7
        assert len(review_pks) == 3
        assert await store.count_pks("postgres://h/d", "products") == 7

    async def test_isolation_between_source_urls(self, store: SyncStateStore):
        # Two different DBs with a same table+pk shouldn't collide
        await store.upsert_pk("postgres://h/db1", "products", "P001", "node-A")
        await store.upsert_pk("postgres://h/db2", "products", "P001", "node-B")
        assert await store.get_node_id("postgres://h/db1", "products", "P001") == "node-A"
        assert await store.get_node_id("postgres://h/db2", "products", "P001") == "node-B"


class TestSchemaInstall:
    async def test_install_schema_idempotent(self, backend):
        # Calling ensure_cdc_tables twice should not raise
        await backend.ensure_cdc_tables()
        await backend.ensure_cdc_tables()
        # Verify tables exist
        async with backend._db().execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'syn_cdc_%'"
        ) as cur:
            tables = {row[0] for row in await cur.fetchall()}
        assert tables == {"syn_cdc_state", "syn_cdc_pk_index"}
