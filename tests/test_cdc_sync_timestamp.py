"""End-to-end tests for timestamp-based CDC sync.

Spins up a real SQLite source database, points a SynapticGraph at
it with ``mode="cdc"``, and exercises the full cycle:

- initial load records watermark + PK index
- a second call with no changes is a no-op
- inserting a new row advances the watermark and bumps ``added``
- updating an existing row bumps ``updated`` and keeps the node count stable
- node IDs are stable across syncs
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from synaptic import SynapticGraph
from synaptic.extensions.cdc.ids import deterministic_row_id
from synaptic.extensions.cdc.sync import detect_change_column

pytest.importorskip("aiosqlite")


def _seed_source_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price INTEGER,
            updated_at INTEGER NOT NULL
        );
        INSERT INTO products VALUES (1, '운동화A', 89000, 1000);
        INSERT INTO products VALUES (2, '티셔츠B', 35000, 1100);
        INSERT INTO products VALUES (3, '모자C',   15000, 1200);
        """
    )
    con.commit()
    con.close()


class TestDetectChangeColumn:
    def test_prefers_updated_at(self):
        cols = [
            {"name": "id", "type": "int"},
            {"name": "created_at", "type": "date"},
            {"name": "updated_at", "type": "date"},
        ]
        assert detect_change_column(cols) == "updated_at"

    def test_returns_none_when_absent(self):
        cols = [{"name": "id", "type": "int"}, {"name": "name", "type": "str"}]
        assert detect_change_column(cols) is None

    def test_case_insensitive(self):
        cols = [{"name": "id", "type": "int"}, {"name": "UpdatedAt", "type": "date"}]
        assert detect_change_column(cols) == "UpdatedAt"


class TestTimestampSync:
    @pytest.fixture
    async def graph_and_src(self):
        with tempfile.TemporaryDirectory() as d:
            src_path = Path(d) / "source.db"
            graph_db = Path(d) / "graph.db"
            _seed_source_db(src_path)
            # SQLAlchemy-style URL: sqlite:///relative or sqlite:////absolute
            conn_str = f"sqlite:////{str(src_path).lstrip('/')}"

            graph = await SynapticGraph.from_database(
                conn_str,
                db=str(graph_db),
                mode="cdc",
            )
            yield graph, conn_str, src_path
            await graph.backend.close()

    async def test_initial_load_seeds_state_and_pk_index(self, graph_and_src):
        graph, conn_str, _ = graph_and_src

        # 3 products in source → 3 deterministic nodes in graph.
        assert await graph.backend.count_nodes() >= 3

        store = graph.backend.cdc_state_store()
        state = await store.load_state(conn_str, "products")
        assert state is not None
        assert state.strategy == "timestamp"
        assert state.change_col == "updated_at"
        assert state.last_watermark == "1200"
        assert state.row_count == 3

        # PK index knows about all three rows.
        assert await store.count_pks(conn_str, "products") == 3
        assert await store.get_node_id(conn_str, "products", "1") == deterministic_row_id(
            conn_str, "products", "1"
        )

    async def test_second_sync_with_no_changes_is_noop(self, graph_and_src):
        graph, conn_str, _ = graph_and_src
        count_before = await graph.backend.count_nodes()

        result = await graph.sync_from_database(conn_str)

        # Watermark-inclusive reads will surface rows whose change
        # column exactly equals the prior watermark — they re-upsert
        # via the same deterministic IDs so node count is unchanged.
        assert await graph.backend.count_nodes() == count_before
        # No rows advance the watermark (no changes after last sync),
        # so 'added' must stay at zero.
        products_stats = next(t for t in result.tables if t.table == "products")
        assert products_stats.added == 0

    async def test_insert_propagates(self, graph_and_src):
        graph, conn_str, src_path = graph_and_src
        count_before = await graph.backend.count_nodes()

        con = sqlite3.connect(src_path)
        con.execute("INSERT INTO products VALUES (4, '지갑D', 55000, 2000)")
        con.commit()
        con.close()

        result = await graph.sync_from_database(conn_str)

        assert await graph.backend.count_nodes() == count_before + 1
        products = next(t for t in result.tables if t.table == "products")
        assert products.added == 1

        store = graph.backend.cdc_state_store()
        state = await store.load_state(conn_str, "products")
        assert state.last_watermark == "2000"

    async def test_update_propagates_as_upsert(self, graph_and_src):
        graph, conn_str, src_path = graph_and_src
        count_before = await graph.backend.count_nodes()
        pk_node_id = deterministic_row_id(conn_str, "products", "1")

        con = sqlite3.connect(src_path)
        con.execute(
            "UPDATE products SET name = ?, updated_at = 3000 WHERE id = 1",
            ("운동화A-리뉴얼",),
        )
        con.commit()
        con.close()

        result = await graph.sync_from_database(conn_str)

        # No new node — same deterministic ID upserted.
        assert await graph.backend.count_nodes() == count_before
        products = next(t for t in result.tables if t.table == "products")
        # `>=` watermark inclusivity means the row at the old
        # watermark is always re-read alongside the actually-changed
        # row; the useful invariants are no new nodes, and the
        # updated content is visible.
        assert products.added == 0
        assert products.updated >= 1

        updated_node = await graph.backend.get_node(pk_node_id)
        assert updated_node is not None
        assert "리뉴얼" in updated_node.properties.get("name", "")

        # Watermark advanced to the new maximum.
        store = graph.backend.cdc_state_store()
        state = await store.load_state(conn_str, "products")
        assert state.last_watermark == "3000"

    async def test_schema_drift_triggers_full_reload(self, graph_and_src):
        """ALTER TABLE ADD COLUMN on source → detected on next sync.

        Without this guard, the new column is silently dropped from
        the ingested properties because the row_hash / watermark state
        makes every row look "already synced" under the old shape.
        """
        graph, conn_str, src_path = graph_and_src
        count_before = await graph.backend.count_nodes()

        # Source ALTER TABLE: add a new column.
        con = sqlite3.connect(src_path)
        con.execute("ALTER TABLE products ADD COLUMN description TEXT DEFAULT ''")
        con.execute(
            "UPDATE products SET description = ?, updated_at = 3000 WHERE id = 1",
            ("리뉴얼된 설명",),
        )
        con.commit()
        con.close()

        result = await graph.sync_from_database(conn_str)

        products = next(t for t in result.tables if t.table == "products")
        assert products.schema_changed is True, "drift flag must propagate to SyncResult"

        # After a forced reload, every existing row is re-ingested as
        # "added" (the PK index was wiped) — node count still stable
        # because deterministic IDs upsert over the old nodes.
        assert await graph.backend.count_nodes() == count_before
        assert products.added == 3  # all three re-ingested

        # Fingerprint in state now reflects the new 5-column schema.
        store = graph.backend.cdc_state_store()
        state = await store.load_state(conn_str, "products")
        assert state is not None
        assert "description:" in state.schema_fingerprint

        # The new column is actually present on the node properties.
        pk_node_id = deterministic_row_id(conn_str, "products", "1")
        reloaded = await graph.backend.get_node(pk_node_id)
        assert reloaded is not None
        assert reloaded.properties.get("description") == "리뉴얼된 설명"

    async def test_unchanged_schema_does_not_trigger_reload(self, graph_and_src):
        """Sanity: flag is False and state is preserved when the source
        schema hasn't moved between sync runs."""
        graph, conn_str, _ = graph_and_src

        result = await graph.sync_from_database(conn_str)
        products = next(t for t in result.tables if t.table == "products")
        assert products.schema_changed is False

    async def test_legacy_empty_fingerprint_does_not_force_reload(self, graph_and_src):
        """Upgrading Synaptic on an existing graph should not trigger a
        spurious reload for tables whose stored fingerprint is empty
        (written by a pre-fingerprint version of the syncer)."""
        graph, conn_str, _ = graph_and_src
        store = graph.backend.cdc_state_store()
        state = await store.load_state(conn_str, "products")
        assert state is not None

        # Simulate a legacy state row by wiping the fingerprint.
        state.schema_fingerprint = ""
        await store.save_state(state)

        result = await graph.sync_from_database(conn_str)
        products = next(t for t in result.tables if t.table == "products")
        assert products.schema_changed is False

    async def test_auto_mode_reuses_state(self, graph_and_src):
        graph, conn_str, _src_path = graph_and_src
        db_path = graph.backend._path

        # Close the first graph and reopen with mode="auto".
        await graph.backend.close()
        reopened = await SynapticGraph.from_database(
            conn_str,
            db=db_path,
            mode="auto",
        )

        store = reopened.backend.cdc_state_store()
        state = await store.load_state(conn_str, "products")
        assert state is not None  # prior state survived close / reopen
        assert state.strategy == "timestamp"

        await reopened.backend.close()
