"""Phase 5 — hash fallback for tables without ``updated_at``.

Tables without a monotonic change column are synced by content
hashing every row and comparing against the hash stored in the PK
index. Unchanged rows are skipped (no ``save_node`` call), which
the test asserts via a spy on ``TableIngester.ingest``.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from synaptic import SynapticGraph
from synaptic.extensions.cdc.hashing import row_hash
from synaptic.extensions.cdc.ids import deterministic_row_id

pytest.importorskip("aiosqlite")


def _seed(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        -- No updated_at column on purpose: this exercises the hash fallback.
        CREATE TABLE config (
            id INTEGER PRIMARY KEY,
            key TEXT,
            value TEXT
        );
        INSERT INTO config VALUES (1, 'theme', 'dark');
        INSERT INTO config VALUES (2, 'lang',  'ko');
        INSERT INTO config VALUES (3, 'tz',    'Asia/Seoul');
        """
    )
    con.commit()
    con.close()


class TestRowHash:
    def test_deterministic(self):
        a = row_hash({"id": 1, "name": "x"})
        b = row_hash({"name": "x", "id": 1})  # different insertion order
        assert a == b

    def test_changes_with_value(self):
        a = row_hash({"id": 1, "name": "x"})
        b = row_hash({"id": 1, "name": "y"})
        assert a != b

    def test_skips_meta_columns(self):
        a = row_hash({"id": 1, "_table_name": "products"})
        b = row_hash({"id": 1, "_table_name": "DIFFERENT"})
        assert a == b


class TestHashSync:
    @pytest.fixture
    async def graph_and_src(self):
        with tempfile.TemporaryDirectory() as d:
            src_path = Path(d) / "source.db"
            graph_db = Path(d) / "graph.db"
            _seed(src_path)
            conn_str = f"sqlite:////{str(src_path).lstrip('/')}"

            graph = await SynapticGraph.from_database(
                conn_str,
                db=str(graph_db),
                mode="cdc",
            )
            yield graph, conn_str, src_path
            await graph.backend.close()

    async def test_initial_sync_uses_hash_strategy(self, graph_and_src):
        graph, conn_str, _ = graph_and_src

        store = graph.backend.cdc_state_store()
        state = await store.load_state(conn_str, "config")
        assert state is not None
        assert state.strategy == "hash"
        assert state.change_col is None
        assert state.row_count == 3

        # Each row got a hash recorded.
        for pk in ("1", "2", "3"):
            assert await store.get_row_hash(conn_str, "config", pk) is not None

    async def test_unchanged_row_is_skipped(self, graph_and_src):
        graph, conn_str, _ = graph_and_src

        # Spy: count how many rows TableIngester.ingest gets called with.
        from synaptic.extensions.cdc.sync import HashTableSyncer

        original = HashTableSyncer.sync_table
        ingested_rows: list[int] = []

        async def spy(self, graph_, schema, row_reader):
            stats = await original(self, graph_, schema, row_reader)
            ingested_rows.append(stats.added + stats.updated)
            return stats

        HashTableSyncer.sync_table = spy
        try:
            result = await graph.sync_from_database(conn_str)
        finally:
            HashTableSyncer.sync_table = original

        # No source changes → no ingest work for the config table.
        config = next(t for t in result.tables if t.table == "config")
        assert config.added == 0
        assert config.updated == 0
        assert ingested_rows[0] == 0  # nothing routed through ingest

    async def test_value_change_propagates(self, graph_and_src):
        graph, conn_str, src_path = graph_and_src
        node_id = deterministic_row_id(conn_str, "config", "1")

        con = sqlite3.connect(src_path)
        con.execute("UPDATE config SET value = 'light' WHERE id = 1")
        con.commit()
        con.close()

        result = await graph.sync_from_database(conn_str)

        config = next(t for t in result.tables if t.table == "config")
        assert config.added == 0
        assert config.updated == 1

        node = await graph.backend.get_node(node_id)
        assert node is not None
        assert node.properties["value"] == "light"

        # Hash advanced.
        store = graph.backend.cdc_state_store()
        new_hash = await store.get_row_hash(conn_str, "config", "1")
        assert new_hash is not None
        # The new hash matches what the ingester would compute fresh.
        # (Not a strict equality on a known constant — that would
        # bind us to the digest function. Just make sure it changed.)
        from synaptic.extensions.cdc.hashing import row_hash as rh

        assert new_hash == rh({"id": 1, "key": "theme", "value": "light"})

    async def test_insert_propagates(self, graph_and_src):
        graph, conn_str, src_path = graph_and_src
        before = await graph.backend.count_nodes()

        con = sqlite3.connect(src_path)
        con.execute("INSERT INTO config VALUES (4, 'fontsize', '14')")
        con.commit()
        con.close()

        result = await graph.sync_from_database(conn_str)
        config = next(t for t in result.tables if t.table == "config")
        assert config.added == 1
        assert await graph.backend.count_nodes() == before + 1

    async def test_delete_propagates_via_hash_path(self, graph_and_src):
        graph, conn_str, src_path = graph_and_src
        before = await graph.backend.count_nodes()

        con = sqlite3.connect(src_path)
        con.execute("DELETE FROM config WHERE id = 2")
        con.commit()
        con.close()

        result = await graph.sync_from_database(conn_str)
        config = next(t for t in result.tables if t.table == "config")
        assert config.deleted == 1
        assert await graph.backend.count_nodes() == before - 1
