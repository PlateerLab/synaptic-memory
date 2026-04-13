"""Phase 3 — delete detection.

Verifies that rows removed from the source database also disappear
from the graph on the next sync, including their FK edges (cascade).
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from synaptic import SynapticGraph
from synaptic.extensions.cdc.ids import deterministic_row_id

pytest.importorskip("aiosqlite")


def _seed(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE category (
            id INTEGER PRIMARY KEY,
            name TEXT,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category_id INTEGER REFERENCES category(id),
            updated_at INTEGER NOT NULL
        );
        INSERT INTO category VALUES (1, '신발', 1000);
        INSERT INTO category VALUES (2, '의류', 1100);
        INSERT INTO products VALUES (1, '운동화A',   1, 1000);
        INSERT INTO products VALUES (2, '티셔츠B',   2, 1100);
        INSERT INTO products VALUES (3, '모자C',     2, 1200);
        """
    )
    con.commit()
    con.close()


class TestDeleteDetection:
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

    async def test_delete_row_removes_node_and_edges(self, graph_and_src):
        graph, conn_str, src_path = graph_and_src
        deleted_pk_node = deterministic_row_id(conn_str, "products", "1")

        # Sanity: the node exists with at least one outgoing edge.
        assert await graph.backend.get_node(deleted_pk_node) is not None
        edges_before = await graph.backend.get_edges(deleted_pk_node, direction="outgoing")
        assert len(edges_before) >= 1

        # Drop product 1 from the source.
        con = sqlite3.connect(src_path)
        con.execute("DELETE FROM products WHERE id = 1")
        con.commit()
        con.close()

        result = await graph.sync_from_database(conn_str)

        products = next(t for t in result.tables if t.table == "products")
        assert products.deleted == 1
        assert result.deleted == 1

        # Node and its outbound edges are gone.
        assert await graph.backend.get_node(deleted_pk_node) is None
        edges_after = await graph.backend.get_edges(deleted_pk_node, direction="outgoing")
        assert edges_after == []

        # PK index forgot the row.
        store = graph.backend.cdc_state_store()
        assert (
            await store.get_node_id(conn_str, "products", "1")
        ) is None

    async def test_other_rows_untouched(self, graph_and_src):
        graph, conn_str, src_path = graph_and_src

        con = sqlite3.connect(src_path)
        con.execute("DELETE FROM products WHERE id = 1")
        con.commit()
        con.close()

        await graph.sync_from_database(conn_str)

        kept_node_2 = deterministic_row_id(conn_str, "products", "2")
        kept_node_3 = deterministic_row_id(conn_str, "products", "3")
        assert await graph.backend.get_node(kept_node_2) is not None
        assert await graph.backend.get_node(kept_node_3) is not None

    async def test_no_deletes_when_source_unchanged(self, graph_and_src):
        graph, conn_str, _src_path = graph_and_src

        result = await graph.sync_from_database(conn_str)
        assert result.deleted == 0
        for table_stats in result.tables:
            assert table_stats.deleted == 0

    async def test_first_sync_skips_delete_detection(self, graph_and_src):
        # The fixture already ran the initial load via mode="cdc".
        # If delete detection had been incorrectly run on the first
        # pass it would have wiped everything (the PK index was empty
        # before the load). Verify that didn't happen.
        graph, conn_str, _ = graph_and_src
        store = graph.backend.cdc_state_store()
        assert await store.count_pks(conn_str, "products") == 3
        assert await store.count_pks(conn_str, "category") == 2
