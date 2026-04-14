"""Phase 4 — FK edge re-computation.

When a row's FK value changes between syncs, the old RELATED edge
must be torn down and a new one created in its place. The PK index
keeps a JSON snapshot of each row's FK values so the diff can be
computed without re-reading the source database for the prior state.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from synaptic import EdgeKind, SynapticGraph
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
        INSERT INTO category VALUES (3, '액세서리', 1200);
        INSERT INTO products VALUES (1, '운동화A',   1, 1000);
        INSERT INTO products VALUES (2, '티셔츠B',   2, 1100);
        """
    )
    con.commit()
    con.close()


async def _outgoing_targets(graph, source_id: str) -> set[str]:
    edges = await graph.backend.get_edges(source_id, direction="outgoing")
    return {e.target_id for e in edges if e.kind == EdgeKind.RELATED}


class TestFKReWiring:
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

    async def test_initial_fk_snapshot_recorded(self, graph_and_src):
        graph, conn_str, _ = graph_and_src
        store = graph.backend.cdc_state_store()

        snap = await store.get_fk_edges(conn_str, "products", "1")
        assert snap == {"category_id": "1"}

    async def test_fk_repoint_removes_old_edge_and_creates_new(self, graph_and_src):
        graph, conn_str, src_path = graph_and_src

        product_node = deterministic_row_id(conn_str, "products", "1")
        old_cat_node = deterministic_row_id(conn_str, "category", "1")
        new_cat_node = deterministic_row_id(conn_str, "category", "3")

        # Sanity: initial edge points at category 1.
        before = await _outgoing_targets(graph, product_node)
        assert old_cat_node in before
        assert new_cat_node not in before

        # Repoint product 1 from category 1 → category 3.
        con = sqlite3.connect(src_path)
        con.execute("UPDATE products SET category_id = 3, updated_at = 5000 WHERE id = 1")
        con.commit()
        con.close()

        result = await graph.sync_from_database(conn_str)

        after = await _outgoing_targets(graph, product_node)
        assert new_cat_node in after
        assert old_cat_node not in after

        products = next(t for t in result.tables if t.table == "products")
        assert products.fk_edges_removed >= 1
        assert products.fk_edges_added >= 1

        # FK snapshot now reflects the new value.
        store = graph.backend.cdc_state_store()
        snap = await store.get_fk_edges(conn_str, "products", "1")
        assert snap == {"category_id": "3"}

    async def test_unchanged_fk_does_not_churn_edges(self, graph_and_src):
        graph, conn_str, src_path = graph_and_src
        product_node = deterministic_row_id(conn_str, "products", "1")
        before = await _outgoing_targets(graph, product_node)

        # Touch the row but leave category_id alone.
        con = sqlite3.connect(src_path)
        con.execute(
            "UPDATE products SET name = ?, updated_at = 5000 WHERE id = 1",
            ("운동화A-리뉴얼",),
        )
        con.commit()
        con.close()

        result = await graph.sync_from_database(conn_str)
        after = await _outgoing_targets(graph, product_node)
        assert after == before

        products = next(t for t in result.tables if t.table == "products")
        assert products.fk_edges_removed == 0
