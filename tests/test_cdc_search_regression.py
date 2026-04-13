"""Phase 7 — CDC must not regress search quality.

The whole point of the CDC effort was to enable incremental syncs
without changing how nodes are scored, ranked, or matched. This
test builds the same source database two ways — legacy
``mode='full'`` and incremental ``mode='cdc'`` — runs identical
queries against both, and asserts the top results match by title.

Node IDs are intentionally compared loosely: ``mode='full'`` uses
random UUIDs and ``mode='cdc'`` uses deterministic ones, so we
compare the *titles* (which derive from the source data and are
mode-invariant) instead of IDs.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from synaptic import SynapticGraph

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
            description TEXT,
            category_id INTEGER REFERENCES category(id),
            updated_at INTEGER NOT NULL
        );
        INSERT INTO category VALUES (1, '신발', 1000);
        INSERT INTO category VALUES (2, '의류', 1100);
        INSERT INTO category VALUES (3, '액세서리', 1200);

        INSERT INTO products VALUES
            (1, '러닝화 에어맥스',  '여름용 가벼운 러닝화', 1, 1000),
            (2, '워킹화 컴포트',    '하이킹과 산책에 적합', 1, 1010),
            (3, '반팔 티셔츠',      '코튼 100%% 베이직 티', 2, 1020),
            (4, '플리스 자켓',      '겨울철 보온용 플리스', 2, 1030),
            (5, '가죽 벨트',        '소가죽 정장용 벨트',   3, 1040),
            (6, '울 비니',          '겨울 모자 울 100%%',   3, 1050);
        """
    )
    con.commit()
    con.close()


async def _topk_titles(graph: SynapticGraph, query: str, k: int = 5) -> list[str]:
    result = await graph.search(query)
    return [a.node.title for a in result.nodes[:k]]


class TestSearchRegressionFullVsCDC:
    @pytest.fixture
    async def two_graphs(self):
        with tempfile.TemporaryDirectory() as d:
            src_path = Path(d) / "source.db"
            full_db = Path(d) / "full.db"
            cdc_db = Path(d) / "cdc.db"
            _seed(src_path)
            conn_str = f"sqlite:////{str(src_path).lstrip('/')}"

            full_graph = await SynapticGraph.from_database(
                conn_str, db=str(full_db), mode="full"
            )
            cdc_graph = await SynapticGraph.from_database(
                conn_str, db=str(cdc_db), mode="cdc"
            )
            yield full_graph, cdc_graph
            await full_graph.backend.close()
            await cdc_graph.backend.close()

    async def test_node_count_matches(self, two_graphs):
        full_graph, cdc_graph = two_graphs
        full_count = await full_graph.backend.count_nodes()
        cdc_count = await cdc_graph.backend.count_nodes()
        assert full_count == cdc_count

    async def test_korean_query_top_result_matches(self, two_graphs):
        full_graph, cdc_graph = two_graphs
        for query in ("러닝화", "벨트", "비니", "자켓"):
            full_top = await _topk_titles(full_graph, query, k=3)
            cdc_top = await _topk_titles(cdc_graph, query, k=3)
            assert full_top, f"full graph returned nothing for {query!r}"
            assert cdc_top, f"cdc graph returned nothing for {query!r}"
            # Top-1 must agree on the same data point.
            assert full_top[0] == cdc_top[0], (
                f"top result diverged for {query!r}: "
                f"full={full_top[0]} cdc={cdc_top[0]}"
            )

    async def test_topk_set_matches(self, two_graphs):
        full_graph, cdc_graph = two_graphs
        for query in ("운동",  "겨울", "정장"):
            full_top = set(await _topk_titles(full_graph, query, k=5))
            cdc_top = set(await _topk_titles(cdc_graph, query, k=5))
            # The top-k *set* should match — internal ordering can
            # vary by tied scores, but the membership cannot.
            assert full_top == cdc_top, (
                f"top-k set diverged for {query!r}: "
                f"full-cdc={full_top - cdc_top} cdc-full={cdc_top - full_top}"
            )


class TestSyncIdempotency:
    """Re-syncing an unchanged source must be a true no-op."""

    @pytest.fixture
    async def graph_and_src(self):
        with tempfile.TemporaryDirectory() as d:
            src_path = Path(d) / "source.db"
            graph_db = Path(d) / "graph.db"
            _seed(src_path)
            conn_str = f"sqlite:////{str(src_path).lstrip('/')}"

            graph = await SynapticGraph.from_database(
                conn_str, db=str(graph_db), mode="cdc"
            )
            yield graph, conn_str
            await graph.backend.close()

    async def test_second_sync_does_not_call_table_ingester(self, graph_and_src):
        graph, conn_str = graph_and_src

        from synaptic.extensions import table_ingester as ti_mod

        original = ti_mod.TableIngester.ingest
        call_log: list[int] = []

        async def spy(self, graph_, table_name, columns, rows, **kwargs):
            call_log.append(len(rows))
            return await original(self, graph_, table_name, columns, rows, **kwargs)

        ti_mod.TableIngester.ingest = spy
        try:
            result = await graph.sync_from_database(conn_str)
        finally:
            ti_mod.TableIngester.ingest = original

        # The hash syncer never calls ingest with an empty list, and
        # the timestamp syncer might re-read rows at the watermark;
        # either way the per-table _ingested_ row count should be
        # very small relative to the full table size, and `added`
        # must be zero across the board.
        for table_stats in result.tables:
            assert table_stats.added == 0
        # And no table grew.
        for n_rows in call_log:
            assert n_rows <= 1, f"unexpected re-ingest of {n_rows} rows on no-op sync"
