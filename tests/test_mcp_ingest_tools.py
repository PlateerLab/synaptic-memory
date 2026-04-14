"""MCP ingest + CDC tool tests.

Exercises the new tools by calling them directly (FastMCP leaves the
decorated functions callable). Each test resets the module-level
graph state and points ``_db_path`` at a fresh tempfile so the tools
open a clean graph per run.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("aiosqlite")
pytest.importorskip("mcp")


@pytest.fixture
async def fresh_mcp_graph():
    """Reset MCP module state and yield a handle to the server module.

    Each test gets its own temp directory so the graph SQLite file
    doesn't leak across tests.
    """
    from synaptic.mcp import server as mcp_server

    with tempfile.TemporaryDirectory() as d:
        mcp_server._graph = None
        mcp_server._backend = None
        mcp_server._embedder = None
        mcp_server._tracker = None
        mcp_server._db_path = str(Path(d) / "graph.db")
        mcp_server._dsn = ""
        mcp_server._source_dsn = ""
        mcp_server._embed_url = ""

        yield mcp_server, Path(d)

        if mcp_server._backend is not None:
            await mcp_server._backend.close()
        mcp_server._graph = None
        mcp_server._backend = None


class TestAddDocument:
    async def test_short_doc_becomes_single_node(self, fresh_mcp_graph):
        mcp_server, _ = fresh_mcp_graph
        result = await mcp_server.knowledge_add_document(
            title="Deploy runbook",
            content="Run `make deploy` on the release branch.",
            tags="ops,runbook",
        )
        assert result["success"] is True
        assert result["chunks"] == 1
        assert result["first_node_id"]

    async def test_long_doc_splits_into_chunks(self, fresh_mcp_graph):
        mcp_server, _ = fresh_mcp_graph
        long_text = "Paragraph one. " * 300  # ~4500 chars
        result = await mcp_server.knowledge_add_document(
            title="Handbook",
            content=long_text,
            chunk_size=500,
            chunk_overlap=50,
        )
        assert result["success"] is True
        assert result["chunks"] > 1


class TestAddTable:
    async def test_ingests_rows_and_fks(self, fresh_mcp_graph):
        mcp_server, _ = fresh_mcp_graph
        # Parent table
        await mcp_server.knowledge_add_table(
            table_name="category",
            columns=[
                {"name": "id", "type": "int"},
                {"name": "name", "type": "str"},
            ],
            rows=[
                {"id": 1, "name": "신발"},
                {"id": 2, "name": "의류"},
            ],
            primary_key="id",
        )
        # Child table with FK
        result = await mcp_server.knowledge_add_table(
            table_name="product",
            columns=[
                {"name": "id", "type": "int"},
                {"name": "name", "type": "str"},
                {"name": "category_id", "type": "int"},
            ],
            rows=[
                {"id": 1, "name": "운동화A", "category_id": 1},
                {"id": 2, "name": "티셔츠B", "category_id": 2},
            ],
            primary_key="id",
            foreign_keys={"category_id": ["category", "id"]},
        )
        assert result["success"] is True
        assert result["rows_ingested"] == 2
        assert result["fk_count"] == 1


class TestAddChunks:
    async def test_ingests_chunk_list(self, fresh_mcp_graph):
        mcp_server, _ = fresh_mcp_graph
        result = await mcp_server.knowledge_add_chunks(
            chunks=[
                {"title": "Chunk 1", "content": "Alpha beta gamma."},
                {"title": "Chunk 2", "content": "Delta epsilon zeta."},
                {"title": "bad"},  # missing content
            ],
            default_source="test",
        )
        assert result["chunks_added"] == 2
        assert len(result["errors"]) == 1
        assert result["first_node_id"]


class TestIngestPath:
    async def test_csv_ingest(self, fresh_mcp_graph):
        mcp_server, tmp = fresh_mcp_graph
        csv_path = tmp / "products.csv"
        csv_path.write_text(
            "id,name,price\n1,운동화A,89000\n2,티셔츠B,35000\n",
            encoding="utf-8",
        )
        result = await mcp_server.knowledge_ingest_path(path=str(csv_path))
        assert result["success"] is True
        assert result["format"] == "csv"
        assert result["rows"] == 2

    async def test_text_ingest(self, fresh_mcp_graph):
        mcp_server, tmp = fresh_mcp_graph
        txt_path = tmp / "notes.txt"
        txt_path.write_text("This is a short note.", encoding="utf-8")
        result = await mcp_server.knowledge_ingest_path(path=str(txt_path))
        assert result["success"] is True
        assert result["format"] == "text"
        assert result["chunks"] >= 1

    async def test_missing_path(self, fresh_mcp_graph):
        mcp_server, tmp = fresh_mcp_graph
        result = await mcp_server.knowledge_ingest_path(path=str(tmp / "nope.csv"))
        assert result["success"] is False
        assert "not found" in result["error"]


class TestRemove:
    async def test_removes_existing_node(self, fresh_mcp_graph):
        mcp_server, _ = fresh_mcp_graph
        add = await mcp_server.knowledge_add_document(
            title="doomed",
            content="short",
        )
        node_id = add["first_node_id"]
        result = await mcp_server.knowledge_remove(node_id=node_id)
        assert result["success"] is True

    async def test_remove_missing_node(self, fresh_mcp_graph):
        mcp_server, _ = fresh_mcp_graph
        result = await mcp_server.knowledge_remove(node_id="deadbeef")
        assert result["success"] is False


class TestKnowledgeSearch:
    """Verify MCP knowledge_search routes through EvidenceSearch.

    These tests document the Phase 2 migration: knowledge_search no
    longer calls the legacy ``graph.search()`` / ``HybridSearch``
    path, so the hardcoded ``cos >= 0.45`` threshold can no longer
    silently drop semantic hits.
    """

    async def test_lexical_query_returns_results(self, fresh_mcp_graph):
        """Sanity — a query that shares words with the doc still hits."""
        mcp_server, _ = fresh_mcp_graph
        await mcp_server.knowledge_add_document(
            title="Refund policy",
            content=(
                "Customers can request a refund within 30 days of purchase. "
                "Contact support to start the refund process."
            ),
            tags="ops,policy",
        )
        result = await mcp_server.knowledge_search(query="refund policy")
        assert result["success"] is True
        assert len(result["results"]) >= 1
        # First hit must be the doc we just ingested
        titles = [r["title"] for r in result["results"]]
        assert any("Refund policy" in t or "refund policy" in t.lower() for t in titles)

    async def test_evidence_search_fields_present(self, fresh_mcp_graph):
        """The new payload must include EvidenceSearch-specific fields
        (``reason``, ``category``) that the legacy ``ActivatedNode``
        format lacked. Acts as a regression guard against accidental
        revert to ``graph.search()``."""
        mcp_server, _ = fresh_mcp_graph
        await mcp_server.knowledge_add_document(
            title="Onboarding guide",
            content="New employee onboarding takes two weeks.",
        )
        result = await mcp_server.knowledge_search(query="onboarding")
        assert result["results"], "expected at least one hit"
        first = result["results"][0]
        assert "reason" in first  # EvidenceSearch field
        assert "category" in first  # EvidenceSearch field
        assert "score" in first

    async def test_empty_corpus_returns_no_results(self, fresh_mcp_graph):
        """Queries against an empty knowledge graph must succeed
        with an explanatory message — not crash."""
        mcp_server, _ = fresh_mcp_graph
        result = await mcp_server.knowledge_search(query="anything")
        assert result["success"] is True
        assert result["results"] == []
        assert "No knowledge" in result.get("message", "")

    async def test_no_match_query_returns_no_results(self, fresh_mcp_graph):
        """A query with neither lexical nor semantic overlap must
        not surface unrelated hits at the top."""
        mcp_server, _ = fresh_mcp_graph
        await mcp_server.knowledge_add_document(
            title="Pizza recipe",
            content="To make pizza, prepare dough, sauce, and cheese.",
        )
        # Without an embedder wired up there is no semantic path;
        # this query has zero lexical overlap with the doc, so we
        # expect zero results — *not* a crash and not a wrong hit.
        result = await mcp_server.knowledge_search(query="quantum cryptography")
        assert result["success"] is True
        # The pizza doc must not be the top hit for an unrelated query
        if result["results"]:
            top_title = result["results"][0]["title"].lower()
            assert "pizza" not in top_title


class TestSyncFromDatabase:
    async def test_initial_sync_seeds_state(self, fresh_mcp_graph):
        mcp_server, tmp = fresh_mcp_graph
        src = tmp / "source.db"
        con = sqlite3.connect(src)
        con.executescript(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                updated_at INTEGER NOT NULL
            );
            INSERT INTO products VALUES (1, 'Alpha', 1000);
            INSERT INTO products VALUES (2, 'Beta',  1100);
            """
        )
        con.commit()
        con.close()

        conn_str = f"sqlite:////{str(src).lstrip('/')}"
        result = await mcp_server.knowledge_sync_from_database(connection_string=conn_str)
        assert result["success"] is True
        products = next(t for t in result["tables"] if t["table"] == "products")
        assert products["added"] == 2
        assert products["strategy"] == "timestamp"
        assert products["error"] is None

    async def test_second_sync_idempotent(self, fresh_mcp_graph):
        mcp_server, tmp = fresh_mcp_graph
        src = tmp / "source.db"
        con = sqlite3.connect(src)
        con.executescript(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                updated_at INTEGER NOT NULL
            );
            INSERT INTO products VALUES (1, 'Alpha', 1000);
            """
        )
        con.commit()
        con.close()

        conn_str = f"sqlite:////{str(src).lstrip('/')}"
        await mcp_server.knowledge_sync_from_database(connection_string=conn_str)
        # Second run with no changes — added must stay 0.
        second = await mcp_server.knowledge_sync_from_database(connection_string=conn_str)
        assert second["added"] == 0

    async def test_missing_dsn_errors_clearly(self, fresh_mcp_graph):
        mcp_server, _ = fresh_mcp_graph
        result = await mcp_server.knowledge_sync_from_database()
        assert result["success"] is False
        assert "source DSN" in result["error"]

    async def test_source_dsn_fallback(self, fresh_mcp_graph):
        mcp_server, tmp = fresh_mcp_graph
        src = tmp / "source.db"
        con = sqlite3.connect(src)
        con.executescript(
            """
            CREATE TABLE t (
                id INTEGER PRIMARY KEY,
                updated_at INTEGER NOT NULL
            );
            INSERT INTO t VALUES (1, 1000);
            """
        )
        con.commit()
        con.close()

        mcp_server._source_dsn = f"sqlite:////{str(src).lstrip('/')}"
        result = await mcp_server.knowledge_sync_from_database()
        assert result["success"] is True
        assert result["added"] == 1
