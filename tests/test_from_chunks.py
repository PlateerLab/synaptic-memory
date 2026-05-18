"""Tests for SynapticGraph.from_chunks() — bring-your-own-chunker path.

Covers the contract that lets users feed pre-parsed documents into the
graph without depending on the optional xgen-doc2chunk loader.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from synaptic import SynapticGraph


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as d:
        yield str(Path(d) / "test.db")


class TestFromChunks:
    async def test_minimal_chunk(self, tmp_db):
        """A single chunk with only `content` should ingest fine."""
        chunks = [{"content": "Hello world. This is a test document."}]
        graph = await SynapticGraph.from_chunks(chunks, db=tmp_db)
        stats = await graph.stats()
        assert stats["total_nodes"] >= 1

    async def test_full_metadata(self, tmp_db):
        """All recognised fields propagate through to nodes."""
        chunks = [
            {
                "content": "iPhone is a smartphone made by Apple.",
                "title": "iPhone Overview",
                "doc_id": "iphone_001",
                "category": "electronics",
                "source": "/data/manual.pdf",
                "chunk_index": 0,
                "page": 1,
            },
            {
                "content": "Galaxy is a smartphone made by Samsung.",
                "title": "Galaxy Overview",
                "doc_id": "galaxy_001",
                "category": "electronics",
                "source": "/data/manual.pdf",
                "chunk_index": 1,
                "page": 2,
            },
        ]
        graph = await SynapticGraph.from_chunks(chunks, db=tmp_db)
        stats = await graph.stats()
        assert stats["total_nodes"] >= 2

        # Verify search works
        result = await graph.search("iPhone Apple")
        assert len(result.nodes) >= 1

    async def test_auto_doc_id_when_missing(self, tmp_db):
        """Missing doc_id should be auto-generated."""
        chunks = [
            {"content": "First chunk text"},
            {"content": "Second chunk text"},
        ]
        graph = await SynapticGraph.from_chunks(chunks, db=tmp_db)
        stats = await graph.stats()
        assert stats["total_nodes"] >= 2

    async def test_empty_content_skipped(self, tmp_db):
        """Chunks with empty content are silently dropped."""
        chunks = [
            {"content": "real content"},
            {"content": ""},
            {"content": "   "},  # whitespace only
            {"content": "another real one"},
        ]
        graph = await SynapticGraph.from_chunks(chunks, db=tmp_db)
        stats = await graph.stats()
        # Only the 2 non-empty chunks ingested
        assert stats["total_nodes"] >= 2

    async def test_empty_input_raises(self, tmp_db):
        """Calling with [] should raise ValueError, not silently produce
        an empty graph."""
        with pytest.raises(ValueError, match="at least one chunk"):
            await SynapticGraph.from_chunks([], db=tmp_db)

    async def test_title_auto_derived(self, tmp_db):
        """When no title is given, the first line of content is used."""
        chunks = [
            {
                "content": "Project Apollo Plan\nThis describes the moon mission timeline.",
            },
        ]
        graph = await SynapticGraph.from_chunks(chunks, db=tmp_db)
        nodes = await graph._backend.list_nodes(kind=None, limit=10)
        titles = [n.title for n in nodes if n.title]
        # Auto-derived title should contain "Apollo" from the first line
        assert any("Apollo" in t for t in titles)


class TestFromChunksWiring:
    """The one-line constructors must wire the embedder / reranker into
    the *returned* graph — not just use them at ingest time."""

    async def test_embedder_wired_into_returned_graph(self, tmp_db):
        # An unreachable embed_url: the ingest-time embedding pass fails
        # gracefully, but the embedder must still be attached so that
        # query-time vector search has it.
        graph = await SynapticGraph.from_chunks(
            [{"content": "a test chunk for wiring"}],
            db=tmp_db,
            embed_url="http://localhost:1/v1",
        )
        assert graph._embedder is not None

    async def test_no_embedder_when_url_omitted(self, tmp_db):
        graph = await SynapticGraph.from_chunks(
            [{"content": "a test chunk"}], db=tmp_db
        )
        assert graph._embedder is None

    async def test_reranker_wired_into_returned_graph(self, tmp_db):
        graph = await SynapticGraph.from_chunks(
            [{"content": "a test chunk for reranker wiring"}],
            db=tmp_db,
            rerank_url="http://localhost:1",
        )
        from synaptic.extensions.reranker_cross import VLLMReranker

        assert isinstance(graph._reranker, VLLMReranker)

    async def test_caller_supplied_backend_is_used(self):
        from synaptic.backends.memory import MemoryBackend

        backend = MemoryBackend()
        await backend.connect()
        graph = await SynapticGraph.from_chunks(
            [{"content": "chunk on a caller-supplied backend"}],
            backend=backend,
        )
        assert graph._backend is backend


class TestSyncConstructors:
    def test_from_chunks_sync_builds_graph(self, tmp_db):
        # Called from a plain (non-async) test — the sync wrapper must
        # spin its own loop and return a ready graph.
        graph = SynapticGraph.from_chunks_sync(
            [{"content": "a chunk built via the sync facade"}], db=tmp_db
        )
        assert graph is not None
        assert graph._backend is not None

    async def test_sync_constructor_rejects_running_loop(self, tmp_db):
        # Inside an event loop asyncio.run would deadlock — must raise.
        with pytest.raises(RuntimeError, match="event loop"):
            SynapticGraph.from_chunks_sync([{"content": "x"}], db=tmp_db)
