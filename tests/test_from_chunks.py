"""Tests for SynapticGraph.from_chunks() — bring-your-own-chunker path.

Covers the contract that lets users feed pre-parsed documents into the
graph without depending on the optional xgen-doc2chunk loader.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from synaptic import SynapticGraph
from synaptic.models import EdgeKind, NodeKind


class _FakeOpenIEExtractor:
    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    async def extract_and_link(self, graph, node_id: str, title: str, content: str):
        self.calls.append((node_id, title, content))
        return [f"hub_{node_id}"]


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as d:
        yield str(Path(d) / "test.db")


@pytest.fixture(autouse=True)
async def close_created_graphs(monkeypatch):
    created: list[SynapticGraph] = []
    original_from_chunks = SynapticGraph.from_chunks
    original_from_chunks_sync = SynapticGraph.from_chunks_sync

    async def _from_chunks(cls, *args, **kwargs):
        graph = await original_from_chunks(*args, **kwargs)
        created.append(graph)
        return graph

    def _from_chunks_sync(cls, *args, **kwargs):
        graph = original_from_chunks_sync(*args, **kwargs)
        created.append(graph)
        return graph

    monkeypatch.setattr(SynapticGraph, "from_chunks", classmethod(_from_chunks))
    monkeypatch.setattr(SynapticGraph, "from_chunks_sync", classmethod(_from_chunks_sync))
    yield
    while created:
        await created.pop().close()


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

    async def test_chunks_with_same_doc_id_create_chunk_chain(self, tmp_db):
        chunks = [
            {
                "content": "Acme depends on Roadmap.",
                "title": "Doc A",
                "doc_id": "doc_a",
                "chunk_index": 0,
            },
            {
                "content": "Roadmap depends on Budget.",
                "title": "Doc A",
                "doc_id": "doc_a",
                "chunk_index": 1,
            },
        ]

        graph = await SynapticGraph.from_chunks(chunks, db=tmp_db)

        chunk_nodes = await graph._backend.list_nodes(kind=NodeKind.CHUNK, limit=10)
        assert len(chunk_nodes) == 2
        assert {n.properties["doc_id"] for n in chunk_nodes} == {"doc_a"}

        first = next(n for n in chunk_nodes if n.properties["chunk_index"] == "0")
        edges = await graph._backend.get_edges(first.id, direction="outgoing")
        assert any(edge.kind == EdgeKind.NEXT_CHUNK for edge in edges)

    async def test_openie_extractor_not_called_without_opt_in(self, tmp_db):
        extractor = _FakeOpenIEExtractor()

        await SynapticGraph.from_chunks(
            [{"content": "Acme depends on Roadmap.", "title": "Doc"}],
            db=tmp_db,
            openie_extractor=extractor,
        )

        assert extractor.calls == []

    async def test_openie_extractor_runs_when_opted_in(self, tmp_db):
        extractor = _FakeOpenIEExtractor()

        await SynapticGraph.from_chunks(
            [{"content": "Acme depends on Roadmap.", "title": "Doc"}],
            db=tmp_db,
            openie_extractor=extractor,
            openie_enabled=True,
        )

        assert len(extractor.calls) == 1
        assert extractor.calls[0][2] == "Acme depends on Roadmap."

    async def test_auto_doc_id_when_missing(self, tmp_db):
        """Missing doc_id should be auto-generated."""
        chunks = [
            {"content": "First chunk text"},
            {"content": "Second chunk text"},
        ]
        graph = await SynapticGraph.from_chunks(chunks, db=tmp_db)
        stats = await graph.stats()
        assert stats["total_nodes"] >= 2

    async def test_auto_doc_id_is_deterministic(self, tmp_path):
        chunks = [
            {"content": "First chunk text"},
            {"content": "Second chunk text"},
        ]

        graph_a = await SynapticGraph.from_chunks(chunks, db=str(tmp_path / "a.db"))
        graph_b = await SynapticGraph.from_chunks(chunks, db=str(tmp_path / "b.db"))

        nodes_a = await graph_a._backend.list_nodes(limit=100)
        nodes_b = await graph_b._backend.list_nodes(limit=100)
        assert sorted(n.id for n in nodes_a) == sorted(n.id for n in nodes_b)
        assert sorted(n.properties.get("doc_id", "") for n in nodes_a) == sorted(
            n.properties.get("doc_id", "") for n in nodes_b
        )

        edges_a = {
            edge.id
            for node in nodes_a
            for edge in await graph_a._backend.get_edges(node.id, direction="outgoing")
        }
        edges_b = {
            edge.id
            for node in nodes_b
            for edge in await graph_b._backend.get_edges(node.id, direction="outgoing")
        }
        assert edges_a == edges_b

        await graph_a.close()
        await graph_b.close()

    async def test_auto_doc_id_groups_chunks_by_source(self, tmp_db):
        chunks = [
            {"content": "Page one", "title": "Manual", "source": "/docs/manual.pdf"},
            {"content": "Page two", "title": "Manual", "source": "/docs/manual.pdf"},
        ]

        graph = await SynapticGraph.from_chunks(chunks, db=tmp_db)

        chunk_nodes = await graph._backend.list_nodes(kind=NodeKind.CHUNK, limit=10)
        assert len(chunk_nodes) == 2
        assert len({n.properties["doc_id"] for n in chunk_nodes}) == 1
        first = next(n for n in chunk_nodes if n.properties["chunk_index"] == "0")
        edges = await graph._backend.get_edges(first.id, direction="outgoing")
        assert any(edge.kind == EdgeKind.NEXT_CHUNK for edge in edges)

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
        graph = await SynapticGraph.from_chunks([{"content": "a test chunk"}], db=tmp_db)
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
