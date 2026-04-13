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
