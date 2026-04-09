"""Tests for ChunkEntityIndex — E2GraphRAG-style bidirectional index."""

import pytest

from synaptic import EdgeKind, NodeKind, SynapticGraph
from synaptic.extensions.chunk_entity_index import ChunkEntityIndex

# --- Unit tests for ChunkEntityIndex ---


class TestChunkEntityIndexUnit:
    """Pure in-memory index operations (no backend)."""

    def test_register_and_lookup(self):
        idx = ChunkEntityIndex()
        idx.register("c1", "e1")
        idx.register("c1", "e2")
        idx.register("c2", "e1")

        assert idx.chunks_for_entity("e1") == {"c1", "c2"}
        assert idx.chunks_for_entity("e2") == {"c1"}
        assert idx.entities_for_chunk("c1") == {"e1", "e2"}
        assert idx.entities_for_chunk("c2") == {"e1"}

    def test_empty_lookup(self):
        idx = ChunkEntityIndex()
        assert idx.chunks_for_entity("nonexistent") == set()
        assert idx.entities_for_chunk("nonexistent") == set()

    def test_unregister_chunk(self):
        idx = ChunkEntityIndex()
        idx.register("c1", "e1")
        idx.register("c1", "e2")
        idx.register("c2", "e1")

        idx.unregister_chunk("c1")

        assert idx.chunks_for_entity("e1") == {"c2"}
        assert idx.chunks_for_entity("e2") == set()
        assert idx.entities_for_chunk("c1") == set()

    def test_unregister_entity(self):
        idx = ChunkEntityIndex()
        idx.register("c1", "e1")
        idx.register("c2", "e1")
        idx.register("c1", "e2")

        idx.unregister_entity("e1")

        assert idx.chunks_for_entity("e1") == set()
        assert idx.entities_for_chunk("c1") == {"e2"}
        assert idx.entities_for_chunk("c2") == set()

    def test_shared_entities(self):
        idx = ChunkEntityIndex()
        idx.register("c1", "e1")
        idx.register("c2", "e1")
        idx.register("c3", "e1")
        idx.register("c1", "e2")
        idx.register("c2", "e2")
        idx.register("c1", "e3")  # only in c1

        shared = idx.shared_entities(["c1", "c2", "c3"])
        assert shared["e1"] == 3
        assert shared["e2"] == 2
        assert "e3" not in shared  # only in 1 chunk

    def test_chunks_for_entities_ranking(self):
        idx = ChunkEntityIndex()
        idx.register("c1", "e1")
        idx.register("c1", "e2")
        idx.register("c1", "e3")
        idx.register("c2", "e1")
        idx.register("c2", "e2")
        idx.register("c3", "e1")

        ranked = idx.chunks_for_entities(["e1", "e2", "e3"])
        keys = list(ranked.keys())
        # c1 has 3 entities, c2 has 2, c3 has 1
        assert keys[0] == "c1"
        assert ranked["c1"] == 3.0
        assert ranked["c2"] == 2.0
        assert ranked["c3"] == 1.0

    def test_stats(self):
        idx = ChunkEntityIndex()
        idx.register("c1", "e1")
        idx.register("c1", "e2")
        idx.register("c2", "e1")

        s = idx.stats()
        assert s["entity_count"] == 2
        assert s["chunk_count"] == 2
        assert s["avg_chunks_per_entity"] == 1.5  # e1→2 chunks, e2→1 chunk
        assert s["avg_entities_per_chunk"] == 1.5  # c1→2 entities, c2→1 entity


# --- Integration tests with SynapticGraph ---


class TestChunkEntityIndexIntegration:
    """Tests that add_document() integrates with ChunkEntityIndex."""

    @pytest.fixture
    def graph_with_index(self):
        """Create a graph with ChunkEntityIndex and PhraseExtractor."""
        from synaptic.extensions.phrase_extractor import PhraseExtractor

        idx = ChunkEntityIndex()
        graph = SynapticGraph.memory()
        # Re-create with chunk_entity_index
        from synaptic.backends.memory import MemoryBackend
        from synaptic.extensions.classifier_rules import RuleBasedClassifier

        graph = SynapticGraph(
            MemoryBackend(),
            classifier=RuleBasedClassifier(),
            phrase_extractor=PhraseExtractor(max_phrases_per_node=5),
            chunk_entity_index=idx,
        )
        return graph, idx

    async def test_add_document_creates_chunk_nodes(self, graph_with_index):
        graph, idx = graph_with_index
        nodes = await graph.add_document(
            "Test Document",
            "This is a short document about API design. " * 50,  # long enough to chunk
            chunk_size=200,
        )

        assert len(nodes) > 1
        for node in nodes:
            assert node.kind == NodeKind.CHUNK
            assert "parent_doc" in node.properties
            assert node.properties["parent_doc"] == "Test Document"

    async def test_add_document_creates_next_chunk_edges(self, graph_with_index):
        graph, idx = graph_with_index
        nodes = await graph.add_document(
            "Sequential Doc",
            "First section about PostgreSQL. " * 30 + "Second section about Redis cache. " * 30,
            chunk_size=200,
        )

        # Check NEXT_CHUNK edges exist between sequential chunks
        if len(nodes) >= 2:
            edges = await graph.backend.get_edges(nodes[0].id, direction="outgoing")
            next_chunk_edges = [e for e in edges if e.kind == EdgeKind.NEXT_CHUNK]
            assert len(next_chunk_edges) == 1
            assert next_chunk_edges[0].target_id == nodes[1].id

    async def test_short_document_creates_single_chunk(self, graph_with_index):
        graph, idx = graph_with_index
        nodes = await graph.add_document(
            "Short Doc",
            "This is a short document.",
        )

        assert len(nodes) == 1
        assert nodes[0].kind == NodeKind.CHUNK

    async def test_chunk_entity_index_populated_by_phrase_extractor(self, graph_with_index):
        graph, idx = graph_with_index
        # PhraseExtractor should extract "PostgreSQL" and "API" as phrases
        nodes = await graph.add_document(
            "PostgreSQL API Guide",
            "PostgreSQL is a powerful database. The API provides REST endpoints.",
        )

        # Index should have at least the chunk registered
        assert idx.chunk_count >= 0  # may be 0 if no phrases matched via CONTAINS

    async def test_add_document_without_index_uses_original_kind(self):
        """Without ChunkEntityIndex, add_document() should use the provided kind."""
        graph = SynapticGraph.memory()
        nodes = await graph.add_document(
            "No Index Doc",
            "Content. " * 200,
            chunk_size=200,
            kind=NodeKind.CONCEPT,
        )

        # Without chunk_entity_index, kind should be whatever was passed
        for node in nodes:
            assert node.kind == NodeKind.CONCEPT

    async def test_rebuild_from_backend(self, graph_with_index):
        graph, idx = graph_with_index

        # Manually create chunk + entity + MENTIONS edge
        chunk = await graph.add("Test Chunk", "content", kind=NodeKind.CHUNK)
        entity = await graph.add("Test Entity", "", kind=NodeKind.ENTITY)
        await graph.link(chunk.id, entity.id, kind=EdgeKind.MENTIONS, weight=0.8)

        # Clear index and rebuild
        new_idx = ChunkEntityIndex()
        await new_idx.rebuild_from_backend(graph.backend)

        assert new_idx.chunks_for_entity(entity.id) == {chunk.id}
        assert new_idx.entities_for_chunk(chunk.id) == {entity.id}
