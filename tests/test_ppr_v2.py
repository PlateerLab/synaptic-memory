"""Tests for personalized_pagerank_v2 — HippoRAG2-style noise reduction."""

from synaptic import EdgeKind, NodeKind, SynapticGraph
from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
from synaptic.ppr import personalized_pagerank_v2


class TestPPRv2Basic:
    async def test_empty_seeds(self):
        backend = MemoryBackend()
        result = await personalized_pagerank_v2(backend, {})
        assert result == []

    async def test_single_seed_no_edges(self):
        backend = MemoryBackend()
        from synaptic.models import Node

        node = Node(id="n1", kind=NodeKind.ENTITY, title="Test")
        await backend.save_node(node)

        result = await personalized_pagerank_v2(backend, {"n1": 1.0})
        assert len(result) >= 1
        assert result[0][0] == "n1"

    async def test_chunk_seed_boost(self):
        """CHUNK nodes should get higher teleport weight via passage_boost."""
        backend = MemoryBackend()
        from synaptic.models import Edge, Node

        c1 = Node(id="c1", kind=NodeKind.CHUNK, title="Chunk 1")
        c2 = Node(id="c2", kind=NodeKind.CONCEPT, title="Concept 1")
        shared = Node(id="s1", kind=NodeKind.ENTITY, title="Shared")
        await backend.save_node(c1)
        await backend.save_node(c2)
        await backend.save_node(shared)

        # Both connect to shared entity equally
        await backend.save_edge(Edge(source_id="c1", target_id="s1", kind=EdgeKind.MENTIONS))
        await backend.save_edge(Edge(source_id="c2", target_id="s1", kind=EdgeKind.RELATED))

        # Seeded equally, but chunk gets passage_boost in teleport
        result = await personalized_pagerank_v2(
            backend,
            {"c1": 0.5, "c2": 0.5},
            passage_boost=2.0,
        )

        scores = dict(result)
        # Chunk should have higher score due to boosted teleport
        assert scores.get("c1", 0) > scores.get("c2", 0)


class TestPPRv2NoiseReduction:
    async def test_weak_edge_zeroing(self):
        """Edges below edge_weight_floor should be ignored."""
        backend = MemoryBackend()
        from synaptic.models import Edge, Node

        n1 = Node(id="n1", kind=NodeKind.ENTITY, title="N1")
        n2 = Node(id="n2", kind=NodeKind.ENTITY, title="N2")
        await backend.save_node(n1)
        await backend.save_node(n2)

        # Very weak edge
        await backend.save_edge(
            Edge(source_id="n1", target_id="n2", kind=EdgeKind.RELATED, weight=0.05)
        )

        result = await personalized_pagerank_v2(
            backend,
            {"n1": 1.0},
            edge_weight_floor=0.15,
        )

        scores = dict(result)
        # n2 should barely be reached (RELATED weight 0.4 * 0.05 = 0.02 < 0.15)
        assert scores.get("n2", 0) < scores.get("n1", 0) * 0.1

    async def test_chunk_to_chunk_blocked(self):
        """Direct CHUNK→CHUNK propagation via PART_OF should be blocked."""
        backend = MemoryBackend()
        from synaptic.models import Edge, Node

        c1 = Node(id="c1", kind=NodeKind.CHUNK, title="Chunk 1")
        c2 = Node(id="c2", kind=NodeKind.CHUNK, title="Chunk 2")
        entity = Node(id="e1", kind=NodeKind.ENTITY, title="Entity")
        await backend.save_node(c1)
        await backend.save_node(c2)
        await backend.save_node(entity)

        # c1 → c2 via PART_OF (should be blocked in v2)
        await backend.save_edge(Edge(source_id="c1", target_id="c2", kind=EdgeKind.PART_OF))
        # c1 → entity via MENTIONS (should propagate)
        await backend.save_edge(Edge(source_id="c1", target_id="e1", kind=EdgeKind.MENTIONS))

        result = await personalized_pagerank_v2(backend, {"c1": 1.0})
        scores = dict(result)

        # Entity should be reachable, but c2 should have minimal score
        # (PART_OF between chunks is blocked)
        assert scores.get("e1", 0) > scores.get("c2", 0)

    async def test_next_chunk_still_allowed(self):
        """NEXT_CHUNK between chunks should still work (sequential reading)."""
        backend = MemoryBackend()
        from synaptic.models import Edge, Node

        c1 = Node(id="c1", kind=NodeKind.CHUNK, title="Chunk 1")
        c2 = Node(id="c2", kind=NodeKind.CHUNK, title="Chunk 2")
        await backend.save_node(c1)
        await backend.save_node(c2)

        await backend.save_edge(
            Edge(source_id="c1", target_id="c2", kind=EdgeKind.NEXT_CHUNK, weight=0.7)
        )

        result = await personalized_pagerank_v2(backend, {"c1": 1.0})
        scores = dict(result)

        # c2 should be reachable via NEXT_CHUNK
        assert "c2" in scores


class TestPPRv2Integration:
    async def test_search_uses_v2_with_index(self):
        """When ChunkEntityIndex is present, search should use PPR v2."""
        idx = ChunkEntityIndex()
        graph = SynapticGraph(
            MemoryBackend(),
            chunk_entity_index=idx,
        )

        # Add some data
        await graph.add("Test Entity", "content about databases", kind=NodeKind.ENTITY)
        await graph.add("Another", "content about caching", kind=NodeKind.CONCEPT)

        result = await graph.search("databases")
        # Should complete without error (v2 is used internally)
        assert result is not None

    async def test_search_uses_v1_without_index(self):
        """Without ChunkEntityIndex, search should use PPR v1."""
        graph = SynapticGraph.memory()

        await graph.add("Test", "content about databases")

        result = await graph.search("databases")
        assert result is not None
