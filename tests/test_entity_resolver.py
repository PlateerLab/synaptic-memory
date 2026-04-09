"""Tests for EntityResolver — duplicate entity merging."""

import pytest

from synaptic import EdgeKind, NodeKind, SynapticGraph
from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
from synaptic.extensions.entity_resolver import EntityResolver, _title_similarity


class TestTitleSimilarity:
    def test_identical(self):
        assert _title_similarity("PostgreSQL", "PostgreSQL") == 1.0

    def test_case_insensitive(self):
        assert _title_similarity("PostgreSQL", "postgresql") == 1.0

    def test_similar(self):
        sim = _title_similarity("PostgreSQL", "Postgresql DB")
        assert sim > 0.6

    def test_different(self):
        sim = _title_similarity("PostgreSQL", "Redis")
        assert sim <= 0.4

    def test_empty(self):
        assert _title_similarity("", "test") == 0.0
        assert _title_similarity("test", "") == 0.0


class TestEntityResolver:
    @pytest.fixture
    def graph(self):
        idx = ChunkEntityIndex()
        return SynapticGraph(MemoryBackend(), chunk_entity_index=idx), idx

    async def test_resolve_exact_duplicates(self, graph):
        g, idx = graph
        e1 = await g.add("PostgreSQL", "relational database", kind=NodeKind.ENTITY)
        e2 = await g.add("PostgreSQL", "postgres database", kind=NodeKind.ENTITY)

        resolver = EntityResolver(threshold=0.85)
        merged = await resolver.resolve(g)

        assert len(merged) == 1
        # One should remain, one should be deleted
        remaining = await g.backend.get_node(merged[0][0])
        removed = await g.backend.get_node(merged[0][1])
        assert remaining is not None
        assert removed is None

    async def test_resolve_similar_titles(self, graph):
        g, _ = graph
        await g.add("PostgreSQL Database", "content1", kind=NodeKind.ENTITY)
        await g.add("PostgreSQL DB", "content2", kind=NodeKind.ENTITY)

        resolver = EntityResolver(threshold=0.7)
        merged = await resolver.resolve(g)

        assert len(merged) >= 1

    async def test_no_merge_different_entities(self, graph):
        g, _ = graph
        await g.add("PostgreSQL", "database", kind=NodeKind.ENTITY)
        await g.add("Redis", "cache", kind=NodeKind.ENTITY)

        resolver = EntityResolver(threshold=0.85)
        merged = await resolver.resolve(g)

        assert len(merged) == 0

    async def test_merge_preserves_edges(self, graph):
        g, _ = graph
        e1 = await g.add("PostgreSQL", "db1", kind=NodeKind.ENTITY)
        e2 = await g.add("PostgreSQL", "db2", kind=NodeKind.ENTITY)
        other = await g.add("SQL Guide", "guide", kind=NodeKind.CONCEPT)

        await g.link(e2.id, other.id, kind=EdgeKind.RELATED, weight=0.8)

        resolver = EntityResolver(threshold=0.85)
        merged = await resolver.resolve(g)

        assert len(merged) == 1
        kept_id = merged[0][0]
        # The kept entity should now have the edge to "SQL Guide"
        edges = await g.backend.get_edges(kept_id)
        edge_targets = {e.target_id for e in edges} | {e.source_id for e in edges}
        assert other.id in edge_targets

    async def test_merge_updates_chunk_entity_index(self, graph):
        g, idx = graph
        chunk = await g.add("Test Chunk", "content", kind=NodeKind.CHUNK)
        e1 = await g.add("PostgreSQL", "db1", kind=NodeKind.ENTITY)
        e2 = await g.add("PostgreSQL", "db2", kind=NodeKind.ENTITY)

        idx.register(chunk.id, e2.id)

        resolver = EntityResolver(threshold=0.85)
        merged = await resolver.resolve(g)

        assert len(merged) == 1
        kept_id = merged[0][0]
        removed_id = merged[0][1]

        # Chunk should now point to kept entity
        assert kept_id in idx.entities_for_chunk(chunk.id)
        assert removed_id not in idx.entities_for_chunk(chunk.id)

    async def test_find_candidates_dry_run(self, graph):
        g, _ = graph
        await g.add("PostgreSQL", "db", kind=NodeKind.ENTITY)
        await g.add("PostgreSQL DB", "database", kind=NodeKind.ENTITY)
        await g.add("Redis", "cache", kind=NodeKind.ENTITY)

        resolver = EntityResolver(threshold=0.7)
        candidates = await resolver.find_candidates(g)

        # Should find PostgreSQL pair but not Redis
        assert len(candidates) >= 1
        # All candidates should be above threshold
        for _, _, sim in candidates:
            assert sim >= 0.7

    async def test_skip_phrase_nodes(self, graph):
        g, _ = graph
        await g.add("Test", "content", kind=NodeKind.ENTITY, tags=["_phrase"])
        await g.add("Test", "content2", kind=NodeKind.ENTITY, tags=["_phrase"])

        resolver = EntityResolver(threshold=0.85)
        merged = await resolver.resolve(g)

        assert len(merged) == 0  # Phrase nodes should be skipped

    async def test_empty_graph(self, graph):
        g, _ = graph
        resolver = EntityResolver(threshold=0.85)
        merged = await resolver.resolve(g)
        assert merged == []

    async def test_single_entity(self, graph):
        g, _ = graph
        await g.add("PostgreSQL", "db", kind=NodeKind.ENTITY)

        resolver = EntityResolver(threshold=0.85)
        merged = await resolver.resolve(g)
        assert merged == []
