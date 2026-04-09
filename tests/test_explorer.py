"""Tests for GraphExplorer — interactive data exploration API."""

import pytest

from synaptic import EdgeKind, NodeKind, SynapticGraph
from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
from synaptic.models import (
    GraphData,
    GraphStats,
    NodeDetail,
)


@pytest.fixture
async def populated_graph():
    """Graph with chunks, entities, table rows, and a community."""
    idx = ChunkEntityIndex()
    from synaptic.extensions.phrase_extractor import PhraseExtractor

    graph = SynapticGraph(
        MemoryBackend(),
        phrase_extractor=PhraseExtractor(max_phrases_per_node=5),
        chunk_entity_index=idx,
    )

    # Add a document (creates CHUNK nodes)
    chunks = await graph.add_document(
        "PostgreSQL Guide",
        "PostgreSQL is a powerful relational database. "
        "It supports ACID transactions and JSON storage. " * 20,
        chunk_size=200,
    )

    # Add entities manually
    entity = await graph.add("PostgreSQL", "relational database system", kind=NodeKind.ENTITY)
    await graph.link(chunks[0].id, entity.id, kind=EdgeKind.MENTIONS, weight=0.8)
    idx.register(chunks[0].id, entity.id)

    entity2 = await graph.add("Redis", "in-memory cache", kind=NodeKind.ENTITY)
    await graph.link(entity.id, entity2.id, kind=EdgeKind.RELATED, weight=0.6)

    # Add table data
    await graph.add_table(
        "service",
        [
            {"name": "id", "type": "int"},
            {"name": "name", "type": "str"},
            {"name": "port", "type": "int"},
        ],
        [
            {"id": 1, "name": "PostgreSQL", "port": 5432},
            {"id": 2, "name": "Redis", "port": 6379},
        ],
    )

    # Add community node
    comm = await graph.add(
        "Community DB",
        "database related nodes",
        kind=NodeKind.COMMUNITY,
        properties={"member_count": "3"},
    )
    await graph.link(entity.id, comm.id, kind=EdgeKind.PART_OF, weight=0.5)

    return graph, idx, {"chunks": chunks, "entity": entity, "entity2": entity2, "community": comm}


class TestGraphDataOverview:
    async def test_get_graph_data(self, populated_graph):
        graph, idx, refs = populated_graph
        explorer = graph.explorer

        data = await explorer.get_graph_data(max_nodes=100)
        assert isinstance(data, GraphData)
        assert len(data.nodes) > 0
        assert len(data.edges) > 0
        assert "total_nodes" in data.stats

    async def test_excludes_chunks_by_default(self, populated_graph):
        graph, _, _ = populated_graph
        data = await graph.explorer.get_graph_data()
        kinds = {n["kind"] for n in data.nodes}
        assert "chunk" not in kinds

    async def test_include_chunks(self, populated_graph):
        graph, _, _ = populated_graph
        data = await graph.explorer.get_graph_data(include_chunks=True)
        kinds = {n["kind"] for n in data.nodes}
        assert "chunk" in kinds

    async def test_filter_by_kind(self, populated_graph):
        graph, _, _ = populated_graph
        data = await graph.explorer.get_graph_data(node_kinds=["entity"])
        for n in data.nodes:
            assert n["kind"] == "entity"


class TestNodeDetail:
    async def test_get_node_detail(self, populated_graph):
        graph, _, refs = populated_graph
        detail = await graph.explorer.get_node_detail(refs["entity"].id)
        assert detail is not None
        assert detail.node.title == "PostgreSQL"
        assert len(detail.neighbors) > 0

    async def test_entity_chunk_count(self, populated_graph):
        graph, _, refs = populated_graph
        detail = await graph.explorer.get_node_detail(refs["entity"].id)
        assert detail.chunk_count >= 1

    async def test_nonexistent_node(self, populated_graph):
        graph, _, _ = populated_graph
        detail = await graph.explorer.get_node_detail("nonexistent")
        assert detail is None


class TestEntityContext:
    async def test_get_entity_context(self, populated_graph):
        graph, _, refs = populated_graph
        ctx = await graph.explorer.get_entity_context(refs["entity"].id)
        assert ctx is not None
        assert ctx.entity.title == "PostgreSQL"
        assert len(ctx.source_chunks) >= 1

    async def test_related_entities(self, populated_graph):
        graph, _, refs = populated_graph
        ctx = await graph.explorer.get_entity_context(refs["entity"].id)
        related_titles = {n.title for n, _ in ctx.related_entities}
        assert "Redis" in related_titles

    async def test_community_membership(self, populated_graph):
        graph, _, refs = populated_graph
        ctx = await graph.explorer.get_entity_context(refs["entity"].id)
        assert ctx.community is not None
        assert ctx.community["id"] == refs["community"].id


class TestChunkDetail:
    async def test_get_chunk_detail(self, populated_graph):
        graph, _, refs = populated_graph
        chunk_id = refs["chunks"][0].id
        detail = await graph.explorer.get_chunk_detail(chunk_id)
        assert detail is not None
        assert detail.chunk.kind == NodeKind.CHUNK
        assert detail.parent_doc == "PostgreSQL Guide"

    async def test_chunk_entities(self, populated_graph):
        graph, _, refs = populated_graph
        chunk_id = refs["chunks"][0].id
        detail = await graph.explorer.get_chunk_detail(chunk_id)
        # Should have at least the manually linked entity
        assert len(detail.extracted_entities) >= 1

    async def test_next_chunk_navigation(self, populated_graph):
        graph, _, refs = populated_graph
        if len(refs["chunks"]) >= 2:
            detail = await graph.explorer.get_chunk_detail(refs["chunks"][0].id)
            assert detail.next_chunk is not None


class TestTableRowDetail:
    async def test_get_table_row(self, populated_graph):
        graph, _, _ = populated_graph
        # Find a table row node
        all_nodes = await graph.backend.list_nodes(kind=NodeKind.ENTITY, limit=100)
        table_rows = [n for n in all_nodes if n.properties.get("_table_name")]
        assert len(table_rows) > 0

        detail = await graph.explorer.get_table_row_detail(table_rows[0].id)
        assert detail is not None
        assert detail.table_name == "service"
        assert len(detail.columns) > 0
        assert "name" in detail.columns


class TestGraphStats:
    async def test_get_stats(self, populated_graph):
        graph, _, _ = populated_graph
        stats = await graph.explorer.get_graph_stats()
        assert isinstance(stats, GraphStats)
        assert stats.total_nodes > 0
        assert stats.entity_count > 0

    async def test_stats_by_kind(self, populated_graph):
        graph, _, _ = populated_graph
        stats = await graph.explorer.get_graph_stats()
        assert "entity" in stats.nodes_by_kind


class TestSearchInGraph:
    async def test_search(self, populated_graph):
        graph, _, _ = populated_graph
        results = await graph.explorer.search_in_graph("PostgreSQL")
        assert len(results) > 0
        assert all(isinstance(r, NodeDetail) for r in results)

    async def test_search_empty(self, populated_graph):
        graph, _, _ = populated_graph
        results = await graph.explorer.search_in_graph("zzzznonexistent")
        assert results == []
