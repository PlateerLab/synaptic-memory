"""Tests for TableIngester — structured data ingestion."""

import pytest

from synaptic import EdgeKind, NodeKind, SynapticGraph
from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
from synaptic.extensions.table_ingester import TableIngester, _row_to_natural_language


class TestRowToNaturalLanguage:
    def test_basic(self):
        result = _row_to_natural_language("product", {"name": "운동화A", "price": 89000})
        assert "product" in result
        assert "운동화A" in result
        assert "89000" in result

    def test_skips_none(self):
        result = _row_to_natural_language("t", {"a": 1, "b": None})
        assert "1" in result
        assert "b" not in result


class TestTableIngesterIntegration:
    @pytest.fixture
    def graph_with_ontology(self):
        from synaptic.backends.memory import MemoryBackend
        from synaptic.extensions.classifier_rules import RuleBasedClassifier
        from synaptic.ontology import build_agent_ontology

        idx = ChunkEntityIndex()
        return SynapticGraph(
            MemoryBackend(),
            classifier=RuleBasedClassifier(),
            ontology=build_agent_ontology(),
            chunk_entity_index=idx,
        ), idx

    async def test_ingest_creates_entity_nodes(self, graph_with_ontology):
        graph, idx = graph_with_ontology
        columns = [
            {"name": "id", "type": "int"},
            {"name": "name", "type": "str"},
            {"name": "price", "type": "int"},
        ]
        rows = [
            {"id": 1, "name": "운동화A", "price": 89000},
            {"id": 2, "name": "티셔츠B", "price": 35000},
        ]

        nodes = await graph.add_table("product", columns, rows)

        assert len(nodes) == 2
        for node in nodes:
            assert node.kind == NodeKind.ENTITY
            assert "_table:product" in node.tags
            assert node.properties["_table_name"] == "product"

    async def test_ingest_stores_column_values(self, graph_with_ontology):
        graph, _ = graph_with_ontology
        columns = [{"name": "id", "type": "int"}, {"name": "name", "type": "str"}]
        rows = [{"id": 1, "name": "테스트"}]

        nodes = await graph.add_table("test_table", columns, rows)

        assert nodes[0].properties["name"] == "테스트"
        assert nodes[0].properties["id"] == "1"

    async def test_ingest_creates_fk_edges(self, graph_with_ontology):
        graph, _ = graph_with_ontology

        # First ingest categories
        cat_cols = [{"name": "id", "type": "int"}, {"name": "name", "type": "str"}]
        cat_rows = [
            {"id": 1, "name": "신발"},
            {"id": 2, "name": "의류"},
        ]
        await graph.add_table("category", cat_cols, cat_rows)

        # Then ingest products with FK
        prod_cols = [
            {"name": "id", "type": "int"},
            {"name": "name", "type": "str"},
            {"name": "category_id", "type": "int"},
        ]
        prod_rows = [
            {"id": 1, "name": "운동화A", "category_id": 1},
            {"id": 2, "name": "티셔츠B", "category_id": 2},
        ]

        ingester = TableIngester()
        # Need to use same ingester instance so node_cache is shared
        await ingester.ingest(graph, "category", cat_cols, cat_rows)
        prod_nodes = await ingester.ingest(
            graph,
            "product2",
            prod_cols,
            prod_rows,
            foreign_keys={"category_id": ("category", "id")},
        )

        # Check FK edge exists
        edges = await graph.backend.get_edges(prod_nodes[0].id, direction="outgoing")
        fk_edges = [e for e in edges if e.kind == EdgeKind.RELATED]
        assert len(fk_edges) >= 1

    async def test_ingest_natural_language_content(self, graph_with_ontology):
        graph, _ = graph_with_ontology
        columns = [{"name": "id", "type": "int"}, {"name": "name", "type": "str"}]
        rows = [{"id": 1, "name": "운동화A"}]

        nodes = await graph.add_table("product", columns, rows)

        # Content should be natural language for FTS
        assert "운동화A" in nodes[0].content
        assert "product" in nodes[0].content

    async def test_ingest_title_uses_name_column(self, graph_with_ontology):
        graph, _ = graph_with_ontology
        columns = [{"name": "id", "type": "int"}, {"name": "name", "type": "str"}]
        rows = [{"id": 1, "name": "운동화A"}]

        nodes = await graph.add_table("product", columns, rows)

        assert "운동화A" in nodes[0].title

    async def test_ingest_searchable(self, graph_with_ontology):
        graph, _ = graph_with_ontology
        columns = [
            {"name": "id", "type": "int"},
            {"name": "name", "type": "str"},
            {"name": "category", "type": "str"},
        ]
        rows = [
            {"id": 1, "name": "PostgreSQL", "category": "database"},
            {"id": 2, "name": "Redis", "category": "cache"},
        ]

        await graph.add_table("service", columns, rows)

        # Should be searchable via FTS
        result = await graph.search("PostgreSQL database")
        found_titles = [a.node.title for a in result.nodes]
        assert any("PostgreSQL" in t for t in found_titles)

    async def test_ingest_empty_rows(self, graph_with_ontology):
        graph, _ = graph_with_ontology
        columns = [{"name": "id", "type": "int"}]
        nodes = await graph.add_table("empty", columns, [])
        assert nodes == []


class TestDeterministicReingest:
    """CDC Phase 1: same (source_url, table, pk) → same node_id across runs."""

    @pytest.fixture
    def graph(self):
        from synaptic.backends.memory import MemoryBackend
        from synaptic.extensions.classifier_rules import RuleBasedClassifier
        from synaptic.ontology import build_agent_ontology

        return SynapticGraph(
            MemoryBackend(),
            classifier=RuleBasedClassifier(),
            ontology=build_agent_ontology(),
            chunk_entity_index=ChunkEntityIndex(),
        )

    async def test_reingest_produces_stable_node_ids(self, graph):
        columns = [{"name": "id", "type": "int"}, {"name": "name", "type": "str"}]
        rows = [
            {"id": 1, "name": "운동화A"},
            {"id": 2, "name": "티셔츠B"},
        ]
        source_url = "postgres://host/db"

        # First ingest
        first = await TableIngester().ingest(graph, "product", columns, rows, source_url=source_url)
        first_ids = {n.properties["id"]: n.id for n in first}

        # Second ingest with a fresh instance — same IDs must come back
        second = await TableIngester().ingest(
            graph, "product", columns, rows, source_url=source_url
        )
        second_ids = {n.properties["id"]: n.id for n in second}

        assert first_ids == second_ids
        # Upsert: total node count unchanged after the second run
        assert await graph.backend.count_nodes() == len(rows)

    async def test_source_url_isolates_node_ids(self, graph):
        columns = [{"name": "id", "type": "int"}, {"name": "name", "type": "str"}]
        rows = [{"id": 1, "name": "운동화A"}]

        a = await TableIngester().ingest(
            graph, "product", columns, rows, source_url="postgres://host/db1"
        )
        b = await TableIngester().ingest(
            graph, "product", columns, rows, source_url="postgres://host/db2"
        )
        assert a[0].id != b[0].id

    async def test_legacy_mode_still_random(self, graph):
        columns = [{"name": "id", "type": "int"}, {"name": "name", "type": "str"}]
        rows = [{"id": 1, "name": "운동화A"}]

        a = await TableIngester().ingest(graph, "product", columns, rows)
        b = await TableIngester().ingest(graph, "product", columns, rows)
        # No source_url → legacy random UUIDs, two different nodes
        assert a[0].id != b[0].id

    async def test_fk_resolves_across_ingester_instances(self, graph):
        source_url = "postgres://host/db"
        cat_cols = [{"name": "id", "type": "int"}, {"name": "name", "type": "str"}]
        cat_rows = [{"id": 1, "name": "신발"}]
        await TableIngester().ingest(graph, "category", cat_cols, cat_rows, source_url=source_url)

        # Separate TableIngester instance — normally its _node_cache would
        # be empty, but deterministic ID derivation lets FK edges resolve.
        prod_cols = [
            {"name": "id", "type": "int"},
            {"name": "name", "type": "str"},
            {"name": "category_id", "type": "int"},
        ]
        prod_rows = [{"id": 1, "name": "운동화A", "category_id": 1}]
        prod_nodes = await TableIngester().ingest(
            graph,
            "product",
            prod_cols,
            prod_rows,
            foreign_keys={"category_id": ("category", "id")},
            source_url=source_url,
        )

        edges = await graph.backend.get_edges(prod_nodes[0].id, direction="outgoing")
        assert any(e.kind == EdgeKind.RELATED for e in edges)
