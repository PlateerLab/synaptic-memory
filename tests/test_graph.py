"""Tests for SynapticGraph facade."""

from __future__ import annotations

import pytest

from synaptic.agent_search import SearchIntent, suggest_intent
from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import DigestResult, EdgeKind, NodeKind
from synaptic.ontology import (
    OntologyRegistry,
    PropertyDef,
    RelationConstraint,
    TypeDef,
    build_agent_ontology,
)


class TestGraphCRUD:
    async def test_add_and_get(self, graph: SynapticGraph) -> None:
        node = await graph.add("Test Node", "Test content", kind=NodeKind.LESSON)
        assert node.title == "Test Node"
        assert node.kind == NodeKind.LESSON

        fetched = await graph.get(node.id)
        assert fetched is not None
        assert fetched.title == "Test Node"

    async def test_get_nonexistent(self, graph: SynapticGraph) -> None:
        result = await graph.get("nonexistent")
        assert result is None

    async def test_remove(self, graph: SynapticGraph) -> None:
        node = await graph.add("To Remove", "Content")
        assert await graph.remove(node.id) is True
        assert await graph.get(node.id) is None

    async def test_remove_nonexistent(self, graph: SynapticGraph) -> None:
        assert await graph.remove("nonexistent") is False

    async def test_link(self, graph: SynapticGraph) -> None:
        n1 = await graph.add("Node A", "Content A")
        n2 = await graph.add("Node B", "Content B")
        edge = await graph.link(n1.id, n2.id, kind=EdgeKind.CAUSED)
        assert edge.source_id == n1.id
        assert edge.target_id == n2.id
        assert edge.kind == EdgeKind.CAUSED


class TestGraphSearch:
    async def test_search_by_title(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.search("배포")
        assert len(result.nodes) > 0
        assert any("배포" in n.node.title for n in result.nodes)

    async def test_search_by_content(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.search("N+1")
        assert len(result.nodes) > 0

    async def test_search_empty_query(self, graph: SynapticGraph) -> None:
        result = await graph.search("")
        assert result.nodes == []

    async def test_search_no_results(self, graph: SynapticGraph) -> None:
        result = await graph.search("completely_nonexistent_term_xyz")
        assert result.nodes == []

    async def test_search_stages(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.search("테스트")
        assert "fts" in result.stages_used


class TestGraphReinforce:
    async def test_reinforce_success(self, graph: SynapticGraph) -> None:
        n1 = await graph.add("Node A", "Content")
        n2 = await graph.add("Node B", "Content")
        await graph.reinforce([n1.id, n2.id], success=True)

        updated = await graph.get(n1.id)
        assert updated is not None
        assert updated.success_count >= 1

    async def test_reinforce_failure(self, graph: SynapticGraph) -> None:
        n1 = await graph.add("Node A", "Content")
        await graph.reinforce([n1.id], success=False)

        updated = await graph.get(n1.id)
        assert updated is not None
        assert updated.failure_count >= 1


class TestGraphConsolidate:
    async def test_consolidate_without_digester(self, graph: SynapticGraph) -> None:
        result = await graph.consolidate()
        assert isinstance(result, DigestResult)


class TestGraphMaintenance:
    async def test_prune(self, graph: SynapticGraph) -> None:
        n1 = await graph.add("A", "a")
        n2 = await graph.add("B", "b")
        edge = await graph.link(n1.id, n2.id)
        # Manually set low weight
        edge.weight = 0.01
        await graph.backend.update_edge(edge)
        pruned = await graph.prune()
        assert pruned == 1

    async def test_decay(self, populated_graph: SynapticGraph) -> None:
        count = await populated_graph.decay()
        assert count > 0

    async def test_stats(self, populated_graph: SynapticGraph) -> None:
        stats = await populated_graph.stats()
        assert stats["total_nodes"] == 5
        assert stats.get("kind_lesson", 0) == 2
        assert stats.get("kind_rule", 0) == 2

    async def test_export_markdown(self, populated_graph: SynapticGraph) -> None:
        md = await populated_graph.export_markdown()
        assert "# Knowledge Graph" in md
        assert "배포 자동화" in md


class TestLinkOntologyValidation:
    async def test_link_validates_edge_constraints(self) -> None:
        """link() should reject edges that violate ontology constraints."""
        backend = MemoryBackend()
        await backend.connect()
        ontology = build_agent_ontology()
        graph = SynapticGraph(backend, ontology=ontology)

        concept = await graph.add("Some Concept", "content", kind=NodeKind.CONCEPT)
        outcome = await graph.add("Some Outcome", "content", kind=NodeKind.OUTCOME)

        # resulted_in is constrained: decision → outcome only
        with pytest.raises(ValueError, match="Ontology validation failed"):
            await graph.link(concept.id, outcome.id, kind=EdgeKind.RESULTED_IN)

    async def test_link_allows_valid_edges(self) -> None:
        backend = MemoryBackend()
        await backend.connect()
        ontology = build_agent_ontology()
        graph = SynapticGraph(backend, ontology=ontology)

        decision = await graph.add("Choose DB", "rationale", kind=NodeKind.DECISION)
        outcome = await graph.add("Success", "ok", kind=NodeKind.OUTCOME)

        # This should pass — decision → outcome is valid for resulted_in
        edge = await graph.link(decision.id, outcome.id, kind=EdgeKind.RESULTED_IN)
        assert edge is not None

    async def test_link_without_ontology_always_passes(self) -> None:
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(backend)  # no ontology

        n1 = await graph.add("A", "a", kind=NodeKind.CONCEPT)
        n2 = await graph.add("B", "b", kind=NodeKind.OUTCOME)

        # No ontology = no validation = always passes
        edge = await graph.link(n1.id, n2.id, kind=EdgeKind.RESULTED_IN)
        assert edge is not None


class TestOntologyPersistence:
    async def test_save_and_load_ontology(self) -> None:
        backend = MemoryBackend()
        await backend.connect()
        ontology = build_agent_ontology()
        graph = SynapticGraph(backend, ontology=ontology)

        await graph.save_ontology()

        # Create a new graph without ontology
        graph2 = SynapticGraph(backend)
        assert graph2.ontology is None

        # Load from storage
        loaded = await graph2.load_ontology()
        assert loaded is not None
        assert loaded.get_type("session") is not None
        assert loaded.get_type("tool_call") is not None
        assert loaded.is_a("tool_call", "agent_activity")

    async def test_load_ontology_when_not_saved(self) -> None:
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(backend)

        loaded = await graph.load_ontology()
        assert loaded is None


class TestSuggestIntent:
    def test_failure_keywords(self) -> None:
        assert suggest_intent("배포 실패 원인") == SearchIntent.PAST_FAILURES
        assert suggest_intent("error in production") == SearchIntent.PAST_FAILURES

    def test_decision_keywords(self) -> None:
        assert suggest_intent("DB 선택 어떻게") == SearchIntent.SIMILAR_DECISIONS

    def test_rule_keywords(self) -> None:
        assert suggest_intent("배포 규칙 정책") == SearchIntent.RELATED_RULES

    def test_reasoning_keywords(self) -> None:
        assert suggest_intent("왜 이런 결과가") == SearchIntent.REASONING_CHAIN

    def test_explore_keywords(self) -> None:
        assert suggest_intent("관련된 것들 연결") == SearchIntent.CONTEXT_EXPLORE

    def test_no_match_returns_general(self) -> None:
        assert suggest_intent("xyzzy foobar") == SearchIntent.GENERAL

    def test_agent_search_auto_intent(self) -> None:
        """agent_search with intent='auto' should infer from query."""
        # Just verify the default is "auto" — actual search tested elsewhere
        from synaptic.graph import SynapticGraph
        import inspect
        sig = inspect.signature(SynapticGraph.agent_search)
        assert sig.parameters["intent"].default == "auto"
