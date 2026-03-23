"""Tests for SynapticGraph facade."""

from __future__ import annotations

import pytest

from synaptic.agent_search import SearchIntent, suggest_intent
from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import DigestResult, EdgeKind, MaintenanceResult, NodeKind
from synaptic.ontology import (
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
        import inspect

        from synaptic.graph import SynapticGraph

        sig = inspect.signature(SynapticGraph.agent_search)
        assert sig.parameters["intent"].default == "auto"


class TestGraphList:
    async def test_list_all(self, populated_graph: SynapticGraph) -> None:
        nodes = await populated_graph.list()
        assert len(nodes) == 5

    async def test_list_by_kind(self, populated_graph: SynapticGraph) -> None:
        lessons = await populated_graph.list(kind=NodeKind.LESSON)
        assert len(lessons) == 2
        assert all(n.kind == NodeKind.LESSON for n in lessons)

    async def test_list_with_limit(self, populated_graph: SynapticGraph) -> None:
        nodes = await populated_graph.list(limit=2)
        assert len(nodes) == 2

    async def test_list_empty(self, graph: SynapticGraph) -> None:
        nodes = await graph.list()
        assert nodes == []


class TestGraphUpdate:
    async def test_update_title(self, graph: SynapticGraph) -> None:
        node = await graph.add("Original", "Content", kind=NodeKind.CONCEPT)
        updated = await graph.update(node.id, title="Updated Title")
        assert updated is not None
        assert updated.title == "Updated Title"
        assert updated.content == "Content"

    async def test_update_content(self, graph: SynapticGraph) -> None:
        node = await graph.add("Title", "Old content")
        updated = await graph.update(node.id, content="New content")
        assert updated is not None
        assert updated.content == "New content"

    async def test_update_multiple_fields(self, graph: SynapticGraph) -> None:
        node = await graph.add("T", "C", kind=NodeKind.CONCEPT, tags=["old"])
        updated = await graph.update(
            node.id,
            title="New T",
            content="New C",
            tags=["new1", "new2"],
        )
        assert updated is not None
        assert updated.title == "New T"
        assert updated.content == "New C"
        assert updated.tags == ["new1", "new2"]

    async def test_update_nonexistent(self, graph: SynapticGraph) -> None:
        result = await graph.update("nonexistent", title="X")
        assert result is None

    async def test_update_refreshes_cache(self, graph: SynapticGraph) -> None:
        node = await graph.add("Cached", "Content")
        # Populate cache
        await graph.get(node.id)
        # Update
        await graph.update(node.id, title="Updated")
        # Get should return updated version
        fetched = await graph.get(node.id)
        assert fetched is not None
        assert fetched.title == "Updated"


class TestMaintain:
    async def test_maintain_returns_unified_result(self, graph: SynapticGraph) -> None:
        await graph.add("Node", "Content")
        result = await graph.maintain()
        assert isinstance(result, MaintenanceResult)
        assert result.consolidated is not None
        assert isinstance(result.decayed, int)
        assert isinstance(result.pruned, int)
        assert isinstance(result.total_affected, int)


class TestAddTurn:
    async def test_add_turn_creates_session(self, graph: SynapticGraph) -> None:
        session, user_n, asst_n = await graph.add_turn(
            "안녕하세요",
            "안녕하세요! 무엇을 도와드릴까요?",
            session_id="test_session_1",
        )
        assert session.kind == NodeKind.SESSION
        assert user_n.content == "안녕하세요"
        assert asst_n.content == "안녕하세요! 무엇을 도와드릴까요?"

    async def test_add_turn_reuses_session(self, graph: SynapticGraph) -> None:
        sid = "test_session_2"
        s1, _, _ = await graph.add_turn("Hi", "Hello", session_id=sid)
        s2, _, _ = await graph.add_turn("How?", "Like this", session_id=sid)
        assert s1.id == s2.id

    async def test_add_turn_links_turns(self, graph: SynapticGraph) -> None:
        sid = "test_session_3"
        _, u1, a1 = await graph.add_turn("First", "Reply1", session_id=sid)
        _, u2, a2 = await graph.add_turn("Second", "Reply2", session_id=sid)

        # a1 → u2 should be linked via FOLLOWED_BY
        edges = await graph.backend.get_edges(a1.id, direction="outgoing")
        followed = [e for e in edges if e.kind == EdgeKind.FOLLOWED_BY]
        assert len(followed) == 1
        assert followed[0].target_id == u2.id

    async def test_add_turn_auto_session_id(self, graph: SynapticGraph) -> None:
        session, _, _ = await graph.add_turn("Q", "A")
        assert session.id.startswith("session_")


class TestCustomKind:
    async def test_add_with_custom_kind(self, graph: SynapticGraph) -> None:
        node = await graph.add("한국 문화", "인사 예절", kind="culture")
        assert node.kind == "culture"

    async def test_get_custom_kind(self, graph: SynapticGraph) -> None:
        node = await graph.add("다크모드", "선호", kind="preference")
        fetched = await graph.get(node.id)
        assert fetched is not None
        assert fetched.kind == "preference"

    async def test_list_by_custom_kind(self, graph: SynapticGraph) -> None:
        await graph.add("A", "a", kind="culture")
        await graph.add("B", "b", kind="culture")
        await graph.add("C", "c", kind=NodeKind.CONCEPT)
        nodes = await graph.list(kind="culture")
        assert len(nodes) == 2
        assert all(n.kind == "culture" for n in nodes)

    async def test_update_to_custom_kind(self, graph: SynapticGraph) -> None:
        node = await graph.add("X", "x", kind=NodeKind.CONCEPT)
        updated = await graph.update(node.id, kind="preference")
        assert updated is not None
        assert updated.kind == "preference"

    async def test_builtin_kind_still_works(self, graph: SynapticGraph) -> None:
        node = await graph.add("Y", "y", kind=NodeKind.LESSON)
        assert node.kind == NodeKind.LESSON
        assert node.kind == "lesson"  # StrEnum == str
