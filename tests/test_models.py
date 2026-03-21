"""Tests for synaptic.models."""

from __future__ import annotations

from synaptic.models import (
    ActivatedNode,
    ConsolidationLevel,
    DigestResult,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
    SearchResult,
)


class TestNode:
    def test_default_values(self) -> None:
        node = Node()
        assert len(node.id) == 16
        assert node.kind == NodeKind.CONCEPT
        assert node.title == ""
        assert node.content == ""
        assert node.tags == []
        assert node.level == ConsolidationLevel.L0_RAW
        assert node.vitality == 1.0
        assert node.access_count == 0
        assert node.success_count == 0
        assert node.failure_count == 0

    def test_custom_values(self) -> None:
        node = Node(
            kind=NodeKind.LESSON,
            title="Test",
            content="Content",
            tags=["a", "b"],
            level=ConsolidationLevel.L3_PERMANENT,
        )
        assert node.kind == NodeKind.LESSON
        assert node.title == "Test"
        assert node.tags == ["a", "b"]
        assert node.level == ConsolidationLevel.L3_PERMANENT

    def test_unique_ids(self) -> None:
        ids = {Node().id for _ in range(100)}
        assert len(ids) == 100


class TestEdge:
    def test_default_values(self) -> None:
        edge = Edge()
        assert len(edge.id) == 16
        assert edge.kind == EdgeKind.RELATED
        assert edge.weight == 1.0

    def test_custom_kind(self) -> None:
        edge = Edge(kind=EdgeKind.CAUSED, weight=2.5)
        assert edge.kind == EdgeKind.CAUSED
        assert edge.weight == 2.5


class TestEnums:
    def test_consolidation_levels(self) -> None:
        assert ConsolidationLevel.L0_RAW.value == "L0"
        assert ConsolidationLevel.L3_PERMANENT.value == "L3"

    def test_node_kinds(self) -> None:
        assert NodeKind.CONCEPT.value == "concept"
        assert NodeKind.AGENT.value == "agent"

    def test_edge_kinds(self) -> None:
        assert EdgeKind.SUPERSEDES.value == "supersedes"


class TestActivatedNode:
    def test_defaults(self) -> None:
        node = Node(title="Test")
        activated = ActivatedNode(node=node)
        assert activated.activation == 0.0
        assert activated.resonance == 0.0
        assert activated.path == []


class TestSearchResult:
    def test_empty(self) -> None:
        result = SearchResult()
        assert result.nodes == []
        assert result.total_candidates == 0


class TestDigestResult:
    def test_empty(self) -> None:
        result = DigestResult()
        assert result.nodes_created == []
        assert result.edges_created == []
        assert result.tokens_used == 0
