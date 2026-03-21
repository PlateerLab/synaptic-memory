"""Tests for agent-optimized search."""

import pytest

from synaptic.activity import ActivityTracker
from synaptic.agent_search import SearchIntent
from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind, NodeKind


@pytest.fixture
async def graph() -> SynapticGraph:
    backend = MemoryBackend()
    await backend.connect()
    return SynapticGraph(backend)


@pytest.fixture
async def populated_graph(graph: SynapticGraph) -> SynapticGraph:
    """Graph with decisions, outcomes, lessons, and rules."""
    # Decisions
    d1 = await graph.add(
        "Use PostgreSQL for storage",
        "PostgreSQL supports vector search and FTS",
        kind=NodeKind.DECISION,
        tags=["database", "postgresql"],
        properties={"rationale": "Need vector + FTS"},
    )
    d2 = await graph.add(
        "Skip integration tests",
        "Time pressure before release",
        kind=NodeKind.DECISION,
        tags=["testing", "ci"],
        properties={"rationale": "Deadline approaching"},
    )

    # Outcomes
    o1 = await graph.add(
        "PostgreSQL migration successful",
        "Zero downtime migration completed",
        kind=NodeKind.OUTCOME,
        tags=["database", "success"],
        properties={"success": "true"},
    )
    o1.success_count = 3
    await graph.backend.update_node(o1)

    o2 = await graph.add(
        "Production bug from missing tests",
        "Critical bug found after release",
        kind=NodeKind.OUTCOME,
        tags=["testing", "failure"],
        properties={"success": "false"},
    )
    o2.failure_count = 2
    await graph.backend.update_node(o2)

    # Link decisions → outcomes
    await graph.link(d1.id, o1.id, kind=EdgeKind.RESULTED_IN)
    await graph.link(d2.id, o2.id, kind=EdgeKind.RESULTED_IN)

    # Lessons
    lesson = await graph.add(
        "Always run integration tests before release",
        "Skipping tests led to production bugs",
        kind=NodeKind.LESSON,
        tags=["testing", "best-practice"],
    )
    await graph.link(lesson.id, o2.id, kind=EdgeKind.LEARNED_FROM)

    # Rules
    await graph.add(
        "All PRs must have tests",
        "Mandatory test coverage for all pull requests",
        kind=NodeKind.RULE,
        tags=["testing", "ci", "policy"],
    )

    # Concepts
    await graph.add(
        "CI/CD Pipeline",
        "Continuous integration and deployment pipeline",
        kind=NodeKind.CONCEPT,
        tags=["ci", "deploy"],
    )

    return graph


class TestGeneralSearch:
    @pytest.mark.asyncio
    async def test_general_intent_works(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.agent_search("database", intent="general")
        assert result.nodes
        assert result.total_candidates > 0

    @pytest.mark.asyncio
    async def test_general_is_default(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.agent_search("database")
        assert result.nodes


class TestSimilarDecisions:
    @pytest.mark.asyncio
    async def test_finds_decisions(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.agent_search(
            "database storage choice", intent="similar_decisions",
        )
        assert result.nodes
        kinds = {n.node.kind for n in result.nodes}
        # Should contain DECISION and/or OUTCOME nodes
        assert kinds & {NodeKind.DECISION, NodeKind.OUTCOME}

    @pytest.mark.asyncio
    async def test_expands_to_outcomes(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.agent_search(
            "PostgreSQL", intent="similar_decisions",
        )
        # Should find the decision AND the outcome via RESULTED_IN
        kinds = {n.node.kind for n in result.nodes}
        assert NodeKind.OUTCOME in kinds or NodeKind.DECISION in kinds


class TestPastFailures:
    @pytest.mark.asyncio
    async def test_finds_failures(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.agent_search(
            "testing", intent="past_failures",
        )
        # Should include failed outcomes and related decisions
        assert result.nodes
        has_failure = any(
            n.node.failure_count > 0 for n in result.nodes
        )
        assert has_failure

    @pytest.mark.asyncio
    async def test_includes_lessons(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.agent_search(
            "testing", intent="past_failures",
        )
        kinds = {n.node.kind for n in result.nodes}
        assert NodeKind.LESSON in kinds


class TestRelatedRules:
    @pytest.mark.asyncio
    async def test_finds_rules(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.agent_search(
            "testing", intent="related_rules",
        )
        assert result.nodes
        kinds = {n.node.kind for n in result.nodes}
        assert NodeKind.RULE in kinds or NodeKind.LESSON in kinds


class TestReasoningChain:
    @pytest.mark.asyncio
    async def test_chain_ordering(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.agent_search(
            "testing", intent="reasoning_chain",
        )
        assert result.nodes
        # Decisions should come before outcomes, which come before lessons
        kinds = [n.node.kind for n in result.nodes]
        if NodeKind.DECISION in kinds and NodeKind.OUTCOME in kinds:
            dec_idx = kinds.index(NodeKind.DECISION)
            out_idx = kinds.index(NodeKind.OUTCOME)
            assert dec_idx < out_idx


class TestContextExplore:
    @pytest.mark.asyncio
    async def test_expands_neighborhood(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.agent_search(
            "CI/CD", intent="context_explore",
        )
        assert result.total_candidates > 0

    @pytest.mark.asyncio
    async def test_context_tags_boost(self, populated_graph: SynapticGraph) -> None:
        # With context tags matching, scores should be higher
        result_no_ctx = await populated_graph.agent_search(
            "database", intent="context_explore",
        )
        result_with_ctx = await populated_graph.agent_search(
            "database", intent="context_explore",
            context_tags=["database", "postgresql"],
        )
        # Both should return results
        assert result_no_ctx.nodes
        assert result_with_ctx.nodes


class TestWithActivityTracker:
    @pytest.mark.asyncio
    async def test_search_agent_sessions(self, graph: SynapticGraph) -> None:
        tracker = ActivityTracker(graph)
        session = await tracker.start_session(agent_id="test-agent")
        await tracker.log_tool_call(
            session.id, tool_name="database_query",
            result="Found 10 results", success=True,
        )
        decision = await tracker.record_decision(
            session.id, title="Use caching",
            rationale="Reduce DB load",
        )
        await tracker.record_outcome(
            decision.id, title="Cache hit rate 95%",
            content="Performance improved", success=True,
        )

        # Agent search should find the decision
        result = await graph.agent_search(
            "caching", intent="similar_decisions",
        )
        assert any("caching" in n.node.title.lower() for n in result.nodes)

    @pytest.mark.asyncio
    async def test_search_reasoning_chain_from_activity(
        self, graph: SynapticGraph,
    ) -> None:
        tracker = ActivityTracker(graph)
        session = await tracker.start_session(agent_id="agent-1")
        decision = await tracker.record_decision(
            session.id, title="Deploy to production",
            rationale="All tests pass",
        )
        await tracker.record_outcome(
            decision.id, title="Deploy success",
            content="Zero errors", success=True,
        )

        result = await graph.agent_search(
            "deploy", intent="reasoning_chain",
        )
        assert result.nodes
        kinds = {n.node.kind for n in result.nodes}
        assert NodeKind.DECISION in kinds
