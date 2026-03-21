"""Tests for the agent activity tracker."""

import pytest

from synaptic.activity import ActivityTracker
from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind, NodeKind


@pytest.fixture
async def graph() -> SynapticGraph:
    backend = MemoryBackend()
    await backend.connect()
    return SynapticGraph(backend)


@pytest.fixture
def tracker(graph: SynapticGraph) -> ActivityTracker:
    return ActivityTracker(graph)


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_start_session(self, tracker: ActivityTracker) -> None:
        session = await tracker.start_session(agent_id="test-agent", description="Test run")
        assert session.kind == NodeKind.SESSION
        assert session.properties["agent_id"] == "test-agent"
        assert session.properties["status"] == "active"

    @pytest.mark.asyncio
    async def test_end_session(self, tracker: ActivityTracker, graph: SynapticGraph) -> None:
        session = await tracker.start_session(agent_id="agent-1")
        await tracker.end_session(session.id, outcome="success")
        updated = await graph.backend.get_node(session.id)
        assert updated is not None
        assert updated.properties["status"] == "completed"
        assert "end_time" in updated.properties

    @pytest.mark.asyncio
    async def test_end_nonexistent_session(self, tracker: ActivityTracker) -> None:
        # Should not raise
        await tracker.end_session("nonexistent-id")


class TestToolCallLogging:
    @pytest.mark.asyncio
    async def test_log_tool_call(self, tracker: ActivityTracker, graph: SynapticGraph) -> None:
        session = await tracker.start_session(agent_id="agent-1")
        tc = await tracker.log_tool_call(
            session.id,
            tool_name="file_read",
            parameters={"path": "/etc/hosts"},
            result="127.0.0.1 localhost",
            success=True,
            duration_ms=42.5,
        )
        assert tc.kind == NodeKind.TOOL_CALL
        assert tc.properties["tool_name"] == "file_read"
        assert tc.properties["success"] == "true"
        assert tc.properties["duration_ms"] == "42.5"

        # Verify PART_OF edge to session
        edges = await graph.backend.get_edges(tc.id, direction="outgoing")
        part_of = [e for e in edges if e.kind == EdgeKind.PART_OF]
        assert len(part_of) == 1
        assert part_of[0].target_id == session.id

    @pytest.mark.asyncio
    async def test_followed_by_chain(self, tracker: ActivityTracker, graph: SynapticGraph) -> None:
        session = await tracker.start_session(agent_id="agent-1")
        tc1 = await tracker.log_tool_call(session.id, tool_name="search", result="found")
        tc2 = await tracker.log_tool_call(session.id, tool_name="read", result="content")

        # tc1 → tc2 via FOLLOWED_BY
        edges = await graph.backend.get_edges(tc1.id, direction="outgoing")
        followed = [e for e in edges if e.kind == EdgeKind.FOLLOWED_BY]
        assert len(followed) == 1
        assert followed[0].target_id == tc2.id


class TestDecisionOutcome:
    @pytest.mark.asyncio
    async def test_record_decision(self, tracker: ActivityTracker) -> None:
        session = await tracker.start_session(agent_id="agent-1")
        decision = await tracker.record_decision(
            session.id,
            title="Use PostgreSQL",
            rationale="Need vector search support",
            alternatives=["SQLite", "MongoDB"],
        )
        assert decision.kind == NodeKind.DECISION
        assert decision.properties["rationale"] == "Need vector search support"
        assert "SQLite" in decision.properties["alternatives"]

    @pytest.mark.asyncio
    async def test_record_decision_with_context(
        self, tracker: ActivityTracker, graph: SynapticGraph,
    ) -> None:
        session = await tracker.start_session(agent_id="agent-1")
        ctx = await graph.add("DB Requirements", "Need ACID + vector", kind=NodeKind.CONCEPT)
        decision = await tracker.record_decision(
            session.id,
            title="Choose DB",
            rationale="Based on requirements",
            context_node_ids=[ctx.id],
        )
        edges = await graph.backend.get_edges(decision.id, direction="outgoing")
        deps = [e for e in edges if e.kind == EdgeKind.DEPENDS_ON]
        assert len(deps) == 1
        assert deps[0].target_id == ctx.id

    @pytest.mark.asyncio
    async def test_record_outcome_success(
        self, tracker: ActivityTracker, graph: SynapticGraph,
    ) -> None:
        session = await tracker.start_session(agent_id="agent-1")
        decision = await tracker.record_decision(
            session.id, title="Deploy v2", rationale="Ready",
        )
        outcome = await tracker.record_outcome(
            decision.id, title="Deploy succeeded", content="Zero downtime", success=True,
        )
        assert outcome.kind == NodeKind.OUTCOME
        assert outcome.properties["success"] == "true"

        # Verify RESULTED_IN edge
        edges = await graph.backend.get_edges(decision.id, direction="outgoing")
        resulted = [e for e in edges if e.kind == EdgeKind.RESULTED_IN]
        assert len(resulted) == 1
        assert resulted[0].target_id == outcome.id

    @pytest.mark.asyncio
    async def test_record_outcome_failure_reinforcement(
        self, tracker: ActivityTracker, graph: SynapticGraph,
    ) -> None:
        session = await tracker.start_session(agent_id="agent-1")
        decision = await tracker.record_decision(
            session.id, title="Skip tests", rationale="Time pressure",
        )
        await tracker.record_outcome(
            decision.id, title="Bug in prod", content="Caused incident", success=False,
        )
        # Decision should have failure_count increased via Hebbian
        updated_decision = await graph.backend.get_node(decision.id)
        assert updated_decision is not None
        assert updated_decision.failure_count > 0


class TestObservation:
    @pytest.mark.asyncio
    async def test_record_observation(self, tracker: ActivityTracker) -> None:
        session = await tracker.start_session(agent_id="agent-1")
        obs = await tracker.record_observation(
            session.id, title="High CPU usage", content="CPU at 95%",
        )
        assert obs.kind == NodeKind.OBSERVATION

    @pytest.mark.asyncio
    async def test_observation_with_source(
        self, tracker: ActivityTracker, graph: SynapticGraph,
    ) -> None:
        session = await tracker.start_session(agent_id="agent-1")
        tc = await tracker.log_tool_call(session.id, tool_name="monitor", result="CPU 95%")
        obs = await tracker.record_observation(
            session.id, title="High CPU", content="95%", source_node_id=tc.id,
        )
        edges = await graph.backend.get_edges(tc.id, direction="outgoing")
        produced = [e for e in edges if e.kind == EdgeKind.PRODUCED]
        assert any(e.target_id == obs.id for e in produced)


class TestTimeline:
    @pytest.mark.asyncio
    async def test_session_timeline(self, tracker: ActivityTracker) -> None:
        session = await tracker.start_session(agent_id="agent-1")
        tc1 = await tracker.log_tool_call(session.id, tool_name="search", result="found")
        d1 = await tracker.record_decision(session.id, title="Use X", rationale="best fit")
        tc2 = await tracker.log_tool_call(session.id, tool_name="apply", result="done")

        timeline = await tracker.get_session_timeline(session.id)
        assert len(timeline) == 3
        # Should be ordered by created_at
        ids = [n.id for n in timeline]
        assert ids == [tc1.id, d1.id, tc2.id]

    @pytest.mark.asyncio
    async def test_decision_chain(self, tracker: ActivityTracker, graph: SynapticGraph) -> None:
        session = await tracker.start_session(agent_id="agent-1")
        decision = await tracker.record_decision(
            session.id, title="Deploy v2", rationale="Ready",
        )
        outcome = await tracker.record_outcome(
            decision.id, title="Success", content="Done", success=True,
        )
        # Add a lesson learned from this outcome
        lesson = await graph.add("Always test first", "Pre-deploy checklist", kind=NodeKind.LESSON)
        await graph.link(lesson.id, outcome.id, kind=EdgeKind.LEARNED_FROM)

        chain = await tracker.get_decision_chain(decision.id)
        chain_ids = [n.id for n, _ in chain]
        assert outcome.id in chain_ids
        assert lesson.id in chain_ids
