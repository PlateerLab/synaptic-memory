"""Agent activity tracker — records sessions, tool calls, decisions, outcomes."""

from __future__ import annotations

import json
from time import time
from typing import TYPE_CHECKING

from synaptic.models import (
    Edge,
    EdgeKind,
    Node,
    NodeKind,
)

if TYPE_CHECKING:
    from synaptic.graph import SynapticGraph


class ActivityTracker:
    """Records agent activities as nodes/edges in the knowledge graph.

    Uses only the SynapticGraph public API — no internal coupling.
    """

    __slots__ = ("_graph", "_prev_activity")

    def __init__(self, graph: SynapticGraph) -> None:
        self._graph = graph
        # Track last activity per session for FOLLOWED_BY chains
        self._prev_activity: dict[str, str] = {}

    async def start_session(
        self,
        *,
        agent_id: str = "",
        description: str = "",
        metadata: dict[str, str] | None = None,
    ) -> Node:
        """Create a Session node."""
        props: dict[str, str] = {
            "agent_id": agent_id,
            "start_time": str(time()),
            "status": "active",
        }
        if metadata:
            props.update(metadata)

        node = await self._graph.add(
            title=f"Session: {description or agent_id or 'unnamed'}",
            content=description,
            kind=NodeKind.SESSION,
            tags=["session", agent_id] if agent_id else ["session"],
            properties=props,
        )
        return node

    async def end_session(
        self,
        session_id: str,
        *,
        outcome: str = "",
    ) -> None:
        """Close a session, record end time and outcome."""
        node = await self._graph.get(session_id)
        if node is None:
            return
        node.properties["end_time"] = str(time())
        node.properties["status"] = "completed"
        if outcome:
            node.properties["outcome"] = outcome
        await self._graph.backend.update_node(node)
        self._prev_activity.pop(session_id, None)

    async def log_tool_call(
        self,
        session_id: str,
        *,
        tool_name: str,
        parameters: dict[str, object] | None = None,
        result: str = "",
        success: bool = True,
        duration_ms: float = 0.0,
    ) -> Node:
        """Record a tool invocation. Links to session via PART_OF."""
        props: dict[str, str] = {
            "tool_name": tool_name,
            "success": str(success).lower(),
            "duration_ms": str(duration_ms),
        }
        if parameters:
            props["parameters"] = json.dumps(parameters, default=str)
        if result:
            props["result_summary"] = result[:1000]

        node = await self._graph.add(
            title=f"Tool: {tool_name}",
            content=result[:2000] if result else "",
            kind=NodeKind.TOOL_CALL,
            tags=["tool_call", tool_name],
            properties=props,
        )
        await self._link_to_session(node.id, session_id)
        return node

    async def record_decision(
        self,
        session_id: str,
        *,
        title: str,
        rationale: str,
        alternatives: list[str] | None = None,
        context_node_ids: list[str] | None = None,
    ) -> Node:
        """Record a decision with rationale."""
        props: dict[str, str] = {
            "rationale": rationale,
        }
        if alternatives:
            props["alternatives"] = json.dumps(alternatives)

        node = await self._graph.add(
            title=title,
            content=rationale,
            kind=NodeKind.DECISION,
            tags=["decision"],
            properties=props,
        )

        await self._link_to_session(node.id, session_id)

        # Link to context nodes
        if context_node_ids:
            for ctx_id in context_node_ids:
                await self._graph.link(
                    node.id,
                    ctx_id,
                    kind=EdgeKind.DEPENDS_ON,
                    weight=0.8,
                )

        return node

    async def record_observation(
        self,
        session_id: str,
        *,
        title: str,
        content: str,
        source_node_id: str = "",
    ) -> Node:
        """Record an observation from tool output or environment."""
        node = await self._graph.add(
            title=title,
            content=content,
            kind=NodeKind.OBSERVATION,
            tags=["observation"],
        )
        await self._link_to_session(node.id, session_id)

        if source_node_id:
            await self._graph.link(
                source_node_id,
                node.id,
                kind=EdgeKind.PRODUCED,
                weight=0.7,
            )

        return node

    async def record_outcome(
        self,
        decision_id: str,
        *,
        title: str,
        content: str,
        success: bool,
    ) -> Node:
        """Link an outcome to a prior decision. Triggers Hebbian reinforcement."""
        props: dict[str, str] = {
            "success": str(success).lower(),
        }

        node = await self._graph.add(
            title=title,
            content=content,
            kind=NodeKind.OUTCOME,
            tags=["outcome", "success" if success else "failure"],
            properties=props,
        )

        # Link decision → outcome
        await self._graph.link(
            decision_id,
            node.id,
            kind=EdgeKind.RESULTED_IN,
            weight=1.0,
        )

        # Hebbian reinforcement on the decision and outcome
        await self._graph.reinforce([decision_id, node.id], success=success)

        return node

    async def get_session_timeline(self, session_id: str) -> list[Node]:
        """Return all activity nodes in a session, ordered by created_at."""
        edges = await self._graph.backend.get_edges(session_id, direction="incoming")
        part_of_edges = [e for e in edges if e.kind == EdgeKind.PART_OF]

        nodes: list[Node] = []
        for edge in part_of_edges:
            node = await self._graph.backend.get_node(edge.source_id)
            if node is not None:
                nodes.append(node)

        nodes.sort(key=lambda n: n.created_at)
        return nodes

    async def get_decision_chain(self, decision_id: str) -> list[tuple[Node, Edge]]:
        """Follow decision → outcome → lesson chain."""
        chain: list[tuple[Node, Edge]] = []

        decision = await self._graph.backend.get_node(decision_id)
        if decision is None:
            return chain

        # Find outcomes
        edges = await self._graph.backend.get_edges(decision_id, direction="outgoing")
        for edge in edges:
            if edge.kind == EdgeKind.RESULTED_IN:
                outcome = await self._graph.backend.get_node(edge.target_id)
                if outcome is not None:
                    chain.append((outcome, edge))
                    # Find lessons learned from outcomes
                    lesson_edges = await self._graph.backend.get_edges(
                        outcome.id, direction="incoming"
                    )
                    for le in lesson_edges:
                        if le.kind == EdgeKind.LEARNED_FROM:
                            lesson = await self._graph.backend.get_node(le.source_id)
                            if lesson is not None:
                                chain.append((lesson, le))

        return chain

    # --- Internal helpers ---

    async def _link_to_session(self, node_id: str, session_id: str) -> None:
        """Link activity node to session via PART_OF + FOLLOWED_BY chain."""
        # PART_OF → session
        await self._graph.link(
            node_id,
            session_id,
            kind=EdgeKind.PART_OF,
            weight=1.0,
        )
        # FOLLOWED_BY chain (temporal ordering)
        prev_id = self._prev_activity.get(session_id)
        if prev_id:
            await self._graph.link(
                prev_id,
                node_id,
                kind=EdgeKind.FOLLOWED_BY,
                weight=1.0,
            )
        self._prev_activity[session_id] = node_id
