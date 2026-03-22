"""벤치마크 fixtures — 엔터프라이즈 시나리오 인덱싱 + 에이전트 활동 시뮬레이션."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from synaptic.activity import ActivityTracker
from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind, NodeKind

DATA_DIR = Path(__file__).parent / "data"

# NodeKind 매핑
_KIND_MAP: dict[str, NodeKind] = {
    "CONCEPT": NodeKind.CONCEPT,
    "ENTITY": NodeKind.ENTITY,
    "LESSON": NodeKind.LESSON,
    "DECISION": NodeKind.DECISION,
    "RULE": NodeKind.RULE,
    "ARTIFACT": NodeKind.ARTIFACT,
}

_EDGE_MAP: dict[str, EdgeKind] = {
    "RELATED": EdgeKind.RELATED,
    "DEPENDS_ON": EdgeKind.DEPENDS_ON,
    "LEARNED_FROM": EdgeKind.LEARNED_FROM,
    "CAUSED": EdgeKind.CAUSED,
    "PRODUCED": EdgeKind.PRODUCED,
}


def _load_scenario() -> dict:
    path = DATA_DIR / "enterprise_scenario.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
async def enterprise_graph() -> AsyncGenerator[tuple[SynapticGraph, dict[str, str]]]:
    """Phase 1+2: 엔터프라이즈 지식 인덱싱 + 에이전트 활동 시뮬레이션.

    Returns:
        (graph, id_map) — id_map은 시나리오 ID → 실제 노드 ID 매핑.
    """
    scenario = _load_scenario()

    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend)
    tracker = ActivityTracker(graph)

    id_map: dict[str, str] = {}

    # ── Phase 1: 엔터프라이즈 지식 인덱싱 ──
    for doc in scenario["knowledge_sources"]:
        kind = _KIND_MAP.get(doc["kind"], NodeKind.CONCEPT)
        node = await graph.add(
            title=doc["title"],
            content=doc["content"],
            kind=kind,
            tags=doc.get("tags", []),
            source=doc.get("source", ""),
            properties=doc.get("properties"),
        )
        id_map[doc["id"]] = node.id

    # 지식 간 관계 연결
    for link in scenario["knowledge_links"]:
        src = id_map.get(link["source"])
        tgt = id_map.get(link["target"])
        if src and tgt:
            edge_kind = _EDGE_MAP.get(link["kind"], EdgeKind.RELATED)
            await graph.link(src, tgt, kind=edge_kind)

    # ── Phase 2: 에이전트 활동 시뮬레이션 ──
    for session_data in scenario["agent_sessions"]:
        session = await tracker.start_session(
            agent_id=session_data["agent_id"],
            description=session_data["description"],
        )

        # Tool calls
        for tc in session_data["tool_calls"]:
            await tracker.log_tool_call(
                session.id,
                tool_name=tc["tool"],
                parameters=tc.get("params"),
                result=tc.get("result", ""),
                success=tc.get("success", True),
                duration_ms=tc.get("duration_ms", 0.0),
            )

        # Decisions + Outcomes
        for dec_data in session_data.get("decisions", []):
            decision = await tracker.record_decision(
                session.id,
                title=dec_data["title"],
                rationale=dec_data["rationale"],
                alternatives=dec_data.get("alternatives"),
            )

            if "outcome" in dec_data:
                out = dec_data["outcome"]
                await tracker.record_outcome(
                    decision.id,
                    title=out["title"],
                    content=out["content"],
                    success=out["success"],
                )

        # 접근한 지식 노드에 대한 Hebbian 강화
        accessed = session_data.get("knowledge_accessed", [])
        accessed_ids = [id_map[a] for a in accessed if a in id_map]
        if accessed_ids:
            await graph.reinforce(accessed_ids, success=True)

        await tracker.end_session(session.id)

    # ── Phase 2.5: 실패 세션의 관련 노드 약화 ──
    for session_data in scenario["agent_sessions"]:
        for dec_data in session_data.get("decisions", []):
            if "outcome" in dec_data and not dec_data["outcome"]["success"]:
                accessed = session_data.get("knowledge_accessed", [])
                accessed_ids = [id_map[a] for a in accessed if a in id_map]
                if accessed_ids:
                    await graph.reinforce(accessed_ids, success=False)

    yield graph, id_map
    await backend.close()
