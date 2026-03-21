"""Shared fixtures for synaptic-memory tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind, NodeKind


@pytest.fixture
async def backend() -> AsyncGenerator[MemoryBackend]:
    b = MemoryBackend()
    await b.connect()
    yield b
    await b.close()


@pytest.fixture
async def graph(backend: MemoryBackend) -> SynapticGraph:
    return SynapticGraph(backend)


@pytest.fixture
async def populated_graph(graph: SynapticGraph) -> SynapticGraph:
    """Graph with sample nodes and edges for search/traversal tests."""
    n1 = await graph.add(
        "배포 자동화",
        "CI/CD 파이프라인으로 배포 자동화 구현",
        kind=NodeKind.LESSON,
        tags=["deploy", "ci/cd"],
    )
    n2 = await graph.add(
        "테스트 커버리지",
        "테스트 커버리지 80% 이상 유지 필요",
        kind=NodeKind.RULE,
        tags=["test", "quality"],
    )
    n3 = await graph.add(
        "API 설계 원칙",
        "REST API는 리소스 중심으로 설계",
        kind=NodeKind.DECISION,
        tags=["api", "design"],
    )
    n4 = await graph.add(
        "성능 최적화",
        "데이터베이스 쿼리 N+1 문제 해결",
        kind=NodeKind.LESSON,
        tags=["performance", "database"],
    )
    n5 = await graph.add(
        "보안 점검",
        "OWASP Top 10 기반 보안 점검 필수",
        kind=NodeKind.RULE,
        tags=["security"],
    )

    await graph.link(n1.id, n2.id, kind=EdgeKind.RELATED)
    await graph.link(n2.id, n3.id, kind=EdgeKind.DEPENDS_ON)
    await graph.link(n3.id, n4.id, kind=EdgeKind.RELATED)
    await graph.link(n4.id, n5.id, kind=EdgeKind.LEARNED_FROM)

    return graph
