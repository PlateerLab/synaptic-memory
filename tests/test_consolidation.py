"""Tests for memory consolidation cascade."""

from __future__ import annotations

from time import time

from synaptic.backends.memory import MemoryBackend
from synaptic.consolidation import (
    L0_TTL_HOURS,
    L1_PROMOTION_ACCESS,
    L3_PROMOTION_SUCCESS,
    ConsolidationCascade,
)
from synaptic.models import ConsolidationLevel, DigestResult, Node, NodeKind


class TestConsolidationCascade:
    async def test_l0_promotion_on_access(self, backend: MemoryBackend) -> None:
        cascade = ConsolidationCascade()
        node = Node(
            title="Promoted",
            level=ConsolidationLevel.L0_RAW,
            access_count=L1_PROMOTION_ACCESS,
        )
        await backend.save_node(node)

        result = await cascade.consolidate(backend)
        updated = await backend.get_node(node.id)
        assert updated is not None
        assert updated.level == ConsolidationLevel.L1_SPRINT
        assert node.id in result.nodes_updated

    async def test_l0_ttl_expiry(self, backend: MemoryBackend) -> None:
        cascade = ConsolidationCascade()
        old_time = time() - (L0_TTL_HOURS + 1) * 3600
        node = Node(
            title="Expired",
            level=ConsolidationLevel.L0_RAW,
            access_count=0,
            created_at=old_time,
            updated_at=old_time,
        )
        await backend.save_node(node)

        await cascade.consolidate(backend)
        assert await backend.get_node(node.id) is None

    async def test_l0_survives_within_ttl(self, backend: MemoryBackend) -> None:
        cascade = ConsolidationCascade()
        node = Node(
            title="Fresh",
            level=ConsolidationLevel.L0_RAW,
            access_count=0,
        )
        await backend.save_node(node)

        await cascade.consolidate(backend)
        assert await backend.get_node(node.id) is not None

    async def test_l3_promotion_on_success_rate(self, backend: MemoryBackend) -> None:
        cascade = ConsolidationCascade()
        old_time = time() - 400 * 86400  # Over L2 TTL
        node = Node(
            title="Proven Knowledge",
            level=ConsolidationLevel.L2_MONTHLY,
            access_count=20,
            success_count=L3_PROMOTION_SUCCESS,
            failure_count=1,
            created_at=old_time,
            updated_at=old_time,
        )
        await backend.save_node(node)

        await cascade.consolidate(backend)
        updated = await backend.get_node(node.id)
        assert updated is not None
        assert updated.level == ConsolidationLevel.L3_PERMANENT

    async def test_l3_never_expires(self, backend: MemoryBackend) -> None:
        cascade = ConsolidationCascade()
        very_old = time() - 10000 * 86400
        node = Node(
            title="Permanent",
            level=ConsolidationLevel.L3_PERMANENT,
            created_at=very_old,
            updated_at=very_old,
        )
        await backend.save_node(node)

        await cascade.consolidate(backend)
        assert await backend.get_node(node.id) is not None

    async def test_l3_demotion_on_low_success_rate(self, backend: MemoryBackend) -> None:
        """L3 node with success rate below 60% should be demoted to L2."""
        cascade = ConsolidationCascade()
        node = Node(
            title="Degraded Knowledge",
            level=ConsolidationLevel.L3_PERMANENT,
            access_count=30,
            success_count=L3_PROMOTION_SUCCESS,  # 10
            failure_count=20,  # rate = 10/30 = 33% < 60%
        )
        await backend.save_node(node)

        await cascade.consolidate(backend)
        updated = await backend.get_node(node.id)
        assert updated is not None
        assert updated.level == ConsolidationLevel.L2_MONTHLY

    async def test_l3_no_demotion_when_rate_ok(self, backend: MemoryBackend) -> None:
        """L3 node with acceptable success rate stays at L3."""
        cascade = ConsolidationCascade()
        node = Node(
            title="Still Valid",
            level=ConsolidationLevel.L3_PERMANENT,
            access_count=20,
            success_count=15,
            failure_count=5,  # rate = 15/20 = 75% > 60%
        )
        await backend.save_node(node)

        await cascade.consolidate(backend)
        updated = await backend.get_node(node.id)
        assert updated is not None
        assert updated.level == ConsolidationLevel.L3_PERMANENT

    async def test_consolidate_with_digester(self, backend: MemoryBackend) -> None:
        cascade = ConsolidationCascade()

        class MockDigester:
            async def digest(self, context: dict[str, object]) -> DigestResult:
                new_node = Node(title="Digested", kind=NodeKind.LESSON)
                return DigestResult(nodes_created=[new_node])

        result = await cascade.consolidate(backend, MockDigester())
        assert len(result.nodes_created) == 1
        assert result.nodes_created[0].title == "Digested"
        # Node should be persisted
        saved = await backend.get_node(result.nodes_created[0].id)
        assert saved is not None
