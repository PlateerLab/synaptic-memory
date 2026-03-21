"""Memory consolidation cascade: L0 → L1 → L2 → L3."""

from __future__ import annotations

from time import time

from synaptic.models import ConsolidationLevel, DigestResult, Node
from synaptic.protocols import Digester, StorageBackend

L0_TTL_HOURS = 72
L1_PROMOTION_ACCESS = 3
L1_TTL_DAYS = 90
L2_PROMOTION_ACCESS = 10
L2_TTL_DAYS = 365
L3_PROMOTION_SUCCESS = 10
L3_PROMOTION_RATE = 0.8


class ConsolidationCascade:
    """Manages memory lifecycle: TTL expiry + promotion based on usage."""

    __slots__ = (
        "l0_ttl_hours",
        "l1_promotion_access",
        "l1_ttl_days",
        "l2_promotion_access",
        "l2_ttl_days",
        "l3_promotion_rate",
        "l3_promotion_success",
    )

    def __init__(
        self,
        *,
        l0_ttl_hours: float = L0_TTL_HOURS,
        l1_promotion_access: int = L1_PROMOTION_ACCESS,
        l1_ttl_days: float = L1_TTL_DAYS,
        l2_promotion_access: int = L2_PROMOTION_ACCESS,
        l2_ttl_days: float = L2_TTL_DAYS,
        l3_promotion_success: int = L3_PROMOTION_SUCCESS,
        l3_promotion_rate: float = L3_PROMOTION_RATE,
    ) -> None:
        self.l0_ttl_hours = l0_ttl_hours
        self.l1_promotion_access = l1_promotion_access
        self.l1_ttl_days = l1_ttl_days
        self.l2_promotion_access = l2_promotion_access
        self.l2_ttl_days = l2_ttl_days
        self.l3_promotion_success = l3_promotion_success
        self.l3_promotion_rate = l3_promotion_rate

    async def consolidate(
        self,
        backend: StorageBackend,
        digester: Digester | None = None,
        *,
        context: dict[str, object] | None = None,
    ) -> DigestResult:
        result = DigestResult()
        now = time()

        # 1. Run digester if provided (creates new nodes/edges)
        if digester is not None:
            result = await digester.digest(context or {})
            for node in result.nodes_created:
                await backend.save_node(node)
            for edge in result.edges_created:
                await backend.save_edge(edge)

        # 2. Process existing nodes: TTL expiry + level promotion
        levels = (
            ConsolidationLevel.L0_RAW,
            ConsolidationLevel.L1_SPRINT,
            ConsolidationLevel.L2_MONTHLY,
            ConsolidationLevel.L3_PERMANENT,
        )
        page_size = 500
        for level in levels:
            # Paginate to handle large node counts
            while True:
                nodes = await backend.list_nodes(level=level, limit=page_size)
                if not nodes:
                    break
                processed = 0
                for node in nodes:
                    action = self._evaluate(node, now=now)
                    match action:
                        case "delete":
                            await backend.delete_node(node.id)
                            processed += 1
                        case "promote":
                            node.level = self._next_level(node.level)
                            node.updated_at = now
                            await backend.update_node(node)
                            result.nodes_updated.append(node.id)
                            processed += 1
                        case "demote":
                            node.level = self._prev_level(node.level)
                            node.updated_at = now
                            await backend.update_node(node)
                            result.nodes_updated.append(node.id)
                            processed += 1
                        case _:
                            pass
                # If nothing was modified in this batch, no point re-fetching
                if processed == 0:
                    break

        return result

    def _evaluate(self, node: Node, *, now: float) -> str:
        age_hours = (now - node.created_at) / 3600

        match node.level:
            case ConsolidationLevel.L0_RAW:
                if age_hours > self.l0_ttl_hours and node.access_count < self.l1_promotion_access:
                    return "delete"
                if node.access_count >= self.l1_promotion_access:
                    return "promote"
            case ConsolidationLevel.L1_SPRINT:
                age_days = age_hours / 24
                if age_days > self.l1_ttl_days and node.access_count < self.l2_promotion_access:
                    return "delete"
                if node.access_count >= self.l2_promotion_access:
                    return "promote"
            case ConsolidationLevel.L2_MONTHLY:
                age_days = age_hours / 24
                if age_days > self.l2_ttl_days and not self._qualifies_for_l3(node):
                    return "delete"
                if self._qualifies_for_l3(node):
                    return "promote"
            case ConsolidationLevel.L3_PERMANENT:
                # L3 demotion: if success rate drops below threshold, demote to L2
                if self._should_demote_l3(node):
                    return "demote"

        return "keep"

    def _should_demote_l3(self, node: Node) -> bool:
        """Demote L3 if it accumulated enough failures to drop below threshold."""
        total = node.success_count + node.failure_count
        if total < self.l3_promotion_success:
            return False  # not enough data to judge
        rate = node.success_count / total if total > 0 else 0.0
        # Demote if rate drops below 60% (stricter threshold than promotion at 80%)
        return rate < self.l3_promotion_rate * 0.75

    def _qualifies_for_l3(self, node: Node) -> bool:
        if node.success_count < self.l3_promotion_success:
            return False
        total = node.success_count + node.failure_count
        if total == 0:
            return False
        return (node.success_count / total) >= self.l3_promotion_rate

    def _next_level(self, level: ConsolidationLevel) -> ConsolidationLevel:
        match level:
            case ConsolidationLevel.L0_RAW:
                return ConsolidationLevel.L1_SPRINT
            case ConsolidationLevel.L1_SPRINT:
                return ConsolidationLevel.L2_MONTHLY
            case ConsolidationLevel.L2_MONTHLY:
                return ConsolidationLevel.L3_PERMANENT
            case _:
                return level

    def _prev_level(self, level: ConsolidationLevel) -> ConsolidationLevel:
        match level:
            case ConsolidationLevel.L3_PERMANENT:
                return ConsolidationLevel.L2_MONTHLY
            case ConsolidationLevel.L2_MONTHLY:
                return ConsolidationLevel.L1_SPRINT
            case ConsolidationLevel.L1_SPRINT:
                return ConsolidationLevel.L0_RAW
            case _:
                return level
