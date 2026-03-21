"""Hebbian learning engine — edge weight reinforcement."""

from __future__ import annotations

from itertools import combinations
from time import time

from synaptic.models import Edge, EdgeKind
from synaptic.protocols import StorageBackend

REINFORCE_DELTA = 0.1
WEAKEN_DELTA = 0.15  # Failure learning is stronger
MAX_WEIGHT = 5.0
MIN_WEIGHT = -2.0  # Anti-resonance


class HebbianEngine:
    """Adjusts edge weights based on co-activation success/failure."""

    __slots__ = ()

    async def reinforce(
        self,
        backend: StorageBackend,
        node_ids: list[str],
        *,
        success: bool,
    ) -> None:
        if len(node_ids) < 1:
            return

        now = time()

        # 1. Update each node's success/failure count
        for nid in node_ids:
            node = await backend.get_node(nid)
            if node is None:
                continue
            node.access_count += 1
            if success:
                node.success_count += 1
            else:
                node.failure_count += 1
            node.updated_at = now
            await backend.update_node(node)

        # 2. Adjust edges between co-activated node pairs
        if len(node_ids) < 2:
            return

        delta = REINFORCE_DELTA if success else -WEAKEN_DELTA

        for src_id, tgt_id in combinations(node_ids, 2):
            edge = await self._find_edge(backend, src_id, tgt_id)
            if edge is not None:
                edge.weight = _clamp(edge.weight + delta)
                await backend.update_edge(edge)
            elif success:
                # Create new edge only on success
                new_edge = Edge(
                    source_id=src_id,
                    target_id=tgt_id,
                    kind=EdgeKind.RELATED,
                    weight=REINFORCE_DELTA,
                    created_at=now,
                )
                await backend.save_edge(new_edge)

    async def _find_edge(
        self,
        backend: StorageBackend,
        src_id: str,
        tgt_id: str,
    ) -> Edge | None:
        edges = await backend.get_edges(src_id, direction="outgoing")
        for e in edges:
            if e.target_id == tgt_id:
                return e
        # Check reverse
        edges = await backend.get_edges(tgt_id, direction="outgoing")
        for e in edges:
            if e.target_id == src_id:
                return e
        return None


def _clamp(value: float) -> float:
    return max(MIN_WEIGHT, min(MAX_WEIGHT, value))
