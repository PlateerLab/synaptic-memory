"""Per-search read-through cache for graph traversal.

Graph expansion and PPR are intentionally read-heavy: they walk from the same
seed nodes through several structural paths. This helper keeps those reads
inside one search call so each edge/node lookup is paid for once without
changing backend semantics or persisting any cache state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from synaptic.models import Edge, Node

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

Direction = Literal["both", "incoming", "outgoing"]


class GraphReadCache:
    """Small async read-through cache scoped to one graph traversal/search."""

    __slots__ = ("_backend", "_edge_cache", "_neighbor_cache", "_node_cache")

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend
        self._edge_cache: dict[tuple[str, Direction], list[Edge]] = {}
        self._neighbor_cache: dict[tuple[str, int], list[tuple[Node, Edge]]] = {}
        self._node_cache: dict[str, Node | None] = {}

    async def get_node(self, node_id: str) -> Node | None:
        if node_id not in self._node_cache:
            self._node_cache[node_id] = await self._backend.get_node(node_id)
        return self._node_cache[node_id]

    async def get_nodes(self, node_ids: list[str]) -> dict[str, Node]:
        """Return found nodes keyed by id, batch-loading misses when possible."""
        if not node_ids:
            return {}

        missing = [nid for nid in dict.fromkeys(node_ids) if nid not in self._node_cache]
        if missing:
            get_batch = getattr(self._backend, "get_nodes_batch", None)
            if callable(get_batch):
                nodes = await get_batch(missing)
                found = {node.id: node for node in nodes}
                for nid in missing:
                    self._node_cache[nid] = found.get(nid)
            else:
                for nid in missing:
                    self._node_cache[nid] = await self._backend.get_node(nid)

        return {
            nid: node
            for nid in dict.fromkeys(node_ids)
            if (node := self._node_cache.get(nid)) is not None
        }

    async def get_edges(self, node_id: str, *, direction: Direction = "both") -> list[Edge]:
        key = (node_id, direction)
        cached = self._edge_cache.get(key)
        if cached is None:
            if direction != "both":
                both = self._edge_cache.get((node_id, "both"))
                if both is not None:
                    cached = _filter_edges(node_id, both, direction)
                    self._edge_cache[key] = cached
                    return cached
            else:
                outgoing = self._edge_cache.get((node_id, "outgoing"))
                incoming = self._edge_cache.get((node_id, "incoming"))
                if outgoing is not None and incoming is not None:
                    cached = _merge_edges(outgoing, incoming)
                    self._edge_cache[key] = cached
                    return cached
            cached = await self._backend.get_edges(node_id, direction=direction)
            self._edge_cache[key] = cached
        return cached

    async def get_neighbors(self, node_id: str, *, depth: int = 1) -> list[tuple[Node, Edge]]:
        key = (node_id, depth)
        cached = self._neighbor_cache.get(key)
        if cached is None:
            cached = await self._backend.get_neighbors(node_id, depth=depth)
            self._neighbor_cache[key] = cached
            for node, _edge in cached:
                self._node_cache.setdefault(node.id, node)
        return cached


def _filter_edges(node_id: str, edges: list[Edge], direction: Direction) -> list[Edge]:
    if direction == "outgoing":
        return [edge for edge in edges if edge.source_id == node_id]
    if direction == "incoming":
        return [edge for edge in edges if edge.target_id == node_id]
    return edges


def _merge_edges(first: list[Edge], second: list[Edge]) -> list[Edge]:
    merged: list[Edge] = []
    seen: set[str] = set()
    for edge in first + second:
        if edge.id in seen:
            continue
        seen.add(edge.id)
        merged.append(edge)
    return merged
