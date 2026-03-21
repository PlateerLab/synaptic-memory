"""LRU cache layer for frequently accessed nodes."""

from __future__ import annotations

from collections import OrderedDict

from synaptic.models import Node


class NodeCache:
    """Bounded LRU cache for nodes. Thread-safe is NOT guaranteed — async-only.

    Usage:
        cache = NodeCache(maxsize=256)
        cache.put(node)
        node = cache.get("node_id")  # None if miss
        cache.invalidate("node_id")
    """

    __slots__ = ("_cache", "_hits", "_maxsize", "_misses")

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Node] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, node_id: str) -> Node | None:
        if node_id in self._cache:
            self._cache.move_to_end(node_id)
            self._hits += 1
            return self._cache[node_id]
        self._misses += 1
        return None

    def put(self, node: Node) -> None:
        if node.id in self._cache:
            self._cache.move_to_end(node.id)
        self._cache[node.id] = node
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def invalidate(self, node_id: str) -> None:
        self._cache.pop(node_id, None)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, int | float]:
        return {
            "size": self.size,
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }
