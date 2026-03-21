"""Tests for NodeCache."""

from synaptic.cache import NodeCache
from synaptic.models import Node


class TestNodeCache:
    def test_put_and_get(self) -> None:
        cache = NodeCache(maxsize=10)
        node = Node(id="abc", title="Test")
        cache.put(node)
        assert cache.get("abc") is not None
        assert cache.get("abc").title == "Test"  # type: ignore[union-attr]

    def test_miss(self) -> None:
        cache = NodeCache(maxsize=10)
        assert cache.get("missing") is None

    def test_lru_eviction(self) -> None:
        cache = NodeCache(maxsize=2)
        cache.put(Node(id="a", title="A"))
        cache.put(Node(id="b", title="B"))
        cache.put(Node(id="c", title="C"))
        # "a" should be evicted (oldest)
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_access_refreshes_lru(self) -> None:
        cache = NodeCache(maxsize=2)
        cache.put(Node(id="a", title="A"))
        cache.put(Node(id="b", title="B"))
        cache.get("a")  # Refresh "a"
        cache.put(Node(id="c", title="C"))
        # "b" should be evicted (least recently used)
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_invalidate(self) -> None:
        cache = NodeCache(maxsize=10)
        cache.put(Node(id="a", title="A"))
        cache.invalidate("a")
        assert cache.get("a") is None

    def test_clear(self) -> None:
        cache = NodeCache(maxsize=10)
        cache.put(Node(id="a", title="A"))
        cache.put(Node(id="b", title="B"))
        cache.clear()
        assert cache.size == 0

    def test_hit_rate(self) -> None:
        cache = NodeCache(maxsize=10)
        cache.put(Node(id="a", title="A"))
        cache.get("a")  # Hit
        cache.get("b")  # Miss
        assert cache.hit_rate == 0.5

    def test_stats(self) -> None:
        cache = NodeCache(maxsize=10)
        cache.put(Node(id="a", title="A"))
        cache.get("a")
        cache.get("missing")
        s = cache.stats()
        assert s["size"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5
