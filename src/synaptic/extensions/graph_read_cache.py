"""Per-search read-through cache for graph traversal.

Graph expansion and PPR are intentionally read-heavy: they walk from the same
seed nodes through several structural paths. This helper keeps those reads
inside one search call so each edge/node lookup is paid for once without
changing backend semantics or persisting any cache state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from synaptic.models import Edge, EdgeKind, Node

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

Direction = Literal["both", "incoming", "outgoing"]


class GraphReadCache:
    """Small async read-through cache scoped to one graph traversal/search."""

    __slots__ = (
        "_backend",
        "_edge_cache",
        "_edge_kind_cache",
        "_edge_kind_light_cache",
        "_edge_light_cache",
        "_neighbor_cache",
        "_node_cache",
    )

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend
        self._edge_cache: dict[tuple[str, Direction], list[Edge]] = {}
        self._edge_kind_cache: dict[tuple[str, Direction, tuple[str, ...]], list[Edge]] = {}
        self._edge_kind_light_cache: dict[tuple[str, Direction, tuple[str, ...]], list[Edge]] = {}
        self._edge_light_cache: dict[tuple[str, Direction], list[Edge]] = {}
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
        return (await self.get_edges_many([node_id], direction=direction)).get(node_id, [])

    async def get_edges_many(
        self, node_ids: list[str], *, direction: Direction = "both"
    ) -> dict[str, list[Edge]]:
        """Return edge lists for multiple nodes, using backend batch reads if available."""
        unique_ids = list(dict.fromkeys(node_ids))
        if not unique_ids:
            return {}

        result: dict[str, list[Edge]] = {}
        missing: list[str] = []
        for node_id in unique_ids:
            cached = self._cached_edges(node_id, direction)
            if cached is None:
                missing.append(node_id)
            else:
                result[node_id] = cached

        if missing:
            get_batch = getattr(self._backend, "get_edges_batch", None)
            if callable(get_batch):
                fetched = await get_batch(missing, direction=direction)
                for node_id in missing:
                    edges = list(fetched.get(node_id, []))
                    self._edge_cache[(node_id, direction)] = edges
                    result[node_id] = edges
            else:
                for node_id in missing:
                    edges = await self._backend.get_edges(node_id, direction=direction)
                    self._edge_cache[(node_id, direction)] = edges
                    result[node_id] = edges

        return result

    async def get_edges_many_light(
        self, node_ids: list[str], *, direction: Direction = "both"
    ) -> dict[str, list[Edge]]:
        """Return traversal-only edges, using lightweight backend reads when available.

        PPR only needs source/target/kind/weight. SQLite can skip loading and
        parsing provenance JSON for that path, while callers that need full
        edge metadata continue to use ``get_edges_many``.
        """
        unique_ids = list(dict.fromkeys(node_ids))
        if not unique_ids:
            return {}

        result: dict[str, list[Edge]] = {}
        missing: list[str] = []
        for node_id in unique_ids:
            cached = self._cached_edges_light(node_id, direction)
            if cached is None:
                missing.append(node_id)
            else:
                result[node_id] = cached

        if missing:
            get_light = getattr(self._backend, "get_edges_batch_light", None)
            if callable(get_light):
                fetched = await get_light(missing, direction=direction)
                for node_id in missing:
                    edges = list(fetched.get(node_id, []))
                    self._edge_light_cache[(node_id, direction)] = edges
                    result[node_id] = edges
            else:
                fetched = await self.get_edges_many(missing, direction=direction)
                for node_id in missing:
                    edges = list(fetched.get(node_id, []))
                    self._edge_light_cache[(node_id, direction)] = edges
                    result[node_id] = edges

        return result

    async def get_edges_many_by_kind(
        self,
        node_ids: list[str],
        *,
        direction: Direction = "both",
        kinds: Sequence[EdgeKind | str],
    ) -> dict[str, list[Edge]]:
        """Return edges for multiple nodes, limited to the requested kinds.

        Backends that expose a filtered batch read avoid materializing noisy
        edge kinds just so GraphExpander can discard them. Backends without the
        optional method degrade to the normal full edge read plus in-memory
        filtering, preserving semantics.
        """
        unique_ids = list(dict.fromkeys(node_ids))
        if not unique_ids:
            return {}

        kind_key = _kind_key(kinds)
        if not kind_key:
            return {node_id: [] for node_id in unique_ids}

        result: dict[str, list[Edge]] = {}
        missing: list[str] = []
        for node_id in unique_ids:
            cached = self._cached_edges_by_kind(node_id, direction, kind_key)
            if cached is None:
                missing.append(node_id)
            else:
                result[node_id] = cached

        if missing:
            get_filtered = getattr(self._backend, "get_edges_batch_filtered", None)
            if callable(get_filtered):
                fetched = await get_filtered(missing, direction=direction, kinds=list(kind_key))
                for node_id in missing:
                    edges = _filter_edges_by_kind(fetched.get(node_id, []), kind_key)
                    self._edge_kind_cache[(node_id, direction, kind_key)] = edges
                    result[node_id] = edges
            else:
                fetched = await self.get_edges_many(missing, direction=direction)
                for node_id in missing:
                    edges = _filter_edges_by_kind(fetched.get(node_id, []), kind_key)
                    self._edge_kind_cache[(node_id, direction, kind_key)] = edges
                    result[node_id] = edges

        return result

    async def get_edges_many_by_kind_light(
        self,
        node_ids: list[str],
        *,
        direction: Direction = "both",
        kinds: Sequence[EdgeKind | str],
    ) -> dict[str, list[Edge]]:
        """Return traversal-only edges limited to the requested kinds.

        This mirrors ``get_edges_many_by_kind`` for expansion paths that only
        need source/target/kind/weight. SQLite can skip loading and parsing
        ``properties_json`` for those paths; callers that need provenance
        metadata should keep using ``get_edges_many_by_kind``.
        """
        unique_ids = list(dict.fromkeys(node_ids))
        if not unique_ids:
            return {}

        kind_key = _kind_key(kinds)
        if not kind_key:
            return {node_id: [] for node_id in unique_ids}

        result: dict[str, list[Edge]] = {}
        missing: list[str] = []
        for node_id in unique_ids:
            cached = self._cached_edges_by_kind_light(node_id, direction, kind_key)
            if cached is None:
                missing.append(node_id)
            else:
                result[node_id] = cached

        if missing:
            get_filtered_light = getattr(self._backend, "get_edges_batch_filtered_light", None)
            if callable(get_filtered_light):
                fetched = await get_filtered_light(
                    missing,
                    direction=direction,
                    kinds=list(kind_key),
                )
                for node_id in missing:
                    edges = _filter_edges_by_kind(fetched.get(node_id, []), kind_key)
                    self._edge_kind_light_cache[(node_id, direction, kind_key)] = edges
                    result[node_id] = edges
            else:
                fetched = await self.get_edges_many_by_kind(
                    missing,
                    direction=direction,
                    kinds=kind_key,
                )
                for node_id in missing:
                    edges = list(fetched.get(node_id, []))
                    self._edge_kind_light_cache[(node_id, direction, kind_key)] = edges
                    result[node_id] = edges

        return result

    async def get_edges_many_by_kind_selective_light(
        self,
        node_ids: list[str],
        *,
        direction: Direction = "both",
        light_kinds: Sequence[EdgeKind | str],
        full_kinds: Sequence[EdgeKind | str],
    ) -> dict[str, list[Edge]]:
        """Return mixed metadata edges in one filtered batch read.

        ``light_kinds`` are materialized without properties, while
        ``full_kinds`` keep provenance metadata. This is useful for relation
        expansion where generic RELATED edges only need traversal fields, but
        typed OpenIE edges need ``is_openie`` and ``confidence``.
        """
        unique_ids = list(dict.fromkeys(node_ids))
        if not unique_ids:
            return {}

        light_key = _kind_key(light_kinds)
        full_key = _kind_key(full_kinds)
        if not light_key and not full_key:
            return {node_id: [] for node_id in unique_ids}

        result: dict[str, list[Edge]] = {}
        missing: list[str] = []
        for node_id in unique_ids:
            light_edges = (
                self._cached_edges_by_kind_light(node_id, direction, light_key) if light_key else []
            )
            full_edges = (
                self._cached_edges_by_kind(node_id, direction, full_key) if full_key else []
            )
            if light_edges is None or full_edges is None:
                missing.append(node_id)
            else:
                result[node_id] = _merge_edges(list(light_edges), list(full_edges))

        if missing:
            get_selective = getattr(self._backend, "get_edges_batch_filtered_selective_light", None)
            if callable(get_selective):
                fetched = await get_selective(
                    missing,
                    direction=direction,
                    light_kinds=list(light_key),
                    full_kinds=list(full_key),
                )
            else:
                all_fetched = await self.get_edges_many_by_kind(
                    missing,
                    direction=direction,
                    kinds=[*light_key, *full_key],
                )
                fetched = {
                    node_id: _strip_light_kind_properties(
                        list(all_fetched.get(node_id, [])),
                        light_key,
                    )
                    for node_id in missing
                }

            for node_id in missing:
                edges = list(fetched.get(node_id, []))
                if light_key:
                    self._edge_kind_light_cache[(node_id, direction, light_key)] = (
                        _filter_edges_by_kind(edges, light_key)
                    )
                if full_key:
                    self._edge_kind_cache[(node_id, direction, full_key)] = _filter_edges_by_kind(
                        edges, full_key
                    )
                result[node_id] = edges

        return result

    def _cached_edges(self, node_id: str, direction: Direction) -> list[Edge] | None:
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
        return cached

    def _cached_edges_light(self, node_id: str, direction: Direction) -> list[Edge] | None:
        full = self._cached_edges(node_id, direction)
        if full is not None:
            return full
        cached = self._edge_light_cache.get((node_id, direction))
        if cached is None:
            if direction != "both":
                both = self._edge_light_cache.get((node_id, "both"))
                if both is not None:
                    cached = _filter_edges(node_id, both, direction)
                    self._edge_light_cache[(node_id, direction)] = cached
                    return cached
            else:
                outgoing = self._edge_light_cache.get((node_id, "outgoing"))
                incoming = self._edge_light_cache.get((node_id, "incoming"))
                if outgoing is not None and incoming is not None:
                    cached = _merge_edges(outgoing, incoming)
                    self._edge_light_cache[(node_id, direction)] = cached
                    return cached
        return cached

    def _cached_edges_by_kind(
        self, node_id: str, direction: Direction, kind_key: tuple[str, ...]
    ) -> list[Edge] | None:
        full = self._cached_edges(node_id, direction)
        if full is not None:
            return _filter_edges_by_kind(full, kind_key)
        cached = self._edge_kind_cache.get((node_id, direction, kind_key))
        if cached is not None:
            return cached
        if direction != "both":
            both = self._edge_kind_cache.get((node_id, "both", kind_key))
            if both is not None:
                cached = _filter_edges(node_id, both, direction)
                self._edge_kind_cache[(node_id, direction, kind_key)] = cached
                return cached
        return None

    def _cached_edges_by_kind_light(
        self, node_id: str, direction: Direction, kind_key: tuple[str, ...]
    ) -> list[Edge] | None:
        light = self._cached_edges_light(node_id, direction)
        if light is not None:
            return _filter_edges_by_kind(light, kind_key)
        full = self._cached_edges_by_kind(node_id, direction, kind_key)
        if full is not None:
            return full
        cached = self._edge_kind_light_cache.get((node_id, direction, kind_key))
        if cached is not None:
            return cached
        if direction != "both":
            both = self._edge_kind_light_cache.get((node_id, "both", kind_key))
            if both is not None:
                cached = _filter_edges(node_id, both, direction)
                self._edge_kind_light_cache[(node_id, direction, kind_key)] = cached
                return cached
        return None

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


def _kind_key(kinds: Sequence[EdgeKind | str]) -> tuple[str, ...]:
    return tuple(
        sorted({kind.value if isinstance(kind, EdgeKind) else str(kind) for kind in kinds})
    )


def _filter_edges_by_kind(edges: Sequence[Edge], kind_key: tuple[str, ...]) -> list[Edge]:
    kind_set = set(kind_key)
    return [edge for edge in edges if edge.kind.value in kind_set]


def _strip_light_kind_properties(edges: list[Edge], light_key: tuple[str, ...]) -> list[Edge]:
    if not light_key:
        return edges
    light_set = set(light_key)
    stripped: list[Edge] = []
    for edge in edges:
        if edge.kind.value not in light_set:
            stripped.append(edge)
            continue
        stripped.append(
            Edge(
                id=edge.id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                kind=edge.kind,
                weight=edge.weight,
                properties={},
                created_at=edge.created_at,
            )
        )
    return stripped
