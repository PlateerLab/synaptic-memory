"""CRUD orchestration layer over StorageBackend."""

from __future__ import annotations

from collections.abc import Sequence
from time import time
from typing import Literal

from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
)
from synaptic.protocols import StorageBackend, TagExtractor


class Store:
    """High-level CRUD operations with tag extraction and vitality tracking."""

    __slots__ = ("_backend", "_tag_extractor")

    def __init__(
        self,
        backend: StorageBackend,
        *,
        tag_extractor: TagExtractor | None = None,
    ) -> None:
        self._backend = backend
        self._tag_extractor = tag_extractor

    async def add_node(
        self,
        title: str,
        content: str,
        *,
        kind: NodeKind = NodeKind.CONCEPT,
        tags: list[str] | None = None,
        source: str = "",
        level: ConsolidationLevel = ConsolidationLevel.L0_RAW,
        embedding: list[float] | None = None,
    ) -> Node:
        if tags is None and self._tag_extractor is not None:
            tags = self._tag_extractor.extract(f"{title} {content}")

        node = Node(
            kind=kind,
            title=title,
            content=content,
            tags=tags or [],
            level=level,
            embedding=embedding or [],
            source=source,
        )
        await self._backend.save_node(node)
        return node

    async def get_node(self, node_id: str) -> Node | None:
        node = await self._backend.get_node(node_id)
        if node is not None:
            node.access_count += 1
            node.updated_at = time()
            await self._backend.update_node(node)
        return node

    async def update_node(self, node: Node) -> None:
        node.updated_at = time()
        await self._backend.update_node(node)

    async def delete_node(self, node_id: str) -> None:
        await self._backend.delete_node(node_id)

    async def list_nodes(
        self,
        *,
        kind: NodeKind | None = None,
        level: ConsolidationLevel | None = None,
        limit: int = 100,
    ) -> list[Node]:
        return await self._backend.list_nodes(kind=kind, level=level, limit=limit)

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        kind: EdgeKind = EdgeKind.RELATED,
        weight: float = 1.0,
    ) -> Edge:
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            kind=kind,
            weight=weight,
        )
        await self._backend.save_edge(edge)
        return edge

    async def get_edges(
        self, node_id: str, *, direction: Literal["both", "incoming", "outgoing"] = "both"
    ) -> list[Edge]:
        return await self._backend.get_edges(node_id, direction=direction)

    async def add_nodes_batch(self, nodes: Sequence[Node]) -> None:
        await self._backend.save_nodes_batch(nodes)

    async def add_edges_batch(self, edges: Sequence[Edge]) -> None:
        await self._backend.save_edges_batch(edges)
