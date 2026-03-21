"""SynapticGraph — main entry point (facade)."""

from __future__ import annotations

from synaptic.consolidation import ConsolidationCascade
from synaptic.exporter import MarkdownExporter
from synaptic.hebbian import HebbianEngine
from synaptic.models import (
    DigestResult,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
    SearchResult,
)
from synaptic.protocols import Digester, QueryRewriter, StorageBackend, TagExtractor
from synaptic.search import HybridSearch
from synaptic.store import Store


class SynapticGraph:
    """Facade over the synaptic memory system."""

    __slots__ = (
        "_backend",
        "_consolidation",
        "_exporter",
        "_hebbian",
        "_search",
        "_store",
    )

    def __init__(
        self,
        backend: StorageBackend,
        *,
        query_rewriter: QueryRewriter | None = None,
        tag_extractor: TagExtractor | None = None,
    ) -> None:
        self._backend = backend
        self._store = Store(backend, tag_extractor=tag_extractor)
        self._search = HybridSearch(query_rewriter=query_rewriter)
        self._hebbian = HebbianEngine()
        self._consolidation = ConsolidationCascade()
        self._exporter = MarkdownExporter()

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    async def add(
        self,
        title: str,
        content: str,
        *,
        kind: NodeKind = NodeKind.CONCEPT,
        tags: list[str] | None = None,
        source: str = "",
        embedding: list[float] | None = None,
    ) -> Node:
        return await self._store.add_node(
            title, content, kind=kind, tags=tags, source=source, embedding=embedding
        )

    async def link(
        self,
        source_id: str,
        target_id: str,
        *,
        kind: EdgeKind = EdgeKind.RELATED,
        weight: float = 1.0,
    ) -> Edge:
        return await self._store.add_edge(source_id, target_id, kind=kind, weight=weight)

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        embedding: list[float] | None = None,
    ) -> SearchResult:
        return await self._search.search(self._backend, query, limit=limit, embedding=embedding)

    async def get(self, node_id: str) -> Node | None:
        return await self._store.get_node(node_id)

    async def remove(self, node_id: str) -> bool:
        node = await self._backend.get_node(node_id)
        if node is None:
            return False
        await self._store.delete_node(node_id)
        return True

    async def reinforce(self, node_ids: list[str], *, success: bool = True) -> None:
        await self._hebbian.reinforce(self._backend, node_ids, success=success)

    async def consolidate(
        self,
        digester: Digester | None = None,
        *,
        context: dict[str, object] | None = None,
    ) -> DigestResult:
        return await self._consolidation.consolidate(self._backend, digester, context=context)

    async def prune(self) -> int:
        return await self._backend.prune_edges(weight_below=0.1)

    async def decay(self) -> int:
        return await self._backend.decay_vitality(factor=0.95)

    async def export_markdown(self, *, node_ids: list[str] | None = None) -> str:
        return await self._exporter.export(self._backend, node_ids=node_ids)

    async def stats(self) -> dict[str, int]:
        all_nodes = await self._backend.list_nodes(limit=10000)
        by_kind: dict[str, int] = {}
        by_level: dict[str, int] = {}
        for node in all_nodes:
            by_kind[str(node.kind)] = by_kind.get(str(node.kind), 0) + 1
            by_level[str(node.level)] = by_level.get(str(node.level), 0) + 1

        result: dict[str, int] = {"total_nodes": len(all_nodes)}
        for k, v in sorted(by_kind.items()):
            result[f"kind_{k}"] = v
        for k, v in sorted(by_level.items()):
            result[f"level_{k}"] = v
        return result
