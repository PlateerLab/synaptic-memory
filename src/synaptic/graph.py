"""SynapticGraph — main entry point (facade)."""

from __future__ import annotations

from difflib import SequenceMatcher
from time import time

from synaptic.cache import NodeCache
from synaptic.consolidation import ConsolidationCascade
from synaptic.exporter import JSONExporter, MarkdownExporter
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
        "_cache",
        "_consolidation",
        "_hebbian",
        "_json_exporter",
        "_md_exporter",
        "_search",
        "_store",
    )

    def __init__(
        self,
        backend: StorageBackend,
        *,
        query_rewriter: QueryRewriter | None = None,
        tag_extractor: TagExtractor | None = None,
        cache_size: int = 256,
    ) -> None:
        self._backend = backend
        self._store = Store(backend, tag_extractor=tag_extractor)
        self._search = HybridSearch(query_rewriter=query_rewriter)
        self._hebbian = HebbianEngine()
        self._consolidation = ConsolidationCascade()
        self._md_exporter = MarkdownExporter()
        self._json_exporter = JSONExporter()
        self._cache = NodeCache(maxsize=cache_size)

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    @property
    def cache(self) -> NodeCache:
        return self._cache

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
        node = await self._store.add_node(
            title, content, kind=kind, tags=tags, source=source, embedding=embedding
        )
        self._cache.put(node)
        return node

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
        cached = self._cache.get(node_id)
        if cached is not None:
            return cached
        node = await self._store.get_node(node_id)
        if node is not None:
            self._cache.put(node)
        return node

    async def remove(self, node_id: str) -> bool:
        node = await self._backend.get_node(node_id)
        if node is None:
            return False
        await self._store.delete_node(node_id)
        self._cache.invalidate(node_id)
        return True

    async def reinforce(self, node_ids: list[str], *, success: bool = True) -> None:
        await self._hebbian.reinforce(self._backend, node_ids, success=success)
        # Invalidate cached nodes (counts changed)
        for nid in node_ids:
            self._cache.invalidate(nid)

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
        self._cache.clear()  # Vitality changed globally
        return await self._backend.decay_vitality(factor=0.95)

    async def export_markdown(self, *, node_ids: list[str] | None = None) -> str:
        return await self._md_exporter.export(self._backend, node_ids=node_ids)

    async def export_json(self, *, node_ids: list[str] | None = None) -> str:
        return await self._json_exporter.export(self._backend, node_ids=node_ids)

    async def merge(
        self,
        source_id: str,
        target_id: str,
    ) -> Node | None:
        """Merge source node into target. Combines content, stats, edges.

        Source node is deleted after merge.
        Returns the updated target node, or None if either node is missing.
        """
        source = await self._backend.get_node(source_id)
        target = await self._backend.get_node(target_id)
        if source is None or target is None:
            return None

        # Merge content
        if source.content and source.content not in target.content:
            target.content = f"{target.content}\n\n{source.content}".strip()

        # Merge tags (deduplicate)
        merged_tags = list(dict.fromkeys([*target.tags, *source.tags]))
        target.tags = merged_tags

        # Merge stats
        target.access_count += source.access_count
        target.success_count += source.success_count
        target.failure_count += source.failure_count
        target.vitality = max(target.vitality, source.vitality)
        target.updated_at = time()

        # Re-point source's edges to target
        source_edges = await self._backend.get_edges(source_id)
        for edge in source_edges:
            new_src = target_id if edge.source_id == source_id else edge.source_id
            new_tgt = target_id if edge.target_id == source_id else edge.target_id
            if new_src != new_tgt:  # Avoid self-loops
                new_edge = Edge(
                    source_id=new_src,
                    target_id=new_tgt,
                    kind=edge.kind,
                    weight=edge.weight,
                )
                try:
                    await self._backend.save_edge(new_edge)
                except Exception:  # noqa: S110
                    pass  # Duplicate edge — skip

        await self._backend.update_node(target)
        await self._backend.delete_node(source_id)
        self._cache.invalidate(source_id)
        self._cache.invalidate(target_id)
        return target

    async def find_duplicates(
        self,
        *,
        threshold: float = 0.85,
        limit: int = 50,
    ) -> list[tuple[Node, Node, float]]:
        """Find potential duplicate node pairs based on title similarity.

        Returns list of (node_a, node_b, similarity_score) tuples.
        """
        nodes = await self._backend.list_nodes(limit=limit * 10)
        duplicates: list[tuple[Node, Node, float]] = []

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if nodes[i].kind != nodes[j].kind:
                    continue
                sim = SequenceMatcher(None, nodes[i].title.lower(), nodes[j].title.lower()).ratio()
                if sim >= threshold:
                    duplicates.append((nodes[i], nodes[j], sim))

        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates[:limit]

    async def stats(self) -> dict[str, int | float]:
        all_nodes = await self._backend.list_nodes(limit=10000)
        by_kind: dict[str, int] = {}
        by_level: dict[str, int] = {}
        for node in all_nodes:
            by_kind[str(node.kind)] = by_kind.get(str(node.kind), 0) + 1
            by_level[str(node.level)] = by_level.get(str(node.level), 0) + 1

        result: dict[str, int | float] = {"total_nodes": len(all_nodes)}
        for k, v in sorted(by_kind.items()):
            result[f"kind_{k}"] = v
        for k, v in sorted(by_level.items()):
            result[f"level_{k}"] = v

        cache_stats = self._cache.stats()
        result["cache_hit_rate"] = cache_stats["hit_rate"]
        result["cache_size"] = cache_stats["size"]
        return result
