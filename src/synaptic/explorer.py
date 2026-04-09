"""GraphExplorer — interactive graph data exploration API.

Unlike JSONExporter (static dump), GraphExplorer provides interactive
drill-down: node detail, entity context, chunk navigation, edge evidence,
table row inspection, community overview, and in-graph search.

Designed as the backend API for visualization frontends.

Usage::

    graph = SynapticGraph(backend, chunk_entity_index=idx)
    explorer = graph.explorer

    # Graph overview
    data = await explorer.get_graph_data(max_nodes=200)

    # Click entity → see all source chunks
    ctx = await explorer.get_entity_context("entity_042")

    # Click chunk → see extracted entities with positions
    detail = await explorer.get_chunk_detail("chunk_001")
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from synaptic.models import (
    ChunkDetail,
    CommunityDetail,
    EdgeDetail,
    EdgeKind,
    EntityContext,
    GraphData,
    GraphStats,
    Node,
    NodeDetail,
    NodeKind,
    TableRowDetail,
)

if TYPE_CHECKING:
    from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
    from synaptic.protocols import StorageBackend


class GraphExplorer:
    """Interactive graph data exploration — visualization frontend backend.

    Provides drill-down APIs that go beyond simple node/edge listing:
    - Entity → source chunks (원문 확인)
    - Chunk → extracted entities (추출 결과 확인)
    - Edge → evidence chunks (관계 근거 확인)
    - Table row → column data + FK relations
    - Community → summary + members + key entities
    """

    __slots__ = ("_backend", "_chunk_entity_index")

    def __init__(
        self,
        backend: StorageBackend,
        chunk_entity_index: ChunkEntityIndex | None = None,
    ) -> None:
        self._backend = backend
        self._chunk_entity_index = chunk_entity_index

    # --- Graph overview ---

    async def get_graph_data(
        self,
        *,
        root_node_id: str | None = None,
        source: str | None = None,
        max_nodes: int = 500,
        max_depth: int = 3,
        include_chunks: bool = False,
        include_phrases: bool = False,
        node_kinds: list[str] | None = None,
    ) -> GraphData:
        """Graph visualization data. Defaults to ENTITY+RULE+CONCEPT only."""
        all_nodes = await self._backend.list_nodes(limit=max_nodes * 2)

        # Filter nodes
        filtered: list[Node] = []
        for n in all_nodes:
            if not include_chunks and n.kind == NodeKind.CHUNK:
                continue
            if not include_phrases and "_phrase" in (n.tags or []):
                continue
            if n.kind == NodeKind.TYPE_DEF:
                continue
            if node_kinds and str(n.kind) not in node_kinds:
                continue
            if source and n.source != source:
                # Also check parent_doc property
                if (
                    n.properties.get("parent_doc") != source
                    and n.properties.get("_table_name") != source
                ):
                    continue
            filtered.append(n)

        if root_node_id:
            # BFS from root to limit depth
            filtered = await self._bfs_subgraph(root_node_id, max_depth, max_nodes)

        filtered = filtered[:max_nodes]
        node_ids = {n.id for n in filtered}

        # Gather edges between visible nodes
        edge_dicts: list[dict[str, object]] = []
        seen_edges: set[str] = set()
        for n in filtered:
            edges = await self._backend.get_edges(n.id)
            for e in edges:
                if e.id in seen_edges:
                    continue
                other = e.target_id if e.source_id == n.id else e.source_id
                if other in node_ids:
                    seen_edges.add(e.id)
                    edge_dicts.append(
                        {
                            "id": e.id,
                            "source": e.source_id,
                            "target": e.target_id,
                            "kind": str(e.kind),
                            "weight": e.weight,
                        }
                    )

        # Build node dicts
        node_dicts: list[dict[str, object]] = []
        for n in filtered:
            edges = await self._backend.get_edges(n.id)
            node_dicts.append(
                {
                    "id": n.id,
                    "label": n.title,
                    "kind": str(n.kind),
                    "tags": n.tags,
                    "size": len(edges),  # edge count as size
                    "properties": dict(n.properties),
                    "content_preview": n.content[:100] if n.content else "",
                }
            )

        # Community nodes
        comm_dicts: list[dict[str, object]] = []
        for n in filtered:
            if n.kind == NodeKind.COMMUNITY:
                comm_dicts.append(
                    {
                        "id": n.id,
                        "label": n.title,
                        "member_count": int(n.properties.get("member_count", "0")),
                        "summary_preview": n.content[:200] if n.content else "",
                    }
                )

        stats = await self.get_graph_stats(source=source)
        return GraphData(
            nodes=node_dicts,
            edges=edge_dicts,
            communities=comm_dicts,
            stats={
                "total_nodes": stats.total_nodes,
                "total_edges": stats.total_edges,
                "visible_nodes": len(node_dicts),
                "visible_edges": len(edge_dicts),
            },
        )

    # --- Node detail ---

    async def get_node_detail(self, node_id: str) -> NodeDetail | None:
        """Full node detail with neighbors."""
        node = await self._backend.get_node(node_id)
        if node is None:
            return None

        edges = await self._backend.get_edges(node_id)
        neighbors: list[tuple[Node, object]] = []
        for e in edges:
            other_id = e.target_id if e.source_id == node_id else e.source_id
            other = await self._backend.get_node(other_id)
            if other:
                neighbors.append((other, e))

        chunk_count = 0
        if self._chunk_entity_index and node.kind == NodeKind.ENTITY:
            chunk_count = len(self._chunk_entity_index.chunks_for_entity(node_id))

        # Find community membership
        community_id = ""
        for e in edges:
            other_id = e.target_id if e.source_id == node_id else e.source_id
            if e.kind == EdgeKind.PART_OF:
                other = await self._backend.get_node(other_id)
                if other and other.kind == NodeKind.COMMUNITY:
                    community_id = other.id
                    break

        return NodeDetail(
            node=node,
            neighbors=neighbors,
            chunk_count=chunk_count,
            community_id=community_id,
        )

    # --- Entity context ---

    async def get_entity_context(self, entity_id: str) -> EntityContext | None:
        """Entity with all source chunks and related entities."""
        entity = await self._backend.get_node(entity_id)
        if entity is None:
            return None

        # Source chunks
        source_chunks: list[Node] = []
        if self._chunk_entity_index:
            chunk_ids = self._chunk_entity_index.chunks_for_entity(entity_id)
            for cid in chunk_ids:
                chunk = await self._backend.get_node(cid)
                if chunk:
                    source_chunks.append(chunk)
        else:
            # Fallback: scan edges
            edges = await self._backend.get_edges(entity_id, direction="incoming")
            for e in edges:
                if e.kind in (EdgeKind.MENTIONS, EdgeKind.CONTAINS):
                    chunk = await self._backend.get_node(e.source_id)
                    if chunk:
                        source_chunks.append(chunk)

        # Related entities
        related: list[tuple[Node, object]] = []
        edges = await self._backend.get_edges(entity_id)
        for e in edges:
            other_id = e.target_id if e.source_id == entity_id else e.source_id
            other = await self._backend.get_node(other_id)
            if other and other.kind == NodeKind.ENTITY and other.id != entity_id:
                related.append((other, e))

        # Community
        community: dict[str, object] | None = None
        for e in edges:
            if e.kind == EdgeKind.PART_OF:
                other_id = e.target_id if e.source_id == entity_id else e.source_id
                comm = await self._backend.get_node(other_id)
                if comm and comm.kind == NodeKind.COMMUNITY:
                    community = {
                        "id": comm.id,
                        "title": comm.title,
                        "summary": comm.content,
                    }
                    break

        return EntityContext(
            entity=entity,
            source_chunks=source_chunks,
            related_entities=related,
            community=community,
        )

    # --- Chunk detail ---

    async def get_chunk_detail(self, chunk_id: str) -> ChunkDetail | None:
        """Chunk with extracted entities and prev/next navigation."""
        chunk = await self._backend.get_node(chunk_id)
        if chunk is None:
            return None

        # Extracted entities
        entities: list[dict[str, object]] = []
        edges = await self._backend.get_edges(chunk_id, direction="outgoing")
        for e in edges:
            if e.kind in (EdgeKind.MENTIONS, EdgeKind.CONTAINS):
                ent = await self._backend.get_node(e.target_id)
                if ent:
                    # Try to find mention position in chunk content
                    start = chunk.content.find(ent.title) if chunk.content else -1
                    entities.append(
                        {
                            "entity": {
                                "id": ent.id,
                                "title": ent.title,
                                "kind": str(ent.kind),
                                "tags": ent.tags,
                            },
                            "mention_span": (start, start + len(ent.title)) if start >= 0 else None,
                        }
                    )

        # Prev/next chunks via NEXT_CHUNK edges
        prev_chunk: Node | None = None
        next_chunk: Node | None = None

        for e in edges:
            if e.kind == EdgeKind.NEXT_CHUNK and e.source_id == chunk_id:
                next_chunk = await self._backend.get_node(e.target_id)

        incoming = await self._backend.get_edges(chunk_id, direction="incoming")
        for e in incoming:
            if e.kind == EdgeKind.NEXT_CHUNK and e.target_id == chunk_id:
                prev_chunk = await self._backend.get_node(e.source_id)

        parent_doc = chunk.properties.get("parent_doc", "")

        return ChunkDetail(
            chunk=chunk,
            extracted_entities=entities,
            prev_chunk=prev_chunk,
            next_chunk=next_chunk,
            parent_doc=parent_doc,
        )

    # --- Edge detail ---

    async def get_edge_detail(self, edge_id: str) -> EdgeDetail | None:
        """Edge with source/target and evidence chunks."""
        # Find edge by scanning (no direct edge lookup in protocol)
        all_nodes = await self._backend.list_nodes(limit=10_000)
        target_edge = None
        for n in all_nodes:
            edges = await self._backend.get_edges(n.id)
            for e in edges:
                if e.id == edge_id:
                    target_edge = e
                    break
            if target_edge:
                break

        if target_edge is None:
            return None

        source_node = await self._backend.get_node(target_edge.source_id)
        target_node = await self._backend.get_node(target_edge.target_id)

        # Evidence: find chunks that contain both source and target entities
        evidence_chunks: list[Node] = []
        if self._chunk_entity_index and source_node and target_node:
            src_chunks = self._chunk_entity_index.chunks_for_entity(source_node.id)
            tgt_chunks = self._chunk_entity_index.chunks_for_entity(target_node.id)
            shared = src_chunks & tgt_chunks
            for cid in shared:
                chunk = await self._backend.get_node(cid)
                if chunk:
                    evidence_chunks.append(chunk)

        return EdgeDetail(
            edge=target_edge,
            source_node=source_node,
            target_node=target_node,
            evidence_chunks=evidence_chunks,
        )

    # --- Table row detail ---

    async def get_table_row_detail(self, node_id: str) -> TableRowDetail | None:
        """Table row with column data and FK relations."""
        node = await self._backend.get_node(node_id)
        if node is None:
            return None

        table_name = node.properties.get("_table_name", "")
        primary_key = node.properties.get("_primary_key", "id")

        # Extract column values (exclude internal properties)
        columns = {k: v for k, v in node.properties.items() if not k.startswith("_")}

        # FK-linked rows
        related_rows: list[tuple[Node, object]] = []
        edges = await self._backend.get_edges(node_id)
        for e in edges:
            other_id = e.target_id if e.source_id == node_id else e.source_id
            other = await self._backend.get_node(other_id)
            if other and other.properties.get("_table_name"):
                related_rows.append((other, e))

        # Schema info
        schema: dict[str, object] = {
            "table_name": table_name,
            "primary_key": primary_key,
            "column_count": len(columns),
        }

        return TableRowDetail(
            node=node,
            columns=columns,
            table_name=table_name,
            related_rows=related_rows,
            schema=schema,
        )

    # --- Community detail ---

    async def get_community_detail(self, community_id: str) -> CommunityDetail | None:
        """Community with summary, members, and key entities."""
        comm = await self._backend.get_node(community_id)
        if comm is None or comm.kind != NodeKind.COMMUNITY:
            return None

        # Members: nodes with PART_OF edge to this community
        members: list[Node] = []
        edges = await self._backend.get_edges(community_id, direction="incoming")
        for e in edges:
            if e.kind == EdgeKind.PART_OF:
                member = await self._backend.get_node(e.source_id)
                if member:
                    members.append(member)

        # Key entities: members sorted by edge count (most connected first)
        entity_scores: list[tuple[Node, int]] = []
        for m in members:
            if m.kind == NodeKind.ENTITY:
                m_edges = await self._backend.get_edges(m.id)
                entity_scores.append((m, len(m_edges)))
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        key_entities = [n for n, _ in entity_scores[:10]]

        # Sub-communities
        sub_communities: list[Node] = [m for m in members if m.kind == NodeKind.COMMUNITY]

        return CommunityDetail(
            community=comm,
            summary=comm.content,
            members=members,
            key_entities=key_entities,
            sub_communities=sub_communities,
        )

    # --- In-graph search ---

    async def search_in_graph(
        self,
        query: str,
        *,
        source: str | None = None,
        limit: int = 20,
    ) -> list[NodeDetail]:
        """Search within the graph by keyword/entity name."""
        results = await self._backend.search_fts(query, limit=limit * 2)

        details: list[NodeDetail] = []
        for node in results:
            if source:
                if node.source != source and node.properties.get("parent_doc") != source:
                    continue
            detail = await self.get_node_detail(node.id)
            if detail:
                details.append(detail)
            if len(details) >= limit:
                break

        return details

    # --- Statistics ---

    async def get_graph_stats(self, *, source: str | None = None) -> GraphStats:
        """Graph-level statistics."""
        all_nodes = await self._backend.list_nodes(limit=100_000)

        if source:
            all_nodes = [
                n
                for n in all_nodes
                if n.source == source
                or n.properties.get("parent_doc") == source
                or n.properties.get("_table_name") == source
            ]

        nodes_by_kind: dict[str, int] = defaultdict(int)
        for n in all_nodes:
            nodes_by_kind[str(n.kind)] += 1

        entity_count = nodes_by_kind.get(str(NodeKind.ENTITY), 0)
        chunk_count = nodes_by_kind.get(str(NodeKind.CHUNK), 0)
        community_count = nodes_by_kind.get(str(NodeKind.COMMUNITY), 0)
        table_nodes = [n for n in all_nodes if n.properties.get("_table_name")]
        table_names = {n.properties["_table_name"] for n in table_nodes}

        # Edge stats
        edges_by_kind: dict[str, int] = defaultdict(int)
        total_edges = 0
        entity_edge_counts: list[int] = []
        for n in all_nodes:
            edges = await self._backend.get_edges(n.id, direction="outgoing")
            for e in edges:
                edges_by_kind[str(e.kind)] += 1
                total_edges += 1
            if n.kind == NodeKind.ENTITY:
                entity_edge_counts.append(len(edges))

        avg_entities_per_chunk = 0.0
        if chunk_count > 0 and self._chunk_entity_index:
            s = self._chunk_entity_index.stats()
            avg_entities_per_chunk = s.get("avg_entities_per_chunk", 0.0)

        avg_edges_per_entity = 0.0
        if entity_edge_counts:
            avg_edges_per_entity = sum(entity_edge_counts) / len(entity_edge_counts)

        return GraphStats(
            total_nodes=len(all_nodes),
            total_edges=total_edges,
            nodes_by_kind=dict(nodes_by_kind),
            edges_by_kind=dict(edges_by_kind),
            entity_count=entity_count,
            chunk_count=chunk_count,
            community_count=community_count,
            table_count=len(table_names),
            avg_entities_per_chunk=avg_entities_per_chunk,
            avg_edges_per_entity=avg_edges_per_entity,
        )

    # --- Internal helpers ---

    async def _bfs_subgraph(self, root_id: str, max_depth: int, max_nodes: int) -> list[Node]:
        """BFS from root to collect subgraph."""
        from collections import deque

        visited: set[str] = set()
        result: list[Node] = []
        queue: deque[tuple[str, int]] = deque([(root_id, 0)])

        while queue and len(result) < max_nodes:
            nid, depth = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)

            node = await self._backend.get_node(nid)
            if node is None:
                continue
            if node.kind == NodeKind.TYPE_DEF:
                continue

            result.append(node)

            if depth < max_depth:
                edges = await self._backend.get_edges(nid)
                for e in edges:
                    other = e.target_id if e.source_id == nid else e.source_id
                    if other not in visited:
                        queue.append((other, depth + 1))

        return result
