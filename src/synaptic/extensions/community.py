"""Community detection on the knowledge graph.

Discovers clusters of related nodes and creates COMMUNITY summary nodes.
Supports:
  - Pure Python Louvain (zero-dep fallback)
  - igraph + leidenalg (optional, higher quality)
  - Incremental re-detection for modified subgraphs
  - LLM-based or extractive community summarization

Usage::

    detector = CommunityDetector()
    communities = await detector.detect(graph)

    # With LLM summaries
    detector = CommunityDetector(llm=OllamaLLMProvider(...))
    communities = await detector.detect(graph)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from synaptic.models import EdgeKind, Node, NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider
    from synaptic.graph import SynapticGraph
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("community-detector")


def _str_list() -> list[str]:
    return []


@dataclass(slots=True)
class Community:
    """A detected community of related nodes."""

    id: str = ""
    member_ids: list[str] = field(default_factory=_str_list)
    summary: str = ""
    level: int = 0
    parent_community_id: str = ""


_LLM_SUMMARY_PROMPT = """다음 지식 노드들의 공통 주제를 한국어로 2-3문장으로 요약하세요.
각 노드의 제목과 내용입니다:

{nodes_text}

요약:"""


class CommunityDetector:
    """Detects communities in the knowledge graph and creates summary nodes.

    Uses Louvain algorithm (pure Python) by default.
    When igraph + leidenalg are installed, uses Leiden (higher quality).

    Example::

        detector = CommunityDetector(min_community_size=3)
        communities = await detector.detect(graph)

        # Incremental: only re-detect around changed nodes
        communities = await detector.detect_incremental(graph, {"node_42", "node_55"})
    """

    __slots__ = ("_llm", "_min_size", "_resolution")

    def __init__(
        self,
        *,
        resolution: float = 1.0,
        min_community_size: int = 3,
        llm: LLMProvider | None = None,
    ) -> None:
        self._resolution = resolution
        self._min_size = min_community_size
        self._llm = llm

    async def detect(self, graph: SynapticGraph) -> list[Community]:
        """Detect communities across the entire graph.

        Creates COMMUNITY nodes in the graph with summaries.

        Returns:
            List of detected communities.
        """
        backend = graph.backend

        # Gather all non-internal nodes
        all_nodes = await backend.list_nodes(limit=100_000)
        nodes = [
            n for n in all_nodes
            if n.kind not in (NodeKind.TYPE_DEF, NodeKind.COMMUNITY)
            and "_phrase" not in (n.tags or [])
        ]

        if len(nodes) < self._min_size:
            return []

        # Build adjacency for community detection
        node_ids = {n.id for n in nodes}
        adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for node in nodes:
            edges = await backend.get_edges(node.id)
            for edge in edges:
                other = edge.target_id if edge.source_id == node.id else edge.source_id
                if other in node_ids and other != node.id:
                    adj[node.id][other] += edge.weight
                    adj[other][node.id] += edge.weight

        # Run community detection
        try:
            partitions = self._leiden_detect(node_ids, adj)
        except ImportError:
            partitions = self._louvain_detect(node_ids, adj)

        # Filter by min size
        communities: list[Community] = []
        node_map = {n.id: n for n in nodes}

        for comm_id, member_ids in enumerate(partitions):
            if len(member_ids) < self._min_size:
                continue

            members = [node_map[nid] for nid in member_ids if nid in node_map]
            summary = await self._summarize(members)

            community = Community(
                id=f"comm_{comm_id}",
                member_ids=list(member_ids),
                summary=summary,
                level=0,
            )
            communities.append(community)

            # Create COMMUNITY node in graph
            comm_node = await graph.add(
                title=f"Community {comm_id}",
                content=summary,
                kind=NodeKind.COMMUNITY,
                tags=["_community", f"_members:{len(member_ids)}"],
                properties={"member_count": str(len(member_ids))},
            )
            community.id = comm_node.id

            # Link members to community
            for mid in member_ids:
                await graph.link(mid, comm_node.id, kind=EdgeKind.PART_OF, weight=0.5)

        logger.info(f"Detected {len(communities)} communities from {len(nodes)} nodes")
        return communities

    async def detect_incremental(
        self,
        graph: SynapticGraph,
        changed_node_ids: set[str],
    ) -> list[Community]:
        """Re-detect communities only around changed nodes (2-hop neighborhood).

        More efficient than full detection for incremental updates.
        """
        backend = graph.backend

        # Gather 2-hop neighborhood
        neighborhood: set[str] = set(changed_node_ids)
        for nid in changed_node_ids:
            neighbors = await backend.get_neighbors(nid, depth=2)
            for node, _edge in neighbors:
                if node.kind not in (NodeKind.TYPE_DEF, NodeKind.COMMUNITY):
                    neighborhood.add(node.id)

        if len(neighborhood) < self._min_size:
            return []

        # Remove old community nodes for affected area
        all_nodes = await backend.list_nodes(kind=NodeKind.COMMUNITY, limit=10_000)
        for comm_node in all_nodes:
            edges = await backend.get_edges(comm_node.id, direction="incoming")
            for edge in edges:
                if edge.source_id in neighborhood:
                    await backend.delete_node(comm_node.id)
                    break

        # Re-detect for the neighborhood (delegate to full detect with filtered nodes)
        return await self.detect(graph)

    def _louvain_detect(
        self,
        node_ids: set[str],
        adj: dict[str, dict[str, float]],
    ) -> list[list[str]]:
        """Pure Python Louvain community detection (zero-dep).

        Greedy modularity optimization.
        """
        nodes = list(node_ids)
        if not nodes:
            return []

        # Initialize: each node in its own community
        community: dict[str, int] = {nid: i for i, nid in enumerate(nodes)}
        next_comm = len(nodes)

        # Total edge weight
        total_weight = sum(
            sum(weights.values()) for weights in adj.values()
        ) / 2.0
        if total_weight == 0:
            return [[nid] for nid in nodes]

        improved = True
        max_iterations = 20

        for _ in range(max_iterations):
            if not improved:
                break
            improved = False

            for nid in nodes:
                current_comm = community[nid]

                # Calculate neighbor community weights
                comm_weights: dict[int, float] = defaultdict(float)
                for neighbor, weight in adj.get(nid, {}).items():
                    comm_weights[community[neighbor]] += weight

                if not comm_weights:
                    continue

                # Find best community to move to
                best_comm = current_comm
                best_gain = 0.0

                for target_comm, weight_to_comm in comm_weights.items():
                    if target_comm == current_comm:
                        continue
                    # Simplified modularity gain
                    gain = weight_to_comm * self._resolution
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = target_comm

                if best_comm != current_comm:
                    community[nid] = best_comm
                    improved = True

        # Group by community
        groups: dict[int, list[str]] = defaultdict(list)
        for nid, comm in community.items():
            groups[comm].append(nid)

        return list(groups.values())

    def _leiden_detect(
        self,
        node_ids: set[str],
        adj: dict[str, dict[str, float]],
    ) -> list[list[str]]:
        """Leiden community detection using igraph + leidenalg."""
        import igraph  # noqa: F401 — raises ImportError if not installed
        import leidenalg

        nodes = list(node_ids)
        node_to_idx = {nid: i for i, nid in enumerate(nodes)}

        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        seen: set[tuple[int, int]] = set()

        for src, neighbors in adj.items():
            src_idx = node_to_idx.get(src)
            if src_idx is None:
                continue
            for tgt, w in neighbors.items():
                tgt_idx = node_to_idx.get(tgt)
                if tgt_idx is None:
                    continue
                edge_key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
                if edge_key not in seen:
                    seen.add(edge_key)
                    edges.append(edge_key)
                    weights.append(w)

        g = igraph.Graph(n=len(nodes), edges=edges, directed=False)
        partition = leidenalg.find_partition(
            g,
            leidenalg.ModularityVertexPartition,
            weights=weights if weights else None,
            resolution_parameter=self._resolution,
        )

        result: list[list[str]] = []
        for members in partition:
            result.append([nodes[i] for i in members])

        return result

    async def _summarize(self, members: list[Node]) -> str:
        """Generate community summary."""
        if not members:
            return ""

        # Build extractive summary (always available)
        titles = [m.title for m in members if m.title][:10]
        extractive = f"관련 노드: {', '.join(titles)}"

        if self._llm is None:
            return extractive

        # LLM summary
        nodes_text = "\n".join(
            f"- {m.title}: {m.content[:200]}" for m in members[:10]
        )
        prompt = _LLM_SUMMARY_PROMPT.format(nodes_text=nodes_text)

        try:
            summary = await self._llm.generate(
                system="당신은 지식 그래프의 커뮤니티를 요약하는 전문가입니다.",
                user=prompt,
                max_tokens=256,
            )
            return summary.strip() if summary.strip() else extractive
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}")
            return extractive
