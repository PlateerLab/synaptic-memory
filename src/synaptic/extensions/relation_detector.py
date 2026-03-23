"""Rule-based + embedding similarity automatic relation detection.

Automatically detects relations with existing nodes when a new node is added.
- Title mention detection: RELATED if an existing node's title appears in the new node's content
- Tag overlap detection: RELATED if shared tags exceed a threshold
- Embedding similarity: RELATED if cosine similarity is above threshold
- NodeKind pair rules: RULE→CONCEPT = DEPENDS_ON, LESSON→* = LEARNED_FROM

Maintains an InvertedIndex for fast candidate lookup without full node traversal.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from synaptic.models import EdgeKind, NodeKind

if TYPE_CHECKING:
    from synaptic.models import Node
    from synaptic.protocols import StorageBackend

logger = logging.getLogger(__name__)


class InvertedIndex:
    """Inverted index for tag and title tokens. O(1) lookup during relation detection.

    Not thread-safe as it is used within a single asyncio event loop.
    SynapticGraph must update this index on add/remove operations.
    """

    __slots__ = ("_node_tags", "_node_title", "_tag_index", "_title_index")

    def __init__(self) -> None:
        self._tag_index: dict[str, set[str]] = {}  # tag → {node_id}
        self._title_index: dict[str, str] = {}  # title_lower → node_id (4+ chars only)
        self._node_tags: dict[str, list[str]] = {}  # node_id → tags (for cleanup on remove)
        self._node_title: dict[str, str] = {}  # node_id → title_lower

    def add(self, node: Node) -> None:
        """Register a node's tags and title in the inverted index.

        Args:
            node: Node to register. Its tags and title are indexed.
        """
        # Register in tag index
        self._node_tags[node.id] = list(node.tags)
        for tag in node.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = set()
            self._tag_index[tag_lower].add(node.id)

        # Register in title index (4+ chars only — short titles cause false positives)
        title_lower = node.title.strip().lower()
        if len(title_lower) >= 4:
            self._title_index[title_lower] = node.id
            self._node_title[node.id] = title_lower

    def remove(self, node_id: str) -> None:
        """Remove a node from the inverted index.

        Args:
            node_id: ID of the node to remove.
        """
        # Remove from tag index
        tags = self._node_tags.pop(node_id, [])
        for tag in tags:
            tag_lower = tag.lower()
            node_set = self._tag_index.get(tag_lower)
            if node_set is not None:
                node_set.discard(node_id)
                if not node_set:
                    del self._tag_index[tag_lower]

        # Remove from title index
        title_lower = self._node_title.pop(node_id, "")
        if title_lower and title_lower in self._title_index:
            # Don't delete if another node is registered with the same title
            if self._title_index[title_lower] == node_id:
                del self._title_index[title_lower]

    def find_by_tag_overlap(self, tags: list[str], exclude_id: str = "") -> dict[str, int]:
        """Find nodes with overlapping tags.

        Args:
            tags: List of tags to compare.
            exclude_id: Node ID to exclude from results (usually self).

        Returns:
            {node_id: overlap_count} — nodes with 1+ overlapping tags.
        """
        overlap: dict[str, int] = {}
        for tag in tags:
            tag_lower = tag.lower()
            node_ids = self._tag_index.get(tag_lower)
            if node_ids is None:
                continue
            for nid in node_ids:
                if nid == exclude_id:
                    continue
                overlap[nid] = overlap.get(nid, 0) + 1
        return overlap

    def find_title_mentions(self, text: str) -> list[str]:
        """Return node_ids whose titles are mentioned in the given text.

        Case-insensitive matching. Only titles with 4+ characters are indexed,
        so false positives from short titles do not occur.

        Args:
            text: Text to search (typically the new node's content).

        Returns:
            List of node_ids whose titles are mentioned (no duplicates).
        """
        if not text:
            return []

        text_lower = text.lower()
        mentioned: list[str] = []
        for title_lower, node_id in self._title_index.items():
            if title_lower in text_lower:
                mentioned.append(node_id)
        return mentioned


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RuleBasedRelationDetector:
    """Rule-based + embedding similarity automatic relation detection.

    Call ``detect()`` when a new node is added to get candidate relations
    with existing nodes. Applies 4 rules in order:

    1. **Title 언급**: 새 노드 content에 기존 노드 title이 등장 → RELATED
    2. **Tag overlap**: 공통 tag가 ``tag_overlap_min``개 이상 → RELATED
    3. **Embedding 유사도**: cosine similarity ≥ threshold → RELATED
    4. **NodeKind 쌍 규칙** (title 언급이 있는 경우만):
       - RULE → CONCEPT: DEPENDS_ON
       - LESSON → *: LEARNED_FROM

    Example::

        detector = RuleBasedRelationDetector(max_edges_per_node=5)
        # graph.add() 시 detector.index.add(node) 호출
        edges = await detector.detect(new_node, backend)
        for target_id, edge_kind, weight in edges:
            await backend.save_edge(...)
    """

    __slots__ = (
        "_embedding_threshold",
        "_embedding_weight_scale",
        "_index",
        "_max_edges",
        "_tag_overlap_min",
        "_tag_overlap_weight",
        "_title_mention_weight",
    )

    def __init__(
        self,
        *,
        max_edges_per_node: int = 5,
        tag_overlap_min: int = 2,
        title_mention_weight: float = 0.8,
        tag_overlap_weight: float = 0.5,
        embedding_threshold: float = 0.75,
        embedding_weight_scale: float = 0.7,
    ) -> None:
        """Initialize RuleBasedRelationDetector.

        Args:
            max_edges_per_node: Maximum number of relations to detect per node.
            tag_overlap_min: Minimum shared tags required for a tag overlap relation.
            title_mention_weight: Default weight for title mention relations.
            tag_overlap_weight: Default weight for tag overlap relations.
            embedding_threshold: Embedding similarity threshold (0.0~1.0).
            embedding_weight_scale: Scale factor to convert similarity to edge weight.
        """
        self._index = InvertedIndex()
        self._max_edges = max_edges_per_node
        self._tag_overlap_min = tag_overlap_min
        self._title_mention_weight = title_mention_weight
        self._tag_overlap_weight = tag_overlap_weight
        self._embedding_threshold = embedding_threshold
        self._embedding_weight_scale = embedding_weight_scale

    @property
    def index(self) -> InvertedIndex:
        """Internal inverted index. Used by graph.py to update the index on add/remove."""
        return self._index

    async def detect(
        self, node: Node, backend: StorageBackend
    ) -> list[tuple[str, EdgeKind, float]]:
        """Detect relations between a new node and existing nodes.

        ``self.index.add(node)`` must be completed before calling detect().
        (The node itself is automatically excluded from results)

        Args:
            node: New node to detect relations for.
            backend: StorageBackend for kind lookup.

        Returns:
            [(target_node_id, edge_kind, weight), ...] up to max_edges_per_node.
        """
        relations: list[tuple[str, EdgeKind, float]] = []
        seen_targets: set[str] = set()

        # 1. Existing node title mentioned in new node content → RELATED (default)
        mentioned_ids = self._index.find_title_mentions(node.content)
        for target_id in mentioned_ids:
            if target_id == node.id or target_id in seen_targets:
                continue
            seen_targets.add(target_id)

            # NodeKind 쌍 규칙 (title mention이 있는 경우만 kind 체크)
            edge_kind, weight = await self._resolve_kind_pair(node, target_id, backend)
            relations.append((target_id, edge_kind, weight))

        # 2. Shared tags >= overlap_min → RELATED
        if node.tags:
            overlaps = self._index.find_by_tag_overlap(node.tags, exclude_id=node.id)
            for target_id, count in overlaps.items():
                if count < self._tag_overlap_min:
                    continue
                if target_id in seen_targets:
                    continue
                seen_targets.add(target_id)
                relations.append((target_id, EdgeKind.RELATED, self._tag_overlap_weight))

        # 3. Embedding similarity → RELATED
        if node.embedding:
            try:
                candidates = await backend.search_vector(
                    node.embedding,
                    limit=self._max_edges * 2,
                )
                for candidate in candidates:
                    if candidate.id == node.id or candidate.id in seen_targets:
                        continue
                    if not candidate.embedding:
                        continue
                    sim = _cosine_similarity(node.embedding, candidate.embedding)
                    if sim >= self._embedding_threshold:
                        seen_targets.add(candidate.id)
                        weight = sim * self._embedding_weight_scale
                        relations.append((candidate.id, EdgeKind.RELATED, weight))
            except Exception:
                logger.debug("Embedding search failed, skipping", exc_info=True)

        # weight 내림차순 정렬 후 max_edges 제한
        relations.sort(key=lambda r: r[2], reverse=True)
        return relations[: self._max_edges]

    async def detect_batch(
        self, nodes: list[Node], backend: StorageBackend
    ) -> dict[str, list[tuple[str, EdgeKind, float]]]:
        """Detect relations for multiple nodes in batch.

        Args:
            nodes: List of nodes to detect relations for.
            backend: StorageBackend for kind lookup.

        Returns:
            {node_id: [(target_node_id, edge_kind, weight), ...]} mapping.
        """
        result: dict[str, list[tuple[str, EdgeKind, float]]] = {}
        for node in nodes:
            result[node.id] = await self.detect(node, backend)
        return result

    async def _resolve_kind_pair(
        self,
        source: Node,
        target_id: str,
        backend: StorageBackend,
    ) -> tuple[EdgeKind, float]:
        """Determine edge kind and weight based on NodeKind pair rules.

        - RULE → CONCEPT: DEPENDS_ON (0.6)
        - LESSON → *: LEARNED_FROM (0.7)
        - Otherwise: RELATED (title_mention_weight)

        Args:
            source: Source node (newly added node).
            target_id: Target node ID.
            backend: StorageBackend for target kind lookup.

        Returns:
            (EdgeKind, weight) tuple.
        """
        # LESSON → any node → LEARNED_FROM
        if source.kind == NodeKind.LESSON:
            return EdgeKind.LEARNED_FROM, 0.7

        # RULE → CONCEPT → DEPENDS_ON (target kind 조회 필요)
        if source.kind == NodeKind.RULE:
            target_node = await backend.get_node(target_id)
            if target_node is not None and target_node.kind == NodeKind.CONCEPT:
                return EdgeKind.DEPENDS_ON, 0.6

        # Default: RELATED
        return EdgeKind.RELATED, self._title_mention_weight


# --- NodeKind pair → EdgeKind mapping (reused by EmbeddingRelationDetector) ---

_KIND_PAIR_RULES: dict[tuple[NodeKind, NodeKind | None], EdgeKind] = {
    (NodeKind.LESSON, None): EdgeKind.LEARNED_FROM,
    (NodeKind.RULE, NodeKind.CONCEPT): EdgeKind.DEPENDS_ON,
}


def _resolve_edge_kind(
    source_kind: NodeKind,
    target_kind: NodeKind,
) -> EdgeKind:
    """Determine EdgeKind based on NodeKind pair.

    - LESSON → * : LEARNED_FROM
    - RULE → CONCEPT : DEPENDS_ON
    - Otherwise: RELATED
    """
    if source_kind == NodeKind.LESSON:
        return EdgeKind.LEARNED_FROM
    pair = (source_kind, target_kind)
    return _KIND_PAIR_RULES.get(pair, EdgeKind.RELATED)


class EmbeddingRelationDetector:
    """Embedding cosine similarity-based automatic relation creation.

    When a node with an embedding vector is added, computes cosine similarity
    with existing nodes and automatically connects similar ones.
    Uses pure vector operations without LLM calls.

    If ``fallback`` is configured, title/tag-based relations are also detected.
    Since graph.py calls ``relation_detector.index``,
    fallback's index is returned when fallback is set.

    Example::

        rule_detector = RuleBasedRelationDetector()
        detector = EmbeddingRelationDetector(
            similarity_threshold=0.7,
            fallback=rule_detector,
        )
        # graph.add() 시 detector.index.add(node) 호출 → fallback.index에 위임
        edges = await detector.detect(new_node, backend)
    """

    __slots__ = ("_fallback", "_max_edges", "_threshold", "index")

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.7,
        max_edges_per_node: int = 5,
        fallback: RuleBasedRelationDetector | None = None,
    ) -> None:
        """Initialize EmbeddingRelationDetector.

        Args:
            similarity_threshold: Minimum cosine similarity for a relation (0.0~1.0).
            max_edges_per_node: Maximum number of relations to detect per node.
            fallback: Title/tag-based relation detector. None = embedding only.
        """
        self._threshold = similarity_threshold
        self._max_edges = max_edges_per_node
        self._fallback = fallback
        # graph.py calls relation_detector.index.add(node), so provide
        # fallback's index if available, otherwise an empty InvertedIndex
        self.index = fallback.index if fallback is not None else InvertedIndex()

    async def detect(
        self,
        node: Node,
        backend: StorageBackend,
    ) -> list[tuple[str, EdgeKind, float]]:
        """Detect relations between a new node and existing nodes.

        1. If node has an embedding, search for similar nodes via backend.search_vector()
        2. Create RELATED edges for nodes above similarity_threshold
        3. Adjust EdgeKind based on NodeKind pair rules (LESSON→* = LEARNED_FROM, etc.)
        4. Add title/tag-based relations if fallback is available
        5. Deduplicate and return top max_edges_per_node results

        Args:
            node: New node to detect relations for.
            backend: StorageBackend for similar node search and kind lookup.

        Returns:
            [(target_node_id, edge_kind, weight), ...] up to max_edges_per_node.
        """
        relations: list[tuple[str, EdgeKind, float]] = []
        seen_targets: set[str] = set()

        # 1. Embedding-based similarity detection
        if node.embedding:
            try:
                candidates = await backend.search_vector(
                    node.embedding,
                    limit=self._max_edges * 2,
                )
                for candidate in candidates:
                    if candidate.id == node.id or candidate.id in seen_targets:
                        continue
                    if not candidate.embedding:
                        continue
                    sim = _cosine_similarity(node.embedding, candidate.embedding)
                    if sim >= self._threshold:
                        seen_targets.add(candidate.id)
                        edge_kind = _resolve_edge_kind(node.kind, candidate.kind)
                        relations.append((candidate.id, edge_kind, sim))
            except Exception:
                logger.debug(
                    "Embedding search failed in EmbeddingRelationDetector",
                    exc_info=True,
                )

        # 2. Fallback: add title/tag-based relations (deduplicated)
        if self._fallback is not None:
            fallback_relations = await self._fallback.detect(node, backend)
            for target_id, edge_kind, weight in fallback_relations:
                if target_id not in seen_targets:
                    seen_targets.add(target_id)
                    relations.append((target_id, edge_kind, weight))

        # weight 내림차순 정렬 후 max_edges 제한
        relations.sort(key=lambda r: r[2], reverse=True)
        return relations[: self._max_edges]
