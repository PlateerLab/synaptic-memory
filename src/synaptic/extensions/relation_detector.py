"""규칙 기반 + 임베딩 유사도 관계 자동 탐지.

새 노드 추가 시 기존 노드와의 관계를 자동 탐지한다.
- title 언급 탐지: 새 노드 content에 기존 노드 title이 등장하면 RELATED
- tag overlap 탐지: 공통 tag가 임계값 이상이면 RELATED
- embedding 유사도: cosine similarity가 threshold 이상이면 RELATED
- NodeKind 쌍 규칙: RULE→CONCEPT은 DEPENDS_ON, LESSON→* 은 LEARNED_FROM

InvertedIndex를 유지하여 전체 노드 순회 없이 후보를 빠르게 찾는다.
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
    """tag와 title 토큰의 역인덱스. 관계 탐지 시 O(1) 조회.

    asyncio 단일 이벤트 루프에서 사용하므로 thread-safe가 아님.
    SynapticGraph가 add/remove 시 이 인덱스를 함께 갱신해야 한다.
    """

    __slots__ = ("_tag_index", "_title_index", "_node_tags", "_node_title")

    def __init__(self) -> None:
        self._tag_index: dict[str, set[str]] = {}  # tag → {node_id}
        self._title_index: dict[str, str] = {}  # title_lower → node_id (4글자 이상)
        self._node_tags: dict[str, list[str]] = {}  # node_id → tags (remove 시 정리용)
        self._node_title: dict[str, str] = {}  # node_id → title_lower

    def add(self, node: Node) -> None:
        """노드의 tags와 title을 역인덱스에 등록한다.

        Args:
            node: 등록할 노드. tags와 title을 인덱싱한다.
        """
        # tag 인덱스 등록
        self._node_tags[node.id] = list(node.tags)
        for tag in node.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = set()
            self._tag_index[tag_lower].add(node.id)

        # title 인덱스 등록 (4글자 이상만 — 짧은 title은 false positive 유발)
        title_lower = node.title.strip().lower()
        if len(title_lower) >= 4:
            self._title_index[title_lower] = node.id
            self._node_title[node.id] = title_lower

    def remove(self, node_id: str) -> None:
        """노드를 역인덱스에서 제거한다.

        Args:
            node_id: 제거할 노드 ID.
        """
        # tag 인덱스에서 제거
        tags = self._node_tags.pop(node_id, [])
        for tag in tags:
            tag_lower = tag.lower()
            node_set = self._tag_index.get(tag_lower)
            if node_set is not None:
                node_set.discard(node_id)
                if not node_set:
                    del self._tag_index[tag_lower]

        # title 인덱스에서 제거
        title_lower = self._node_title.pop(node_id, "")
        if title_lower and title_lower in self._title_index:
            # 동일 title로 다른 노드가 등록된 경우 삭제하지 않음
            if self._title_index[title_lower] == node_id:
                del self._title_index[title_lower]

    def find_by_tag_overlap(
        self, tags: list[str], exclude_id: str = ""
    ) -> dict[str, int]:
        """주어진 tags와 겹치는 노드들을 찾는다.

        Args:
            tags: 비교할 tag 목록.
            exclude_id: 결과에서 제외할 노드 ID (보통 자기 자신).

        Returns:
            {node_id: overlap_count} — 겹치는 tag 수가 1 이상인 노드들.
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
        """text에 언급된 기존 노드의 title에 해당하는 node_id를 반환한다.

        대소문자 무시 매칭. title이 4글자 이상인 것만 인덱스에 등록되어 있으므로
        짧은 title에 의한 false positive는 발생하지 않는다.

        Args:
            text: 검색할 텍스트 (보통 새 노드의 content).

        Returns:
            언급된 title에 해당하는 node_id 목록 (중복 없음).
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
    """두 벡터의 코사인 유사도를 계산한다."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RuleBasedRelationDetector:
    """규칙 기반 + 임베딩 유사도 관계 자동 탐지.

    새 노드가 추가될 때 ``detect()``를 호출하면 기존 노드와의
    관계 후보를 반환한다. 4가지 규칙을 순서대로 적용한다:

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
        "_title_mention_weight",
        "_tag_overlap_weight",
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
        """RuleBasedRelationDetector를 초기화한다.

        Args:
            max_edges_per_node: 한 노드에서 탐지할 최대 관계 수.
            tag_overlap_min: tag overlap 관계로 인정할 최소 공통 tag 수.
            title_mention_weight: title 언급 관계의 기본 weight.
            tag_overlap_weight: tag overlap 관계의 기본 weight.
            embedding_threshold: 임베딩 유사도 관계 임계값 (0.0~1.0).
            embedding_weight_scale: 유사도를 edge weight로 변환할 스케일 팩터.
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
        """내부 역인덱스. graph.py에서 add/remove 시 인덱스 갱신에 사용."""
        return self._index

    async def detect(
        self, node: Node, backend: StorageBackend
    ) -> list[tuple[str, EdgeKind, float]]:
        """새 노드와 기존 노드 사이의 관계를 탐지한다.

        detect() 호출 전에 ``self.index.add(node)``가 완료되어 있어야 한다.
        (자기 자신은 결과에서 자동 제외됨)

        Args:
            node: 관계를 탐지할 새 노드.
            backend: 노드 kind 조회용 StorageBackend.

        Returns:
            [(target_node_id, edge_kind, weight), ...] 최대 max_edges_per_node개.
        """
        relations: list[tuple[str, EdgeKind, float]] = []
        seen_targets: set[str] = set()

        # 1. 새 노드 content에 기존 노드 title 언급 → RELATED (기본)
        mentioned_ids = self._index.find_title_mentions(node.content)
        for target_id in mentioned_ids:
            if target_id == node.id or target_id in seen_targets:
                continue
            seen_targets.add(target_id)

            # NodeKind 쌍 규칙 (title mention이 있는 경우만 kind 체크)
            edge_kind, weight = await self._resolve_kind_pair(
                node, target_id, backend
            )
            relations.append((target_id, edge_kind, weight))

        # 2. 공통 tag overlap_min개 이상 → RELATED
        if node.tags:
            overlaps = self._index.find_by_tag_overlap(
                node.tags, exclude_id=node.id
            )
            for target_id, count in overlaps.items():
                if count < self._tag_overlap_min:
                    continue
                if target_id in seen_targets:
                    continue
                seen_targets.add(target_id)
                relations.append(
                    (target_id, EdgeKind.RELATED, self._tag_overlap_weight)
                )

        # 3. Embedding 유사도 → RELATED
        if node.embedding:
            try:
                candidates = await backend.search_vector(
                    node.embedding, limit=self._max_edges * 2,
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
                        relations.append(
                            (candidate.id, EdgeKind.RELATED, weight)
                        )
            except Exception:
                logger.debug("Embedding search failed, skipping", exc_info=True)

        # weight 내림차순 정렬 후 max_edges 제한
        relations.sort(key=lambda r: r[2], reverse=True)
        return relations[: self._max_edges]

    async def detect_batch(
        self, nodes: list[Node], backend: StorageBackend
    ) -> dict[str, list[tuple[str, EdgeKind, float]]]:
        """여러 노드의 관계를 일괄 탐지한다.

        Args:
            nodes: 관계를 탐지할 노드 목록.
            backend: 노드 kind 조회용 StorageBackend.

        Returns:
            {node_id: [(target_node_id, edge_kind, weight), ...]} 매핑.
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
        """NodeKind 쌍 규칙에 따라 edge kind와 weight를 결정한다.

        - RULE → CONCEPT: DEPENDS_ON (0.6)
        - LESSON → *: LEARNED_FROM (0.7)
        - 그 외: RELATED (title_mention_weight)

        Args:
            source: 소스 노드 (새로 추가된 노드).
            target_id: 타겟 노드 ID.
            backend: 타겟 노드 kind 조회용.

        Returns:
            (EdgeKind, weight) 튜플.
        """
        # LESSON → 어떤 노드든 → LEARNED_FROM
        if source.kind == NodeKind.LESSON:
            return EdgeKind.LEARNED_FROM, 0.7

        # RULE → CONCEPT → DEPENDS_ON (target kind 조회 필요)
        if source.kind == NodeKind.RULE:
            target_node = await backend.get_node(target_id)
            if target_node is not None and target_node.kind == NodeKind.CONCEPT:
                return EdgeKind.DEPENDS_ON, 0.6

        # 기본: RELATED
        return EdgeKind.RELATED, self._title_mention_weight


# --- NodeKind 쌍 → EdgeKind 매핑 (EmbeddingRelationDetector에서도 재사용) ---

_KIND_PAIR_RULES: dict[tuple[NodeKind, NodeKind | None], EdgeKind] = {
    (NodeKind.LESSON, None): EdgeKind.LEARNED_FROM,
    (NodeKind.RULE, NodeKind.CONCEPT): EdgeKind.DEPENDS_ON,
}


def _resolve_edge_kind(
    source_kind: NodeKind,
    target_kind: NodeKind,
) -> EdgeKind:
    """NodeKind 쌍에 따라 EdgeKind를 결정한다.

    - LESSON → * : LEARNED_FROM
    - RULE → CONCEPT : DEPENDS_ON
    - 그 외: RELATED
    """
    if source_kind == NodeKind.LESSON:
        return EdgeKind.LEARNED_FROM
    pair = (source_kind, target_kind)
    return _KIND_PAIR_RULES.get(pair, EdgeKind.RELATED)


class EmbeddingRelationDetector:
    """Embedding cosine similarity 기반 관계 자동 생성.

    노드 추가 시 embedding vector가 있으면, 기존 노드와의
    cosine similarity를 계산하여 유사한 노드끼리 자동 연결.
    LLM 호출 없이 순수 벡터 연산만 사용.

    ``fallback``\ 이 설정되어 있으면 title/tag 기반 관계도 함께 탐지한다.
    graph.py는 ``relation_detector.index``\ 를 호출하므로,
    fallback이 있으면 fallback의 index를 반환한다.

    Example::

        rule_detector = RuleBasedRelationDetector()
        detector = EmbeddingRelationDetector(
            similarity_threshold=0.7,
            fallback=rule_detector,
        )
        # graph.add() 시 detector.index.add(node) 호출 → fallback.index에 위임
        edges = await detector.detect(new_node, backend)
    """

    __slots__ = ("_threshold", "_max_edges", "_fallback", "index")

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.7,
        max_edges_per_node: int = 5,
        fallback: RuleBasedRelationDetector | None = None,
    ) -> None:
        """EmbeddingRelationDetector를 초기화한다.

        Args:
            similarity_threshold: 관계로 인정할 최소 cosine similarity (0.0~1.0).
            max_edges_per_node: 한 노드에서 탐지할 최대 관계 수.
            fallback: title/tag 기반 관계 탐지기. None이면 embedding만 사용.
        """
        self._threshold = similarity_threshold
        self._max_edges = max_edges_per_node
        self._fallback = fallback
        # graph.py가 relation_detector.index.add(node) 호출하므로
        # fallback이 있으면 fallback의 index를, 없으면 빈 InvertedIndex를 제공
        self.index = fallback.index if fallback is not None else InvertedIndex()

    async def detect(
        self,
        node: Node,
        backend: StorageBackend,
    ) -> list[tuple[str, EdgeKind, float]]:
        """새 노드와 기존 노드 사이의 관계를 탐지한다.

        1. node의 embedding이 있으면 backend.search_vector()로 유사 노드 검색
        2. similarity_threshold 이상인 노드와 RELATED 엣지 생성
        3. NodeKind 쌍에 따라 EdgeKind 조정 (LESSON→* = LEARNED_FROM 등)
        4. fallback이 있으면 title/tag 기반 관계도 추가
        5. 중복 제거 후 상위 max_edges_per_node개 반환

        Args:
            node: 관계를 탐지할 새 노드.
            backend: 유사 노드 검색 및 kind 조회용 StorageBackend.

        Returns:
            [(target_node_id, edge_kind, weight), ...] 최대 max_edges_per_node개.
        """
        relations: list[tuple[str, EdgeKind, float]] = []
        seen_targets: set[str] = set()

        # 1. Embedding 기반 유사도 탐지
        if node.embedding:
            try:
                candidates = await backend.search_vector(
                    node.embedding, limit=self._max_edges * 2,
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

        # 2. Fallback: title/tag 기반 관계 추가 (중복 제거)
        if self._fallback is not None:
            fallback_relations = await self._fallback.detect(node, backend)
            for target_id, edge_kind, weight in fallback_relations:
                if target_id not in seen_targets:
                    seen_targets.add(target_id)
                    relations.append((target_id, edge_kind, weight))

        # weight 내림차순 정렬 후 max_edges 제한
        relations.sort(key=lambda r: r[2], reverse=True)
        return relations[: self._max_edges]
