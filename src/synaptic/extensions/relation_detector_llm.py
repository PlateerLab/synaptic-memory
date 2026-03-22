"""LLM 기반 관계 탐지기.

규칙 기반 후보 추출(InvertedIndex + vector search) 후
LLM이 의미적 관계를 판단하여 EdgeKind와 weight를 결정한다.

LLM 호출 실패 시 fallback(RuleBasedRelationDetector)으로 자동 전환.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from synaptic.extensions.relation_detector import InvertedIndex
from synaptic.models import EdgeKind

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider
    from synaptic.extensions.relation_detector import RuleBasedRelationDetector
    from synaptic.models import Node
    from synaptic.protocols import StorageBackend

logger = logging.getLogger(__name__)

_RELATION_MAP: dict[str, EdgeKind] = {
    "related": EdgeKind.RELATED,
    "caused": EdgeKind.CAUSED,
    "learned_from": EdgeKind.LEARNED_FROM,
    "depends_on": EdgeKind.DEPENDS_ON,
    "produced": EdgeKind.PRODUCED,
    "contradicts": EdgeKind.CONTRADICTS,
    "supersedes": EdgeKind.SUPERSEDES,
}

_SYSTEM_PROMPT = """\
주어진 새 지식 노드와 기존 후보 노드들 사이의 관계를 분석하라.

관계 종류:
- related: 주제가 관련됨
- caused: 새 노드가 후보를 야기함 (또는 반대)
- learned_from: 후보 경험에서 새 노드의 교훈을 얻음
- depends_on: 새 노드가 후보에 의존
- contradicts: 서로 모순
- supersedes: 새 노드가 후보를 대체

관계가 있는 것만 JSON 배열로 응답:
[
  {"target": 0, "relation": "depends_on", "weight": 0.8, "reason": "간단한 이유"}
]

불확실하면 포함하지 마라. 빈 배열 []도 가능.
반드시 JSON만 출력하라. 설명이나 사고 과정을 쓰지 마라. /no_think"""


class LLMRelationDetector:
    """LLM 기반 관계 탐지기.

    후보 추출은 InvertedIndex(title mention) + vector search로 수행하고,
    관계 판단은 LLM에게 위임한다. LLM 호출 실패 시 fallback detector를 사용.

    Example::

        from synaptic.extensions.llm_provider import OllamaLLMProvider
        from synaptic.extensions.relation_detector import RuleBasedRelationDetector

        llm = OllamaLLMProvider(model="qwen3:0.6b")
        fallback = RuleBasedRelationDetector()
        detector = LLMRelationDetector(llm, fallback=fallback)

        edges = await detector.detect(new_node, backend)
    """

    __slots__ = ("_fallback", "_index", "_llm", "_max_candidates", "_max_edges")

    def __init__(
        self,
        llm: LLMProvider,
        *,
        fallback: RuleBasedRelationDetector | None = None,
        max_candidates: int = 10,
        max_edges_per_node: int = 5,
    ) -> None:
        """LLMRelationDetector를 초기화한다.

        Args:
            llm: LLM 텍스트 생성 프로바이더.
            fallback: LLM 실패 시 사용할 규칙 기반 탐지기.
                      None이면 LLM 실패 시 빈 리스트 반환.
            max_candidates: LLM에게 보낼 최대 후보 노드 수.
            max_edges_per_node: 최종 반환할 최대 관계 수.
        """
        self._llm = llm
        self._fallback = fallback
        self._max_candidates = max_candidates
        self._max_edges = max_edges_per_node
        # fallback이 있으면 index를 공유하여 이중 인덱스 방지
        self._index = fallback.index if fallback is not None else InvertedIndex()

    @property
    def index(self) -> InvertedIndex:
        """내부 역인덱스. graph.py에서 add/remove 시 인덱스 갱신에 사용."""
        return self._index

    async def detect(
        self, node: Node, backend: StorageBackend
    ) -> list[tuple[str, EdgeKind, float]]:
        """새 노드와 기존 노드 사이의 관계를 LLM으로 탐지한다.

        1. InvertedIndex + vector search로 후보 추출
        2. LLM에게 관계 판단 요청
        3. JSON 파싱 → EdgeKind 매핑

        LLM 호출/파싱 실패 시 fallback detector를 사용한다.

        Args:
            node: 관계를 탐지할 새 노드.
            backend: 후보 조회용 StorageBackend.

        Returns:
            [(target_node_id, edge_kind, weight), ...] 최대 max_edges_per_node개.
        """
        # 1. 후보 추출
        candidates = await self._gather_candidates(node, backend)
        if not candidates:
            return []

        # 2. LLM 관계 판단
        try:
            prompt = self._build_prompt(node, candidates)
            raw = await self._llm.generate(
                system=_SYSTEM_PROMPT,
                user=prompt,
                max_tokens=512,
            )
            relations = self._parse_response(raw, candidates)
        except Exception:
            logger.warning(
                "LLM relation detection failed, using fallback",
                exc_info=True,
            )
            if self._fallback is not None:
                return await self._fallback.detect(node, backend)
            return []

        # 3. weight 내림차순 정렬 + max_edges 제한
        relations.sort(key=lambda r: r[2], reverse=True)
        return relations[: self._max_edges]

    async def _gather_candidates(
        self, node: Node, backend: StorageBackend
    ) -> list[Node]:
        """InvertedIndex + vector search로 후보 노드를 수집한다.

        Args:
            node: 새로 추가된 노드.
            backend: 후보 조회용 StorageBackend.

        Returns:
            중복 제거된 후보 노드 목록 (최대 max_candidates개).
        """
        seen: set[str] = {node.id}
        candidates: list[Node] = []

        # title mention으로 후보 추출
        mentioned_ids = self._index.find_title_mentions(node.content)
        for nid in mentioned_ids:
            if nid in seen:
                continue
            seen.add(nid)
            n = await backend.get_node(nid)
            if n is not None:
                candidates.append(n)

        # vector search로 후보 추출
        if node.embedding:
            try:
                vec_results = await backend.search_vector(
                    node.embedding, limit=self._max_candidates
                )
                for n in vec_results:
                    if n.id in seen:
                        continue
                    seen.add(n.id)
                    candidates.append(n)
            except Exception:
                logger.debug(
                    "Vector search failed during candidate gathering",
                    exc_info=True,
                )

        return candidates[: self._max_candidates]

    def _build_prompt(self, node: Node, candidates: list[Node]) -> str:
        """LLM에게 보낼 user 프롬프트를 구성한다.

        Args:
            node: 새로 추가된 노드.
            candidates: 관계 판단 대상 후보 노드들.

        Returns:
            포맷팅된 프롬프트 문자열.
        """
        lines = [
            "새 노드:",
            f"- 제목: {node.title}",
            f"- 종류: {node.kind}",
            f"- 내용: {node.content[:800]}",
            "",
            "후보 노드:",
        ]
        for i, c in enumerate(candidates):
            lines.append(f"[{i}] {c.title} ({c.kind}): {c.content[:200]}")
        return "\n".join(lines)

    def _parse_response(
        self, raw: str, candidates: list[Node]
    ) -> list[tuple[str, EdgeKind, float]]:
        """LLM JSON 응답을 파싱하여 관계 목록을 반환한다.

        Args:
            raw: LLM이 반환한 JSON 문자열.
            candidates: 후보 노드 목록 (인덱스 매핑용).

        Returns:
            [(target_node_id, edge_kind, weight), ...] 유효한 항목만.

        Raises:
            ValueError: JSON 파싱 실패 또는 배열이 아닌 경우.
        """
        # JSON 배열 추출 — LLM이 감싼 텍스트가 있을 수 있으므로 [ ] 사이를 탐색
        text = raw.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            msg = f"JSON array not found in LLM response: {text[:100]}"
            raise ValueError(msg)

        data = json.loads(text[start : end + 1])
        if not isinstance(data, list):
            msg = f"Expected JSON array, got {type(data).__name__}"
            raise ValueError(msg)

        relations: list[tuple[str, EdgeKind, float]] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            # target 인덱스 검증
            target_idx = item.get("target")
            if not isinstance(target_idx, int) or target_idx < 0 or target_idx >= len(candidates):
                logger.debug("Invalid target index: %s", target_idx)
                continue

            # relation → EdgeKind 매핑
            relation_str = str(item.get("relation", "")).lower().strip()
            edge_kind = _RELATION_MAP.get(relation_str)
            if edge_kind is None:
                logger.debug("Unknown relation type: %s", relation_str)
                continue

            # weight 검증 (0.0~1.0)
            weight = item.get("weight", 0.5)
            if not isinstance(weight, (int, float)):
                weight = 0.5
            weight = max(0.0, min(1.0, float(weight)))

            target_id = candidates[target_idx].id
            relations.append((target_id, edge_kind, weight))

        return relations
