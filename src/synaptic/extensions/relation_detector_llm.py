"""LLM-based relation detector.

After rule-based candidate extraction (InvertedIndex + vector search),
the LLM judges semantic relations and determines EdgeKind and weight.

Automatically falls back to RuleBasedRelationDetector on LLM call failure.
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
    """LLM-based relation detector.

    Candidate extraction uses InvertedIndex (title mention) + vector search,
    and relation judgment is delegated to the LLM. Falls back to fallback detector on LLM failure.

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
        """Initialize LLMRelationDetector.

        Args:
            llm: LLM text generation provider.
            fallback: Rule-based detector to use on LLM failure.
                      None = return empty list on LLM failure.
            max_candidates: Maximum candidate nodes to send to LLM.
            max_edges_per_node: Maximum relations to return.
        """
        self._llm = llm
        self._fallback = fallback
        self._max_candidates = max_candidates
        self._max_edges = max_edges_per_node
        # Share index with fallback to prevent duplicate indexing
        self._index = fallback.index if fallback is not None else InvertedIndex()

    @property
    def index(self) -> InvertedIndex:
        """Internal inverted index. Used by graph.py to update the index on add/remove."""
        return self._index

    async def detect(
        self, node: Node, backend: StorageBackend
    ) -> list[tuple[str, EdgeKind, float]]:
        """Detect relations between a new node and existing nodes via LLM.

        1. Extract candidates via InvertedIndex + vector search
        2. Request relation judgment from LLM
        3. Parse JSON → EdgeKind mapping

        Uses fallback detector on LLM call/parsing failure.

        Args:
            node: New node to detect relations for.
            backend: StorageBackend for candidate lookup.

        Returns:
            [(target_node_id, edge_kind, weight), ...] up to max_edges_per_node.
        """
        # 1. Extract candidates
        candidates = await self._gather_candidates(node, backend)
        if not candidates:
            return []

        # 2. LLM relation judgment
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

    async def _gather_candidates(self, node: Node, backend: StorageBackend) -> list[Node]:
        """Collect candidate nodes via InvertedIndex + vector search.

        Args:
            node: Newly added node.
            backend: StorageBackend for candidate lookup.

        Returns:
            Deduplicated list of candidate nodes (up to max_candidates).
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
        """Build user prompt to send to the LLM.

        Args:
            node: Newly added node.
            candidates: Candidate nodes for relation judgment.

        Returns:
            Formatted prompt string.
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
        """Parse LLM JSON response and return a list of relations.

        Args:
            raw: JSON string returned by the LLM.
            candidates: Candidate node list (for index mapping).

        Returns:
            [(target_node_id, edge_kind, weight), ...] valid entries only.

        Raises:
            ValueError: JSON parsing failure or non-array response.
        """
        # Extract JSON array — search between [ ] as LLM may wrap the output in text
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

            # Validate target index
            target_idx = item.get("target")
            if not isinstance(target_idx, int) or target_idx < 0 or target_idx >= len(candidates):
                logger.debug("Invalid target index: %s", target_idx)
                continue

            # Map relation → EdgeKind
            relation_str = str(item.get("relation", "")).lower().strip()
            edge_kind = _RELATION_MAP.get(relation_str)
            if edge_kind is None:
                logger.debug("Unknown relation type: %s", relation_str)
                continue

            # Validate weight (0.0~1.0)
            weight = item.get("weight", 0.5)
            if not isinstance(weight, (int, float)):
                weight = 0.5
            weight = max(0.0, min(1.0, float(weight)))

            target_id = candidates[target_idx].id
            relations.append((target_id, edge_kind, weight))

        return relations
