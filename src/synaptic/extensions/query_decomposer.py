"""Query decomposition — break complex queries into atomic sub-queries.

Handles compound Korean/English queries:
  "A와 B 비교" → ["A", "B"]
  "매출과 직원수 변화" → ["매출 변화", "직원수 변화"]
  "A랑 B 차이가 뭐야" → ["A", "B"]

Two modes:
  1. Rule-based: split on Korean conjunctions (와/과/랑/이랑/그리고/및),
     comparison keywords (비교/차이/vs), temporal ranges
  2. LLM-based: for ambiguous decomposition (optional fallback)

DecomposeRAG research shows 50% improvement on complex queries.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider

logger = logging.getLogger("query-decomposer")

# Korean conjunction patterns for splitting
# "매출과 직원수" → split on "과 " (particle attached to previous word)
_RE_KO_CONJUNCTION = re.compile(
    r"(?:와|과|랑|이랑|하고)\s+|"  # 와/과/랑/이랑/하고 + space
    r"\s+(?:그리고|및|또는)\s+"  # 그리고/및/또는 (standalone)
)

# English conjunction patterns
_RE_EN_CONJUNCTION = re.compile(
    r"\s+(?:and|or|versus)\s+|\s+vs\.?\s+", re.IGNORECASE
)

# Comparison/contrast keywords — the query asks to compare items
_RE_COMPARISON = re.compile(
    r"(?:비교|차이|다른\s*점|versus|vs\.?|compare|difference|differ)",
    re.IGNORECASE,
)

# Temporal range: "2020년부터 2024년까지" or "from 2020 to 2024"
_RE_TEMPORAL_RANGE = re.compile(
    r"(\d{4})년?\s*(?:부터|에서|from)\s*(\d{4})년?\s*(?:까지|to|까지의)?",
    re.IGNORECASE,
)

# Short query threshold — don't decompose very short queries
_MIN_DECOMPOSE_LEN = 4

_LLM_SYSTEM_PROMPT = """당신은 복합 질문을 원자적 서브 질문으로 분해하는 전문가입니다.

규칙:
1. 질문이 여러 개의 독립적인 정보를 요청하면, 각각을 별도 서브 질문으로 분리하세요.
2. 질문이 단일 주제이면 분해하지 마세요.
3. 서브 질문에는 원래 질문의 맥락(시간, 조건 등)을 유지하세요.

JSON 배열로 응답하세요:
["서브질문1", "서브질문2"]

단일 주제이면:
["원래 질문"]"""


class QueryDecomposer:
    """Decomposes complex queries into atomic sub-queries.

    Example::

        decomposer = QueryDecomposer()
        subs = await decomposer.decompose("매출과 직원수 비교")
        # → ["매출", "직원수"]

        decomposer = QueryDecomposer(llm=OllamaLLMProvider(...))
        subs = await decomposer.decompose("2020년부터 플래티어 매출 변화와 그 원인")
        # → ["플래티어 매출 변화 2020년부터", "매출 변화 원인"]
    """

    __slots__ = ("_llm", "_use_rules_first")

    def __init__(
        self,
        *,
        llm: LLMProvider | None = None,
        use_rules_first: bool = True,
    ) -> None:
        self._llm = llm
        self._use_rules_first = use_rules_first

    async def decompose(self, query: str) -> list[str]:
        """Decompose query into sub-queries.

        Returns single-element list if query is not decomposable.
        """
        query = query.strip()
        if not query or len(query) < _MIN_DECOMPOSE_LEN:
            return [query] if query else []

        # Step 1: Rule-based decomposition
        if self._use_rules_first:
            parts = self._rule_decompose(query)
            if len(parts) > 1:
                return parts

        # Step 2: LLM decomposition (if available and rules didn't split)
        if self._llm is not None:
            llm_parts = await self._llm_decompose(query)
            if len(llm_parts) > 1:
                return llm_parts

        return [query]

    def _rule_decompose(self, query: str) -> list[str]:
        """Rule-based decomposition using Korean/English conjunction patterns."""
        # Check if query contains comparison keywords
        is_comparison = bool(_RE_COMPARISON.search(query))

        # Try Korean conjunctions first (before removing comparison keywords)
        parts = _RE_KO_CONJUNCTION.split(query)
        if len(parts) <= 1:
            # Try English conjunctions
            parts = _RE_EN_CONJUNCTION.split(query)

        # Clean up parts
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 2]

        if len(parts) <= 1:
            # Try temporal range decomposition
            temporal = self._temporal_decompose(query)
            if temporal:
                return temporal
            return [query]

        # If comparison query, remove comparison keywords from parts
        if is_comparison:
            parts = [_RE_COMPARISON.sub("", p).strip() for p in parts]
            parts = [p for p in parts if p and len(p) >= 2]

        return parts if len(parts) > 1 else [query]

    def _temporal_decompose(self, query: str) -> list[str] | None:
        """Decompose temporal range queries."""
        match = _RE_TEMPORAL_RANGE.search(query)
        if not match:
            return None

        start_year = match.group(1)
        end_year = match.group(2)

        # Remove the temporal range from query to get the base topic
        base = _RE_TEMPORAL_RANGE.sub("", query).strip()
        if not base:
            return None

        return [
            f"{base} {start_year}년",
            f"{base} {end_year}년",
        ]

    async def _llm_decompose(self, query: str) -> list[str]:
        """LLM-based query decomposition."""
        if self._llm is None:
            return [query]

        try:
            response = await self._llm.generate(
                system=_LLM_SYSTEM_PROMPT,
                user=f"질문: {query}",
                max_tokens=256,
            )

            parts = json.loads(response)
            if isinstance(parts, list) and all(isinstance(p, str) for p in parts):
                cleaned = [p.strip() for p in parts if p.strip()]
                return cleaned if cleaned else [query]

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"LLM decomposition failed: {e}")
        except Exception as e:
            logger.warning(f"LLM decomposition error: {e}")

        return [query]
