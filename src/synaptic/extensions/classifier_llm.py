"""LLM-based NodeKind classifier — automatic rich metadata generation.

Stores knowledge in a structure that LLMs can easily retrieve later.
At ingestion time, predicts "when will this knowledge be searched for"
and generates metadata accordingly.

classify() is sync-protocol-compatible — returns cached result or fallback.
classify_async() calls the LLM to produce a ClassificationResult;
results are stored in a content-hash-based LRU cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from synaptic.models import NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider
    from synaptic.protocols import KindClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ClassificationResult:
    """LLM classification result — includes search-optimized metadata."""

    kind: NodeKind
    tags: list[str]
    search_keywords: list[str]
    search_scenarios: list[str]
    summary: str
    confidence: float = 0.8


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
지식 노드의 메타데이터를 JSON으로 생성하라. /no_think

kind 분류 (가장 적합한 하나만):
- rule: "~해야 한다", "~금지", 정책, 규정, 가이드라인, 약관, 제한 조건
- lesson: 장애/실패/성공 사후 분석, 교훈, "원인은~", "다음에는~", postmortem
- decision: "~를 선택", "~를 채택", 대안 비교, trade-off, 의사결정 기록
- artifact: API 명세, 엔드포인트, 스키마, 코드, 시스템 컴포넌트, 도구
- entity: 회사명, 제품명, 인물, 도시, 고유 대상
- concept: 위에 해당 안 되면 concept

예시:
입력: "주문 후 7일 이내 환불 가능. 개봉 제품은 환불 불가."
출력: {"kind":"rule","confidence":0.95,"tags":["환불","refund","정책","주문"],"search_keywords":["환불 가능한 기간","환불 규정","개봉 제품 환불"],"search_scenarios":["고객이 환불을 요청했을 때 규정 확인"],"summary":"7일 이내 환불 가능, 개봉 제품 불가"}

입력: "PG사 API 타임아웃으로 결제 실패. 원인은 트래픽 급증. 교훈: 서킷브레이커 필요."
출력: {"kind":"lesson","confidence":0.95,"tags":["결제","PG","장애","서킷브레이커","circuit breaker"],"search_keywords":["결제 실패 원인","API 타임아웃 대응","PG사 장애 사례"],"search_scenarios":["결제 시스템 장애 발생 시 과거 사례 검색"],"summary":"PG사 타임아웃으로 결제 실패, 서킷브레이커 도입 필요"}

입력: "카나리 배포 채택. 대안 블루그린은 비용 문제로 기각."
출력: {"kind":"decision","confidence":0.9,"tags":["배포","카나리","canary","블루그린","deploy"],"search_keywords":["배포 방식 선택","카나리 vs 블루그린","배포 전략 결정"],"search_scenarios":["새 서비스 배포 전략을 결정할 때"],"summary":"카나리 배포 채택, 블루그린은 비용 문제로 기각"}

반드시 JSON만 출력. tags 3~7개, search_keywords 3~5개."""

# ---------------------------------------------------------------------------
# Batch classification system prompt
# ---------------------------------------------------------------------------

_BATCH_SYSTEM_PROMPT = """\
여러 문서의 메타데이터를 JSON 배열로 생성하라. /no_think

kind 분류 (가장 적합한 하나만):
- rule: "~해야 한다", "~금지", 정책, 규정, 가이드라인, 약관, 제한 조건
- lesson: 장애/실패/성공 사후 분석, 교훈, "원인은~", "다음에는~", postmortem
- decision: "~를 선택", "~를 채택", 대안 비교, trade-off, 의사결정 기록
- artifact: API 명세, 엔드포인트, 스키마, 코드, 시스템 컴포넌트, 도구
- entity: 회사명, 제품명, 인물, 도시, 고유 대상
- concept: 위에 해당 안 되면 concept

반드시 JSON 배열만 출력. 각 객체에 kind, confidence, tags(3~7개), \
search_keywords(3~5개), search_scenarios, summary 포함.
[{"index": 0, "kind": "...", "confidence": 0.9, "tags": [...], \
"search_keywords": [...], "search_scenarios": [...], "summary": "..."}, ...]"""

# Maximum content length (to save tokens)
_MAX_CONTENT_LEN = 2000

# Valid values convertible to NodeKind
_VALID_KINDS = {k.value for k in NodeKind}


# ---------------------------------------------------------------------------
# LLM cache (content-hash-based LRU)
# ---------------------------------------------------------------------------

class _LRUCache:
    """Thread-unsafe LRU cache backed by OrderedDict."""

    __slots__ = ("_maxsize", "_data")

    def __init__(self, maxsize: int = 512) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, ClassificationResult] = OrderedDict()

    def get(self, key: str) -> ClassificationResult | None:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value: ClassificationResult) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)


# ---------------------------------------------------------------------------
# LLMClassifier
# ---------------------------------------------------------------------------

class LLMClassifier:
    """LLM-based NodeKind classifier — automatic search-optimized metadata generation.

    Parameters
    ----------
    llm:
        LLMProvider protocol implementation (OllamaLLMProvider, OpenAILLMProvider, etc.).
    fallback:
        KindClassifier to use on LLM failure. Defaults to None (returns CONCEPT).
    cache_maxsize:
        Content-hash-based LRU cache size.
    """

    __slots__ = ("_llm", "_fallback", "_cache")

    def __init__(
        self,
        llm: LLMProvider,
        *,
        fallback: KindClassifier | None = None,
        cache_maxsize: int = 512,
    ) -> None:
        self._llm = llm
        self._fallback = fallback
        self._cache = _LRUCache(maxsize=cache_maxsize)

    # -- Sync protocol compliance (KindClassifier) --

    def classify(self, title: str, content: str) -> NodeKind:
        """Synchronous classification — returns cached result or fallback.

        Does not use asyncio.run(). For async results, use classify_async()
        and then get_cached_result() to retrieve synchronously.
        """
        cached = self.get_cached_result(title, content)
        if cached is not None:
            return cached.kind

        if self._fallback is not None:
            return self._fallback.classify(title, content)

        return NodeKind.CONCEPT

    # -- Async LLM classification --

    async def classify_async(self, title: str, content: str) -> ClassificationResult:
        """Generate rich classification metadata via LLM call.

        Results are stored in cache and can be retrieved synchronously
        via classify() or get_cached_result().
        """
        cache_key = self._make_cache_key(title, content)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            result = await self._call_llm(title, content)
        except Exception:
            logger.exception("LLM classification failed, using fallback")
            result = self._make_fallback_result(title, content)

        self._cache.put(cache_key, result)
        return result

    # -- Cache lookup --

    def get_cached_result(self, title: str, content: str) -> ClassificationResult | None:
        """Look up classification result from cache. Used after classify_async in graph.py, etc."""
        cache_key = self._make_cache_key(title, content)
        return self._cache.get(cache_key)

    # -- Internal methods --

    async def _call_llm(self, title: str, content: str) -> ClassificationResult:
        """Send classification request to LLM and parse the response."""
        truncated = content[:_MAX_CONTENT_LEN]
        user_msg = f"Title: {title}\nContent: {truncated}"

        raw = await self._llm.generate(
            system=_SYSTEM_PROMPT,
            user=user_msg,
            max_tokens=512,
        )

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> ClassificationResult:
        """Parse LLM response JSON. Falls back to regex extraction on failure."""
        data = self._extract_json(raw)

        kind_str = data.get("kind", "concept")
        if kind_str not in _VALID_KINDS:
            kind_str = "concept"

        return ClassificationResult(
            kind=NodeKind(kind_str),
            tags=self._ensure_str_list(data.get("tags", [])),
            search_keywords=self._ensure_str_list(data.get("search_keywords", [])),
            search_scenarios=self._ensure_str_list(data.get("search_scenarios", [])),
            summary=str(data.get("summary", "")),
            confidence=self._clamp(float(data.get("confidence", 0.8)), 0.0, 1.0),
        )

    @staticmethod
    def _extract_json(raw: str) -> dict[str, object]:
        """JSON parsing — direct attempt, then code block extraction fallback."""
        # 1st: Direct parsing
        try:
            return json.loads(raw)  # type: ignore[return-value]
        except (json.JSONDecodeError, ValueError):
            pass

        # 2nd: Extract ```json ... ``` block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))  # type: ignore[return-value]
            except (json.JSONDecodeError, ValueError):
                pass

        # 3rd: Extract first { ... } block
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))  # type: ignore[return-value]
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning("Failed to parse LLM response as JSON: %s", raw[:200])
        return {}

    def _make_fallback_result(self, title: str, content: str) -> ClassificationResult:
        """Generate fallback-based result on LLM failure."""
        if self._fallback is not None:
            kind = self._fallback.classify(title, content)
        else:
            kind = NodeKind.CONCEPT

        return ClassificationResult(
            kind=kind,
            tags=[],
            search_keywords=[],
            search_scenarios=[],
            summary=title,
            confidence=0.3,
        )

    # -- Batch classification --

    async def classify_batch_async(
        self,
        items: list[tuple[str, str]],
        *,
        content_limit: int = 500,
    ) -> list[ClassificationResult]:
        """Classify multiple documents in a single LLM call.

        Parameters
        ----------
        items:
            ``[(title, content), ...]`` list (max 16 items).
        content_limit:
            Truncate each document's content to this length to reduce token cost.

        Returns
        -------
        list[ClassificationResult]
            Classification results in input order.
        """
        if not items:
            return []

        # Limit to max 16 items
        items = items[:16]

        # Check cache hits
        results: list[ClassificationResult | None] = []
        uncached_indices: list[int] = []
        for i, (title, content) in enumerate(items):
            cached = self.get_cached_result(title, content)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_indices.append(i)

        if not uncached_indices:
            return results  # type: ignore[return-value]

        # Batch LLM call
        try:
            batch_results = await self._call_llm_batch(
                [(items[i][0], items[i][1]) for i in uncached_indices],
                content_limit=content_limit,
            )

            # Map results
            for idx, result in zip(uncached_indices, batch_results):
                results[idx] = result
                title, content = items[idx]
                cache_key = self._make_cache_key(title, content)
                self._cache.put(cache_key, result)

        except Exception:
            logger.exception("Batch LLM classification failed, falling back to individual calls")
            batch_results = []

        # Individual call fallback for missing items
        missing = [i for i in uncached_indices if results[i] is None]
        for i in missing:
            title, content = items[i]
            results[i] = await self.classify_async(title, content)

        return results  # type: ignore[return-value]

    async def _call_llm_batch(
        self,
        items: list[tuple[str, str]],
        *,
        content_limit: int = 500,
    ) -> list[ClassificationResult]:
        """Batch LLM call — classify multiple documents at once."""
        doc_parts: list[str] = []
        for i, (title, content) in enumerate(items):
            truncated = content[:content_limit]
            doc_parts.append(f"Document {i}:\nTitle: {title}\nContent: {truncated}")

        user_msg = "\n\n".join(doc_parts)
        user_msg += "\n\nReturn a JSON array with one object per document."

        raw = await self._llm.generate(
            system=_BATCH_SYSTEM_PROMPT,
            user=user_msg,
            max_tokens=512 * len(items),
        )

        return self._parse_batch_response(raw, len(items))

    def _parse_batch_response(self, raw: str, expected_count: int) -> list[ClassificationResult]:
        """Parse batch LLM response. Expects a JSON array."""
        data_list = self._extract_json_array(raw)

        results: list[ClassificationResult] = []
        for item in data_list[:expected_count]:
            if not isinstance(item, dict):
                continue
            kind_str = item.get("kind", "concept")
            if kind_str not in _VALID_KINDS:
                kind_str = "concept"
            results.append(ClassificationResult(
                kind=NodeKind(kind_str),
                tags=self._ensure_str_list(item.get("tags", [])),
                search_keywords=self._ensure_str_list(item.get("search_keywords", [])),
                search_scenarios=self._ensure_str_list(item.get("search_scenarios", [])),
                summary=str(item.get("summary", "")),
                confidence=self._clamp(float(item.get("confidence", 0.8)), 0.0, 1.0),
            ))

        return results

    @staticmethod
    def _extract_json_array(raw: str) -> list[dict[str, object]]:
        """JSON array parsing — direct attempt, then code block extraction fallback."""
        # 1st: Direct parsing
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed  # type: ignore[return-value]
        except (json.JSONDecodeError, ValueError):
            pass

        # 2nd: Extract ```json ... ``` block
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, list):
                    return parsed  # type: ignore[return-value]
            except (json.JSONDecodeError, ValueError):
                pass

        # 3rd: Extract first [ ... ] block
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return parsed  # type: ignore[return-value]
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning("Failed to parse batch LLM response as JSON array: %s", raw[:200])
        return []

    @staticmethod
    def _make_cache_key(title: str, content: str) -> str:
        """Generate cache key from title + content hash."""
        h = hashlib.sha256()
        h.update(title.encode())
        h.update(content[:_MAX_CONTENT_LEN].encode())
        return h.hexdigest()[:24]

    @staticmethod
    def _ensure_str_list(val: object) -> list[str]:
        """Verify and convert value to list[str]."""
        if isinstance(val, list):
            return [str(v) for v in val]
        return []

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))
