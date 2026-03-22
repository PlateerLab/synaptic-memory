"""LLM-based NodeKind classifier — 풍부한 메타데이터 자동 생성.

LLM이 나중에 꺼내 쓸 지식을, LLM이 잘 찾을 수 있는 구조로 적재한다.
적재 시점에 "이 지식을 나중에 언제 찾게 될지"까지 예측하여 메타데이터 생성.

classify()는 동기 프로토콜 호환 — 캐시 히트면 반환, 아니면 fallback.
classify_async()가 LLM 호출로 ClassificationResult를 생성하며,
결과는 content 해시 기반 LRU 캐시에 보관된다.
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
# 분류 결과
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ClassificationResult:
    """LLM 분류 결과 — 검색 최적화 메타데이터 포함."""

    kind: NodeKind
    tags: list[str]
    search_keywords: list[str]
    search_scenarios: list[str]
    summary: str
    confidence: float = 0.8


# ---------------------------------------------------------------------------
# 시스템 프롬프트
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
너는 지식 그래프 온톨로지 엔지니어다. 주어진 텍스트를 분석하여 지식 노드의 메타데이터를 생성한다.

이 지식은 LLM 에이전트가 나중에 검색해서 사용할 것이다.
"이 지식을 나중에 언제, 어떤 상황에서 찾게 될까?"를 예측하여 메타데이터를 설계하라.

반드시 아래 JSON 형식으로 응답하라:
{
  "kind": "concept|entity|lesson|decision|rule|artifact",
  "confidence": 0.0~1.0,
  "tags": ["태그1", "태그2"],
  "search_keywords": ["이 지식을 찾을 때 사용할 검색어들"],
  "search_scenarios": ["이 지식이 필요한 상황 묘사"],
  "summary": "1줄 요약"
}

kind 분류 기준:
- concept: 일반적인 개념, 정의, 설명
- entity: 회사, 제품, 인물, 시스템 등 고유 대상
- lesson: 경험에서 배운 교훈, 장애 사례, 실패/성공 패턴
- decision: 의사결정과 근거, 대안 비교
- rule: 정책, 규정, 가이드라인, 제약 조건
- artifact: API, 문서, 코드, 시스템 컴포넌트

tags: 도메인/기술 키워드 3~7개, 한글+영어 포함
search_keywords: 이 지식을 찾을 에이전트가 입력할 검색 쿼리 3~5개, "~하려면", "~할 때" 같은 실용적 질문형 포함
search_scenarios: 이 지식이 필요한 구체적 상황 1~2개

JSON만 출력하라. 설명이나 사고 과정을 쓰지 마라. /no_think"""

# content 최대 길이 (토큰 절약)
_MAX_CONTENT_LEN = 2000

# NodeKind로 변환 가능한 값
_VALID_KINDS = {k.value for k in NodeKind}


# ---------------------------------------------------------------------------
# LLM 캐시 (content 해시 기반 LRU)
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
    """LLM 기반 NodeKind 분류기 — 검색 최적화 메타데이터 자동 생성.

    Parameters
    ----------
    llm:
        LLMProvider 프로토콜 구현체 (OllamaLLMProvider, OpenAILLMProvider 등).
    fallback:
        LLM 실패 시 사용할 KindClassifier. 기본값 None이면 CONCEPT 반환.
    cache_maxsize:
        content 해시 기반 LRU 캐시 크기.
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

    # -- 동기 프로토콜 호환 (KindClassifier) --

    def classify(self, title: str, content: str) -> NodeKind:
        """동기 분류 — 캐시 히트면 반환, 아니면 fallback.

        asyncio.run()은 사용하지 않는다. 비동기 결과가 필요하면
        classify_async()를 사용하고, 이후 get_cached_result()로 조회.
        """
        cached = self.get_cached_result(title, content)
        if cached is not None:
            return cached.kind

        if self._fallback is not None:
            return self._fallback.classify(title, content)

        return NodeKind.CONCEPT

    # -- 비동기 LLM 분류 --

    async def classify_async(self, title: str, content: str) -> ClassificationResult:
        """LLM 호출로 풍부한 분류 메타데이터 생성.

        결과는 캐시에 저장되며, 이후 classify()나 get_cached_result()로
        동기적으로 조회할 수 있다.
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

    # -- 캐시 조회 --

    def get_cached_result(self, title: str, content: str) -> ClassificationResult | None:
        """캐시에서 분류 결과 조회. graph.py 등에서 classify_async 이후 사용."""
        cache_key = self._make_cache_key(title, content)
        return self._cache.get(cache_key)

    # -- 내부 메서드 --

    async def _call_llm(self, title: str, content: str) -> ClassificationResult:
        """LLM에 분류 요청 후 응답 파싱."""
        truncated = content[:_MAX_CONTENT_LEN]
        user_msg = f"제목: {title}\n내용: {truncated}"

        raw = await self._llm.generate(
            system=_SYSTEM_PROMPT,
            user=user_msg,
            max_tokens=512,
        )

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> ClassificationResult:
        """LLM 응답 JSON 파싱. 실패 시 정규식 추출 시도."""
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
        """JSON 파싱 — 직접 시도 후 코드블록 추출 fallback."""
        # 1차: 직접 파싱
        try:
            return json.loads(raw)  # type: ignore[return-value]
        except (json.JSONDecodeError, ValueError):
            pass

        # 2차: ```json ... ``` 블록 추출
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))  # type: ignore[return-value]
            except (json.JSONDecodeError, ValueError):
                pass

        # 3차: 첫 번째 { ... } 블록 추출
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))  # type: ignore[return-value]
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning("Failed to parse LLM response as JSON: %s", raw[:200])
        return {}

    def _make_fallback_result(self, title: str, content: str) -> ClassificationResult:
        """LLM 실패 시 fallback 기반 결과 생성."""
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

    @staticmethod
    def _make_cache_key(title: str, content: str) -> str:
        """title + content 해시로 캐시 키 생성."""
        h = hashlib.sha256()
        h.update(title.encode())
        h.update(content[:_MAX_CONTENT_LEN].encode())
        return h.hexdigest()[:24]

    @staticmethod
    def _ensure_str_list(val: object) -> list[str]:
        """값이 list[str]인지 확인하고 변환."""
        if isinstance(val, list):
            return [str(v) for v in val]
        return []

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))
