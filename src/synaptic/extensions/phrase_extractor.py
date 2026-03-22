"""HippoRAG2 Dual-Node KG — passage + phrase 자동 추출 및 연결.

문서에서 핵심 phrase를 추출하고 ENTITY 노드로 그래프에 추가.
Passage 노드와 Phrase 노드를 분리하여 PPR이 phrase를 통해
다른 passage로 도달할 수 있게 한다 (multi-hop bridging).

- Passage → Phrase: CONTAINS 엣지
- 같은 phrase가 여러 passage에 등장하면 자동 bridge 역할

zero-dep: 정규식 기반 phrase 추출, LLM 불필요.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from synaptic.models import EdgeKind, NodeKind

if TYPE_CHECKING:
    from synaptic.graph import SynapticGraph

# --- Phrase 정규화 ---

# 고유명사: 대문자로 시작하는 연속 단어 (2단어 이상 또는 1단어 대문자)
_RE_PROPER_NOUN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)

# 단일 대문자 단어 (3자 이상, 일반 영단어 제외)
_RE_SINGLE_PROPER = re.compile(
    r"\b([A-Z][a-z]{2,})\b"
)

# 괄호 내 약어: (MSU), (API), (LLM) 등
_RE_ABBREVIATION = re.compile(
    r"\(([A-Z]{2,8})\)"
)

# 한국어 고유명사: 따옴표/괄호 내 텍스트
_RE_KO_QUOTED = re.compile(
    "[\u300c\u300e\u201c\u2018]([\u0020-\u007e\uac00-\ud7a3\u3131-\u3163\u00b7\\-]+)[\u300d\u300f\u201d\u2019]"
)

# 한국어 괄호 내 고유명사: (주)플래티어, (재)한국재단 등
_RE_KO_PARENS = re.compile(
    r"\((?:주|사|재|학|재단|사단)\)([\w]+)"
)

# 연도: 4자리 숫자 (1000~2999)
_RE_YEAR = re.compile(
    r"\b([12]\d{3})\b"
)

# 일반적인 영어 stop words (이것들만 있는 구문은 phrase로 인정하지 않음)
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "it", "its", "this", "that",
    "these", "those", "and", "or", "but", "if", "then", "else", "when",
    "where", "how", "what", "which", "who", "whom", "whose", "there",
    "here", "not", "no", "nor", "so", "for", "of", "in", "on", "at",
    "to", "from", "by", "with", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "about", "up", "down", "very", "just",
    "also", "than", "too", "only", "own", "same", "such", "both", "each",
    "few", "more", "most", "other", "some", "all", "any", "every", "new",
})


def _normalize_phrase(phrase: str) -> str:
    """Phrase를 정규화한다: strip + NFC 정규화."""
    return unicodedata.normalize("NFC", phrase.strip())


def _is_meaningful(phrase: str) -> bool:
    """Phrase가 의미 있는지 검사한다.

    제외 조건:
    - stop word만으로 구성된 구문
    - 숫자만으로 구성된 구문 (연도 제외 — 연도는 별도 regex에서 처리)
    - 1글자 phrase
    """
    stripped = phrase.strip()
    if len(stripped) < 2:
        return False
    # 숫자만으로 구성 (연도는 _RE_YEAR에서 이미 처리하므로 여기선 제외 가능)
    if stripped.isdigit():
        return False
    words = phrase.lower().split()
    non_stop = [w for w in words if w not in _STOP_WORDS]
    return len(non_stop) > 0


class PhraseExtractor:
    """문서에서 핵심 phrase를 추출하고 그래프에 phrase 노드로 추가.

    HippoRAG2의 dual-node KG에서 영감.
    passage 노드와 phrase 노드를 분리하여 PPR이 phrase를 통해
    다른 passage로 도달할 수 있게 함 (multi-hop bridging).

    Example::

        extractor = PhraseExtractor(max_phrases_per_node=10)
        graph = SynapticGraph(backend, phrase_extractor=extractor)
        # graph.add() 시 자동으로 phrase 추출 및 연결
        node = await graph.add("Bonn 개요", "Bonn은 독일의 도시...")

    Phrase 노드는 ``NodeKind.ENTITY`` 타입으로 생성되며,
    ``_phrase`` tag가 자동 부여되어 일반 노드와 구분된다.
    """

    __slots__ = ("_min_phrase_len", "_max_phrases", "_phrase_cache")

    def __init__(
        self,
        *,
        min_phrase_length: int = 2,
        max_phrases_per_node: int = 5,
    ) -> None:
        """PhraseExtractor를 초기화한다.

        Args:
            min_phrase_length: phrase 최소 글자 수 (이보다 짧으면 무시).
            max_phrases_per_node: 한 문서에서 추출할 최대 phrase 수.
        """
        self._min_phrase_len = min_phrase_length
        self._max_phrases = max_phrases_per_node
        # phrase 정규화 텍스트 → node_id 캐시 (동일 phrase 재사용)
        self._phrase_cache: dict[str, str] = {}

    async def extract_and_link(
        self,
        graph: SynapticGraph,
        node_id: str,
        title: str,
        content: str,
    ) -> list[str]:
        """passage 노드에서 phrase를 추출하여 ENTITY 노드로 추가하고 연결한다.

        1. title + content에서 핵심 phrase 추출 (정규식 기반, zero-dep)
        2. 각 phrase를 ENTITY 타입 노드로 추가 (이미 있으면 기존 노드 사용)
        3. passage 노드 → phrase 노드 CONTAINS 엣지 생성
        4. phrase가 다른 passage에도 있으면 자동으로 bridge 역할

        Args:
            graph: SynapticGraph 인스턴스 (노드/엣지 추가용).
            node_id: passage 노드 ID.
            title: passage의 제목.
            content: passage의 본문.

        Returns:
            생성된 phrase node ID 리스트.
        """
        phrases = self._extract_phrases(title, content)
        if not phrases:
            return []

        phrase_node_ids: list[str] = []

        for phrase in phrases:
            normalized = _normalize_phrase(phrase).lower()

            # 캐시에서 기존 phrase 노드 ID 조회
            if normalized in self._phrase_cache:
                phrase_node_id = self._phrase_cache[normalized]
                # 노드가 실제로 존재하는지 확인
                existing = await graph.backend.get_node(phrase_node_id)
                if existing is not None:
                    # 기존 phrase 노드에 CONTAINS 엣지만 추가
                    await graph.link(
                        node_id, phrase_node_id,
                        kind=EdgeKind.CONTAINS,
                        weight=0.8,
                    )
                    phrase_node_ids.append(phrase_node_id)
                    continue
                # 캐시 stale → 제거 후 새로 생성
                del self._phrase_cache[normalized]

            # 새 phrase 노드 생성 (relation_detector 중복 방지를 위해
            # graph.add가 아닌 store를 직접 사용)
            phrase_node = await graph._store.add_node(
                title=phrase,
                content="",  # minimal content to avoid FTS noise
                kind=NodeKind.ENTITY,
                tags=["_phrase"],
            )
            await graph.backend.save_node(phrase_node)

            self._phrase_cache[normalized] = phrase_node.id

            # passage → phrase CONTAINS 엣지
            await graph.link(
                node_id, phrase_node.id,
                kind=EdgeKind.CONTAINS,
                weight=0.8,
            )

            phrase_node_ids.append(phrase_node.id)

        return phrase_node_ids

    def _extract_phrases(self, title: str, content: str) -> list[str]:
        """정규식 기반 phrase 추출.

        추출 규칙:
        1. 고유명사 (대문자 시작 연속 단어): "Lomonosov Moscow State University"
        2. 단일 대문자 고유명사 (3자 이상): "Bonn", "Germany"
        3. 괄호 내 약어: "(MSU)", "(API)"
        4. 한국어 고유명사 (따옴표/괄호 내): 「환불 정책」, (주)플래티어
        5. 연도: "1755", "2024"
        6. title 자체를 phrase로 포함

        중복 제거, 정규화(strip), 최대 max_phrases_per_node개 반환.

        Args:
            title: 문서 제목.
            content: 문서 본문.

        Returns:
            추출된 phrase 목록 (정규화, 중복 제거됨).
        """
        text = f"{title}\n{content}"
        seen: set[str] = set()
        phrases: list[str] = []

        def _add(phrase: str) -> None:
            normalized = _normalize_phrase(phrase)
            if len(normalized) < self._min_phrase_len:
                return
            key = normalized.lower()
            if key in seen:
                return
            if not _is_meaningful(normalized):
                return
            seen.add(key)
            phrases.append(normalized)

        # title 자체를 phrase로 포함
        _add(title)

        # 1. 고유명사 (대문자 시작 연속 단어)
        for m in _RE_PROPER_NOUN.finditer(text):
            _add(m.group(1))

        # 2. 단일 대문자 고유명사
        for m in _RE_SINGLE_PROPER.finditer(text):
            word = m.group(1)
            # 문장 시작의 일반 단어 제외 (간단한 휴리스틱)
            if word.lower() not in _STOP_WORDS:
                _add(word)

        # 3. 괄호 내 약어
        for m in _RE_ABBREVIATION.finditer(text):
            _add(m.group(1))

        # 4. 한국어 고유명사
        for m in _RE_KO_QUOTED.finditer(text):
            _add(m.group(1))
        for m in _RE_KO_PARENS.finditer(text):
            _add(m.group(1))

        # 5. 연도
        for m in _RE_YEAR.finditer(text):
            _add(m.group(1))

        return phrases[: self._max_phrases]
