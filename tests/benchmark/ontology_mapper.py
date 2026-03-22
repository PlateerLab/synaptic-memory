"""외부 데이터셋 corpus → 온톨로지 자동 매핑.

키워드 규칙 기반으로 NodeKind 분류, tag 추출, Edge 관계 추출.
LLM 의존성 없이 순수 Python으로 동작하며 deterministic.
"""

from __future__ import annotations

import re
from collections import Counter

from synaptic.models import EdgeKind, NodeKind


# ---------------------------------------------------------------------------
# 키워드 → NodeKind 매핑 사전
# ---------------------------------------------------------------------------
_KIND_KEYWORDS: dict[NodeKind, list[str]] = {
    NodeKind.RULE: [
        "규정", "정책", "규칙", "가이드라인", "약관", "법률", "조항", "기준", "원칙",
        "regulation", "policy", "rule", "guideline", "terms", "law", "clause",
        "standard", "principle",
    ],
    NodeKind.LESSON: [
        "교훈", "장애", "실패", "사고", "사례", "경험", "주의", "오류",
        "lesson", "failure", "incident", "case study", "experience", "caution",
        "error", "postmortem",
    ],
    NodeKind.ENTITY: [
        "회사", "기관", "조직", "제품", "서비스", "인물", "도시", "국가",
        "company", "organization", "institution", "product", "service", "person",
        "city", "country",
    ],
    NodeKind.DECISION: [
        "결정", "선택", "채택", "결론", "판단", "합의",
        "decision", "choice", "adoption", "conclusion", "judgment", "consensus",
    ],
    NodeKind.ARTIFACT: [
        "API", "문서", "보고서", "코드", "시스템", "도구", "프로토콜",
        "document", "report", "code", "system", "tool", "protocol",
        "framework", "library",
    ],
}

# ---------------------------------------------------------------------------
# 도메인 사전 (tag 추출용)
# ---------------------------------------------------------------------------
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "의료": ["의료", "의학", "건강", "질병", "진료", "환자", "병원", "약물", "치료", "증상",
             "medical", "health", "disease", "patient", "hospital", "treatment"],
    "법률": ["법률", "법원", "판결", "소송", "계약", "헌법", "재판", "변호사",
             "legal", "court", "verdict", "lawsuit", "contract", "constitution"],
    "기술": ["기술", "소프트웨어", "하드웨어", "프로그래밍", "알고리즘", "데이터", "클라우드", "서버",
             "technology", "software", "hardware", "programming", "algorithm", "data", "cloud"],
    "금융": ["금융", "은행", "투자", "주식", "보험", "대출", "이자", "자산",
             "finance", "bank", "investment", "stock", "insurance", "loan", "asset"],
    "교육": ["교육", "학교", "대학", "학습", "교사", "학생", "교과", "입학",
             "education", "school", "university", "learning", "teacher", "student"],
    "과학": ["과학", "연구", "실험", "논문", "물리", "화학", "생물", "수학",
             "science", "research", "experiment", "physics", "chemistry", "biology"],
    "환경": ["환경", "기후", "탄소", "오염", "생태", "재활용", "에너지",
             "environment", "climate", "carbon", "pollution", "ecology", "energy"],
    "정치": ["정치", "정부", "국회", "선거", "외교", "정당", "대통령",
             "politics", "government", "parliament", "election", "diplomacy"],
    "경제": ["경제", "GDP", "무역", "수출", "수입", "성장률", "인플레이션",
             "economy", "trade", "export", "import", "growth", "inflation"],
    "문화": ["문화", "예술", "영화", "음악", "문학", "축제", "유산",
             "culture", "art", "film", "music", "literature", "festival", "heritage"],
    "스포츠": ["스포츠", "축구", "야구", "농구", "올림픽", "선수", "경기",
              "sports", "football", "baseball", "basketball", "olympics", "athlete"],
}

# ---------------------------------------------------------------------------
# 한글 단어 추출 패턴
# ---------------------------------------------------------------------------
_HANGUL_WORD_RE = re.compile(r"[가-힣]{2,}")
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]{3,}")

# 전수 비교 최대 corpus 크기
_MAX_FULL_COMPARE = 2000


class OntologyMapper:
    """외부 데이터셋 corpus를 온톨로지 구조로 자동 매핑."""

    def __init__(self, corpus: dict[str, dict[str, str]]) -> None:
        self._corpus = corpus
        # 캐시: doc_id → (title_lower, text_lower)
        self._normalized: dict[str, tuple[str, str]] = {}
        # 캐시: doc_id → NodeKind
        self._kind_cache: dict[str, NodeKind] = {}
        # 캐시: doc_id → tags
        self._tag_cache: dict[str, list[str]] = {}

        for doc_id, doc in corpus.items():
            title = doc.get("title", "")
            text = doc.get("text", "")
            self._normalized[doc_id] = (title.lower(), text.lower())

    def classify(self, doc_id: str) -> NodeKind:
        """title + text 키워드 매칭으로 NodeKind 결정. title 가중치 2x."""
        if doc_id in self._kind_cache:
            return self._kind_cache[doc_id]

        title_lower, text_lower = self._normalized[doc_id]
        combined = title_lower + " " + text_lower

        best_kind = NodeKind.CONCEPT
        best_score = 0

        for kind, keywords in _KIND_KEYWORDS.items():
            score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in title_lower:
                    score += 2  # title 가중치 2x
                if kw_lower in combined:
                    score += 1
            if score > best_score:
                best_score = score
                best_kind = kind

        self._kind_cache[doc_id] = best_kind
        return best_kind

    def extract_tags(self, doc_id: str) -> list[str]:
        """도메인 사전 매칭 + title 핵심 단어 추출. 최대 5개."""
        if doc_id in self._tag_cache:
            return self._tag_cache[doc_id]

        title_lower, text_lower = self._normalized[doc_id]
        combined = title_lower + " " + text_lower
        tags: list[str] = []

        # 1) 도메인 매칭
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in combined:
                    tags.append(domain)
                    break  # 도메인당 하나만

        # 2) title에서 핵심 단어 추출 (2글자 이상 한글, 3글자 이상 영어)
        doc = self._corpus[doc_id]
        title = doc.get("title", "")
        hangul_words = _HANGUL_WORD_RE.findall(title)
        english_words = _ENGLISH_WORD_RE.findall(title)

        # 불용어 제거
        stopwords_ko = {"에서", "으로", "에게", "대한", "에서는", "대해", "하는", "있는", "없는", "되는", "위한", "이는", "통해"}
        stopwords_en = {"the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "has", "have", "been"}

        for w in hangul_words:
            if w not in stopwords_ko and w not in tags:
                tags.append(w)
        for w in english_words:
            if w.lower() not in stopwords_en and w not in tags:
                tags.append(w)

        tags = tags[:5]
        self._tag_cache[doc_id] = tags
        return tags

    def extract_edges(
        self, id_map: dict[str, str]
    ) -> list[tuple[str, str, str, float]]:
        """문서 간 Edge 관계 추출.

        Args:
            id_map: corpus doc_id → graph node_id 매핑

        Returns:
            list of (source_node_id, target_node_id, edge_kind, weight)
        """
        edges: list[tuple[str, str, str, float]] = []
        seen: set[tuple[str, str]] = set()
        # 노드당 edge 수 제한 (과도한 연결 방지)
        edge_count_per_node: Counter[str] = Counter()
        max_edges_per_node = 5

        corpus_ids = [cid for cid in self._corpus if cid in id_map]
        n = len(corpus_ids)
        full_compare = n <= _MAX_FULL_COMPARE

        # 사전 계산: 각 문서의 tags, kind
        tags_map: dict[str, list[str]] = {}
        kind_map: dict[str, NodeKind] = {}
        for cid in corpus_ids:
            tags_map[cid] = self.extract_tags(cid)
            kind_map[cid] = self.classify(cid)

        # title → corpus_id 역인덱스 (빈 title 제외, 짧은 title 무시)
        title_to_cid: dict[str, str] = {}
        for cid in corpus_ids:
            title = self._corpus[cid].get("title", "").strip()
            if title and len(title) >= 4:  # 너무 짧은 title은 false positive 유발
                title_to_cid[title] = cid

        def _add_edge(
            src_cid: str, tgt_cid: str, kind: EdgeKind, weight: float
        ) -> None:
            src_nid = id_map[src_cid]
            tgt_nid = id_map[tgt_cid]
            if src_nid == tgt_nid:
                return
            # 노드당 edge 수 제한
            if edge_count_per_node[src_nid] >= max_edges_per_node:
                return
            if edge_count_per_node[tgt_nid] >= max_edges_per_node:
                return
            pair = (src_nid, tgt_nid)
            reverse_pair = (tgt_nid, src_nid)
            if pair in seen or reverse_pair in seen:
                return
            seen.add(pair)
            edge_count_per_node[src_nid] += 1
            edge_count_per_node[tgt_nid] += 1
            edges.append((src_nid, tgt_nid, kind.value, weight))

        # 방법 1: 문서 A의 content에 문서 B의 title이 언급
        for cid_a in corpus_ids:
            text_a = self._corpus[cid_a].get("text", "")
            if not text_a:
                continue
            text_a_lower = text_a.lower()
            for title_b, cid_b in title_to_cid.items():
                if cid_a == cid_b:
                    continue
                if len(title_b) < 2:
                    continue
                if title_b.lower() in text_a_lower:
                    _add_edge(cid_a, cid_b, EdgeKind.RELATED, 0.8)

        if full_compare:
            # 방법 2: 공통 tag 2개 이상
            for i in range(n):
                cid_a = corpus_ids[i]
                tags_a = set(tags_map[cid_a])
                if len(tags_a) < 2:
                    continue
                for j in range(i + 1, n):
                    cid_b = corpus_ids[j]
                    tags_b = set(tags_map[cid_b])
                    common = tags_a & tags_b
                    if len(common) >= 3:
                        _add_edge(cid_a, cid_b, EdgeKind.RELATED, 0.5)

            # 방법 3: NodeKind 쌍 규칙
            for i in range(n):
                cid_a = corpus_ids[i]
                kind_a = kind_map[cid_a]
                for j in range(n):
                    if i == j:
                        continue
                    cid_b = corpus_ids[j]
                    kind_b = kind_map[cid_b]

                    # RULE → CONCEPT: DEPENDS_ON
                    if kind_a == NodeKind.RULE and kind_b == NodeKind.CONCEPT:
                        # title 참조 관계가 있는 경우만
                        title_b = self._corpus[cid_b].get("title", "").strip()
                        if title_b and title_b.lower() in self._normalized[cid_a][1]:
                            _add_edge(cid_a, cid_b, EdgeKind.DEPENDS_ON, 0.6)

                    # LESSON → 관련 문서: LEARNED_FROM
                    if kind_a == NodeKind.LESSON:
                        title_b = self._corpus[cid_b].get("title", "").strip()
                        if title_b and title_b.lower() in self._normalized[cid_a][1]:
                            _add_edge(cid_a, cid_b, EdgeKind.LEARNED_FROM, 0.7)

        return edges

    def map_all(
        self, id_map: dict[str, str]
    ) -> dict[str, dict[str, NodeKind | list[str]]]:
        """전체 corpus에 대해 classify + extract_tags 일괄 수행.

        Returns:
            {"doc_id": {"kind": NodeKind, "tags": [...]}}
        """
        result: dict[str, dict[str, NodeKind | list[str]]] = {}
        for doc_id in self._corpus:
            if doc_id not in id_map:
                continue
            result[doc_id] = {
                "kind": self.classify(doc_id),
                "tags": self.extract_tags(doc_id),
            }
        return result
