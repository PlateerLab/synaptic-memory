"""Rule-based NodeKind classifier — zero-dep, deterministic.

키워드 사전으로 title + content를 매칭하여 NodeKind를 분류한다.
title 가중치 2x, content 1x. 기본값 CONCEPT.
한글 + 영어 지원, extra_keywords로 사용자 커스텀 확장 가능.
"""

from __future__ import annotations

from synaptic.models import NodeKind

# ---------------------------------------------------------------------------
# 키워드 → NodeKind 매핑 사전
# ---------------------------------------------------------------------------
_KIND_KEYWORDS: dict[NodeKind, list[str]] = {
    NodeKind.RULE: [
        # 한글
        "규정", "정책", "규칙", "가이드라인", "약관", "법률", "조항", "기준", "원칙",
        "의무", "금지", "해야 한다", "하여야 한다", "불허", "준수",
        # 영어
        "regulation", "policy", "rule", "guideline", "terms", "law", "clause",
        "standard", "principle", "must", "shall", "prohibited", "mandatory",
        "compliance", "obligation", "forbidden",
    ],
    NodeKind.LESSON: [
        # 한글
        "교훈", "장애", "실패", "사고", "사례", "경험", "주의", "오류",
        "다음에는", "배운 점", "깨달은 점", "회고", "원인 분석",
        # 영어
        "lesson", "failure", "incident", "case study", "experience", "caution",
        "error", "postmortem", "root cause", "retrospective", "takeaway",
        "lessons learned", "what went wrong",
    ],
    NodeKind.DECISION: [
        # 한글
        "결정", "선택", "채택", "결론", "판단", "합의",
        "대안", "선택한 이유", "의사결정", "결재",
        # 영어
        "decision", "choice", "adoption", "conclusion", "judgment", "consensus",
        "trade-off", "tradeoff", "decided", "alternative", "pros and cons",
        "rationale",
    ],
    NodeKind.ENTITY: [
        # 한글
        "회사", "기관", "조직", "제품", "서비스", "인물", "도시", "국가",
        "주식회사", "법인", "재단",
        # 영어
        "company", "organization", "institution", "product", "service", "person",
        "city", "country", "Inc.", "Corp.", "Ltd.", "LLC", "GmbH", "Co.",
    ],
    NodeKind.ARTIFACT: [
        # 한글
        "API", "문서", "보고서", "코드", "시스템", "도구", "프로토콜",
        "스키마", "엔드포인트", "배포", "릴리즈",
        # 영어
        "document", "report", "code", "system", "tool", "protocol",
        "framework", "library", "endpoint", "schema", "/api/", "v1", "v2",
        "repository", "package", "module", "artifact", "release",
    ],
}


class RuleBasedClassifier:
    """키워드 규칙 기반 NodeKind 분류기.

    Parameters
    ----------
    extra_keywords:
        추가 키워드 사전. ``{NodeKind.RULE: ["커스텀1", "custom2"]}`` 형태로
        기본 사전을 확장할 수 있다.
    """

    def __init__(
        self,
        extra_keywords: dict[NodeKind, list[str]] | None = None,
    ) -> None:
        # 기본 사전 복사 후 확장
        self._keywords: dict[NodeKind, list[str]] = {
            kind: list(kws) for kind, kws in _KIND_KEYWORDS.items()
        }
        if extra_keywords:
            for kind, kws in extra_keywords.items():
                if kind in self._keywords:
                    self._keywords[kind].extend(kws)
                else:
                    self._keywords[kind] = list(kws)

    def classify(self, title: str, content: str) -> NodeKind:
        """title + content 키워드 매칭으로 NodeKind 결정.

        title 매칭 시 가중치 2, content 매칭 시 가중치 1.
        매칭되는 키워드가 없으면 ``NodeKind.CONCEPT`` 반환.
        """
        kind, _ = self.classify_with_confidence(title, content)
        return kind

    def classify_with_confidence(self, title: str, content: str) -> tuple[NodeKind, float]:
        """title + content 키워드 매칭으로 NodeKind와 confidence 반환.

        title 매칭 시 가중치 2, content 매칭 시 가중치 1.
        confidence는 ``min(1.0, total_score / 6.0)`` 으로 정규화.
        매칭되는 키워드가 없으면 ``(NodeKind.CONCEPT, 0.0)`` 반환.
        """
        title_lower = title.lower()
        content_lower = content.lower()

        best_kind = NodeKind.CONCEPT
        best_score = 0

        for kind, keywords in self._keywords.items():
            score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in title_lower:
                    score += 2
                if kw_lower in content_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_kind = kind

        confidence = min(1.0, best_score / 6.0) if best_score > 0 else 0.0
        return best_kind, confidence
