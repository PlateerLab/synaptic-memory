"""Rule-based NodeKind classifier — zero-dep, deterministic.

Matches title + content against keyword dictionaries to classify NodeKind.
Title weight 2x, content 1x. Defaults to CONCEPT.
Supports Korean + English; extensible via extra_keywords.
"""

from __future__ import annotations

from synaptic.models import NodeKind

# ---------------------------------------------------------------------------
# Keyword → NodeKind mapping dictionary
# ---------------------------------------------------------------------------
_KIND_KEYWORDS: dict[NodeKind, list[str]] = {
    NodeKind.RULE: [
        # Korean
        "규정", "정책", "규칙", "가이드라인", "약관", "법률", "조항", "기준", "원칙",
        "의무", "금지", "해야 한다", "하여야 한다", "불허", "준수",
        # English
        "regulation", "policy", "rule", "guideline", "terms", "law", "clause",
        "standard", "principle", "must", "shall", "prohibited", "mandatory",
        "compliance", "obligation", "forbidden",
    ],
    NodeKind.LESSON: [
        # Korean
        "교훈", "장애", "실패", "사고", "사례", "경험", "주의", "오류",
        "다음에는", "배운 점", "깨달은 점", "회고", "원인 분석",
        # English
        "lesson", "failure", "incident", "case study", "experience", "caution",
        "error", "postmortem", "root cause", "retrospective", "takeaway",
        "lessons learned", "what went wrong",
    ],
    NodeKind.DECISION: [
        # Korean
        "결정", "선택", "채택", "결론", "판단", "합의",
        "대안", "선택한 이유", "의사결정", "결재",
        # English
        "decision", "choice", "adoption", "conclusion", "judgment", "consensus",
        "trade-off", "tradeoff", "decided", "alternative", "pros and cons",
        "rationale",
    ],
    NodeKind.ENTITY: [
        # Korean
        "회사", "기관", "조직", "제품", "서비스", "인물", "도시", "국가",
        "주식회사", "법인", "재단",
        # English
        "company", "organization", "institution", "product", "service", "person",
        "city", "country", "Inc.", "Corp.", "Ltd.", "LLC", "GmbH", "Co.",
    ],
    NodeKind.ARTIFACT: [
        # Korean
        "API", "문서", "보고서", "코드", "시스템", "도구", "프로토콜",
        "스키마", "엔드포인트", "배포", "릴리즈",
        # English
        "document", "report", "code", "system", "tool", "protocol",
        "framework", "library", "endpoint", "schema", "/api/", "v1", "v2",
        "repository", "package", "module", "artifact", "release",
    ],
}


class RuleBasedClassifier:
    """Keyword rule-based NodeKind classifier.

    Parameters
    ----------
    extra_keywords:
        Additional keyword dictionary. Extends the default dictionary
        in the form ``{NodeKind.RULE: ["custom1", "custom2"]}``.
    """

    def __init__(
        self,
        extra_keywords: dict[NodeKind, list[str]] | None = None,
    ) -> None:
        # Copy default dictionary and extend
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
        """Determine NodeKind by keyword matching on title + content.

        Title match weight is 2, content match weight is 1.
        Returns ``NodeKind.CONCEPT`` if no keywords match.
        """
        kind, _ = self.classify_with_confidence(title, content)
        return kind

    def classify_with_confidence(self, title: str, content: str) -> tuple[NodeKind, float]:
        """Return NodeKind and confidence by keyword matching on title + content.

        Title match weight is 2, content match weight is 1.
        Confidence is normalized as ``min(1.0, total_score / 6.0)``.
        Returns ``(NodeKind.CONCEPT, 0.0)`` if no keywords match.
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
