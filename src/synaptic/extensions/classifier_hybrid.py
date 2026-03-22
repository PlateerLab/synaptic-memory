"""HybridClassifier — 2단계 분류: 규칙 기반 → LLM fallback.

RuleBasedClassifier로 먼저 분류하고, confidence가 낮으면
LLMClassifier로 위임하여 정확도와 비용 효율을 모두 확보한다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from synaptic.models import NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.classifier_llm import ClassificationResult, LLMClassifier
    from synaptic.extensions.classifier_rules import RuleBasedClassifier

logger = logging.getLogger(__name__)


class HybridClassifier:
    """2단계 분류: 규칙 기반 → LLM fallback.

    Parameters
    ----------
    rule_classifier:
        RuleBasedClassifier 인스턴스 (classify_with_confidence 필요).
    llm_classifier:
        LLMClassifier 인스턴스 (classify_async 사용).
    confidence_threshold:
        이 값 이상이면 규칙 기반 결과를 확정.
        미만이면 LLM에 위임.
    """

    __slots__ = ("rule_classifier", "llm_classifier", "confidence_threshold")

    def __init__(
        self,
        rule_classifier: RuleBasedClassifier,
        llm_classifier: LLMClassifier,
        *,
        confidence_threshold: float = 0.6,
    ) -> None:
        self.rule_classifier = rule_classifier
        self.llm_classifier = llm_classifier
        self.confidence_threshold = confidence_threshold

    def classify(self, title: str, content: str) -> NodeKind:
        """KindClassifier 프로토콜 준수 — 동기 분류.

        LLM은 async이므로 동기 classify에서는 규칙 기반 결과를 반환한다.
        비동기 환경에서는 classify_async()를 사용할 것.
        """
        kind, confidence = self.rule_classifier.classify_with_confidence(title, content)
        if confidence >= self.confidence_threshold:
            return kind
        # LLM fallback은 async → 동기 호출에서는 rule 결과 반환
        return kind

    async def classify_async(self, title: str, content: str) -> ClassificationResult:
        """비동기 2단계 분류 — confidence 부족 시 LLM 위임.

        Returns
        -------
        ClassificationResult
            규칙 기반 확정 시 최소 메타데이터, LLM 위임 시 풍부한 메타데이터.
        """
        from synaptic.extensions.classifier_llm import ClassificationResult

        kind, confidence = self.rule_classifier.classify_with_confidence(title, content)
        if confidence >= self.confidence_threshold:
            return ClassificationResult(
                kind=kind,
                tags=[],
                search_keywords=[],
                search_scenarios=[],
                summary=title,
                confidence=confidence,
            )
        return await self.llm_classifier.classify_async(title, content)
