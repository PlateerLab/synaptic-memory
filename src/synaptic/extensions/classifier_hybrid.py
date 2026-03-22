"""HybridClassifier — two-stage classification: rule-based → LLM fallback.

Classifies with RuleBasedClassifier first; if confidence is low,
delegates to LLMClassifier for both accuracy and cost efficiency.
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
    """Two-stage classification: rule-based → LLM fallback.

    Parameters
    ----------
    rule_classifier:
        RuleBasedClassifier instance (requires classify_with_confidence).
    llm_classifier:
        LLMClassifier instance (uses classify_async).
    confidence_threshold:
        If confidence is at or above this value, the rule-based result is accepted.
        Below this threshold, classification is delegated to the LLM.
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
        """KindClassifier protocol compliance — synchronous classification.

        Since LLM is async, synchronous classify returns the rule-based result.
        Use classify_async() in async environments.
        """
        kind, confidence = self.rule_classifier.classify_with_confidence(title, content)
        if confidence >= self.confidence_threshold:
            return kind
        # LLM fallback is async → return rule result in sync call
        return kind

    async def classify_async(self, title: str, content: str) -> ClassificationResult:
        """Async two-stage classification — delegates to LLM when confidence is low.

        Returns
        -------
        ClassificationResult
            Minimal metadata when rule-based is accepted; rich metadata when delegated to LLM.
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
