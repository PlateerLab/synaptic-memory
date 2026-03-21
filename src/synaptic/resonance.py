"""4-axis resonance scoring: relevance x importance x recency x vitality."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import time

from synaptic.models import Node


@dataclass(slots=True)
class ResonanceWeights:
    relevance: float = 0.4
    importance: float = 0.25
    recency: float = 0.2
    vitality: float = 0.15


_DEFAULT_WEIGHTS = ResonanceWeights()
_RECENCY_DECAY = 0.05  # Per-day decay factor


class ResonanceScorer:
    """Scores nodes using 4-axis resonance."""

    __slots__ = ("_weights",)

    def __init__(self, weights: ResonanceWeights | None = None) -> None:
        self._weights = weights or _DEFAULT_WEIGHTS

    def score(
        self,
        node: Node,
        *,
        search_score: float = 0.0,
        now: float | None = None,
        weights: ResonanceWeights | None = None,
    ) -> float:
        w = weights or self._weights
        ts = now or time()

        # Importance: net success rate
        total = max(node.access_count, 1)
        importance = (node.success_count - node.failure_count) / total
        importance = max(0.0, min(1.0, (importance + 1.0) / 2.0))  # Normalize to [0, 1]

        # Recency: exponential decay based on days since last update
        days_since = max(0.0, (ts - node.updated_at) / 86400.0)
        recency = math.exp(-_RECENCY_DECAY * days_since)

        # Vitality: already normalized [0, 1]
        vitality = max(0.0, min(1.0, node.vitality))

        # Relevance: search score, already [0, 1]
        relevance = max(0.0, min(1.0, search_score))

        return (
            w.relevance * relevance
            + w.importance * importance
            + w.recency * recency
            + w.vitality * vitality
        )
