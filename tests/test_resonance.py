"""Tests for resonance scoring."""

from __future__ import annotations

from time import time

from synaptic.models import Node
from synaptic.resonance import ResonanceScorer, ResonanceWeights


class TestResonanceScorer:
    def test_default_weights(self) -> None:
        scorer = ResonanceScorer()
        node = Node(title="Test", vitality=1.0)
        score = scorer.score(node, search_score=0.5)
        assert 0.0 <= score <= 1.0

    def test_higher_search_score_increases_resonance(self) -> None:
        scorer = ResonanceScorer()
        node = Node(title="Test")
        low = scorer.score(node, search_score=0.1)
        high = scorer.score(node, search_score=0.9)
        assert high > low

    def test_success_increases_importance(self) -> None:
        scorer = ResonanceScorer()
        now = time()
        bad = Node(title="Bad", access_count=10, success_count=2, failure_count=8, updated_at=now)
        good = Node(title="Good", access_count=10, success_count=9, failure_count=1, updated_at=now)
        bad_score = scorer.score(bad, search_score=0.5, now=now)
        good_score = scorer.score(good, search_score=0.5, now=now)
        assert good_score > bad_score

    def test_recency_decay(self) -> None:
        scorer = ResonanceScorer()
        now = time()
        recent = Node(title="Recent", updated_at=now)
        old = Node(title="Old", updated_at=now - 30 * 86400)  # 30 days ago
        recent_score = scorer.score(recent, search_score=0.5, now=now)
        old_score = scorer.score(old, search_score=0.5, now=now)
        assert recent_score > old_score

    def test_vitality_factor(self) -> None:
        scorer = ResonanceScorer()
        now = time()
        healthy = Node(title="Healthy", vitality=1.0, updated_at=now)
        weak = Node(title="Weak", vitality=0.1, updated_at=now)
        healthy_score = scorer.score(healthy, search_score=0.5, now=now)
        weak_score = scorer.score(weak, search_score=0.5, now=now)
        assert healthy_score > weak_score

    def test_custom_weights(self) -> None:
        # All weight on relevance
        weights = ResonanceWeights(relevance=1.0, importance=0.0, recency=0.0, vitality=0.0)
        scorer = ResonanceScorer(weights=weights)
        node = Node(title="Test")
        score = scorer.score(node, search_score=0.7, weights=weights)
        assert abs(score - 0.7) < 0.01

    def test_score_bounds(self) -> None:
        scorer = ResonanceScorer()
        node = Node(title="Test", vitality=1.0, success_count=100, access_count=100)
        score = scorer.score(node, search_score=1.0)
        assert 0.0 <= score <= 1.1  # Small float tolerance
