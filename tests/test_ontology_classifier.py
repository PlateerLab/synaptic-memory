"""Tests for OntologyClassifier — BYO embedder path.

These tests use a hand-crafted fake embedder that returns vectors close
to predetermined NodeKind targets. The goal is to validate the routing
logic (cosine selection, threshold, batch path, lazy warm-up) without
pulling in torch or a real model — the production classifier is
embedder-agnostic, so any object matching the protocol works.
"""

from __future__ import annotations

import pytest

from synaptic.extensions.ontology_classifier import (
    DEFAULT_NODE_KIND_DESCRIPTIONS,
    OntologyClassifier,
    _cosine,
)
from synaptic.models import NodeKind


# --- Cosine helper ---


class TestCosine:
    def test_identical_vectors_return_one(self):
        assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert _cosine([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_length_mismatch_returns_zero(self):
        assert _cosine([1.0], [1.0, 0.0]) == 0.0

    def test_empty_returns_zero(self):
        assert _cosine([], [1.0]) == 0.0


# --- Fake embedder ---
#
# Assigns each NodeKind its own basis vector (one-hot) and maps query
# labels to those vectors via an explicit lookup table. This gives the
# classifier a fully deterministic signal so we can test routing
# without depending on a real embedding model.


class _FakeEmbedder:
    def __init__(self, label_to_kind: dict[str, NodeKind]) -> None:
        self._label_to_kind = label_to_kind
        self._kinds = list(DEFAULT_NODE_KIND_DESCRIPTIONS.keys())
        self._dim = len(self._kinds)

    def _one_hot(self, kind: NodeKind) -> list[float]:
        vec = [0.0] * self._dim
        vec[self._kinds.index(kind)] = 1.0
        return vec

    async def embed(self, text: str) -> list[float]:
        # First — is this a NodeKind description from the default table?
        for kind, desc in DEFAULT_NODE_KIND_DESCRIPTIONS.items():
            if text == desc:
                return self._one_hot(kind)
        # Then — is it a label the test wired to a kind?
        for label, kind in self._label_to_kind.items():
            if text == label:
                return self._one_hot(kind)
        # Unknown → zero vector (classifier must return None)
        return [0.0] * self._dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


# --- Classifier routing ---


@pytest.mark.asyncio
class TestOntologyClassifier:
    async def test_warm_up_loads_all_kind_vectors(self):
        embedder = _FakeEmbedder({})
        clf = OntologyClassifier(embedder=embedder)
        await clf.warm_up()
        # One vector per default NodeKind description
        assert len(clf._kind_vectors) == len(DEFAULT_NODE_KIND_DESCRIPTIONS)

    async def test_classify_routes_rule_label(self):
        embedder = _FakeEmbedder({"규정 및 지침": NodeKind.RULE})
        clf = OntologyClassifier(embedder=embedder)
        kind = await clf.classify("규정 및 지침")
        assert kind == NodeKind.RULE

    async def test_classify_routes_decision_label(self):
        embedder = _FakeEmbedder({"운영계획": NodeKind.DECISION})
        clf = OntologyClassifier(embedder=embedder)
        kind = await clf.classify("운영계획")
        assert kind == NodeKind.DECISION

    async def test_unknown_label_returns_none(self):
        embedder = _FakeEmbedder({})
        clf = OntologyClassifier(embedder=embedder)
        kind = await clf.classify("unknown label with zero vector")
        assert kind is None

    async def test_empty_label_returns_none(self):
        embedder = _FakeEmbedder({})
        clf = OntologyClassifier(embedder=embedder)
        assert await clf.classify("") is None
        assert await clf.classify("   ") is None

    async def test_classify_many_batches(self):
        embedder = _FakeEmbedder(
            {
                "규정 및 지침": NodeKind.RULE,
                "운영계획": NodeKind.DECISION,
                "조사 및 평가": NodeKind.OBSERVATION,
                "unknown": NodeKind.RULE,  # will still map since FakeEmbedder routes it
            }
        )
        clf = OntologyClassifier(embedder=embedder)
        result = await clf.classify_many([
            "규정 및 지침",
            "운영계획",
            "조사 및 평가",
        ])
        assert result["규정 및 지침"] == NodeKind.RULE
        assert result["운영계획"] == NodeKind.DECISION
        assert result["조사 및 평가"] == NodeKind.OBSERVATION

    async def test_classify_many_skips_empty_labels(self):
        embedder = _FakeEmbedder({"regulation": NodeKind.RULE})
        clf = OntologyClassifier(embedder=embedder)
        result = await clf.classify_many(["", "regulation", "  "])
        assert result == {"regulation": NodeKind.RULE}

    async def test_score_all_returns_full_table(self):
        embedder = _FakeEmbedder({"규정 및 지침": NodeKind.RULE})
        clf = OntologyClassifier(embedder=embedder)
        scores = await clf.score_all("규정 및 지침")
        assert len(scores) == len(DEFAULT_NODE_KIND_DESCRIPTIONS)
        # RULE should have the highest score (1.0 for one-hot)
        top_kind = max(scores.items(), key=lambda kv: kv[1])[0]
        assert top_kind == NodeKind.RULE
        assert scores[NodeKind.RULE] == pytest.approx(1.0)

    async def test_threshold_blocks_weak_matches(self):
        embedder = _FakeEmbedder({"weak": NodeKind.RULE})
        clf = OntologyClassifier(embedder=embedder, threshold=1.5)
        # Even a perfect 1.0 match fails a threshold of 1.5
        kind = await clf.classify("weak")
        assert kind is None

    async def test_custom_descriptions_narrow_to_subset(self):
        subset = {
            NodeKind.RULE: "policy rule regulation",
            NodeKind.DECISION: "plan decision strategy",
        }

        class _SubsetEmbedder:
            async def embed(self, text: str) -> list[float]:
                # RULE → [1, 0], DECISION → [0, 1]
                if text == "policy rule regulation":
                    return [1.0, 0.0]
                if text == "plan decision strategy":
                    return [0.0, 1.0]
                if text == "compliance policy":
                    return [0.9, 0.1]
                return [0.0, 0.0]

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [await self.embed(t) for t in texts]

        clf = OntologyClassifier(
            embedder=_SubsetEmbedder(),
            descriptions=subset,
            threshold=0.5,
        )
        kind = await clf.classify("compliance policy")
        assert kind == NodeKind.RULE

    async def test_warm_up_is_idempotent(self):
        embedder = _FakeEmbedder({})
        clf = OntologyClassifier(embedder=embedder)
        await clf.warm_up()
        snapshot = dict(clf._kind_vectors)
        await clf.warm_up()  # second call should be a no-op
        assert clf._kind_vectors == snapshot
