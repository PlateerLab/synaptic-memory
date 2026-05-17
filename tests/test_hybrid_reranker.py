"""Tests for HybridReranker — pure fusion logic, no IO.

The reranker is a pure function over ``ExpandedNode`` input plus
optional lexical / semantic / anchor signals, so these tests don't
need any backend — we just construct nodes directly.
"""

from __future__ import annotations

import pytest

from synaptic.extensions.graph_expander import ExpandedNode
from synaptic.extensions.hybrid_reranker import (
    HybridReranker,
    RerankerWeights,
    _cosine,
    _normalise,
)
from synaptic.models import ConsolidationLevel, Node, NodeKind


def _node(
    id_: str,
    *,
    kind: NodeKind = NodeKind.CHUNK,
    title: str = "",
    tags: list[str] | None = None,
    category: str | None = None,
    embedding: list[float] | None = None,
) -> Node:
    props = {"category": category} if category else {}
    return Node(
        id=id_,
        kind=kind,
        title=title or id_,
        content=title or id_,
        tags=tags or [],
        properties=props,
        embedding=embedding or [],
        level=ConsolidationLevel.L0_RAW,
    )


def _expanded(node: Node, reason: str = "seed") -> ExpandedNode:
    return ExpandedNode(node=node, reason=reason)


# --- Helpers ---


class TestHelpers:
    def test_cosine_basic(self):
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_zero_safe(self):
        assert _cosine([], [1.0, 2.0]) == 0.0
        assert _cosine([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_normalise_minmax(self):
        out = _normalise({"a": 1.0, "b": 3.0, "c": 5.0})
        assert out["a"] == pytest.approx(0.0)
        assert out["c"] == pytest.approx(1.0)
        assert out["b"] == pytest.approx(0.5)

    def test_normalise_flat_returns_ones(self):
        out = _normalise({"a": 2.0, "b": 2.0})
        assert out == {"a": 1.0, "b": 1.0}

    def test_normalise_empty_returns_empty(self):
        assert _normalise({}) == {}


# --- Core rerank path ---


class TestRerankCore:
    def test_empty_returns_empty(self):
        reranker = HybridReranker()
        assert reranker.rerank(expanded=[]) == []

    def test_lexical_dominates_under_default_weights(self):
        reranker = HybridReranker()
        expanded = [
            _expanded(_node("a")),
            _expanded(_node("b")),
        ]
        fts_scores = {"a": 10.0, "b": 1.0}
        scored = reranker.rerank(expanded=expanded, fts_scores=fts_scores)
        assert scored[0].node.id == "a"
        assert scored[0].lexical > scored[1].lexical

    def test_seed_outranks_category_sibling_on_graph_signal(self):
        reranker = HybridReranker()
        seed = _expanded(_node("seed"), reason="seed")
        sibling = _expanded(_node("sibling"), reason="category_sibling")
        # No lexical difference — the graph prior decides the order
        scored = reranker.rerank(
            expanded=[seed, sibling],
            fts_scores={"seed": 1.0, "sibling": 1.0},
        )
        assert scored[0].node.id == "seed"
        assert scored[0].graph > scored[1].graph

    def test_missing_fts_score_defaults_to_zero(self):
        reranker = HybridReranker()
        expanded = [_expanded(_node("a")), _expanded(_node("b"))]
        scored = reranker.rerank(expanded=expanded, fts_scores={"a": 1.0})
        a = next(s for s in scored if s.node.id == "a")
        b = next(s for s in scored if s.node.id == "b")
        assert a.lexical == 1.0
        assert b.lexical == 0.0

    def test_semantic_signal_kicks_in_with_embedding(self):
        reranker = HybridReranker()
        # Two nodes — one perfectly aligned with the query vector, one orthogonal
        aligned = _node("aligned", embedding=[1.0, 0.0])
        orthogonal = _node("orthogonal", embedding=[0.0, 1.0])
        scored = reranker.rerank(
            expanded=[_expanded(aligned), _expanded(orthogonal)],
            query_embedding=[1.0, 0.0],
        )
        top = next(s for s in scored if s.node.id == "aligned")
        other = next(s for s in scored if s.node.id == "orthogonal")
        assert top.semantic > other.semantic

    def test_no_embedding_zeroes_semantic_component(self):
        reranker = HybridReranker()
        expanded = [_expanded(_node("a", embedding=[1.0, 0.0]))]
        scored = reranker.rerank(expanded=expanded)
        assert scored[0].semantic == 0.0


# --- Structural signal ---


class TestStructuralSignal:
    def test_category_match_boosts_structural_score(self):
        reranker = HybridReranker()
        matched = _node("matched", category="규정 및 지침")
        unmatched = _node("unmatched", category="운영계획")
        scored = reranker.rerank(
            expanded=[_expanded(matched), _expanded(unmatched)],
            anchor_categories={"규정 및 지침"},
        )
        m = next(s for s in scored if s.node.id == "matched")
        u = next(s for s in scored if s.node.id == "unmatched")
        assert m.structural > u.structural

    def test_kind_match_boosts_structural_score(self):
        reranker = HybridReranker()
        rule = _node("rule", kind=NodeKind.RULE)
        chunk = _node("chunk", kind=NodeKind.CHUNK)
        scored = reranker.rerank(
            expanded=[_expanded(rule), _expanded(chunk)],
            anchor_kinds={NodeKind.RULE},
        )
        r = next(s for s in scored if s.node.id == "rule")
        c = next(s for s in scored if s.node.id == "chunk")
        assert r.structural > c.structural


# --- Custom weights ---


class TestWeightOverride:
    def test_zeroing_lexical_lets_semantic_decide(self):
        weights = RerankerWeights(lexical=0.0, semantic=1.0, graph=0.0, structural=0.0)
        reranker = HybridReranker(weights=weights)
        a = _node("a", embedding=[1.0, 0.0])
        b = _node("b", embedding=[0.0, 1.0])
        scored = reranker.rerank(
            expanded=[_expanded(a), _expanded(b)],
            fts_scores={"a": 0.1, "b": 0.9},  # b wins on lexical
            query_embedding=[1.0, 0.0],  # a wins on semantic
        )
        # Semantic should dominate
        assert scored[0].node.id == "a"


# --- Output shape invariants ---


class TestOutputShape:
    def test_all_candidates_retained(self):
        reranker = HybridReranker()
        expanded = [_expanded(_node(f"n{i}")) for i in range(5)]
        scored = reranker.rerank(expanded=expanded)
        assert len(scored) == 5

    def test_sorted_descending_by_total(self):
        reranker = HybridReranker()
        expanded = [_expanded(_node(f"n{i}")) for i in range(5)]
        fts_scores = {f"n{i}": float(i) for i in range(5)}
        scored = reranker.rerank(expanded=expanded, fts_scores=fts_scores)
        totals = [s.total for s in scored]
        assert totals == sorted(totals, reverse=True)

    def test_reason_preserved(self):
        reranker = HybridReranker()
        expanded = [
            ExpandedNode(node=_node("x"), reason="chunk_next"),
            ExpandedNode(node=_node("y"), reason="seed"),
        ]
        scored = reranker.rerank(expanded=expanded)
        reasons = {s.node.id: s.reason for s in scored}
        assert reasons == {"x": "chunk_next", "y": "seed"}


class TestReferenceCompanionLift:
    """v0.24 WS-B — a REFERENCES-expanded node is lifted to a fraction of
    its anchor seed's score so it survives reranking despite zero lexical
    overlap with the query."""

    def test_reference_node_lifted_to_anchor_fraction(self):
        reranker = HybridReranker()
        seed = _node("seed_a")
        ref = _node("ref_b")
        expanded = [
            ExpandedNode(node=seed, reason="seed"),
            ExpandedNode(node=ref, reason="references", anchor_hit="seed_a"),
        ]
        scored = reranker.rerank(expanded=expanded, fts_scores={"seed_a": 1.0})
        by_id = {s.node.id: s for s in scored}
        # ref_b has no lexical/semantic signal — without the lift its total
        # is ~0; with the lift it sits at 0.9× its anchor's total.
        assert by_id["ref_b"].total >= 0.9 * by_id["seed_a"].total - 1e-6

    def test_lift_requires_anchor(self):
        """A references node with no anchor_hit is not lifted."""
        reranker = HybridReranker()
        seed = _node("seed_a")
        orphan = _node("orphan_ref")
        expanded = [
            ExpandedNode(node=seed, reason="seed"),
            ExpandedNode(node=orphan, reason="references"),  # no anchor_hit
        ]
        scored = reranker.rerank(expanded=expanded, fts_scores={"seed_a": 1.0})
        by_id = {s.node.id: s for s in scored}
        # No anchor → no companion lift → orphan stays well below what an
        # anchored references node would reach (0.9× the anchor).
        assert by_id["orphan_ref"].total < 0.9 * by_id["seed_a"].total
