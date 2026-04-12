"""Tests for EvidenceAggregator — selection with diversity + coverage."""

from __future__ import annotations

from synaptic.extensions.evidence_aggregator import (
    EvidenceAggregator,
    _jaccard,
    _tokens,
)
from synaptic.extensions.hybrid_reranker import ScoredCandidate
from synaptic.models import ConsolidationLevel, Node, NodeKind


def _node(
    id_: str,
    *,
    title: str = "",
    content: str = "",
    doc_id: str = "",
    category: str = "",
) -> Node:
    props: dict[str, str] = {}
    if doc_id:
        props["doc_id"] = doc_id
    if category:
        props["category"] = category
    return Node(
        id=id_,
        kind=NodeKind.CHUNK,
        title=title or id_,
        content=content or id_,
        properties=props,
        level=ConsolidationLevel.L0_RAW,
    )


def _scored(
    id_: str,
    *,
    total: float = 0.5,
    content: str = "",
    doc_id: str = "",
    category: str = "",
    reason: str = "seed",
) -> ScoredCandidate:
    node = _node(id_, content=content or id_, doc_id=doc_id, category=category)
    return ScoredCandidate(
        node=node,
        total=total,
        lexical=total,
        semantic=0.0,
        graph=0.5,
        structural=0.0,
        reason=reason,
    )


# --- Helpers ---


class TestHelpers:
    def test_tokens_extracts_hangul_and_latin(self):
        toks = _tokens("경마 운영계획 machine learning x")
        assert "경마" in toks
        assert "운영계획" in toks
        assert "machine" in toks
        assert "learning" in toks
        # 2-char minimum — "x" dropped
        assert "x" not in toks

    def test_tokens_empty_returns_empty_set(self):
        assert _tokens("") == set()
        assert _tokens(None) == set()  # type: ignore[arg-type]

    def test_jaccard_identical_sets_return_one(self):
        a = {"foo", "bar"}
        assert _jaccard(a, a) == 1.0

    def test_jaccard_disjoint_sets_return_zero(self):
        assert _jaccard({"foo"}, {"bar"}) == 0.0

    def test_jaccard_zero_safe(self):
        assert _jaccard(set(), {"foo"}) == 0.0
        assert _jaccard({"foo"}, set()) == 0.0


# --- Basic selection ---


class TestBasicSelection:
    def test_empty_input_returns_empty(self):
        agg = EvidenceAggregator()
        assert agg.aggregate(scored=[], k=5) == []

    def test_k_zero_returns_empty(self):
        agg = EvidenceAggregator()
        scored = [_scored("a", total=1.0)]
        assert agg.aggregate(scored=scored, k=0) == []

    def test_single_candidate_returned(self):
        agg = EvidenceAggregator()
        scored = [_scored("a", total=0.9, content="unique content one")]
        result = agg.aggregate(scored=scored, k=5)
        assert len(result) == 1
        assert result[0].node.id == "a"
        assert result[0].score == 0.9

    def test_top_k_order_preserved_when_no_diversity_conflict(self):
        agg = EvidenceAggregator()
        scored = [
            _scored("c", total=0.6, content="apple orange fresh fruit"),
            _scored("a", total=0.95, content="python programming tutorial"),
            _scored("b", total=0.8, content="korean history textbook"),
        ]
        result = agg.aggregate(scored=scored, k=3)
        ids = [e.node.id for e in result]
        # Highest score first
        assert ids[0] == "a"


# --- MMR duplicate suppression ---


class TestMMRDuplicateSuppression:
    def test_near_duplicate_content_dropped(self):
        agg = EvidenceAggregator(similarity_threshold=0.5)
        # Two chunks with almost identical text
        scored = [
            _scored(
                "original",
                total=1.0,
                content="경마산업 운영계획 규정 지침 준수",
            ),
            _scored(
                "duplicate",
                total=0.99,
                content="경마산업 운영계획 규정 지침 준수 내용",
            ),
            _scored(
                "different",
                total=0.5,
                content="완전히 다른 내용 인권경영 보고서 평가",
            ),
        ]
        result = agg.aggregate(scored=scored, k=3)
        ids = [e.node.id for e in result]
        # The exact duplicate is dropped, the different one is kept
        assert "original" in ids
        assert "different" in ids
        assert "duplicate" not in ids

    def test_high_lambda_prefers_relevance_over_diversity(self):
        agg = EvidenceAggregator(mmr_lambda=1.0, similarity_threshold=0.99)
        scored = [
            _scored("a", total=1.0, content="foo bar baz"),
            _scored("b", total=0.9, content="foo bar baz qux"),  # similar
            _scored("c", total=0.1, content="totally unrelated"),
        ]
        result = agg.aggregate(scored=scored, k=2)
        ids = [e.node.id for e in result]
        # With lambda=1.0 and high similarity threshold, the two high-scored
        # candidates are both kept — diversity penalty disabled
        assert "a" in ids
        assert "b" in ids


# --- Per-document cap ---


class TestPerDocumentCap:
    def test_cap_prevents_document_monopoly(self):
        agg = EvidenceAggregator()
        # Five chunks from the same doc with genuinely different text,
        # plus one chunk from a second doc. The cap should keep at
        # most two from doc_A and also admit the doc_B chunk.
        doc_a_contents = [
            "규정 준수 의무 사항 설명",
            "예외 조항 적용 기준 해설",
            "위반 시 제재 조치 절차",
            "내부 감사 수행 방법",
            "개정 이력 관리 방식",
        ]
        scored = [
            _scored(
                f"chunk_{i}",
                total=0.9 - 0.01 * i,
                content=doc_a_contents[i],
                doc_id="doc_A",
            )
            for i in range(5)
        ]
        scored.append(
            _scored(
                "outsider",
                total=0.3,
                content="완전히 다른 주제 고객 만족도",
                doc_id="doc_B",
            )
        )
        result = agg.aggregate(scored=scored, k=4, per_document_cap=2)
        doc_ids = [e.document_id for e in result]
        assert doc_ids.count("doc_A") == 2
        assert "doc_B" in doc_ids


# --- Category coverage ---


class TestCategoryCoverage:
    def test_each_category_gets_at_least_one_representative(self):
        agg = EvidenceAggregator()
        scored = [
            _scored("r1", total=0.9, content="규정 내용 하나", category="규정 및 지침"),
            _scored("r2", total=0.8, content="규정 내용 둘", category="규정 및 지침"),
            _scored("o1", total=0.7, content="운영 내용 하나", category="운영계획"),
        ]
        result = agg.aggregate(
            scored=scored,
            k=2,
            anchor_categories={"규정 및 지침", "운영계획"},
        )
        categories = {e.category for e in result}
        assert "규정 및 지침" in categories
        assert "운영계획" in categories

    def test_coverage_picks_highest_scored_in_category(self):
        agg = EvidenceAggregator()
        scored = [
            _scored("low", total=0.3, content="content A unique first", category="규정"),
            _scored("high", total=0.95, content="content B unique second", category="규정"),
        ]
        result = agg.aggregate(
            scored=scored,
            k=1,
            anchor_categories={"규정"},
        )
        assert result[0].node.id == "high"

    def test_coverage_reason_tagged(self):
        agg = EvidenceAggregator()
        scored = [
            _scored("a", total=0.9, content="규정 문서 내용 one", category="규정"),
        ]
        result = agg.aggregate(scored=scored, k=1, anchor_categories={"규정"})
        assert result[0].reason == "category_coverage"


# --- Output shape ---


class TestOutputShape:
    def test_evidence_preserves_score_from_reranker(self):
        agg = EvidenceAggregator()
        scored = [_scored("a", total=0.77, content="some content")]
        result = agg.aggregate(scored=scored, k=1)
        assert result[0].score == 0.77

    def test_document_id_extracted(self):
        agg = EvidenceAggregator()
        scored = [_scored("a", total=0.5, content="x", doc_id="doc_42")]
        result = agg.aggregate(scored=scored, k=1)
        assert result[0].document_id == "doc_42"
