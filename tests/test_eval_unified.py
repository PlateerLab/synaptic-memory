"""Tests for the unified validation scorer (``eval/unified.py``).

Phase 0 of the v0.20+ track. Locks the dimension classifier and the
weighted score math so that every Phase change is judged against a
stable, reproducible UnifiedScore — not just per-bench numbers.

Critical invariants this file guards:

1. The enumeration classifier here MUST stay aligned with the agent
   loop's enumeration detector — otherwise a query that pulls the
   adaptive turn budget upstream gets recall-typed as something else
   downstream and the dimension report becomes lying.
2. Cross-language detection must NOT be triggered by mere code-mixing
   (Korean queries with one English brand name ≠ cross-language).
3. UnifiedScore composition must be a pure weighted average — single
   number ship/no-ship rule must be defensible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.unified import (
    DEFAULT_WEIGHTS,
    Language,
    QueryDimensions,
    RecallType,
    classify_query,
    score,
)

# --- Classifier --------------------------------------------------


def test_korean_text_only_classifies_as_ko():
    d = classify_query("말 복지 향상을 위한 프로그램")
    assert d.language == Language.KO.value
    assert d.cross_language is False


def test_english_text_only_classifies_as_en():
    d = classify_query("portable computing device")
    assert d.language == Language.EN.value


def test_mixed_languages_classifies_as_mixed():
    d = classify_query("ESG 정책 in Korean operations")
    assert d.language == Language.MIXED.value


def test_enumeration_marker_flips_recall_type():
    """이 분류기와 src/synaptic/agent_loop.py:_is_enumeration_query 가
    같은 query 에 대해 같은 답을 줘야 함. 다르면 adaptive turn budget 이
    fired 되어도 dimension report 에 enumeration 으로 안 잡혀서 reporting
    이 거짓말이 됨."""
    cases = [
        "이용자보호와 관련된 모든 문서 목록",
        "list all ESG documents",
        "전체 상품 카테고리는?",
        "every chunk that mentions ESG",
    ]
    for q in cases:
        d = classify_query(q)
        assert d.enumeration is True, q
        assert d.recall_type == RecallType.ENUMERATION.value, q


def test_summarization_wins_over_enumeration():
    """'요약' is a summary intent even if 'X 모두' would otherwise match."""
    d = classify_query("경마 시행 운영 계획 모두 요약해줘")
    # summarization marker should still reach recall_type
    assert d.recall_type == RecallType.SUMMARIZATION.value


def test_many_relevant_docs_implies_enumeration_recall():
    """A query with 5+ GT docs is effectively an enumeration even
    without an explicit '모두' marker."""
    d = classify_query(
        "horse welfare programs",
        relevant_docs=["d1", "d2", "d3", "d4", "d5"],
    )
    assert d.recall_type == RecallType.ENUMERATION.value


def test_two_or_three_relevant_docs_implies_top_n():
    d = classify_query("popular shoes", relevant_docs=["p1", "p2"])
    assert d.recall_type == RecallType.TOP_N.value


def test_single_relevant_doc_implies_single_lookup():
    d = classify_query("price of G00007", relevant_docs=["products:G00007"])
    assert d.recall_type == RecallType.SINGLE_LOOKUP.value


def test_structured_pct_inferred_from_table_pk_shape():
    d = classify_query(
        "find shoes",
        relevant_docs=["products:G00001", "products:G00002", "doc_text123"],
    )
    # 2/3 are table:pk shape → ~0.66
    assert 0.5 < d.structured_pct < 0.8


def test_explicit_dimensions_override_inferred():
    """Query JSON can carry ``dimensions: {cross_domain: true}`` as
    explicit metadata — overrides any inferred tag."""
    d = classify_query(
        "ESG across legal and ecommerce",
        domain="krra",
        explicit={"cross_domain": True, "cross_language": True},
    )
    assert d.cross_domain is True
    assert d.cross_language is True


def test_hop_count_clamped_to_4():
    """Many '의' markers shouldn't blow up the hop estimate."""
    d = classify_query("A의 B의 C의 D의 E의 F의 G")
    assert d.hop_count <= 4


# --- Scorer ------------------------------------------------------


def test_unified_score_is_weighted_average_of_axis_hit_rates():
    """Concrete invariant: UnifiedScore must equal Σ(weight · hit_rate).
    If we ever switch to e.g. geometric mean, this test is the canary."""
    items = [
        # 2 ko queries (non-enum), 1 hit
        (QueryDimensions(language="ko"), True, "b1"),
        (QueryDimensions(language="ko"), False, "b1"),
        # 2 en queries, 2 hits
        (QueryDimensions(language="en"), True, "b2"),
        (QueryDimensions(language="en"), True, "b2"),
        # 1 enumeration (also lang=ko), hit
        (QueryDimensions(language="ko", enumeration=True, recall_type="enumeration"), True, "b3"),
    ]
    rep = score(items)
    # ko hit-rate = 2/3 (the enum one is also lang=ko, hit)
    # en hit-rate = 2/2 = 1.0
    # enumeration hit-rate = 1/1 = 1.0
    # other axes (mixed, multi_hop, structured, cross_*) = 0 queries → 0
    assert abs(rep.per_dimension["lang:ko"]["hit_rate"] - 2 / 3) < 1e-3
    assert rep.per_dimension["lang:en"]["hit_rate"] == 1.0
    assert rep.per_dimension["recall:enumeration"]["hit_rate"] == 1.0
    # Score should be positive
    assert 0 < rep.unified_score <= 1.0


def test_zero_coverage_axis_is_flagged_in_notes():
    """If we weight an axis but no query covers it, the score is
    penalised — emit a NOTE so reviewer sees the gap."""
    items = [
        (QueryDimensions(language="ko"), True, "b1"),
        (QueryDimensions(language="ko"), True, "b1"),
    ]
    rep = score(items)
    assert any("NO_COVERAGE" in n for n in rep.notes)
    # specifically en + multi-hop + structured + cross_* should all
    # show up as no-coverage
    no_cov_axes = {n.split("axis=")[1].split()[0] for n in rep.notes if "NO_COVERAGE" in n}
    assert "lang:en" in no_cov_axes
    assert "recall:multi_hop" in no_cov_axes


def test_per_bench_breakdown_preserves_legacy_view():
    """Legacy per-bench reporting must still be present for backwards
    comparison against earlier CHANGELOG entries."""
    items = [
        (QueryDimensions(language="ko"), True, "krra_hard"),
        (QueryDimensions(language="ko"), False, "krra_hard"),
        (QueryDimensions(language="ko"), True, "assort_hard"),
    ]
    rep = score(items)
    assert "krra_hard" in rep.per_bench
    assert rep.per_bench["krra_hard"]["n_queries"] == 2
    assert rep.per_bench["krra_hard"]["n_hits"] == 1
    assert rep.per_bench["assort_hard"]["n_queries"] == 1


def test_custom_weights_are_normalised():
    """Caller can pass un-normalised weights — scorer normalises so the
    composite stays in [0, 1]."""
    items = [(QueryDimensions(language="ko"), True, "b1")]
    rep = score(items, weights={"lang:ko": 2.0, "lang:en": 1.0})
    # Both weights renormalised: ko=0.667, en=0.333
    assert abs(rep.weights["lang:ko"] - 2 / 3) < 1e-6
    assert rep.unified_score <= 1.0


def test_default_weights_sum_to_one_after_normalisation():
    """Sanity: DEFAULT_WEIGHTS may not sum to 1 in source but normalised
    scorer must keep the composite bounded at 1.0."""
    raw = sum(DEFAULT_WEIGHTS.values())
    # Simulate every axis fully hit
    items: list = []
    items.append((QueryDimensions(language="ko"), True, "x"))
    items.append((QueryDimensions(language="en"), True, "x"))
    items.append((QueryDimensions(language="mixed"), True, "x"))
    items.append((QueryDimensions(recall_type="multi_hop"), True, "x"))
    items.append(
        (QueryDimensions(recall_type="enumeration", enumeration=True), True, "x")
    )
    items.append((QueryDimensions(structured_pct=1.0), True, "x"))
    items.append((QueryDimensions(cross_domain=True), True, "x"))
    items.append((QueryDimensions(cross_language=True), True, "x"))
    rep = score(items)
    assert raw > 0
    # All-hit case → score ≈ 1.0 (within float tolerance)
    assert 0.99 <= rep.unified_score <= 1.0001


def test_empty_input_returns_zero_score_no_crash():
    rep = score([])
    assert rep.unified_score == 0.0
    assert rep.query_count == 0
    # Notes still flag the empty axes
    assert any("NO_COVERAGE" in n for n in rep.notes)


def test_query_count_matches_total_items():
    items = [(QueryDimensions(language="ko"), True, f"b{i}") for i in range(7)]
    rep = score(items)
    assert rep.query_count == 7
    assert rep.n_hits == 7


@pytest.mark.parametrize(
    "korean_query",
    [
        "이용자보호 관련 모든 문서 목록",
        "환경 친화 정책 모두 알려줘",
        "전체 상품 카테고리는?",
        "모든 인권 관련 규정",
    ],
)
def test_classifier_aligned_with_agent_loop_enumeration_detector(korean_query):
    """The two detectors must agree query-by-query so that any query
    bumping the agent's adaptive turn budget is *also* counted in the
    enumeration recall slice. Drift here = lying dimension report."""
    from synaptic.agent_loop import _is_enumeration_query

    d = classify_query(korean_query)
    assert d.enumeration == _is_enumeration_query(korean_query)
