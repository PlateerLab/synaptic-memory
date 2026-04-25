"""Adaptive turn-budget classifier — `_is_enumeration_query`.

Phase A of the v0.20 exhaustive-recall track. The classifier decides
whether to bump ``max_turns`` from 5 to 15 so the agent has room to
walk pagination cursors. Both false positives and false negatives are
costly:

  - false positive: spends a few extra turns on a query that didn't
    need them (cheap)
  - false negative: agent runs out of turns mid-pagination on an
    enumeration query (expensive — what we're trying to prevent)

So the classifier is biased toward recall: a single enumeration
marker is enough to flip the budget. Tests below lock the marker set
and the start-anchored "모든" guard.
"""

from __future__ import annotations

import pytest

from synaptic.agent_loop import _is_enumeration_query


@pytest.mark.parametrize(
    "query",
    [
        "이용자보호와 관련된 모든 문서 목록",
        "환경 친화 정책 모두 알려줘",
        "전체 상품 카테고리는?",
        "list all products under 100",
        "모든 인권 관련 규정",
        "모든상품의 평균 가격",  # no space after 모든
        "X 관련 자료 전수 정리",
        "Show me all categories",
        "every chunk that mentions ESG",
        "all of the products in 25SS",
        "all the documents from 2023",
        "전체 문서 리스트",
    ],
)
def test_enumeration_queries_are_detected(query):
    assert _is_enumeration_query(query) is True, query


@pytest.mark.parametrize(
    "query",
    [
        "가장 많이 팔린 상품의 리뷰",  # top-N, not enumeration
        "친구 생일 선물 5만원 이하 추천",  # filter, not enumeration
        "경마 시행 운영 계획을 요약해줘",  # summary
        "How does 인권경영 conflict with 운영계획?",  # comparison
        "What is the price of G00007?",  # single-row lookup
        "",  # empty
    ],
)
def test_non_enumeration_queries_are_not_flipped(query):
    assert _is_enumeration_query(query) is False, query


def test_modeun_only_at_start_avoids_false_positives():
    """The 모든 token check is start-anchored (first 6 chars). A query
    that mentions 모든 mid-sentence as part of a quoted phrase or in
    passing should NOT be flipped — only the leading construct
    "모든 X" is a strong enumeration signal."""
    # mid-sentence: agent is asking about *one* thing
    assert _is_enumeration_query("규정 위반에 대해 모든 부서가 책임진다") is False
    # but at start, this IS enumeration
    assert _is_enumeration_query("모든 부서의 책임 사항") is True


def test_classifier_is_case_insensitive():
    assert _is_enumeration_query("LIST ALL ESG documents") is True
    assert _is_enumeration_query("Show ME All") is True
