"""Unit tests for ``synaptic.router.decide_route`` — tier-0 routing signals.

``decide_route`` is a pure function (zero LLM calls). These tests lock the
conservative default pending E2 validation (PLAN-v0.29 §E2): ONLY
structured-operation lexis over a corpus with typed table nodes promotes to
the agent route. Everything else stays single_shot — tier-1 escalation is
``graph.ask()``'s job and is tested in test_graph_ask.py.
"""

from __future__ import annotations

from synaptic.router import RouteDecision, decide_route

# --- promoting signal: structured lexis × typed table nodes -----------


def test_aggregation_lexis_with_tables_routes_agent():
    d = decide_route("색상별 상품 변형 개수", has_table_nodes=True)
    assert d.route == "agent"
    assert d.signals["aggregation"] is True
    assert d.signals["structured_lexis"] is True
    assert d.reasons  # auditable — never an unexplained route


def test_enumeration_with_tables_routes_agent():
    d = decide_route("24FW 시즌 전체 상품 목록", has_table_nodes=True)
    assert d.route == "agent"
    assert d.signals["enumeration"] is True


def test_comparison_filter_with_number_routes_agent():
    d = decide_route("9만원 이상 고가 상품", has_table_nodes=True)
    assert d.route == "agent"
    assert d.signals["comparison_filter"] is True


def test_temporal_filter_with_tables_routes_agent():
    d = decide_route("2024년 11월에 방송된 상품", has_table_nodes=True)
    assert d.route == "agent"
    assert d.signals["temporal_filter"] is True


def test_english_aggregation_routes_agent():
    d = decide_route("how many beers are in the catalog?", has_table_nodes=True)
    assert d.route == "agent"
    assert d.signals["aggregation"] is True


def test_top_n_routes_agent():
    d = decide_route("가장 많이 팔린 상품 TOP 3", has_table_nodes=True)
    assert d.route == "agent"
    assert d.signals["aggregation"] is True


# --- conservative negatives -------------------------------------------


def test_plain_question_stays_single_shot():
    d = decide_route("온실가스 감축 및 에너지 절약 계획은 어떻게 되어있나", has_table_nodes=True)
    assert d.route == "single_shot"
    assert d.signals["structured_lexis"] is False
    assert d.reasons


def test_structured_lexis_without_tables_stays_single_shot():
    # Docs-only corpus: nothing for filter/aggregate tools to run on, so the
    # tier-1 gate (not tier-0) decides whether the agent is needed.
    d = decide_route("가장 많이 팔린 상품 TOP 3", has_table_nodes=False)
    assert d.route == "single_shot"
    assert d.signals["aggregation"] is True
    assert d.signals["has_table_nodes"] is False


def test_comparison_token_without_number_does_not_fire():
    d = decide_route("그 이상 자세한 내용은 어디서 확인하나", has_table_nodes=True)
    assert d.signals["comparison_filter"] is False
    assert d.route == "single_shot"


def test_laptop_does_not_match_top_n():
    # "top N" is anchored on the digit — "laptop" must never fire it.
    d = decide_route("laptop stand product info", has_table_nodes=True)
    assert d.signals["aggregation"] is False
    assert d.route == "single_shot"


def test_empty_query_stays_single_shot():
    d = decide_route("", has_table_nodes=True)
    assert d.route == "single_shot"


# --- signals dict is the E2 diagnostic surface ------------------------


def test_signals_dict_is_fully_populated():
    d = decide_route("아무 질문", has_table_nodes=False)
    for key in (
        "enumeration",
        "aggregation",
        "comparison_filter",
        "temporal_filter",
        "structured_lexis",
        "has_table_nodes",
        "anchor_present",
    ):
        assert key in d.signals


def test_search_scores_recorded_but_not_routed_on():
    # Score signals are embedder-dependent — recorded for the E2 AUC
    # harness, but they must NOT change the route in the conservative default.
    d = decide_route("아무 질문", has_table_nodes=True, search_scores=[0.9, 0.5, 0.1])
    assert d.signals["top1_score"] == 0.9
    assert abs(d.signals["score_margin"] - 0.4) < 1e-9
    assert d.route == "single_shot"


def test_anchor_presence_recorded():
    d = decide_route("아무 질문", has_table_nodes=False, anchor=object())
    assert d.signals["anchor_present"] is True


def test_route_decision_dataclass_shape():
    d = RouteDecision(route="agent")
    assert d.route == "agent"
    assert d.reasons == []
    assert d.signals == {}
