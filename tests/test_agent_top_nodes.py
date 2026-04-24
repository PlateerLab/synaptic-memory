"""Tests for ``top_nodes_tool`` — single-call top-N ranking over a table.

Introduced to collapse the "가장 X한" / "top N" pattern into one tool
call. Before this, the agent had to compose aggregate_nodes with a
metric_property trick that worked but was frequently mis-used on
multi-hop queries (assort Hard a003, a039, a040).
"""

from __future__ import annotations

import pytest

from synaptic.agent_tools_structured import top_nodes_tool
from synaptic.backends.memory import MemoryBackend
from synaptic.models import ConsolidationLevel, Node, NodeKind
from synaptic.search_session import SearchSession


async def _backend_with_products() -> MemoryBackend:
    b = MemoryBackend()
    await b.connect()
    rows = [
        {"product_code": "A", "cumulative_sales": "100", "season": "25SS"},
        {"product_code": "B", "cumulative_sales": "500", "season": "25SS"},
        {"product_code": "C", "cumulative_sales": "300", "season": "24FW"},
        {"product_code": "D", "cumulative_sales": "900", "season": "25SS"},
        {"product_code": "E", "cumulative_sales": "50", "season": "24FW"},
    ]
    for i, r in enumerate(rows):
        await b.save_node(
            Node(
                id=f"n_{i}",
                kind=NodeKind.ENTITY,
                title=f"products:{r['product_code']}",
                properties={**r, "_table_name": "products"},
                level=ConsolidationLevel.L0_RAW,
            )
        )
    return b


@pytest.mark.asyncio
async def test_top_nodes_desc_returns_highest_first():
    backend = await _backend_with_products()
    session = SearchSession(budget_tool_calls=5)
    r = await top_nodes_tool(
        backend,
        session,
        table="products",
        sort_by="cumulative_sales",
        order="desc",
        limit=3,
    )
    titles = [x["title"] for x in r.data["results"]]
    assert titles == ["products:D", "products:B", "products:C"]
    # sort_value is carried through for the agent's next call
    assert r.data["results"][0]["sort_value"] == 900.0


@pytest.mark.asyncio
async def test_top_nodes_asc_returns_lowest_first():
    backend = await _backend_with_products()
    session = SearchSession(budget_tool_calls=5)
    r = await top_nodes_tool(
        backend,
        session,
        table="products",
        sort_by="cumulative_sales",
        order="asc",
        limit=2,
    )
    titles = [x["title"] for x in r.data["results"]]
    assert titles == ["products:E", "products:A"]


@pytest.mark.asyncio
async def test_top_nodes_with_where_prefilter_limits_pool():
    backend = await _backend_with_products()
    session = SearchSession(budget_tool_calls=5)
    # Only rank 25SS products
    r = await top_nodes_tool(
        backend,
        session,
        table="products",
        sort_by="cumulative_sales",
        order="desc",
        limit=5,
        where_property="season",
        where_op="==",
        where_value="25SS",
    )
    titles = [x["title"] for x in r.data["results"]]
    assert titles == ["products:D", "products:B", "products:A"]
    assert r.data["total"] == 3


@pytest.mark.asyncio
async def test_top_nodes_missing_column_emits_fuzzy_verification_hint():
    backend = await _backend_with_products()
    session = SearchSession(budget_tool_calls=5)
    r = await top_nodes_tool(
        backend,
        session,
        table="products",
        sort_by="nonexistent_column",
        order="desc",
        limit=5,
    )
    assert r.data["total"] == 0
    # Hint should point at verifying the column
    actions = [h.action for h in r.hints]
    assert "filter_nodes" in actions


@pytest.mark.asyncio
async def test_top_nodes_with_empty_where_filter_hints_dropping_it():
    """If the where clause filters everything out, suggest retry w/o where."""
    backend = await _backend_with_products()
    session = SearchSession(budget_tool_calls=5)
    r = await top_nodes_tool(
        backend,
        session,
        table="products",
        sort_by="cumulative_sales",
        order="desc",
        where_property="season",
        where_op="==",
        where_value="NONEXISTENT_SEASON",
    )
    assert r.data["total"] == 0
    reasons = [h.reason for h in r.hints]
    assert any("WHERE" in r or "pre-filter" in r for r in reasons)


@pytest.mark.asyncio
async def test_top_nodes_invalid_order_returns_error():
    backend = await _backend_with_products()
    session = SearchSession(budget_tool_calls=5)
    r = await top_nodes_tool(
        backend,
        session,
        table="products",
        sort_by="cumulative_sales",
        order="sideways",
    )
    assert r.ok is False
    assert "order" in (r.error or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "alias,expected_order",
    [
        ("DESC", "desc"),
        ("descending", "desc"),
        ("max", "desc"),
        ("largest", "desc"),
        ("top", "desc"),
        ("ASC", "asc"),
        ("ascending", "asc"),
        ("min", "asc"),
        ("smallest", "asc"),
        ("bottom", "asc"),
    ],
)
async def test_top_nodes_order_aliases_are_normalised(alias, expected_order):
    """LLMs interchange 'DESC' / 'descending' / 'max' / 'top' with
    'desc'. Tolerate all of them to avoid burning a turn on a wording
    mismatch."""
    backend = await _backend_with_products()
    session = SearchSession(budget_tool_calls=5)
    r = await top_nodes_tool(
        backend,
        session,
        table="products",
        sort_by="cumulative_sales",
        order=alias,
        limit=5,
    )
    assert r.ok is True
    # Sanity: desc → highest first, asc → lowest first
    vals = [x["sort_value"] for x in r.data["results"]]
    if expected_order == "desc":
        assert vals == sorted(vals, reverse=True)
    else:
        assert vals == sorted(vals)


@pytest.mark.asyncio
async def test_top_nodes_from_ids_restricts_ranking_pool():
    """Multi-hop chaining — pass node_titles from a prior step and rank
    only within that subset."""
    backend = await _backend_with_products()
    session = SearchSession(budget_tool_calls=5)
    # Rank only the two specific products we "carried over".
    r = await top_nodes_tool(
        backend,
        session,
        table="products",
        sort_by="cumulative_sales",
        order="desc",
        limit=5,
        from_ids=["products:A", "products:C"],
    )
    assert r.data["total"] == 2
    titles = [x["title"] for x in r.data["results"]]
    assert titles == ["products:C", "products:A"]  # C has 300, A has 100


@pytest.mark.asyncio
async def test_top_nodes_result_feeds_extract_ids_correctly():
    """The agent loop's _extract_ids must pick up the ranked titles so
    the breakthrough query "가장 많이 팔린 상품의 리뷰" can surface the
    winning product id into found_ids (which is what the bench matches
    against GT)."""
    from synaptic.agent_loop import _extract_ids

    backend = await _backend_with_products()
    session = SearchSession(budget_tool_calls=5)
    r = await top_nodes_tool(
        backend,
        session,
        table="products",
        sort_by="cumulative_sales",
        order="desc",
        limit=3,
    )
    found: set[str] = set()
    _extract_ids(r.data, found, known_tables={"products"})
    # All three ranked titles must land in found_ids
    assert "products:D" in found
    assert "products:B" in found
    assert "products:C" in found


@pytest.mark.asyncio
async def test_top_nodes_budget_enforcement():
    backend = await _backend_with_products()
    session = SearchSession(budget_tool_calls=0)  # exhausted
    r = await top_nodes_tool(
        backend,
        session,
        table="products",
        sort_by="cumulative_sales",
    )
    assert r.ok is False
    assert r.error == "budget_exceeded"
