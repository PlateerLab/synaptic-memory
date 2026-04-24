"""0-result recovery hints for structured tools.

When ``filter_nodes`` / ``aggregate_nodes`` / ``join_related`` return
no matches, they emit :class:`Hint` entries that suggest the most
likely corrective action (try ``contains``, drop a WHERE, verify
the FK column). These hints surface through ``project_tool_result``
so the LLM sees them in the next turn instead of flailing blindly.

Why this matters: measured agent benchmarks show retry-loops — when
the first tool call returns 0, the agent tends to re-issue nearly-
identical variants instead of switching operators. A concrete hint
short-circuits that loop.
"""

from __future__ import annotations

import pytest

from synaptic.agent_tools_structured import (
    aggregate_nodes_tool,
    filter_nodes_tool,
    join_related_tool,
)
from synaptic.backends.memory import MemoryBackend
from synaptic.models import ConsolidationLevel, Node, NodeKind
from synaptic.search_session import SearchSession


async def _backend_with_rows(rows: list[dict[str, str]]) -> MemoryBackend:
    b = MemoryBackend()
    await b.connect()
    for i, row in enumerate(rows):
        await b.save_node(
            Node(
                id=f"n_{i}",
                kind=NodeKind.ENTITY,
                title=f"products:{row.get('id', i)}",
                properties=row,
                level=ConsolidationLevel.L0_RAW,
            )
        )
    return b


@pytest.mark.asyncio
async def test_filter_zero_result_on_equality_emits_contains_hint():
    backend = await _backend_with_rows(
        [{"id": "1", "name": "티셔츠", "_table_name": "products"}]
    )
    session = SearchSession(budget_tool_calls=5)
    r = await filter_nodes_tool(
        backend,
        session,
        table="products",
        property="name",
        op="==",
        value="셔츠",  # substring of "티셔츠" but != "티셔츠"
    )
    assert r.data["total"] == 0
    # Must include a contains-based retry hint
    hint_ops = {h.args.get("op") for h in r.hints}
    assert "contains" in hint_ops
    # Same property / value carried over
    contains_hint = next(h for h in r.hints if h.args.get("op") == "contains")
    assert contains_hint.args["property"] == "name"
    assert contains_hint.args["value"] == "셔츠"


@pytest.mark.asyncio
async def test_filter_zero_result_on_multiword_contains_emits_first_token_hint():
    backend = await _backend_with_rows(
        [{"id": "1", "name": "운동화", "_table_name": "products"}]
    )
    session = SearchSession(budget_tool_calls=5)
    r = await filter_nodes_tool(
        backend,
        session,
        table="products",
        property="name",
        op="contains",
        value="파란 운동화",
    )
    assert r.data["total"] == 0
    # Expect a hint falling back to just "파란"
    reasons = [h.reason for h in r.hints]
    assert any("first keyword" in r.lower() for r in reasons)


@pytest.mark.asyncio
async def test_filter_hit_emits_no_recovery_hints():
    backend = await _backend_with_rows(
        [{"id": "1", "name": "티셔츠", "_table_name": "products"}]
    )
    session = SearchSession(budget_tool_calls=5)
    r = await filter_nodes_tool(
        backend,
        session,
        table="products",
        property="name",
        op="contains",
        value="티셔츠",
    )
    assert r.data["total"] == 1
    assert r.hints == []  # success path stays silent


@pytest.mark.asyncio
async def test_aggregate_zero_groups_suggests_dropping_where_filter():
    backend = await _backend_with_rows(
        [{"id": str(i), "color": "red", "_table_name": "products"} for i in range(3)]
    )
    session = SearchSession(budget_tool_calls=5)
    r = await aggregate_nodes_tool(
        backend,
        session,
        table="products",
        group_by="color",
        metric="count",
        where_property="size",
        where_op="==",
        where_value="XL",
    )
    assert r.data["total_groups"] == 0
    # Must include a "retry without WHERE" hint
    reasons = [h.reason for h in r.hints]
    assert any("pre-filter" in r or "WHERE" in r for r in reasons)


@pytest.mark.asyncio
async def test_aggregate_with_results_no_hints():
    backend = await _backend_with_rows(
        [{"id": str(i), "color": "red", "_table_name": "products"} for i in range(3)]
    )
    session = SearchSession(budget_tool_calls=5)
    r = await aggregate_nodes_tool(
        backend,
        session,
        table="products",
        group_by="color",
        metric="count",
    )
    assert r.data["total_groups"] == 1
    assert r.hints == []


@pytest.mark.asyncio
async def test_filter_with_misspelled_column_suggests_closest_real_column():
    """Agent passes ``property="sale"`` when the real column is
    ``selling_price``. Expect a fuzzy-match hint pointing to the
    real column."""
    backend = await _backend_with_rows(
        [
            {"id": "1", "selling_price": "100", "_table_name": "products"},
            {"id": "2", "selling_price": "200", "_table_name": "products"},
        ]
    )
    session = SearchSession(budget_tool_calls=5)
    r = await filter_nodes_tool(
        backend,
        session,
        table="products",
        property="sell_price",  # typo / near-miss
        op=">=",
        value="100",
    )
    assert r.data["total"] == 0
    reasons = [h.reason for h in r.hints]
    # Must mention did-you-mean + the real column
    joined = " | ".join(reasons)
    assert "selling_price" in joined
    assert "did you mean" in joined.lower()


@pytest.mark.asyncio
async def test_filter_with_valid_column_no_fuzzy_hint():
    """When the column exists but values don't match, we should NOT
    emit a fuzzy-column hint (that would mislead the agent)."""
    backend = await _backend_with_rows(
        [{"id": "1", "selling_price": "100", "_table_name": "products"}]
    )
    session = SearchSession(budget_tool_calls=5)
    r = await filter_nodes_tool(
        backend,
        session,
        table="products",
        property="selling_price",
        op=">=",
        value="10000",  # no match
    )
    assert r.data["total"] == 0
    reasons = [h.reason for h in r.hints]
    joined = " | ".join(reasons)
    # No "did you mean" — column is valid, just no matching rows
    assert "did you mean" not in joined.lower()


@pytest.mark.asyncio
async def test_join_zero_rows_suggests_fk_verification():
    """join_related with a non-existent FK column should hint the
    agent to verify the FK column name via filter_nodes."""
    backend = await _backend_with_rows(
        [{"id": "1", "name": "product-a", "_table_name": "products"}]
    )
    session = SearchSession(budget_tool_calls=5)
    r = await join_related_tool(
        backend,
        session,
        from_value="1",
        fk_property="nonexistent_fk",
        target_table="reviews",
    )
    assert r.data["total"] == 0
    # Must include a filter_nodes verification hint
    actions = [h.action for h in r.hints]
    assert "filter_nodes" in actions
