"""Pagination contract for structured agent tools.

Phase A of the v0.20+ exhaustive-recall track. Locks in the cursor /
has_more / next_cursor surface across filter_nodes (today) and the
remaining structured tools (top_nodes / aggregate_nodes / join_related)
once each is wired.

Why this matters: before pagination, the agent had no way to retrieve
results [21, 100] of a 100-row enumeration query — the tool returned 20
with `truncated=true` and the agent gave up. Tests here lock the
contract so the agent's next_cursor follow-through path stays correct.
"""

from __future__ import annotations

import pytest

from synaptic.agent_tools_structured import (
    aggregate_nodes_tool,
    filter_nodes_tool,
    join_related_tool,
    top_nodes_tool,
)
from synaptic.backends.memory import MemoryBackend
from synaptic.models import ConsolidationLevel, Node, NodeKind
from synaptic.search_session import SearchSession


async def _backend_with_n_rows(n: int) -> MemoryBackend:
    """Build a backend with N products that all match a single filter."""
    b = MemoryBackend()
    await b.connect()
    for i in range(n):
        await b.save_node(
            Node(
                id=f"n_{i:04d}",
                kind=NodeKind.ENTITY,
                title=f"products:G{i:05d}",
                properties={
                    "_table_name": "products",
                    "category": "shoes",
                    "price": str(10000 + i * 100),
                },
                level=ConsolidationLevel.L0_RAW,
            )
        )
    return b


@pytest.mark.asyncio
async def test_filter_first_page_signals_has_more():
    backend = await _backend_with_n_rows(50)
    session = SearchSession(budget_tool_calls=10)
    r = await filter_nodes_tool(
        backend, session, table="products", property="category",
        op="==", value="shoes", limit=20,
    )
    assert r.ok is True
    assert r.data["total"] == 50
    assert r.data["showing"] == 20
    assert r.data["offset"] == 0
    assert r.data["has_more"] is True
    assert r.data["next_cursor"] == "20"


@pytest.mark.asyncio
async def test_filter_second_page_continues_from_cursor():
    backend = await _backend_with_n_rows(50)
    session = SearchSession(budget_tool_calls=10)
    r = await filter_nodes_tool(
        backend, session, table="products", property="category",
        op="==", value="shoes", limit=20, cursor="20",
    )
    assert r.data["offset"] == 20
    assert r.data["showing"] == 20
    assert r.data["has_more"] is True
    assert r.data["next_cursor"] == "40"
    # Critical: page 2 results are disjoint from page 1
    titles = {item["title"] for item in r.data["results"]}
    assert "products:G00020" in titles
    assert "products:G00039" in titles
    assert "products:G00000" not in titles  # page 1 territory


@pytest.mark.asyncio
async def test_filter_last_page_signals_no_more():
    backend = await _backend_with_n_rows(50)
    session = SearchSession(budget_tool_calls=10)
    r = await filter_nodes_tool(
        backend, session, table="products", property="category",
        op="==", value="shoes", limit=20, cursor="40",
    )
    assert r.data["offset"] == 40
    assert r.data["showing"] == 10  # only 10 left
    assert r.data["has_more"] is False
    assert r.data["next_cursor"] is None


@pytest.mark.asyncio
async def test_filter_cursor_beyond_total_returns_empty_no_more():
    backend = await _backend_with_n_rows(50)
    session = SearchSession(budget_tool_calls=10)
    r = await filter_nodes_tool(
        backend, session, table="products", property="category",
        op="==", value="shoes", limit=20, cursor="100",
    )
    assert r.data["showing"] == 0
    assert r.data["has_more"] is False
    assert r.data["next_cursor"] is None
    # Total still reported correctly even past the end
    assert r.data["total"] == 50


@pytest.mark.asyncio
async def test_filter_invalid_cursor_falls_back_to_first_page():
    """Malformed cursor must degrade to first page, not error out — a
    multi-turn agent that mangles a cursor token should still make
    progress instead of stalling."""
    backend = await _backend_with_n_rows(50)
    session = SearchSession(budget_tool_calls=10)
    for bad in ("abc", "-5", "", "  "):
        r = await filter_nodes_tool(
            backend, session, table="products", property="category",
            op="==", value="shoes", limit=20, cursor=bad,
        )
        assert r.ok is True
        assert r.data["offset"] == 0, f"cursor={bad!r} should fall back to 0"


@pytest.mark.asyncio
async def test_filter_no_cursor_matches_legacy_behavior():
    """Calls without cursor must behave identically to pre-pagination —
    backwards compat for callers that haven't adopted the protocol."""
    backend = await _backend_with_n_rows(15)
    session = SearchSession(budget_tool_calls=5)
    r = await filter_nodes_tool(
        backend, session, table="products", property="category",
        op="==", value="shoes", limit=20,
    )
    assert r.data["total"] == 15
    assert r.data["showing"] == 15
    assert r.data["has_more"] is False
    assert r.data["next_cursor"] is None
    assert r.data["truncated"] is False  # legacy field still correct


@pytest.mark.asyncio
async def test_filter_total_remains_accurate_across_pages():
    """The agent uses ``total`` to plan how many cursor follow-throughs
    to issue. It must NOT decrease as we paginate — it's the size of
    the matched set, not what's left."""
    backend = await _backend_with_n_rows(50)
    session = SearchSession(budget_tool_calls=10)
    pages = []
    cursor = None
    for _ in range(3):
        r = await filter_nodes_tool(
            backend, session, table="products", property="category",
            op="==", value="shoes", limit=20, cursor=cursor,
        )
        pages.append(r.data["total"])
        cursor = r.data["next_cursor"]
        if cursor is None:
            break
    assert pages == [50, 50, 50]


@pytest.mark.asyncio
async def test_filter_pagination_with_from_ids_constrains_pool():
    """Pagination must respect from_ids (multi-hop chaining): cursor
    walks the *filtered* pool, not the global one."""
    backend = await _backend_with_n_rows(50)
    session = SearchSession(budget_tool_calls=10)
    subset = [f"products:G{i:05d}" for i in range(30)]  # 30 of 50
    r = await filter_nodes_tool(
        backend, session, table="products", property="category",
        op="==", value="shoes", limit=20, from_ids=subset,
    )
    assert r.data["total"] == 30  # not 50
    assert r.data["next_cursor"] == "20"
    r2 = await filter_nodes_tool(
        backend, session, table="products", property="category",
        op="==", value="shoes", limit=20, from_ids=subset, cursor="20",
    )
    assert r2.data["showing"] == 10
    assert r2.data["has_more"] is False


# --- top_nodes pagination ---------------------------------------------------


async def _backend_with_priced_rows(n: int) -> MemoryBackend:
    """Like _backend_with_n_rows but the price column is sortable."""
    b = MemoryBackend()
    await b.connect()
    for i in range(n):
        await b.save_node(
            Node(
                id=f"n_{i:04d}",
                kind=NodeKind.ENTITY,
                title=f"products:G{i:05d}",
                properties={
                    "_table_name": "products",
                    "price": str(i),  # so sort_value matches index
                },
                level=ConsolidationLevel.L0_RAW,
            )
        )
    return b


@pytest.mark.asyncio
async def test_top_nodes_first_page_signals_has_more():
    backend = await _backend_with_priced_rows(30)
    session = SearchSession(budget_tool_calls=10)
    r = await top_nodes_tool(
        backend, session, table="products", sort_by="price",
        order="desc", limit=10,
    )
    assert r.data["total"] == 30
    assert r.data["showing"] == 10
    assert r.data["has_more"] is True
    assert r.data["next_cursor"] == "10"
    # Highest 10 first
    sort_vals = [item["sort_value"] for item in r.data["results"]]
    assert sort_vals == [29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0]


@pytest.mark.asyncio
async def test_top_nodes_second_page_continues_ranking():
    backend = await _backend_with_priced_rows(30)
    session = SearchSession(budget_tool_calls=10)
    r = await top_nodes_tool(
        backend, session, table="products", sort_by="price",
        order="desc", limit=10, cursor="10",
    )
    assert r.data["offset"] == 10
    assert r.data["showing"] == 10
    sort_vals = [item["sort_value"] for item in r.data["results"]]
    assert sort_vals == [19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0]
    assert r.data["next_cursor"] == "20"


# --- aggregate_nodes pagination --------------------------------------------


async def _backend_with_grouped_rows() -> MemoryBackend:
    """40 rows split across 25 distinct categories — last page should be 5."""
    b = MemoryBackend()
    await b.connect()
    for i in range(40):
        await b.save_node(
            Node(
                id=f"n_{i:04d}",
                kind=NodeKind.ENTITY,
                title=f"products:G{i:05d}",
                properties={
                    "_table_name": "products",
                    "category": f"cat_{i % 25:02d}",
                },
                level=ConsolidationLevel.L0_RAW,
            )
        )
    return b


@pytest.mark.asyncio
async def test_aggregate_first_page_signals_has_more():
    backend = await _backend_with_grouped_rows()
    session = SearchSession(budget_tool_calls=10)
    r = await aggregate_nodes_tool(
        backend, session, table="products", group_by="category",
        metric="count", limit=10,
    )
    assert r.data["total_groups"] == 25
    assert r.data["showing"] == 10
    assert r.data["has_more"] is True
    assert r.data["next_cursor"] == "10"


@pytest.mark.asyncio
async def test_aggregate_last_page_signals_no_more():
    backend = await _backend_with_grouped_rows()
    session = SearchSession(budget_tool_calls=10)
    r = await aggregate_nodes_tool(
        backend, session, table="products", group_by="category",
        metric="count", limit=10, cursor="20",
    )
    assert r.data["showing"] == 5  # 25 - 20
    assert r.data["has_more"] is False
    assert r.data["next_cursor"] is None


# --- join_related pagination -----------------------------------------------


async def _backend_for_join(n_reviews: int) -> MemoryBackend:
    """One product + n_reviews reviews, all FK'd to the product."""
    b = MemoryBackend()
    await b.connect()
    await b.save_node(
        Node(
            id="p_1", kind=NodeKind.ENTITY, title="products:G00001",
            properties={"_table_name": "products", "_primary_key": "G00001"},
            level=ConsolidationLevel.L0_RAW,
        )
    )
    for i in range(n_reviews):
        await b.save_node(
            Node(
                id=f"r_{i:04d}",
                kind=NodeKind.ENTITY,
                title=f"reviews:R{i:05d}",
                properties={"_table_name": "reviews", "goods_no": "G00001"},
                level=ConsolidationLevel.L0_RAW,
            )
        )
    return b


@pytest.mark.asyncio
async def test_join_related_first_page_signals_has_more():
    backend = await _backend_for_join(50)
    session = SearchSession(budget_tool_calls=10)
    r = await join_related_tool(
        backend, session, from_value="G00001",
        fk_property="goods_no", target_table="reviews", limit=20,
    )
    assert r.data["total"] == 50
    assert r.data["showing"] == 20
    assert r.data["has_more"] is True
    assert r.data["next_cursor"] == "20"


@pytest.mark.asyncio
async def test_join_related_walks_all_pages_and_terminates():
    """End-to-end: cursor follow-through must yield exactly 50 distinct
    review IDs across 3 pages with no duplicates and a clean stop."""
    backend = await _backend_for_join(50)
    session = SearchSession(budget_tool_calls=20)
    seen: set[str] = set()
    cursor = None
    pages = 0
    while True:
        r = await join_related_tool(
            backend, session, from_value="G00001",
            fk_property="goods_no", target_table="reviews",
            limit=20, cursor=cursor,
        )
        pages += 1
        for item in r.data["results"]:
            assert item["title"] not in seen, "duplicate across pages"
            seen.add(item["title"])
        cursor = r.data["next_cursor"]
        if cursor is None:
            break
        if pages > 5:
            pytest.fail("pagination did not terminate")
    assert len(seen) == 50
    assert pages == 3  # 20 + 20 + 10
