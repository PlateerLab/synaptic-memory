"""Structured data tools — filter, aggregate, join on typed properties.

For structured data (CSV/RDB) ingested via ``TableIngester``, the node
properties contain typed values (prices, dates, categories) stored in
``properties_json``. These tools query those properties with SQL
``json_extract`` — no FTS, no embedding, pure structural queries.

This is what makes structured data queryable:
- "10만원 이상 상품" → filter_nodes(property="selling_price", op=">=", value="100000")
- "색상별 상품 수" → aggregate_nodes(group_by="color_id", metric="count")
- "상품의 리뷰" → join_related(from_table="products", fk="product_code", to_table="reviews")

All tools use SQLite ``json_extract(properties_json, '$.key')`` which
is fast enough for ~100K rows without additional indexing. For larger
scales, add generated columns + indexes.

These tools are domain-agnostic — they work with any TableIngester
output regardless of the source schema.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from synaptic.agent_tools import Hint, ToolResult, _budget_check, _node_to_summary
from synaptic.search_session import SearchSession

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("agent-tools-structured")

# Supported comparison operators
_OPS = {
    ">=": ">=",
    "<=": "<=",
    ">": ">",
    "<": "<",
    "==": "=",
    "=": "=",
    "!=": "!=",
    "contains": "LIKE",
}


async def filter_nodes_tool(
    backend: StorageBackend,
    session: SearchSession,
    *,
    table: str = "",
    property: str,
    op: str = "contains",
    value: str,
    limit: int = 20,
) -> ToolResult:
    """Filter nodes by a typed property value.

    Queries ``properties_json`` via SQLite ``json_extract``. Supports
    numeric comparison (>=, <=, >, <, ==) and text containment.

    Args:
        table: Optional table name filter (e.g. "products", "reviews").
            When empty, searches all nodes.
        property: Property key to filter on (e.g. "selling_price").
        op: Comparison operator. One of: >=, <=, >, <, ==, !=, contains.
        value: Value to compare against. Numbers are cast automatically.
        limit: Max results to return.

    Examples:
        - filter_nodes(property="selling_price", op=">=", value="100000")
        - filter_nodes(table="reviews", property="attribute_2_value", op="contains", value="타이트")
        - filter_nodes(property="broadcast_date", op="contains", value="2024-11")
    """
    budget = _budget_check(session, "filter_nodes")
    if budget is not None:
        return budget

    sql_op = _OPS.get(op)
    if sql_op is None:
        return ToolResult(
            tool="filter_nodes", ok=False, data={},
            session=session.summary(),
            error=f"unknown operator: {op}. Use: {list(_OPS.keys())}",
        )

    # Build SQL
    conditions = []
    params: list[Any] = []

    if table:
        conditions.append("json_extract(properties_json, '$._table_name') = ?")
        params.append(table)

    prop_path = f"$.{property}"
    if op == "contains":
        conditions.append(f"json_extract(properties_json, ?) LIKE ?")
        params.extend([prop_path, f"%{value}%"])
    else:
        # Try numeric comparison first, fall back to string
        try:
            num_val = float(value)
            conditions.append(f"CAST(json_extract(properties_json, ?) AS REAL) {sql_op} ?")
            params.extend([prop_path, num_val])
        except ValueError:
            conditions.append(f"json_extract(properties_json, ?) {sql_op} ?")
            params.extend([prop_path, value])

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"SELECT * FROM syn_nodes WHERE {where} LIMIT ?"
    params.append(limit)

    try:
        db = backend._db()
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()

        from synaptic.backends.sqlite import _row_to_node
        nodes = [_row_to_node(r) for r in rows]
    except Exception as exc:
        return ToolResult(
            tool="filter_nodes", ok=False, data={},
            session=session.summary(),
            error=f"query_failed: {exc}",
        )

    session.mark_seen(n.id for n in nodes)

    return ToolResult(
        tool="filter_nodes",
        ok=True,
        data={
            "filter": {"table": table, "property": property, "op": op, "value": value},
            "count": len(nodes),
            "results": [_node_to_summary(n) for n in nodes],
        },
        hints=[],
        session=session.summary(),
    )


async def aggregate_nodes_tool(
    backend: StorageBackend,
    session: SearchSession,
    *,
    table: str = "",
    group_by: str,
    metric: str = "count",
    limit: int = 50,
) -> ToolResult:
    """Aggregate nodes by a property — GROUP BY + COUNT/SUM/AVG/MAX/MIN.

    Args:
        table: Optional table name filter.
        group_by: Property to group by (e.g. "color_id", "season").
        metric: "count", "sum", "avg", "max", "min". For sum/avg/max/min,
            a ``metric_property`` is needed (defaults to the first
            numeric property found).
        limit: Max groups to return.

    Examples:
        - aggregate_nodes(table="products", group_by="season", metric="count")
        - aggregate_nodes(table="product_variants", group_by="color_id", metric="count")
    """
    budget = _budget_check(session, "aggregate_nodes")
    if budget is not None:
        return budget

    metric_upper = metric.upper()
    if metric_upper not in ("COUNT", "SUM", "AVG", "MAX", "MIN"):
        return ToolResult(
            tool="aggregate_nodes", ok=False, data={},
            session=session.summary(),
            error=f"unknown metric: {metric}. Use: count, sum, avg, max, min",
        )

    conditions = []
    params: list[Any] = []
    if table:
        conditions.append("json_extract(properties_json, '$._table_name') = ?")
        params.append(table)

    where = " AND ".join(conditions) if conditions else "1=1"
    group_path = f"$.{group_by}"

    if metric_upper == "COUNT":
        sql = f"""
            SELECT json_extract(properties_json, ?) as grp, COUNT(*) as val
            FROM syn_nodes WHERE {where}
            GROUP BY grp ORDER BY val DESC LIMIT ?
        """
        params_full = [group_path, *params, limit]
    else:
        # For SUM/AVG/MAX/MIN, aggregate on the group_by property itself
        sql = f"""
            SELECT json_extract(properties_json, ?) as grp,
                   {metric_upper}(CAST(json_extract(properties_json, ?) AS REAL)) as val
            FROM syn_nodes WHERE {where}
            GROUP BY grp ORDER BY val DESC LIMIT ?
        """
        params_full = [group_path, group_path, *params, limit]

    try:
        db = backend._db()
        async with db.execute(sql, params_full) as cur:
            rows = await cur.fetchall()
        groups = [{"group": r[0], "value": r[1]} for r in rows if r[0] is not None]
    except Exception as exc:
        return ToolResult(
            tool="aggregate_nodes", ok=False, data={},
            session=session.summary(),
            error=f"aggregate_failed: {exc}",
        )

    return ToolResult(
        tool="aggregate_nodes",
        ok=True,
        data={
            "aggregation": {"table": table, "group_by": group_by, "metric": metric},
            "groups": groups,
            "total_groups": len(groups),
        },
        hints=[],
        session=session.summary(),
    )


async def join_related_tool(
    backend: StorageBackend,
    session: SearchSession,
    *,
    from_value: str,
    fk_property: str,
    target_table: str,
    limit: int = 20,
) -> ToolResult:
    """Follow a foreign key relationship to find related nodes.

    Given a value (e.g. product_code="12800000"), finds all nodes in
    ``target_table`` that have the same value in ``fk_property``.

    This is the graph-tool equivalent of SQL JOIN:
    ``SELECT * FROM reviews WHERE product_code = '12800000'``

    Args:
        from_value: The FK value to look up (e.g. "12800000").
        fk_property: The property name that holds the FK
            (e.g. "product_code").
        target_table: The table to search in (e.g. "reviews").
        limit: Max results.

    Examples:
        - join_related(from_value="12800000", fk_property="product_code", target_table="reviews")
          → all reviews for product 12800000
        - join_related(from_value="1", fk_property="color_id", target_table="product_variants")
          → all variants with color_id=1
    """
    budget = _budget_check(session, "join_related")
    if budget is not None:
        return budget

    sql = """
        SELECT * FROM syn_nodes
        WHERE json_extract(properties_json, '$._table_name') = ?
          AND json_extract(properties_json, ?) = ?
        LIMIT ?
    """
    fk_path = f"$.{fk_property}"

    try:
        db = backend._db()
        async with db.execute(sql, [target_table, fk_path, from_value, limit]) as cur:
            rows = await cur.fetchall()

        from synaptic.backends.sqlite import _row_to_node
        nodes = [_row_to_node(r) for r in rows]
    except Exception as exc:
        return ToolResult(
            tool="join_related", ok=False, data={},
            session=session.summary(),
            error=f"join_failed: {exc}",
        )

    session.mark_seen(n.id for n in nodes)

    return ToolResult(
        tool="join_related",
        ok=True,
        data={
            "join": {
                "from_value": from_value,
                "fk_property": fk_property,
                "target_table": target_table,
            },
            "count": len(nodes),
            "results": [_node_to_summary(n) for n in nodes],
        },
        hints=[],
        session=session.summary(),
    )
