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

import logging
from typing import TYPE_CHECKING, Any

from synaptic.agent_tools import ToolResult, _budget_check, _node_to_summary
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
        conditions.append("json_extract(properties_json, ?) LIKE ?")
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
        # Use the StorageBackend protocol (list_nodes + Python filter)
        # instead of raw SQL. Works with ANY backend, not just SQLite.
        # Full scan so we can report the accurate total count, not just
        # the limited sample — this matters for "how many X?" questions.
        all_nodes = await backend.list_nodes(kind=None, limit=200_000)
        matched_nodes: list[Any] = []
        for n in all_nodes:
            props = n.properties or {}
            if table and props.get("_table_name") != table:
                continue
            raw_val = props.get(property)
            if raw_val is None:
                continue
            if op == "contains":
                if value.lower() in str(raw_val).lower():
                    matched_nodes.append(n)
            else:
                try:
                    cmp_a = float(raw_val)
                    cmp_b = float(value)
                except ValueError:
                    cmp_a, cmp_b = str(raw_val), str(value)
                matched = (
                    (op in (">=",) and cmp_a >= cmp_b)
                    or (op in ("<=",) and cmp_a <= cmp_b)
                    or (op in (">",) and cmp_a > cmp_b)
                    or (op in ("<",) and cmp_a < cmp_b)
                    or (op in ("==", "=") and cmp_a == cmp_b)
                    or (op in ("!=",) and cmp_a != cmp_b)
                )
                if matched:
                    matched_nodes.append(n)

        total = len(matched_nodes)
        nodes = matched_nodes[:limit]
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
            "total": total,
            "showing": len(nodes),
            "truncated": total > len(nodes),
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
    metric_property: str = "",
    where_property: str = "",
    where_op: str = "",
    where_value: str = "",
    limit: int = 50,
) -> ToolResult:
    """Aggregate nodes by a property — GROUP BY + COUNT/SUM/AVG/MAX/MIN.

    Args:
        table: Optional table name filter.
        group_by: Property to group by (e.g. "color_id", "season").
        metric: "count", "sum", "avg", "max", "min".
        metric_property: For sum/avg/max/min — the numeric property to
            aggregate. If empty, aggregates the group_by values themselves.
        where_property: Optional pre-filter property (e.g. "score").
        where_op: Pre-filter operator (>=, <=, >, <, ==, !=, contains).
        where_value: Pre-filter value (e.g. "5").
        limit: Max groups to return.

    Examples:
        - aggregate_nodes(table="products", group_by="season", metric="count")
        - aggregate_nodes(table="feedback", group_by="goods_no", metric="count",
              where_property="score", where_op="==", where_value="5")
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

    try:
        all_nodes = await backend.list_nodes(kind=None, limit=200_000)
        buckets: dict[str, list[float]] = {}
        # Detect FK target table: find which table uses group_by column as PK
        fk_target_table: str = ""
        pk_by_table: dict[str, str] = {}
        for n in all_nodes:
            props = n.properties or {}
            tbl = props.get("_table_name")
            pk = props.get("_primary_key")
            if tbl and pk and tbl not in pk_by_table:
                pk_by_table[tbl] = pk
        for tbl, pk in pk_by_table.items():
            if pk == group_by and tbl != table:
                fk_target_table = tbl
                break

        # Pre-filter operator lookup
        where_sql_op = _OPS.get(where_op) if where_op else None

        for n in all_nodes:
            props = n.properties or {}
            if table and props.get("_table_name") != table:
                continue

            # Apply WHERE pre-filter
            if where_property and where_sql_op:
                raw_w = props.get(where_property)
                if raw_w is None:
                    continue
                if where_op == "contains":
                    if where_value.lower() not in str(raw_w).lower():
                        continue
                else:
                    try:
                        cmp_a = float(raw_w)
                        cmp_b = float(where_value)
                    except (ValueError, TypeError):
                        cmp_a, cmp_b = str(raw_w), str(where_value)  # type: ignore[assignment]
                    passed = (
                        (where_op in (">=",) and cmp_a >= cmp_b)
                        or (where_op in ("<=",) and cmp_a <= cmp_b)
                        or (where_op in (">",) and cmp_a > cmp_b)
                        or (where_op in ("<",) and cmp_a < cmp_b)
                        or (where_op in ("==", "=") and cmp_a == cmp_b)
                        or (where_op in ("!=",) and cmp_a != cmp_b)
                    )
                    if not passed:
                        continue

            grp_val = props.get(group_by)
            if grp_val is None:
                continue
            grp_key = str(grp_val)

            # Value for metric: use metric_property if provided
            if metric_property:
                raw_m = props.get(metric_property)
                try:
                    num = float(raw_m) if raw_m is not None else 0.0
                except (ValueError, TypeError):
                    num = 0.0
            else:
                try:
                    num = float(grp_val)
                except (ValueError, TypeError):
                    num = 1.0
            buckets.setdefault(grp_key, []).append(num)

        groups = []
        for grp_key, vals in buckets.items():
            if metric_upper == "COUNT":
                agg_val = len(vals)
            elif metric_upper == "SUM":
                agg_val = sum(vals)
            elif metric_upper == "AVG":
                agg_val = sum(vals) / len(vals) if vals else 0
            elif metric_upper == "MAX":
                agg_val = max(vals)
            elif metric_upper == "MIN":
                agg_val = min(vals)
            else:
                agg_val = len(vals)
            grp_entry: dict[str, Any] = {"group": grp_key, "value": agg_val}
            if fk_target_table:
                grp_entry["node_title"] = f"{fk_target_table}:{grp_key}"
            groups.append(grp_entry)

        groups.sort(key=lambda g: -g["value"])
        total_groups = len(groups)
        groups = groups[:limit]
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
            "aggregation": {
                "table": table, "group_by": group_by, "metric": metric,
                **({"where": f"{where_property} {where_op} {where_value}"} if where_property else {}),
            },
            "groups": groups,
            "total_groups": total_groups,
            "showing": len(groups),
            "truncated": total_groups > len(groups),
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

    try:
        nodes = []

        # Strategy 1: Graph edge traversal — find source node, follow RELATED edges.
        # Much faster than full scan: O(degree) instead of O(N).
        source_node = None
        # Try to find the source node by title pattern (table:pk)
        try:
            # Guess source table: find a table where from_value is the PK
            sample_nodes = await backend.list_nodes(kind=None, limit=1000)
            for n in sample_nodes:
                props = n.properties or {}
                pk_col = props.get("_primary_key", "")
                if pk_col == fk_property and str(props.get(fk_property, "")) == str(from_value):
                    source_node = n
                    break
        except Exception:
            pass

        matched_nodes: list[Any] = []

        if source_node:
            from synaptic.models import EdgeKind
            try:
                edges = await backend.get_edges(source_node.id, direction="both")
                for edge in edges:
                    if edge.kind != EdgeKind.RELATED:
                        continue
                    other_id = edge.target_id if edge.source_id == source_node.id else edge.source_id
                    other = await backend.get_node(other_id)
                    if other is None:
                        continue
                    if target_table and (other.properties or {}).get("_table_name") != target_table:
                        continue
                    matched_nodes.append(other)
            except Exception:
                pass

        # Strategy 2: Property scan fallback — always run to get accurate total.
        # Full scan so we report the true count, not just edge-traversed subset.
        seen = {n.id for n in matched_nodes}
        all_nodes_list = await backend.list_nodes(kind=None, limit=200_000)
        for n in all_nodes_list:
            if n.id in seen:
                continue
            props = n.properties or {}
            if props.get("_table_name") != target_table:
                continue
            if str(props.get(fk_property, "")) == str(from_value):
                matched_nodes.append(n)
                seen.add(n.id)

        total = len(matched_nodes)
        nodes = matched_nodes[:limit]
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
            "total": total,
            "showing": len(nodes),
            "truncated": total > len(nodes),
            "count": len(nodes),
            "results": [_node_to_summary(n) for n in nodes],
        },
        hints=[],
        session=session.summary(),
    )
