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
    "date_range": "DATE_RANGE",  # value format: "YYYY-MM-DD..YYYY-MM-DD"
    "starts_with": "STARTS_WITH",  # efficient prefix match (dates: "2023-12")
}


def _eval_op(op: str, raw_val: object, value: str) -> bool:
    """Evaluate a comparison operator against a property value.

    Handles numeric/string/date comparisons uniformly. Returns True when
    the condition matches.
    """
    if raw_val is None:
        return False
    if op == "contains":
        return value.lower() in str(raw_val).lower()
    if op == "starts_with":
        return str(raw_val).startswith(value)
    if op == "date_range":
        # value format: "YYYY-MM-DD..YYYY-MM-DD" (inclusive)
        if ".." not in value:
            return False
        start, end = value.split("..", 1)
        s = str(raw_val)[: len(start)]
        return start <= s <= end

    try:
        cmp_a: float | str = float(raw_val)  # type: ignore[assignment]
        cmp_b: float | str = float(value)  # type: ignore[assignment]
    except (ValueError, TypeError):
        cmp_a, cmp_b = str(raw_val), str(value)

    if op == ">=":
        return cmp_a >= cmp_b  # type: ignore[operator]
    if op == "<=":
        return cmp_a <= cmp_b  # type: ignore[operator]
    if op == ">":
        return cmp_a > cmp_b  # type: ignore[operator]
    if op == "<":
        return cmp_a < cmp_b  # type: ignore[operator]
    if op in ("==", "="):
        return cmp_a == cmp_b
    if op == "!=":
        return cmp_a != cmp_b
    return False


async def filter_nodes_tool(
    backend: StorageBackend,
    session: SearchSession,
    *,
    table: str = "",
    property: str,
    op: str = "contains",
    value: str,
    limit: int = 20,
    from_ids: list[str] | None = None,
) -> ToolResult:
    """Filter nodes by a typed property value.

    Queries ``properties_json``. Supports numeric comparison
    (>=, <=, >, <, ==), text containment, date ranges, and prefix
    matching.

    Args:
        table: Optional table name filter (e.g. "products", "reviews").
            When empty, searches all nodes.
        property: Property key to filter on (e.g. "selling_price").
        op: One of ``>=``, ``<=``, ``>``, ``<``, ``==``, ``!=``,
            ``contains``, ``starts_with``, ``date_range``.
        value: Value to compare against. For ``date_range`` use
            ``YYYY-MM-DD..YYYY-MM-DD``. For ``starts_with`` pass the
            prefix (useful for month buckets: ``2023-12``).
        limit: Max results to return.
        from_ids: Optional list of node titles/IDs to restrict the
            search to — used for multi-hop chaining (pass previous
            step's ``node_title`` or title values).

    Examples:
        - filter_nodes(property="selling_price", op=">=", value="100000")
        - filter_nodes(table="reviews", property="attribute_2_value", op="contains", value="타이트")
        - filter_nodes(property="broadcast_date", op="starts_with", value="2024-11")
        - filter_nodes(property="sold_dtm", op="date_range", value="2023-06-01..2023-08-31")
        - filter_nodes(from_ids=["products:12800000","products:12800004"], property="discount_rate", op=">", value="30")
    """
    budget = _budget_check(session, "filter_nodes")
    if budget is not None:
        return budget

    sql_op = _OPS.get(op)
    if sql_op is None:
        return ToolResult(
            tool="filter_nodes",
            ok=False,
            data={},
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

        # Pre-filter by from_ids for multi-hop chaining
        id_filter: set[str] | None = None
        if from_ids:
            id_filter = {str(fid) for fid in from_ids}

        matched_nodes: list[Any] = []
        for n in all_nodes:
            props = n.properties or {}
            if table and props.get("_table_name") != table:
                continue
            if id_filter is not None and n.title not in id_filter and n.id not in id_filter:
                continue
            raw_val = props.get(property)
            if raw_val is None:
                continue
            if _eval_op(op, raw_val, value):
                matched_nodes.append(n)

        total = len(matched_nodes)
        nodes = matched_nodes[:limit]

        # Column presence check — used by the 0-result hint builder
        # below to suggest the closest real column when the caller
        # used a typo / near-miss name.
        prop_present_on_table = False
        table_columns: set[str] = set()
        if total == 0 and table and property:
            for n in all_nodes:
                props = n.properties or {}
                if props.get("_table_name") != table:
                    continue
                if property in props:
                    prop_present_on_table = True
                    break
                for k in props:
                    if not k.startswith("_"):
                        table_columns.add(k)
    except Exception as exc:
        return ToolResult(
            tool="filter_nodes",
            ok=False,
            data={},
            session=session.summary(),
            error=f"query_failed: {exc}",
        )

    session.mark_seen(n.id for n in nodes)

    hints: list[Hint] = []
    # Empty-result recovery — the agent often retries with a subtly
    # different predicate when told which alternatives to try. These
    # hints surface through ``project_tool_result`` so the LLM sees
    # them in the next turn.
    if total == 0:
        # Priority hint: column typo / near-miss. If ``property`` is
        # missing on every row in ``table``, the op-level hints are
        # useless — only a column rename will return rows. Emit
        # fuzzy-match candidates first so the agent's next turn
        # targets a real column.
        if table and property and not prop_present_on_table and table_columns:
            import difflib

            candidates = difflib.get_close_matches(
                property, sorted(table_columns), n=2, cutoff=0.5
            )
            for cand in candidates:
                hints.append(
                    Hint(
                        action="filter_nodes",
                        args={"table": table, "property": cand, "op": op, "value": value},
                        reason=(
                            f"column {property!r} not found on {table!r}; "
                            f"did you mean {cand!r}?"
                        ),
                    )
                )
        if op in ("==", "!=", "=") and isinstance(value, str) and value:
            hints.append(
                Hint(
                    action="filter_nodes",
                    args={
                        "table": table,
                        "property": property,
                        "op": "contains",
                        "value": value,
                    },
                    reason="0 exact matches — try contains for substring / partial match",
                )
            )
        if op == "contains" and isinstance(value, str) and " " in value:
            first_tok = value.split(maxsplit=1)[0]
            if first_tok and first_tok != value:
                hints.append(
                    Hint(
                        action="filter_nodes",
                        args={
                            "table": table,
                            "property": property,
                            "op": "contains",
                            "value": first_tok,
                        },
                        reason="0 matches on the full phrase — try the first keyword alone",
                    )
                )
        hints.append(
            Hint(
                action="search",
                args={"query": str(value)},
                reason=f"no structured match on property={property!r}; try FTS across all nodes",
            )
        )

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
        hints=hints,
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
    group_by_format: str = "",
    limit: int = 50,
    from_ids: list[str] | None = None,
) -> ToolResult:
    """Aggregate nodes by a property — GROUP BY + COUNT/SUM/AVG/MAX/MIN.

    Args:
        table: Optional table name filter.
        group_by: Property to group by (e.g. "color_id", "season").
        metric: "count", "sum", "avg", "max", "min".
        metric_property: For sum/avg/max/min — the numeric property to
            aggregate. If empty, aggregates the group_by values themselves.
        where_property: Optional pre-filter property (e.g. "score").
        where_op: Pre-filter operator (>=, <=, >, <, ==, !=, contains,
            starts_with, date_range).
        where_value: Pre-filter value (e.g. "5", "2023-12",
            "2023-06-01..2023-08-31").
        group_by_format: Optional bucketing for date-like values.
            ``"YYYY-MM"`` buckets by month, ``"YYYY"`` by year,
            ``"YYYY-MM-DD"`` by day. Uses string prefix extraction so
            works for ISO-format strings.
        limit: Max groups to return.
        from_ids: Optional list of node titles/IDs to restrict the
            aggregation to — used for multi-hop chaining (pass the
            result of a previous filter/aggregate call).

    Examples:
        - aggregate_nodes(table="products", group_by="season", metric="count")
        - aggregate_nodes(table="feedback", group_by="goods_no", metric="count",
              where_property="score", where_op="==", where_value="5")
        - aggregate_nodes(table="sold_hist", group_by="sold_dtm",
              group_by_format="YYYY-MM", metric="count")  # monthly buckets
        - aggregate_nodes(from_ids=prev_top_products, group_by="category", metric="count")
    """
    budget = _budget_check(session, "aggregate_nodes")
    if budget is not None:
        return budget

    metric_upper = metric.upper()
    if metric_upper not in ("COUNT", "SUM", "AVG", "MAX", "MIN"):
        return ToolResult(
            tool="aggregate_nodes",
            ok=False,
            data={},
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

        # Pre-filter lookup for id_filter (multi-hop chaining)
        id_filter: set[str] | None = None
        if from_ids:
            id_filter = {str(fid) for fid in from_ids}

        # Bucketing length for date-format group_by
        bucket_len = 0
        if group_by_format:
            bucket_len = {
                "YYYY": 4,
                "YYYY-MM": 7,
                "YYYY-MM-DD": 10,
                "YYYY-MM-DD HH": 13,
            }.get(group_by_format, 0)

        for n in all_nodes:
            props = n.properties or {}
            if table and props.get("_table_name") != table:
                continue
            if id_filter is not None and n.title not in id_filter and n.id not in id_filter:
                continue

            # Apply WHERE pre-filter
            if where_property and where_op:
                raw_w = props.get(where_property)
                if raw_w is None:
                    continue
                if not _eval_op(where_op, raw_w, where_value):
                    continue

            grp_val = props.get(group_by)
            if grp_val is None:
                continue
            grp_key = str(grp_val)
            # Apply date/string bucketing when group_by_format is set
            if bucket_len > 0:
                grp_key = grp_key[:bucket_len]

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
            tool="aggregate_nodes",
            ok=False,
            data={},
            session=session.summary(),
            error=f"aggregate_failed: {exc}",
        )

    return ToolResult(
        tool="aggregate_nodes",
        ok=True,
        data={
            "aggregation": {
                "table": table,
                "group_by": group_by,
                "metric": metric,
                **(
                    {"where": f"{where_property} {where_op} {where_value}"}
                    if where_property
                    else {}
                ),
                **({"group_by_format": group_by_format} if group_by_format else {}),
                **({"from_ids_count": len(from_ids)} if from_ids else {}),
            },
            "groups": groups,
            "total_groups": total_groups,
            "showing": len(groups),
            "truncated": total_groups > len(groups),
        },
        hints=_aggregate_hints(
            table=table,
            group_by=group_by,
            metric=metric,
            where_property=where_property,
            where_op=where_op,
            where_value=where_value,
            total_groups=total_groups,
        ),
        session=session.summary(),
    )


def _aggregate_hints(
    *,
    table: str,
    group_by: str,
    metric: str,
    where_property: str,
    where_op: str,
    where_value: str,
    total_groups: int,
) -> list[Hint]:
    """Recovery hints when aggregate_nodes returns 0 groups.

    The most common failure modes are (a) the ``group_by`` column
    doesn't exist in any row, (b) the ``where`` pre-filter is too
    strict. Both are correctable with one more tool call if the agent
    is told which direction to move.
    """
    if total_groups > 0:
        return []
    hints: list[Hint] = []
    if where_property:
        hints.append(
            Hint(
                action="aggregate_nodes",
                args={
                    "table": table,
                    "group_by": group_by,
                    "metric": metric,
                },
                reason="0 groups under this WHERE — retry without the pre-filter first to verify the group_by column",
            )
        )
    if table:
        hints.append(
            Hint(
                action="filter_nodes",
                args={"table": table, "property": group_by, "op": "contains", "value": ""},
                reason=f"verify {group_by!r} is a real column on {table!r} by listing a few rows",
            )
        )
    return hints


async def join_related_tool(
    backend: StorageBackend,
    session: SearchSession,
    *,
    from_value: str = "",
    fk_property: str,
    target_table: str,
    limit: int = 20,
    from_values: list[str] | None = None,
) -> ToolResult:
    """Follow a foreign key relationship to find related nodes.

    Given a value (e.g. product_code="12800000"), finds all nodes in
    ``target_table`` that have the same value in ``fk_property``.
    Accepts either a single ``from_value`` or a list of
    ``from_values`` — useful for multi-hop chaining where the previous
    step produced multiple IDs.

    This is the graph-tool equivalent of SQL JOIN:
    ``SELECT * FROM reviews WHERE product_code IN (...)``

    Args:
        from_value: Single FK value to look up (e.g. "12800000").
        fk_property: The property name that holds the FK
            (e.g. "product_code").
        target_table: The table to search in (e.g. "reviews").
        limit: Max results.
        from_values: Optional list of FK values — pass multiple PK
            values from a previous aggregate/filter result.

    Examples:
        - join_related(from_value="12800000", fk_property="product_code", target_table="reviews")
        - join_related(from_values=["G00001","G00007"], fk_property="goods_no", target_table="pr_goods_sold_hist")
    """
    budget = _budget_check(session, "join_related")
    if budget is not None:
        return budget

    # Normalize to a set of target values (support single + batch).
    # Strip "table:pk" prefixes so agents can pass raw node titles.
    target_values: set[str] = set()
    for raw in (from_values or []) + ([from_value] if from_value else []):
        s = str(raw).strip()
        if not s:
            continue
        # If value looks like a node title "table:pk_val", extract pk_val
        if ":" in s and not s.replace("-", "").isdigit():
            s = s.split(":", 1)[1]
        target_values.add(s)

    if not target_values:
        return ToolResult(
            tool="join_related",
            ok=False,
            data={},
            session=session.summary(),
            error="join_related requires from_value or from_values",
        )

    try:
        matched_nodes: list[Any] = []
        seen: set[str] = set()

        # Full scan by property — robust for both single and batch lookups.
        all_nodes_list = await backend.list_nodes(kind=None, limit=200_000)
        for n in all_nodes_list:
            if n.id in seen:
                continue
            props = n.properties or {}
            if target_table and props.get("_table_name") != target_table:
                continue
            if str(props.get(fk_property, "")) in target_values:
                matched_nodes.append(n)
                seen.add(n.id)

        total = len(matched_nodes)
        nodes = matched_nodes[:limit]
    except Exception as exc:
        return ToolResult(
            tool="join_related",
            ok=False,
            data={},
            session=session.summary(),
            error=f"join_failed: {exc}",
        )

    session.mark_seen(n.id for n in nodes)

    return ToolResult(
        tool="join_related",
        ok=True,
        data={
            "join": {
                "from_values": sorted(target_values)[:10],
                "from_count": len(target_values),
                "fk_property": fk_property,
                "target_table": target_table,
            },
            "total": total,
            "showing": len(nodes),
            "truncated": total > len(nodes),
            "count": len(nodes),
            "results": [_node_to_summary(n) for n in nodes],
        },
        hints=_join_hints(
            fk_property=fk_property,
            target_table=target_table,
            target_values=target_values,
            total=total,
        ),
        session=session.summary(),
    )


def _join_hints(
    *,
    fk_property: str,
    target_table: str,
    target_values: set[str],
    total: int,
) -> list[Hint]:
    """Recovery hints when join_related returns 0 rows.

    Typical cause: the ``fk_property`` name doesn't match the column
    on ``target_table``. Suggest filter_nodes to list rows in the
    target table so the agent can inspect real column names.
    """
    if total > 0 or not target_values:
        return []
    hints: list[Hint] = []
    sample_val = next(iter(target_values))
    hints.append(
        Hint(
            action="filter_nodes",
            args={"table": target_table, "property": fk_property, "op": "==", "value": sample_val},
            reason=f"0 joined rows — verify {fk_property!r} matches the FK column on {target_table!r}",
        )
    )
    return hints
