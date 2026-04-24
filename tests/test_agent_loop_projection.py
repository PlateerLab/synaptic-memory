"""Tests for ``agent_loop.project_tool_result``.

Covers the projection that replaces the old
``json.dumps(result)[:5000]`` truncation at the agent-loop → tool-result
boundary.  Agent benchmark regression (5.8 % fails on vLLM 16k) came
from oversized results accumulating across turns.  These tests lock
down the projected shape + size guarantees.
"""

from __future__ import annotations

import json

import pytest

from synaptic.agent_loop import _TOOL_RESULT_BUDGET, project_tool_result


def test_non_dict_input_falls_back_to_truncation():
    result = project_tool_result("plain string" * 500, max_chars=100)
    assert len(result) <= 100
    # Still a valid JSON string
    assert json.loads(result).startswith("plain string")


def test_small_result_passes_through_unchanged():
    r = {
        "tool": "filter_nodes",
        "ok": True,
        "data": {"results": [{"id": "a", "title": "A"}], "total": 1},
    }
    out = project_tool_result(r)
    parsed = json.loads(out)
    assert parsed["tool"] == "filter_nodes"
    assert parsed["data"]["results"] == [{"id": "a", "title": "A"}]


def test_filter_result_strips_verbose_fields():
    r = {
        "tool": "filter_nodes",
        "ok": True,
        "data": {
            "total": 1,
            "showing": 1,
            "results": [
                {
                    "id": "p1",
                    "kind": "ENTITY",
                    "title": "Product 1",
                    "preview": "X" * 500,
                    "tags": ["a", "b", "c"],
                    "properties": {
                        "selling_price": 12345,
                        "description": "Y" * 500,
                        "nested": {"junk": "Z" * 200},  # dropped (non-scalar)
                    },
                }
            ],
        },
    }
    out = project_tool_result(r)
    parsed = json.loads(out)
    result = parsed["data"]["results"][0]
    # Identifiers preserved
    assert result["id"] == "p1"
    assert result["title"] == "Product 1"
    # Preview truncated
    assert len(result["preview"]) <= 121  # 120 + …
    # Tags + kind dropped
    assert "tags" not in result
    assert "kind" not in result
    # Scalar properties preserved (truncated), nested dropped
    assert result["properties"]["selling_price"] == "12345"
    assert "nested" not in result["properties"]
    assert len(result["properties"]["description"]) <= 81


def test_aggregate_groups_pass_through():
    """Aggregate groups are already compact — pass through, no mutation."""
    r = {
        "tool": "aggregate_nodes",
        "ok": True,
        "data": {
            "aggregation": {"table": "t", "group_by": "k", "metric": "count"},
            "groups": [{"group": "a", "value": 3}, {"group": "b", "value": 2}],
            "total_groups": 2,
            "showing": 2,
        },
    }
    out = project_tool_result(r)
    parsed = json.loads(out)
    assert parsed["data"]["groups"] == [
        {"group": "a", "value": 3},
        {"group": "b", "value": 2},
    ]


def test_deep_search_evidence_trimmed():
    r = {
        "tool": "deep_search",
        "ok": True,
        "data": {
            "evidence": [
                {
                    "id": "n1",
                    "title": "T1",
                    "snippet": "Q" * 500,
                    "properties": {"doc_id": "d1", "unused": "W" * 100},
                }
            ],
            "document_excerpts": [
                {
                    "document": {
                        "id": "d1",
                        "title": "Doc 1",
                        "properties": {"doc_id": "d1", "source": "X" * 500},
                    },
                    "chunks": [{"id": "c1", "title": "Chunk 1", "text": "Z" * 500}],
                }
            ],
        },
    }
    out = project_tool_result(r)
    parsed = json.loads(out)
    ev = parsed["data"]["evidence"][0]
    assert ev["id"] == "n1"
    assert len(ev["snippet"]) <= 181
    ex = parsed["data"]["document_excerpts"][0]
    assert ex["document"]["id"] == "d1"
    assert len(ex["chunks"][0]["text"]) <= 301


def test_oversized_result_triggers_list_shrink():
    r = {
        "tool": "filter_nodes",
        "ok": True,
        "data": {
            "total": 2000,
            "showing": 200,
            "results": [
                {
                    "id": f"p{i}",
                    "title": f"Product {i}",
                    "preview": "X" * 200,
                    "properties": {"selling_price": i, "name": f"item_{i}"},
                }
                for i in range(200)
            ],
        },
    }
    out = project_tool_result(r, max_chars=1500)
    assert len(out) <= 1500
    parsed = json.loads(out)  # still valid JSON
    assert parsed["tool"] == "filter_nodes"
    assert parsed["data"]["total"] == 2000  # meta preserved
    assert len(parsed["data"]["results"]) < 200  # list was shrunk
    assert parsed["data"].get("_trimmed_for_context") is True


def test_budget_default_is_reasonable():
    # Sanity: 4000 chars ≈ 1000 tokens — 4 turns fits in 16k with slack.
    assert 2000 <= _TOOL_RESULT_BUDGET <= 5000


def test_hints_capped_at_three():
    r = {
        "tool": "search",
        "ok": True,
        "data": {"results": []},
        "hints": [{"action": f"h{i}"} for i in range(10)],
    }
    out = project_tool_result(r)
    parsed = json.loads(out)
    assert len(parsed["hints"]) == 3


def test_error_preserved():
    r = {"tool": "filter_nodes", "ok": False, "data": {}, "error": "bad op"}
    out = project_tool_result(r)
    parsed = json.loads(out)
    assert parsed["ok"] is False
    assert parsed["error"] == "bad op"


def test_top_nodes_result_preserves_sort_value():
    """sort_value on each result is load-bearing for the agent — it
    needs the ranking itself, not just the ids. Projection must keep
    it."""
    r = {
        "tool": "top_nodes",
        "ok": True,
        "data": {
            "query": {"table": "products", "sort_by": "cumulative_sales", "order": "desc"},
            "total": 5,
            "results": [
                {
                    "id": "n1",
                    "title": "products:A",
                    "sort_value": 900.0,
                    "preview": "X" * 500,
                    "properties": {"product_code": "A", "season": "25SS"},
                },
                {
                    "id": "n2",
                    "title": "products:B",
                    "sort_value": 500.0,
                    "preview": "Y" * 500,
                    "properties": {"product_code": "B", "season": "24FW"},
                },
            ],
        },
    }
    out = project_tool_result(r)
    parsed = json.loads(out)
    results = parsed["data"]["results"]
    assert results[0]["sort_value"] == 900.0
    assert results[1]["sort_value"] == 500.0
    assert results[0]["title"] == "products:A"
    # Preview trimmed but sort_value intact
    assert len(results[0]["preview"]) <= 121


def test_search_score_preserved_through_projection():
    """Existing ``search_tool`` results carry ``score`` — same contract."""
    r = {
        "tool": "search",
        "ok": True,
        "data": {
            "results": [{"id": "a", "title": "A", "score": 0.87, "preview": "x"}],
        },
    }
    out = project_tool_result(r)
    parsed = json.loads(out)
    assert parsed["data"]["results"][0]["score"] == 0.87


def test_unknown_tool_result_still_projected():
    r = {
        "tool": "brand_new_tool",
        "ok": True,
        "data": {"results": [{"id": "x", "title": "t", "preview": "Y" * 500}]},
    }
    out = project_tool_result(r)
    parsed = json.loads(out)
    # Generic results[] rule applied
    assert parsed["data"]["results"][0]["id"] == "x"
    assert len(parsed["data"]["results"][0]["preview"]) <= 121


@pytest.mark.parametrize("budget", [500, 1000, 2000, 4000])
def test_projection_respects_budget_on_huge_input(budget):
    r = {
        "tool": "filter_nodes",
        "ok": True,
        "data": {
            "total": 10_000,
            "results": [
                {
                    "id": f"p{i}",
                    "title": f"Product {i}" * 10,
                    "preview": "X" * 1000,
                    "properties": {f"k_{j}": f"v_{j}" * 20 for j in range(20)},
                }
                for i in range(500)
            ],
        },
    }
    out = project_tool_result(r, max_chars=budget)
    assert len(out) <= budget
    # Must round-trip as valid JSON at every budget level
    json.loads(out)
