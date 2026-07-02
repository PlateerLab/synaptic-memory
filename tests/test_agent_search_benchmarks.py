"""Unit tests for agent-style retrieval benchmark helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "ablation" / "run_agent_search_benchmarks.py"
)
SPEC = importlib.util.spec_from_file_location("run_agent_search_benchmarks", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

LOOP_RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "ablation" / "run_agent_loop_benchmarks.py"
)
LOOP_SPEC = importlib.util.spec_from_file_location("run_agent_loop_benchmarks", LOOP_RUNNER_PATH)
assert LOOP_SPEC is not None and LOOP_SPEC.loader is not None
loop_runner = importlib.util.module_from_spec(LOOP_SPEC)
sys.modules[LOOP_SPEC.name] = loop_runner
LOOP_SPEC.loader.exec_module(loop_runner)


def test_reciprocal_rank_uses_first_relevant_top10() -> None:
    assert runner._reciprocal_rank(["a", "b", "gold", "other"], {"gold"}) == 1 / 3


def test_reciprocal_rank_ignores_relevant_after_top10() -> None:
    docs = [f"doc{i}" for i in range(10)] + ["gold"]
    assert runner._reciprocal_rank(docs, {"gold"}) == 0.0


def test_recall_at_k_divides_by_relevant_count() -> None:
    assert runner._recall_at_k(["a", "gold1", "gold2"], {"gold1", "gold2", "gold3"}, 3) == 2 / 3


def test_doc_ids_from_nodes_dedupes_preserving_order() -> None:
    nodes = [
        SimpleNamespace(node=SimpleNamespace(properties={"doc_id": "d1"})),
        SimpleNamespace(node=SimpleNamespace(properties={"doc_id": "d2"})),
        SimpleNamespace(node=SimpleNamespace(properties={"doc_id": "d1"})),
    ]

    assert runner._doc_ids_from_nodes(nodes) == ["d1", "d2"]


def test_doc_ids_from_tool_evidence_prefers_document_id() -> None:
    evidence = [
        {"document_id": "doc_a", "properties": {"doc_id": "ignored"}},
        {"properties": {"doc_id": "doc_b"}},
        {"document_id": "doc_a"},
    ]

    assert runner._doc_ids_from_tool_evidence(evidence) == ["doc_a", "doc_b"]


def test_found_relevant_returns_sorted_intersection() -> None:
    assert loop_runner._found_relevant({"b", "a", "noise"}, {"a", "b", "c"}) == ["a", "b"]


def test_first_relevant_trace_hit_returns_first_turn_and_calls() -> None:
    trace = [
        {"turn": 1, "tool_calls": 2, "found_ids": {"noise"}},
        {"turn": 2, "tool_calls": 4, "found_ids": {"gold", "noise"}},
        {"turn": 3, "tool_calls": 5, "found_ids": {"other"}},
    ]

    assert loop_runner._first_relevant_trace_hit(trace, {"gold"}) == (2, 4)


def test_first_relevant_trace_hit_reports_zero_when_missing() -> None:
    trace = [{"turn": 1, "tool_calls": 2, "found_ids": {"noise"}}]

    assert loop_runner._first_relevant_trace_hit(trace, {"gold"}) == (0, 0)


def test_first_relevant_trace_hit_falls_back_to_final_found_ids() -> None:
    trace = [{"turn": 1, "tool_calls": 2, "found_ids": {"noise"}}]

    assert loop_runner._first_relevant_trace_hit(
        trace,
        {"gold"},
        final_found_ids={"gold"},
        final_turn=3,
        final_tool_calls=5,
    ) == (3, 5)


def test_agent_loop_row_jsonl_roundtrip(tmp_path: Path) -> None:
    row = loop_runner.AgentLoopRow(
        qid="q1",
        query="what happened",
        reached=True,
        relevant_docs=["d1", "d2"],
        found_relevant_docs=["d2"],
        found_ids_count=3,
        turns=2,
        tool_calls=4,
        first_relevant_turn=2,
        first_relevant_tool_calls=3,
        duplicate_tool_calls=1,
        empty_tool_calls=0,
        prompt_tokens=123,
        completion_tokens=45,
        elapsed_sec=6.7,
    )
    path = tmp_path / "agent_loop.jsonl"

    loop_runner._append_jsonl_row(path, row)

    assert loop_runner._load_jsonl_rows(path) == [row]


def test_agent_loop_exploration_metrics_count_query_rewrites() -> None:
    tool_log = [
        {
            "tool": "deep_search",
            "key": 'deep_search:{"query": "what is synaptic memory"}',
            "n_results": 5,
            "duplicate": False,
        },
        {
            "tool": "search",
            "key": 'search:{"query": "synaptic memory event ledger"}',
            "n_results": 5,
            "duplicate": False,
        },
        {
            "tool": "get_document",
            "key": 'get_document:{"doc_id": "doc-1", "query": "event ledger provenance"}',
            "n_results": 1,
            "duplicate": False,
        },
    ]

    metrics = loop_runner._exploration_metrics(tool_log, "what is synaptic memory")

    assert metrics["tool_sequence"] == ["deep_search", "search", "get_document"]
    assert metrics["unique_tools"] == 3
    assert metrics["unique_search_targets"] == 3
    assert metrics["query_rewrites"] == 2


def test_agent_loop_extra_context_preserves_original_query_constraints() -> None:
    context = loop_runner._AGENT_LOOP_EXTRA_CONTEXT

    assert "Preserve the original question's specific entities" in context
    assert "vague one-word target" in context


def test_llm_preflight_error_message_names_endpoint_and_skip_hint() -> None:
    msg = loop_runner._llm_preflight_error_message(
        "http://127.0.0.1:18012/v1",
        "Qwen3.6-27B",
        RuntimeError("connection refused"),
    )

    assert "http://127.0.0.1:18012/v1" in msg
    assert "Qwen3.6-27B" in msg
    assert "--skip-preflight" in msg
