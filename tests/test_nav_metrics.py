"""Agent-navigation-efficiency metrics + the run_agent_loop trace that feeds them.

The trace records, per turn, the cumulative evidence the agent has reached, so
``navigation_efficiency`` can report hops/calls-to-gold and success@budget — the
measurement used to judge whether a graph-structure change actually helps the
agent find faster. Default off → the 81% agent baseline is untouched.
"""

from __future__ import annotations

import json

import pytest

from synaptic.agent_loop import run_agent_loop
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.models import ConsolidationLevel, Node, NodeKind
from synaptic.nav_metrics import aggregate, navigation_efficiency

# --- pure metric -------------------------------------------------------


def _trace(*turns):
    # turns: (turn, tool_calls, [found ids])
    return [{"turn": t, "tool_calls": c, "found_ids": set(ids)} for t, c, ids in turns]


def test_first_and_all_gold():
    tr = _trace((1, 1, ["a"]), (2, 2, ["a", "b"]), (3, 3, ["a", "b", "g1"]), (4, 5, ["g1", "g2"]))
    m = navigation_efficiency(tr, {"g1", "g2"})
    assert m["found_any"] and m["found_all"]
    assert m["hops_to_first_gold"] == 3  # g1 first appears turn 3
    assert m["calls_to_first_gold"] == 3
    assert m["hops_to_all_gold"] == 4  # g2 only by turn 4
    assert m["calls_to_all_gold"] == 5


def test_never_found():
    tr = _trace((1, 1, ["x"]), (2, 2, ["y"]))
    m = navigation_efficiency(tr, {"g1"})
    assert not m["found_any"] and not m["found_all"]
    assert m["hops_to_first_gold"] is None
    assert m["calls_to_first_gold"] is None


def test_success_within_budget():
    tr = _trace((1, 2, ["g1"]))
    assert navigation_efficiency(tr, {"g1"}, call_budget=3)["success_within_budget"] is True
    assert navigation_efficiency(tr, {"g1"}, call_budget=1)["success_within_budget"] is False


def test_empty_gold_is_safe():
    m = navigation_efficiency(_trace((1, 1, ["a"])), set(), call_budget=2)
    assert m["found_any"] is False and m["success_within_budget"] is False


def test_aggregate_means_over_reached_only():
    q1 = navigation_efficiency(_trace((1, 1, ["g"])), {"g"}, call_budget=2)
    q2 = navigation_efficiency(_trace((1, 1, ["x"])), {"g"}, call_budget=2)  # never
    agg = aggregate([q1, q2])
    assert agg["n_queries"] == 2
    assert agg["reach_any_rate"] == 0.5
    assert agg["mean_hops_to_first_gold"] == 1  # mean over the one that reached
    assert agg["success_within_budget_rate"] == 0.5


# --- trace plumbing in the real loop -----------------------------------


class _Func:
    def __init__(self, name, args):
        self.name = name
        self.arguments = json.dumps(args)


class _ToolCall:
    def __init__(self, name, args, tcid="tc1"):
        self.id = tcid
        self.function = _Func(name, args)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        return {"role": "assistant", "content": self.content or ""}


class _Resp:
    def __init__(self, msg):
        self.choices = [type("Ch", (), {"message": msg})()]


class _FakeClient:
    def __init__(self, msgs):
        self._msgs = list(msgs)
        self.chat = self
        self.completions = self

    async def create(self, *, model, messages, tools=None, max_tokens=None):
        return _Resp(self._msgs.pop(0))


async def _backend():
    b = SqliteGraphBackend(":memory:")
    await b.connect()
    await b.save_node(
        Node(
            id="d1",
            kind=NodeKind.CHUNK,
            title="test topic",
            content="test evidence about the topic",
            level=ConsolidationLevel.L0_RAW,
        )
    )
    return b


@pytest.mark.asyncio
async def test_record_trace_off_by_default():
    b = await _backend()
    client = _FakeClient([_Msg(content="answer")])
    res = await run_agent_loop(client=client, backend=b, query="q")
    assert res.trace == []
    await b.close()


@pytest.mark.asyncio
async def test_record_trace_captures_per_turn_evidence():
    b = await _backend()
    client = _FakeClient(
        [_Msg(tool_calls=[_ToolCall("search", {"query": "test"})]), _Msg(content="final")]
    )
    res = await run_agent_loop(client=client, backend=b, query="q", record_trace=True)
    assert len(res.trace) == 1  # one tool-calling turn
    snap = res.trace[0]
    assert snap["turn"] == 1 and snap["tool_calls"] == 1
    # the loop's _extract_ids surfaces the evidence key the agent's tools emit
    # (the chunk title here), not the raw node id — the metric is key-agnostic,
    # so gold matching in the harness must use whatever the tools emit.
    assert "test topic" in snap["found_ids"]
    # the metric reads the trace end-to-end
    m = navigation_efficiency(res.trace, {"test topic"})
    assert m["found_all"] and m["hops_to_first_gold"] == 1
    await b.close()
