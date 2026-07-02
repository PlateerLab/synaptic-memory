"""Agent efficiency telemetry — duplicate / empty-result tool-call signals.

``AgentSearchResult.tool_log`` records one entry per tool call with the
normalized args key, result count, and a duplicate flag (same tool+args
reissued this query). These are the DETERMINISTIC waste signals the efficiency
profiler aggregates (no solve-rate noise floor).
"""

from __future__ import annotations

import json

import pytest

from synaptic.agent_loop import (
    _EFFICIENCY_DIRECTIVE,
    AGENT_SYSTEM,
    AGENT_TOOLS,
    _args_key,
    _bounded_int,
    _dispatch_tool,
    _result_count,
    run_agent_loop,
)
from synaptic.agent_tools import ToolResult
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.models import ConsolidationLevel, Node, NodeKind


def test_result_count():
    assert _result_count({"evidence": [1, 2, 3]}) == 3
    assert _result_count({"results": [1], "groups": [1, 1]}) == 3
    assert _result_count({"document": {"id": "d1"}}) == 1
    assert _result_count({}) == 0
    assert _result_count({"evidence": []}) == 0


def test_args_key_normalizes():
    # whitespace + case + key order collapse to the same key
    a = _args_key("search", {"query": "Foo Bar "})
    b = _args_key("search", {"query": "foo bar"})
    assert a == b
    # different tool or args → different key
    assert _args_key("search", {"query": "x"}) != _args_key("deep_search", {"query": "x"})


def test_agent_system_preserves_query_constraints_for_followups():
    assert "Follow-up searches must preserve the user's constraints" in AGENT_SYSTEM
    assert "named entity, attribute, and relation" in AGENT_SYSTEM


def _tool_schema(name: str) -> dict:
    for tool in AGENT_TOOLS:
        fn = tool["function"]
        if fn["name"] == name:
            return fn
    raise AssertionError(f"missing tool schema: {name}")


def test_agent_tool_schema_exposes_search_limits():
    deep_props = _tool_schema("deep_search")["parameters"]["properties"]
    assert {"query", "category", "limit", "read_top_k"}.issubset(deep_props)

    search_props = _tool_schema("search")["parameters"]["properties"]
    assert {"query", "limit"}.issubset(search_props)


def test_bounded_int_clamps_tool_limits():
    assert _bounded_int("15", default=10, minimum=1, maximum=20) == 15
    assert _bounded_int(" 15 ", default=10, minimum=1, maximum=20) == 15
    assert _bounded_int(None, default=10, minimum=1, maximum=20) == 10
    assert _bounded_int("invalid", default=10, minimum=1, maximum=20) == 10
    assert _bounded_int(99, default=10, minimum=1, maximum=20) == 20
    assert _bounded_int(-5, default=10, minimum=1, maximum=20) == 1


@pytest.mark.asyncio
async def test_dispatch_passes_agent_requested_search_limits(monkeypatch):
    import synaptic.agent_tools as agent_tools
    import synaptic.agent_tools_v2 as agent_tools_v2

    calls: list[tuple] = []

    async def fake_deep_search_tool(
        backend,
        session,
        query,
        *,
        category=None,
        limit=10,
        read_top_k=2,
        embedder=None,
    ):
        calls.append(("deep_search", query, category, limit, read_top_k))
        return ToolResult(tool="deep_search", ok=True, data={"evidence": []})

    async def fake_search_tool(backend, session, query, *, limit=10, embedder=None):
        calls.append(("search", query, limit))
        return ToolResult(tool="search", ok=True, data={"evidence": []})

    monkeypatch.setattr(agent_tools_v2, "deep_search_tool", fake_deep_search_tool)
    monkeypatch.setattr(agent_tools, "search_tool", fake_search_tool)

    await _dispatch_tool(
        "deep_search",
        {"query": "q", "category": "cat", "limit": 99, "read_top_k": 9},
        None,
        None,
    )
    await _dispatch_tool("search", {"query": "q", "limit": -5}, None, None)

    assert calls == [
        ("deep_search", "q", "cat", 20, 5),
        ("search", "q", 1),
    ]


# --- fake client (reused shape) ---------------------------------------


class _Func:
    def __init__(self, name, args):
        self.name = name
        self.arguments = json.dumps(args)


class _ToolCall:
    def __init__(self, name, args, tcid):
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
    def __init__(self, agent_msgs):
        self._agent = list(agent_msgs)
        self.system_seen = ""  # last system message sent on an agent turn
        self.agent_message_log: list[list] = []
        self.chat = self
        self.completions = self

    async def create(self, *, model, messages, tools=None, max_tokens=None):
        if tools is None:  # sufficiency judge
            return _Resp(_Msg(content='{"sufficient": true}'))
        self.agent_message_log.append([dict(m) for m in messages])
        for m in messages:
            if m.get("role") == "system":
                self.system_seen = m["content"]
                break
        return _Resp(self._agent.pop(0))


@pytest.mark.asyncio
async def test_tool_log_flags_duplicate_and_empty():
    b = SqliteGraphBackend(":memory:")
    await b.connect()
    await b.save_node(
        Node(
            id="d1",
            kind=NodeKind.CHUNK,
            title="topic",
            content="evidence about the topic",
            level=ConsolidationLevel.L0_RAW,
        )
    )
    # turn0: search "topic" (hits) + search "zzdoesnotexist" (empty);
    # turn1: search "topic" AGAIN (duplicate); turn2: final answer.
    agent_msgs = [
        _Msg(
            tool_calls=[
                _ToolCall("search", {"query": "topic"}, "t1"),
                _ToolCall("search", {"query": "zzdoesnotexist"}, "t2"),
            ]
        ),
        _Msg(tool_calls=[_ToolCall("search", {"query": "topic"}, "t3")]),
        _Msg(content="done"),
    ]
    res = await run_agent_loop(
        client=_FakeClient(agent_msgs), backend=b, query="q", sufficiency_gate=False
    )

    assert res.tool_calls_made == 3
    log = res.tool_log
    assert len(log) == 3
    # first "topic" search hit, not a duplicate
    assert log[0]["tool"] == "search" and log[0]["n_results"] >= 1
    assert log[0]["duplicate"] is False
    # the nonsense search returned nothing (empty-result waste signal)
    assert log[1]["n_results"] == 0
    # the repeated "topic" search is flagged duplicate (the waste we hunt)
    assert log[2]["duplicate"] is True
    await b.close()


async def _backend_one_node():
    b = SqliteGraphBackend(":memory:")
    await b.connect()
    await b.save_node(
        Node(
            id="d1",
            kind=NodeKind.CHUNK,
            title="topic",
            content="evidence about the topic",
            level=ConsolidationLevel.L0_RAW,
        )
    )
    return b


@pytest.mark.asyncio
async def test_efficiency_directive_on_by_default():
    # Promoted to default-on (measured -15..-26% tool_calls, solve non-negative).
    b = await _backend_one_node()
    client = _FakeClient([_Msg(content="answer")])
    await run_agent_loop(client=client, backend=b, query="q", sufficiency_gate=False)
    assert _EFFICIENCY_DIRECTIVE in client.system_seen
    await b.close()


@pytest.mark.asyncio
async def test_efficiency_directive_can_be_disabled():
    b = await _backend_one_node()
    client = _FakeClient([_Msg(content="answer")])
    await run_agent_loop(
        client=client, backend=b, query="q", sufficiency_gate=False, efficiency_hint=False
    )
    assert _EFFICIENCY_DIRECTIVE not in client.system_seen
    await b.close()


@pytest.mark.asyncio
async def test_efficiency_directive_env_can_disable(monkeypatch):
    # Escape hatch: SYNAPTIC_AGENT_EFFICIENCY=0 turns the default-on directive off.
    monkeypatch.setenv("SYNAPTIC_AGENT_EFFICIENCY", "0")
    b = await _backend_one_node()
    client = _FakeClient([_Msg(content="answer")])
    await run_agent_loop(client=client, backend=b, query="q", sufficiency_gate=False)
    assert _EFFICIENCY_DIRECTIVE not in client.system_seen  # env disabled it
    await b.close()


@pytest.mark.asyncio
async def test_force_first_tool_rejects_zero_tool_answer_once():
    b = await _backend_one_node()
    client = _FakeClient(
        [
            _Msg(content="prior-only answer"),
            _Msg(tool_calls=[_ToolCall("search", {"query": "topic"}, "t1")]),
            _Msg(content="grounded answer"),
        ]
    )

    res = await run_agent_loop(
        client=client,
        backend=b,
        query="topic",
        sufficiency_gate=False,
        force_first_tool=True,
    )

    assert res.final_answer == "grounded answer"
    assert res.tool_calls_made == 1
    assert res.tool_log[0]["tool"] == "search"
    assert "not used the retrieval tools yet" in client.agent_message_log[1][-1]["content"]
    await b.close()
