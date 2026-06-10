"""Token accounting on run_agent_loop — the cost axis of cost-at-quality.

``AgentSearchResult.prompt_tokens / completion_tokens`` must accumulate the
usage of EVERY LLM call the loop makes (main turns, sufficiency judge, forced
final synthesis) and stay 0 — never crash — when responses carry no usage
(fail-open, e.g. test stubs or gateways that strip it).
"""

from __future__ import annotations

import json

import pytest

from synaptic.agent_loop import run_agent_loop
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.models import ConsolidationLevel, Node, NodeKind

# --- fake client (test_agent_efficiency stub shape + usage) -----------


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


class _Usage:
    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Resp:
    def __init__(self, msg, usage=None):
        self.choices = [type("Ch", (), {"message": msg})()]
        self.usage = usage


class _FakeClient:
    """Pops agent-turn responses in order; tool-less calls (sufficiency judge
    or forced synthesis) get ``notool_resp``."""

    def __init__(self, agent_resps, notool_resp=None):
        self._agent = list(agent_resps)
        self._notool = notool_resp
        self.notool_calls = 0
        self.chat = self
        self.completions = self

    async def create(self, *, model, messages, tools=None, max_tokens=None):
        if tools is None:
            self.notool_calls += 1
            return self._notool or _Resp(_Msg(content='{"sufficient": true}'))
        return self._agent.pop(0)


async def _backend_with_doc():
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
async def test_usage_accumulates_across_turns():
    b = await _backend_with_doc()
    resps = [
        _Resp(_Msg(tool_calls=[_ToolCall("search", {"query": "topic"}, "t1")]), _Usage(100, 20)),
        _Resp(_Msg(content="done"), _Usage(50, 10)),
    ]
    res = await run_agent_loop(
        client=_FakeClient(resps), backend=b, query="q", sufficiency_gate=False
    )
    assert res.final_answer == "done"
    assert res.prompt_tokens == 150
    assert res.completion_tokens == 30


@pytest.mark.asyncio
async def test_usage_zero_when_responses_have_none():
    # fail-open: stubs without usage must yield 0, not raise
    b = await _backend_with_doc()
    resps = [
        _Resp(_Msg(tool_calls=[_ToolCall("search", {"query": "topic"}, "t1")])),
        _Resp(_Msg(content="done")),
    ]
    res = await run_agent_loop(
        client=_FakeClient(resps), backend=b, query="q", sufficiency_gate=False
    )
    assert res.prompt_tokens == 0
    assert res.completion_tokens == 0


@pytest.mark.asyncio
async def test_usage_counts_sufficiency_judge():
    b = await _backend_with_doc()
    resps = [
        _Resp(_Msg(tool_calls=[_ToolCall("search", {"query": "topic"}, "t1")]), _Usage(100, 20)),
        _Resp(_Msg(content="answer"), _Usage(50, 10)),
    ]
    judge_resp = _Resp(_Msg(content='{"sufficient": true}'), _Usage(30, 5))
    client = _FakeClient(resps, notool_resp=judge_resp)
    res = await run_agent_loop(client=client, backend=b, query="q", sufficiency_gate=True)
    assert client.notool_calls == 1  # judge ran (evidence existed, non-empty candidate)
    assert res.prompt_tokens == 180
    assert res.completion_tokens == 35


@pytest.mark.asyncio
async def test_usage_counts_forced_final_synthesis():
    # every turn is a tool call → loop exhausts → forced no-tools synthesis
    b = await _backend_with_doc()
    resps = [
        _Resp(_Msg(tool_calls=[_ToolCall("search", {"query": "topic"}, f"t{i}")]), _Usage(100, 20))
        for i in range(2)
    ]
    synth_resp = _Resp(_Msg(content="forced answer"), _Usage(70, 15))
    client = _FakeClient(resps, notool_resp=synth_resp)
    res = await run_agent_loop(
        client=client, backend=b, query="q", max_turns=2, sufficiency_gate=False
    )
    assert res.final_answer == "forced answer"
    assert client.notool_calls == 1
    assert res.prompt_tokens == 270
    assert res.completion_tokens == 55
