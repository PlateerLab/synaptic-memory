"""L29 — query-time sufficiency gate in the agent loop (opt-in).

Instead of accepting the first non-tool message, when ``sufficiency_gate`` is
on the loop asks the LLM once whether the answer is supported by the gathered
evidence; on a clear gap (with turns + retry budget left) it injects the gap
and keeps retrieving. Advisory-only: fail-open, bias-to-sufficient, bounded by
``_MAX_GATE_RETRIES``. Default off → the 81% agent baseline is never changed.
"""

from __future__ import annotations

import json

import pytest

from synaptic.agent_loop import _parse_sufficiency, run_agent_loop
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.models import ConsolidationLevel, Node, NodeKind

# --- fake OpenAI-compatible client -------------------------------------


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
    """``chat.completions.create`` pops an agent message when ``tools`` is
    passed (the agent turn) or a judge text when it isn't (the gate call)."""

    def __init__(self, agent_msgs, judge_texts):
        self._agent = list(agent_msgs)
        self._judge = list(judge_texts)
        self.kinds: list[str] = []
        self.chat = self
        self.completions = self

    async def create(self, *, model, messages, tools=None, max_tokens=None):
        if tools is not None:
            self.kinds.append("agent")
            return _Resp(self._agent.pop(0))
        self.kinds.append("judge")
        return _Resp(_Msg(content=self._judge.pop(0)))


async def _backend_with_evidence():
    b = SqliteGraphBackend(":memory:")
    await b.connect()
    await b.save_node(
        Node(
            id="d1",
            kind=NodeKind.CHUNK,
            title="test topic",
            content="test evidence content about the topic",
            level=ConsolidationLevel.L0_RAW,
        )
    )
    return b


def _search_then(*answers):
    # turn 0 runs a search (produces evidence); later turns emit final text.
    return [_Msg(tool_calls=[_ToolCall("search", {"query": "test"})])] + [
        _Msg(content=a) for a in answers
    ]


# --- parser ------------------------------------------------------------


def test_parse_sufficiency():
    assert _parse_sufficiency('{"sufficient": false, "gap": "the year"}') == {
        "sufficient": False,
        "gap": "the year",
    }
    # tolerant of surrounding prose
    assert _parse_sufficiency('ok: {"sufficient": true, "gap": ""} done')["sufficient"] is True
    assert _parse_sufficiency("not json at all") is None
    assert _parse_sufficiency('{"foo": 1}') is None  # missing key → None (fail open)
    assert _parse_sufficiency("") is None


# --- gate behaviour ----------------------------------------------------


@pytest.mark.asyncio
async def test_gate_off_accepts_first_answer():
    b = await _backend_with_evidence()
    client = _FakeClient(_search_then("answer"), judge_texts=[])
    res = await run_agent_loop(client=client, backend=b, query="q", sufficiency_gate=False)
    assert res.final_answer == "answer"
    assert "judge" not in client.kinds  # gate never fires when off
    await b.close()


@pytest.mark.asyncio
async def test_gate_on_retries_then_accepts():
    b = await _backend_with_evidence()
    client = _FakeClient(
        _search_then("v1", "v2"),
        judge_texts=['{"sufficient": false, "gap": "the year"}', '{"sufficient": true}'],
    )
    res = await run_agent_loop(client=client, backend=b, query="q", sufficiency_gate=True)
    # v1 was judged insufficient → loop continued → v2 accepted after a 2nd
    # (sufficient) judgement.
    assert res.final_answer == "v2"
    assert client.kinds == ["agent", "agent", "judge", "agent", "judge"]
    await b.close()


@pytest.mark.asyncio
async def test_gate_fails_open_on_unparseable_judge():
    b = await _backend_with_evidence()
    client = _FakeClient(_search_then("answer"), judge_texts=["garbage not json"])
    res = await run_agent_loop(client=client, backend=b, query="q", sufficiency_gate=True)
    # judge unparseable → verdict None → answer accepted, never blocked.
    assert res.final_answer == "answer"
    await b.close()


@pytest.mark.asyncio
async def test_env_var_enables_gate(monkeypatch):
    monkeypatch.setenv("SYNAPTIC_SUFFICIENCY_GATE", "1")
    b = await _backend_with_evidence()
    client = _FakeClient(_search_then("answer"), judge_texts=['{"sufficient": true}'])
    await run_agent_loop(client=client, backend=b, query="q")  # gate not passed as arg
    assert "judge" in client.kinds  # env turned it on
    await b.close()
