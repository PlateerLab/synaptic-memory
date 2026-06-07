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

from synaptic.agent_loop import (
    _bridge_is_grounded,
    _parse_sufficiency,
    run_agent_loop,
)
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
        # Messages seen on each agent turn — lets tests assert what the gate
        # injected as the corrective user nudge.
        self.agent_message_log: list[list] = []
        self.chat = self
        self.completions = self

    async def create(self, *, model, messages, tools=None, max_tokens=None):
        if tools is not None:
            self.kinds.append("agent")
            self.agent_message_log.append([dict(m) for m in messages])
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
        "next_query": "",
    }
    # tolerant of surrounding prose
    assert _parse_sufficiency('ok: {"sufficient": true, "gap": ""} done')["sufficient"] is True
    assert _parse_sufficiency("not json at all") is None
    assert _parse_sufficiency('{"foo": 1}') is None  # missing key → None (fail open)
    assert _parse_sufficiency("") is None


def test_parse_sufficiency_next_query():
    # bridge prompt adds next_query — parsed when present, optional otherwise.
    parsed = _parse_sufficiency(
        '{"sufficient": false, "gap": "capital of Y", "next_query": "capital of Korea"}'
    )
    assert parsed == {
        "sufficient": False,
        "gap": "capital of Y",
        "next_query": "capital of Korea",
    }


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
async def test_gate_on_by_default():
    # Promoted to default-on (measured +3.2pp). No arg, no env → gate fires.
    b = await _backend_with_evidence()
    client = _FakeClient(_search_then("answer"), judge_texts=['{"sufficient": true}'])
    res = await run_agent_loop(client=client, backend=b, query="q")
    assert res.final_answer == "answer"
    assert "judge" in client.kinds  # gate ran without being asked
    await b.close()


@pytest.mark.asyncio
async def test_env_var_can_disable_default_gate(monkeypatch):
    # Escape hatch: SYNAPTIC_SUFFICIENCY_GATE=0 turns the default-on gate OFF.
    monkeypatch.setenv("SYNAPTIC_SUFFICIENCY_GATE", "0")
    b = await _backend_with_evidence()
    client = _FakeClient(_search_then("answer"), judge_texts=[])
    res = await run_agent_loop(client=client, backend=b, query="q")
    assert res.final_answer == "answer"
    assert "judge" not in client.kinds  # env disabled it
    await b.close()


@pytest.mark.asyncio
async def test_env_var_enables_gate(monkeypatch):
    monkeypatch.setenv("SYNAPTIC_SUFFICIENCY_GATE", "1")
    b = await _backend_with_evidence()
    client = _FakeClient(_search_then("answer"), judge_texts=['{"sufficient": true}'])
    await run_agent_loop(client=client, backend=b, query="q")  # gate not passed as arg
    assert "judge" in client.kinds  # env turned it on
    await b.close()


# --- L29b: bridge-aware gap injection + grounding ---------------------


def test_bridge_is_grounded():
    ev = "Acme Corp was founded in Berlin. Revenue grew 12%."
    # novel entity present in evidence → grounded
    assert _bridge_is_grounded("where is the HQ", "headquarters of Acme Corp", ev) is True
    assert _bridge_is_grounded("founding city", "Berlin city population", ev) is True
    # hallucinated entity absent from evidence → not grounded
    assert _bridge_is_grounded("where is the HQ", "capital of Korea", ev) is False
    # next_query only restates the question (no novel tokens) → not grounded
    assert _bridge_is_grounded("founding city Berlin", "founding city Berlin", ev) is False
    # empty inputs fail closed
    assert _bridge_is_grounded("q", "", ev) is False
    assert _bridge_is_grounded("q", "Acme", "") is False


@pytest.mark.asyncio
async def test_bridge_relays_grounded_next_query_into_nudge():
    # gate_bridge ON + the proposed bridge entity IS in the evidence ("topic"
    # appears in the test node) → the nudge relays the exact chained query.
    b = await _backend_with_evidence()
    client = _FakeClient(
        _search_then("v1", "v2"),
        judge_texts=[
            '{"sufficient": false, "gap": "the topic detail", "next_query": "topic ownership record"}',
            '{"sufficient": true}',
        ],
    )
    res = await run_agent_loop(
        client=client, backend=b, query="q", sufficiency_gate=True, gate_bridge=True
    )
    assert res.final_answer == "v2"
    # The nudge is injected AFTER turn 1's v1 answer, so it's the last message
    # the agent sees on turn 2 (index 2: turn0 search, turn1 v1, turn2 v2).
    nudge = client.agent_message_log[2][-1]
    assert nudge["role"] == "user"
    assert '"topic ownership record"' in nudge["content"]
    await b.close()


@pytest.mark.asyncio
async def test_bridge_rejects_ungrounded_next_query():
    # gate_bridge ON but the proposed bridge entity is NOT in the evidence
    # (hallucinated) → grounding rejects it → falls back to the generic nudge.
    b = await _backend_with_evidence()
    client = _FakeClient(
        _search_then("v1", "v2"),
        judge_texts=[
            '{"sufficient": false, "gap": "g", "next_query": "capital of Korea"}',
            '{"sufficient": true}',
        ],
    )
    await run_agent_loop(
        client=client, backend=b, query="q", sufficiency_gate=True, gate_bridge=True
    )
    nudge = client.agent_message_log[2][-1]
    assert "capital of Korea" not in nudge["content"]
    assert "use the search tools" in nudge["content"].lower()
    await b.close()


@pytest.mark.asyncio
async def test_plain_gate_does_not_relay_next_query():
    # Without gate_bridge, even a (grounded) next_query in the verdict is ignored
    # — the generic nudge is used (plain judge prompt never asks for next_query).
    b = await _backend_with_evidence()
    client = _FakeClient(
        _search_then("v1", "v2"),
        judge_texts=[
            '{"sufficient": false, "gap": "g", "next_query": "topic ownership record"}',
            '{"sufficient": true}',
        ],
    )
    await run_agent_loop(
        client=client, backend=b, query="q", sufficiency_gate=True, gate_bridge=False
    )
    nudge = client.agent_message_log[2][-1]
    assert "topic ownership record" not in nudge["content"]
    assert "use the search tools" in nudge["content"].lower()
    await b.close()


@pytest.mark.asyncio
async def test_env_var_enables_bridge(monkeypatch):
    monkeypatch.setenv("SYNAPTIC_GATE_BRIDGE", "1")
    b = await _backend_with_evidence()
    client = _FakeClient(
        _search_then("v1", "v2"),
        judge_texts=[
            '{"sufficient": false, "gap": "g", "next_query": "topic registry lookup"}',
            '{"sufficient": true}',
        ],
    )
    await run_agent_loop(client=client, backend=b, query="q")  # bridge not passed as arg
    nudge = client.agent_message_log[2][-1]
    assert "topic registry lookup" in nudge["content"]
    await b.close()
