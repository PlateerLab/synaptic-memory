"""Wiring tests for ``SynapticGraph.ask()`` — routing, escalation, tokens.

Mock-client tests (no real LLM): the fake client dispatches on call shape —
``tools=`` means an agent-loop turn, an "evidence auditor" system prompt
means the tier-1 sufficiency judge, anything else is the cheap-path
synthesis call. What this file guards:

1. ``mode="search"`` / ``mode="agent"`` force their path, no routing.
2. auto + sufficient judge → no escalation, ``escalated=False``.
3. auto + insufficient judge → agent escalation, ``escalated=True``.
4. token totals sum EVERY call (synthesis + judge + agent loop).
5. the answer is never empty when any path produced text (an escalated
   agent run that comes back blank falls back to the cheap synthesis).
6. tier-0 structured lexis × table nodes skips the cheap path entirely.
"""

from __future__ import annotations

import json

import pytest

from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.graph import SynapticGraph
from synaptic.models import ConsolidationLevel, Node, NodeKind

# --- fake client (test_agent_token_usage stub shape + dispatch) -------

_OPEN_GRAPHS: list[SynapticGraph] = []


@pytest.fixture(autouse=True)
async def _close_graphs():
    yield
    while _OPEN_GRAPHS:
        await _OPEN_GRAPHS.pop().close()


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


class _ScriptedClient:
    """Dispatches by call shape: ``tools=`` → agent turn (popped in order),
    "evidence auditor" system → sufficiency judge, otherwise → cheap
    synthesis. Counts each kind so tests can assert which paths ran."""

    def __init__(self, *, synth=None, judge=None, agent=None):
        self._synth = synth or _Resp(_Msg(content="cheap answer"))
        self._judge = judge or _Resp(_Msg(content='{"sufficient": true}'))
        self._agent = list(agent or [])
        self.synth_calls = 0
        self.judge_calls = 0
        self.agent_calls = 0
        self.judge_temps: list = []
        self.synth_messages: list = []
        self.judge_messages: list = []
        self.agent_messages: list = []
        self.chat = self
        self.completions = self

    async def create(self, *, model, messages, tools=None, max_tokens=None, temperature=None):
        if tools is not None:
            self.agent_calls += 1
            self.agent_messages.append(messages)
            return self._agent.pop(0)
        system = str(messages[0].get("content", "")) if messages else ""
        if "evidence auditor" in system:
            self.judge_calls += 1
            self.judge_temps.append(temperature)
            self.judge_messages.append(messages)
            return self._judge
        self.synth_calls += 1
        self.synth_messages.append(messages)
        return self._synth


_INSUFFICIENT = _Resp(_Msg(content='{"sufficient": false, "gap": "order counts"}'), _Usage(30, 5))


async def _make_graph(*, with_table=False) -> SynapticGraph:
    backend = SqliteGraphBackend(":memory:")
    g = SynapticGraph(backend)
    await g.connect()
    _OPEN_GRAPHS.append(g)
    await backend.save_node(
        Node(
            id="d1",
            kind=NodeKind.CHUNK,
            title="topic",
            content="evidence about the topic",
            level=ConsolidationLevel.L0_RAW,
        )
    )
    if with_table:
        await backend.save_node(
            Node(
                id="pr_goods_base:G1",
                kind=NodeKind.ENTITY,
                title="goods G1",
                content="a table row",
                properties={"_table_name": "pr_goods_base"},
                level=ConsolidationLevel.L0_RAW,
            )
        )
    return g


# --- mode forcing ------------------------------------------------------


@pytest.mark.asyncio
async def test_mode_search_forces_cheap_and_never_escalates():
    g = await _make_graph()
    # judge scripted to say insufficient — it must never even be consulted
    client = _ScriptedClient(
        synth=_Resp(_Msg(content="rag answer"), _Usage(100, 10)), judge=_INSUFFICIENT
    )
    res = await g.ask("topic", llm_client=client, mode="search")
    assert res.route == "single_shot"
    assert res.escalated is False
    assert res.answer == "rag answer"
    assert client.judge_calls == 0
    assert client.agent_calls == 0
    assert res.prompt_tokens == 100
    assert res.completion_tokens == 10


@pytest.mark.asyncio
async def test_cheap_prompt_does_not_include_raw_openie_provenance():
    g = await _make_graph()
    await g.update(
        "d1",
        properties={
            "source": "manual.pdf",
            "page": "7",
            "source_event_id": "evt_raw_should_not_enter_prompt",
            "source_chunk_id": "chunk_raw_should_not_enter_prompt",
            "prompt_version": "openie-v1",
            "extractor": "LLMOpenIEExtractor",
            "model": "deepseek-v4-flash",
            "is_openie": "true",
        },
    )
    client = _ScriptedClient(synth=_Resp(_Msg(content="rag answer"), _Usage(100, 10)))

    res = await g.ask("topic", llm_client=client, mode="search")

    assert res.answer == "rag answer"
    prompt = json.dumps(client.synth_messages, ensure_ascii=False)
    assert "evidence about the topic" in prompt
    assert "source_event_id" not in prompt
    assert "source_chunk_id" not in prompt
    assert "prompt_version" not in prompt
    assert "deepseek-v4-flash" not in prompt


@pytest.mark.asyncio
async def test_mode_agent_forces_loop_and_skips_cheap():
    g = await _make_graph()
    client = _ScriptedClient(agent=[_Resp(_Msg(content="agent answer"), _Usage(200, 20))])
    res = await g.ask("topic", llm_client=client, mode="agent")
    assert res.route == "agent"
    assert res.escalated is False
    assert res.answer == "agent answer"
    assert client.synth_calls == 0
    assert res.prompt_tokens == 200
    assert res.completion_tokens == 20


@pytest.mark.asyncio
async def test_invalid_mode_raises():
    g = await _make_graph()
    with pytest.raises(ValueError):
        await g.ask("topic", llm_client=_ScriptedClient(), mode="cheap")


# --- tier-1 gate -------------------------------------------------------


@pytest.mark.asyncio
async def test_cheap_sufficient_does_not_escalate():
    g = await _make_graph()
    client = _ScriptedClient(
        synth=_Resp(_Msg(content="rag answer"), _Usage(100, 10)),
        judge=_Resp(_Msg(content='{"sufficient": true}'), _Usage(30, 5)),
    )
    res = await g.ask("topic", llm_client=client, mode="auto")
    assert res.route == "single_shot"
    assert res.escalated is False
    assert res.answer == "rag answer"
    assert client.judge_calls == 1
    assert client.agent_calls == 0
    # judge must run deterministically (temperature 0)
    assert client.judge_temps == [0.0]
    # tokens: synthesis + judge
    assert res.prompt_tokens == 130
    assert res.completion_tokens == 15
    # cheap path carries its retrieval as evidence
    assert len(res.evidence) >= 1


@pytest.mark.asyncio
async def test_cheap_insufficient_escalates_to_agent():
    g = await _make_graph()
    client = _ScriptedClient(
        synth=_Resp(_Msg(content="rag answer"), _Usage(100, 10)),
        judge=_INSUFFICIENT,
        agent=[_Resp(_Msg(content="agent answer"), _Usage(200, 20))],
    )
    res = await g.ask("topic", llm_client=client, mode="auto")
    assert res.escalated is True
    assert res.route == "agent"
    assert res.answer == "agent answer"
    assert client.judge_calls == 1
    assert client.agent_calls == 1
    assert any("tier-1" in r for r in res.route_reasons)
    # tokens: synthesis (100/10) + judge (30/5) + agent loop (200/20)
    assert res.prompt_tokens == 330
    assert res.completion_tokens == 35


@pytest.mark.asyncio
async def test_escalated_empty_agent_answer_falls_back_to_cheap():
    # Non-empty guarantee: a blank agent run must not erase the cheap answer.
    g = await _make_graph()
    client = _ScriptedClient(
        synth=_Resp(_Msg(content="rag answer"), _Usage(100, 10)),
        judge=_INSUFFICIENT,
        agent=[_Resp(_Msg(content=""), _Usage(200, 20))],
    )
    res = await g.ask("topic", llm_client=client, mode="auto")
    assert res.escalated is True
    assert res.route == "single_shot"  # the path that produced the answer
    assert res.answer == "rag answer"
    assert res.answer.strip()
    # the failed escalation still cost tokens — they must be counted
    assert res.prompt_tokens == 330
    assert res.completion_tokens == 35


# --- tier-0 routing wired through ask() --------------------------------


@pytest.mark.asyncio
async def test_tier0_structured_query_goes_straight_to_agent():
    g = await _make_graph(with_table=True)
    client = _ScriptedClient(agent=[_Resp(_Msg(content="42개"), _Usage(200, 20))])
    res = await g.ask("색상별 상품 변형 개수", llm_client=client, mode="auto")
    assert res.route == "agent"
    assert res.escalated is False  # tier-0 direct routing is not an escalation
    assert res.answer == "42개"
    assert client.synth_calls == 0
    assert client.judge_calls == 0
    assert any("structured-operation lexis" in r for r in res.route_reasons)


@pytest.mark.asyncio
async def test_tier0_structured_lexis_without_tables_stays_cheap():
    # Same structured phrasing over a docs-only corpus → cheap path + tier-1.
    g = await _make_graph(with_table=False)
    client = _ScriptedClient(
        synth=_Resp(_Msg(content="rag answer"), _Usage(100, 10)),
        judge=_Resp(_Msg(content='{"sufficient": true}'), _Usage(30, 5)),
    )
    res = await g.ask("색상별 상품 변형 개수", llm_client=client, mode="auto")
    assert res.route == "single_shot"
    assert client.agent_calls == 0
