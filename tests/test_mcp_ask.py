"""MCP ``knowledge_ask`` tool tests — registration, response fields, no-client behavior.

Follows the ``test_mcp_edit_tools.py`` module-state fixture pattern and
the ``test_graph_ask.py`` scripted-client stub (dispatch by call shape:
``tools=`` → agent turn, "evidence auditor" system → sufficiency judge,
anything else → cheap synthesis). What this file guards:

1. ``knowledge_ask`` is registered on the FastMCP server.
2. No LLM bound (no ``--llm-url``, no injected client) → clear
   ``success=False`` error pointing at ``--llm-url`` — no silent
   search-only degradation.
3. The response carries the routing/cost contract: answer, route,
   route_reasons, escalated, prompt_tokens, completion_tokens, evidence.
4. tier-1 escalation surfaces as ``escalated=True`` + route="agent".
5. ``mode="search"`` never consults the judge or the agent loop.
6. An invalid mode comes back as ``success=False`` (not an exception).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("aiosqlite")
pytest.importorskip("mcp")


# --- fake LLM client (test_graph_ask stub shape) -----------------------


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
        self.chat = self
        self.completions = self

    async def create(self, *, model, messages, tools=None, max_tokens=None, temperature=None):
        if tools is not None:
            self.agent_calls += 1
            return self._agent.pop(0)
        system = str(messages[0].get("content", "")) if messages else ""
        if "evidence auditor" in system:
            self.judge_calls += 1
            return self._judge
        self.synth_calls += 1
        return self._synth


_INSUFFICIENT = _Resp(_Msg(content='{"sufficient": false, "gap": "order counts"}'), _Usage(30, 5))


# --- fixture (test_mcp_edit_tools pattern + LLM state reset) ------------


@pytest.fixture
async def fresh_mcp_graph():
    from synaptic.mcp import server as mcp_server

    with tempfile.TemporaryDirectory() as d:
        mcp_server._graph = None
        mcp_server._backend = None
        mcp_server._embedder = None
        mcp_server._tracker = None
        mcp_server._db_path = str(Path(d) / "graph.db")
        mcp_server._dsn = ""
        mcp_server._source_dsn = ""
        mcp_server._embed_url = ""
        mcp_server._llm_client = None
        mcp_server._llm_url = ""
        mcp_server._llm_model = "gpt-4o-mini"
        mcp_server._llm_api_key = ""

        yield mcp_server

        if mcp_server._backend is not None:
            await mcp_server._backend.close()
        mcp_server._graph = None
        mcp_server._backend = None
        mcp_server._llm_client = None


async def _seed_doc(m) -> None:
    await m.knowledge_add(title="topic", content="evidence about the topic")


# --- registration --------------------------------------------------------


async def test_knowledge_ask_is_registered(fresh_mcp_graph):
    m = fresh_mcp_graph
    tools = await m.server.list_tools()
    assert "knowledge_ask" in {t.name for t in tools}


# --- no LLM bound ---------------------------------------------------------


async def test_no_llm_client_returns_clear_error(fresh_mcp_graph):
    m = fresh_mcp_graph
    result = await m.knowledge_ask(question="anything")
    assert result["success"] is False
    # The error must point the operator at the fix, not degrade silently.
    assert "--llm-url" in result["error"]
    assert "answer" not in result


# --- response field contract ----------------------------------------------


async def test_cheap_path_response_fields(fresh_mcp_graph):
    m = fresh_mcp_graph
    await _seed_doc(m)
    client = _ScriptedClient(
        synth=_Resp(_Msg(content="rag answer"), _Usage(100, 10)),
        judge=_Resp(_Msg(content='{"sufficient": true}'), _Usage(30, 5)),
    )
    m._llm_client = client

    result = await m.knowledge_ask(question="topic", mode="auto")
    assert result["success"] is True
    assert result["answer"] == "rag answer"
    assert result["route"] == "single_shot"
    assert result["escalated"] is False
    assert isinstance(result["route_reasons"], list) and result["route_reasons"]
    # tokens: synthesis (100/10) + tier-1 judge (30/5)
    assert result["prompt_tokens"] == 130
    assert result["completion_tokens"] == 15
    assert client.judge_calls == 1
    assert client.agent_calls == 0
    # evidence is the cheap path's retrieval, in compact dict shape
    assert result["evidence"]
    assert {"id", "kind", "title", "content"} <= set(result["evidence"][0])


async def test_escalation_surfaces_route_and_tokens(fresh_mcp_graph):
    m = fresh_mcp_graph
    await _seed_doc(m)
    client = _ScriptedClient(
        synth=_Resp(_Msg(content="rag answer"), _Usage(100, 10)),
        judge=_INSUFFICIENT,
        agent=[_Resp(_Msg(content="agent answer"), _Usage(200, 20))],
    )
    m._llm_client = client

    result = await m.knowledge_ask(question="topic", mode="auto")
    assert result["success"] is True
    assert result["escalated"] is True
    assert result["route"] == "agent"
    assert result["answer"] == "agent answer"
    assert any("tier-1" in r for r in result["route_reasons"])
    # tokens accumulate across ALL calls: synthesis + judge + agent loop
    assert result["prompt_tokens"] == 330
    assert result["completion_tokens"] == 35
    assert client.agent_calls == 1


async def test_mode_search_never_escalates(fresh_mcp_graph):
    m = fresh_mcp_graph
    await _seed_doc(m)
    # judge scripted to say insufficient — it must never even be consulted
    client = _ScriptedClient(
        synth=_Resp(_Msg(content="rag answer"), _Usage(100, 10)),
        judge=_INSUFFICIENT,
    )
    m._llm_client = client

    result = await m.knowledge_ask(question="topic", mode="search")
    assert result["success"] is True
    assert result["route"] == "single_shot"
    assert result["escalated"] is False
    assert result["answer"] == "rag answer"
    assert client.judge_calls == 0
    assert client.agent_calls == 0


async def test_invalid_mode_returns_message(fresh_mcp_graph):
    m = fresh_mcp_graph
    m._llm_client = _ScriptedClient()
    result = await m.knowledge_ask(question="topic", mode="cheap")
    assert result["success"] is False
    assert "mode" in result["message"]
