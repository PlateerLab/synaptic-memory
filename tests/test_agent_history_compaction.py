"""Agent context-overflow guard — history compaction + reactive retry.

Long (enumeration / deep multi-hop) agent sessions accumulate an unbounded
message history and used to hit the model's context window at turn 12-13, where
the LLM call raised a 400 and the loop BROKE — losing all remaining retrieval.
The guard folds the oldest tool results to a stub before each call (proactive)
and, if a context-length 400 still slips through, compacts harder and retries
once (reactive) instead of dying.
"""

from __future__ import annotations

import json

import pytest

from synaptic.agent_loop import (
    _FOLD_STUB,
    _compact_history,
    _history_chars,
    _is_context_length_error,
    run_agent_loop,
)
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.models import ConsolidationLevel, Node, NodeKind

# --- unit: error detection --------------------------------------------


def test_is_context_length_error():
    assert _is_context_length_error(
        RuntimeError("This model's maximum context length is 32768 tokens")
    )
    assert _is_context_length_error(Exception("prompt contains at least 30721 input_tokens"))
    assert _is_context_length_error(Exception("context_length_exceeded"))
    assert not _is_context_length_error(Exception("connection refused"))
    assert not _is_context_length_error(Exception("rate limit"))


# --- unit: compaction --------------------------------------------------


def _history(n_tools: int, tool_chars: int) -> list:
    msgs = [{"role": "system", "content": "S" * 200}, {"role": "user", "content": "q"}]
    for i in range(n_tools):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"t{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "X" * tool_chars})
    return msgs


def test_compact_history_noop_under_budget():
    msgs = _history(3, 500)
    assert _compact_history(msgs, budget=100_000) == 0
    assert all(_FOLD_STUB not in str(m.get("content")) for m in msgs)


def test_compact_history_folds_oldest_keeps_recent():
    msgs = _history(6, 1000)
    before_ids = [m["tool_call_id"] for m in msgs if m["role"] == "tool"]
    folded = _compact_history(msgs, budget=2500)

    assert folded >= 1
    assert _history_chars(msgs) <= 2500
    # the most recent tool result is preserved full (oldest-first folding)
    assert msgs[-1]["content"] == "X" * 1000
    # the oldest tool result is folded to the stub
    assert msgs[3]["content"] == _FOLD_STUB
    # order + tool_call_id pairing intact (protocol stays valid)
    after_ids = [m["tool_call_id"] for m in msgs if m["role"] == "tool"]
    assert after_ids == before_ids


def test_compact_history_idempotent():
    msgs = _history(6, 1000)
    _compact_history(msgs, budget=2500)
    chars = _history_chars(msgs)
    # running it again changes nothing (already-folded stubs are skipped)
    _compact_history(msgs, budget=2500)
    assert _history_chars(msgs) == chars


# --- integration: reactive retry on overflow --------------------------


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        return {"role": "assistant", "content": self.content or ""}


class _Resp:
    def __init__(self, msg):
        self.choices = [type("Ch", (), {"message": msg})()]


class _Func:
    def __init__(self, name, args):
        self.name = name
        self.arguments = json.dumps(args)


class _ToolCall:
    def __init__(self, name, args, tcid="tc1"):
        self.id = tcid
        self.function = _Func(name, args)


class _OverflowOnceClient:
    """Raises a context-length 400 on the 2nd agent call exactly once; the
    reactive path should compact + retry and then succeed."""

    def __init__(self, agent_msgs):
        self._agent = list(agent_msgs)
        self.agent_calls = 0
        self.raised = False
        self.chat = self
        self.completions = self

    async def create(self, *, model, messages, tools=None, max_tokens=None):
        if tools is None:  # sufficiency judge — keep out of the way
            return _Resp(_Msg(content='{"sufficient": true}'))
        self.agent_calls += 1
        if self.agent_calls == 2 and not self.raised:
            self.raised = True
            raise RuntimeError("This model's maximum context length is 32768 tokens")
        return _Resp(self._agent.pop(0))


async def _backend_with_evidence():
    b = SqliteGraphBackend(":memory:")
    await b.connect()
    await b.save_node(
        Node(
            id="d1",
            kind=NodeKind.CHUNK,
            title="topic",
            content="evidence about the topic " * 300,
            level=ConsolidationLevel.L0_RAW,
        )
    )
    return b


@pytest.mark.asyncio
async def test_reactive_retry_recovers_from_overflow(monkeypatch):
    # Budget chosen so the proactive pass leaves the ~4k-char tool result full
    # (total ≈ 4.2k ≤ 7000) but the reactive pass (budget//2 = 3500) folds it —
    # i.e. the retry only succeeds because the reactive compaction fired.
    monkeypatch.setenv("SYNAPTIC_AGENT_HISTORY_BUDGET", "7000")
    b = await _backend_with_evidence()
    # turn0: search (creates a tool result); turn1: raises once; retry: final.
    agent_msgs = [
        _Msg(tool_calls=[_ToolCall("search", {"query": "topic"})]),
        _Msg(content="final answer after recovery"),
    ]
    client = _OverflowOnceClient(agent_msgs)
    res = await run_agent_loop(
        client=client, backend=b, query="q", sufficiency_gate=False, max_turns=5
    )
    # the loop did NOT break on the 400 — it compacted and retried to a finish.
    assert res.final_answer == "final answer after recovery"
    assert client.raised  # the overflow really happened
    await b.close()
