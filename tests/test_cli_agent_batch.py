"""``synaptic-agent-batch`` CLI — query-file parsing + batch loop.

The agent loop itself (``graph.chat``) is monkeypatched so these tests run
without an LLM endpoint; they pin the CLI's parsing, concurrency, JSONL
output shape, and per-query error capture.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from synaptic.agent_loop import AgentSearchResult
from synaptic.cli import agent_batch


def test_load_queries_formats(tmp_path):
    # 1. plain string list — index becomes the id
    p = tmp_path / "a.json"
    p.write_text(json.dumps(["q1", "q2"]), encoding="utf-8")
    items = agent_batch._load_queries(p)
    assert [i["query"] for i in items] == ["q1", "q2"]
    assert items[0]["id"] == 0

    # 2. object list — explicit id echoed, other keys preserved
    p = tmp_path / "b.json"
    p.write_text(json.dumps([{"id": "x", "query": "qq", "gt": [1]}]), encoding="utf-8")
    items = agent_batch._load_queries(p)
    assert items[0]["id"] == "x" and items[0]["query"] == "qq" and items[0]["gt"] == [1]

    # 3. {"queries": [...]} wrapper + alternate "question" key
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"queries": [{"question": "hi"}]}), encoding="utf-8")
    assert agent_batch._load_queries(p)[0]["query"] == "hi"

    # 4. jsonl — mixed string / object lines
    p = tmp_path / "d.jsonl"
    p.write_text('"line1"\n{"query": "line2"}\n', encoding="utf-8")
    assert [i["query"] for i in agent_batch._load_queries(p)] == ["line1", "line2"]


def test_load_queries_rejects_empty(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps([{"no_query_key": 1}]), encoding="utf-8")
    with pytest.raises(ValueError):
        agent_batch._load_queries(p)


def _make_graph_db(path) -> None:
    async def _build():
        from synaptic.backends.sqlite_graph import SqliteGraphBackend
        from synaptic.graph import SynapticGraph

        backend = SqliteGraphBackend(str(path))
        await backend.connect()
        graph = SynapticGraph(backend)
        await graph.add(title="Doc", content="hello world")
        await backend.close()

    asyncio.run(_build())


def _patch_openai(monkeypatch):
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda *a, **k: object())


def test_batch_run_writes_jsonl(tmp_path, monkeypatch):
    db = tmp_path / "g.sqlite"
    _make_graph_db(db)
    _patch_openai(monkeypatch)

    async def fake_chat(self, query, **kw):
        return AgentSearchResult(
            query=query,
            final_answer=f"answer to {query}",
            found_ids={"doc2", "doc1"},
            turns_used=2,
            tool_calls_made=3,
            elapsed_ms=5.0,
        )

    monkeypatch.setattr("synaptic.graph.SynapticGraph.chat", fake_chat)

    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps(["q one", {"id": "x", "query": "q two"}]), encoding="utf-8")
    out = tmp_path / "out.jsonl"

    rc = agent_batch.main([str(db), "-q", str(qfile), "-o", str(out), "-c", "2"])
    assert rc == 0

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    by_q = {line["query"]: line for line in lines}
    assert by_q["q one"]["answer"] == "answer to q one"
    # found_ids set is serialised sorted
    assert by_q["q one"]["found_ids"] == ["doc1", "doc2"]
    assert by_q["q one"]["turns"] == 2 and by_q["q one"]["tool_calls"] == 3
    assert by_q["q two"]["id"] == "x"
    assert all(line["error"] is None for line in lines)


def test_batch_captures_per_query_errors(tmp_path, monkeypatch):
    db = tmp_path / "g.sqlite"
    _make_graph_db(db)
    _patch_openai(monkeypatch)

    async def boom_chat(self, query, **kw):
        raise RuntimeError("llm exploded")

    monkeypatch.setattr("synaptic.graph.SynapticGraph.chat", boom_chat)

    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps(["only one"]), encoding="utf-8")
    out = tmp_path / "out.jsonl"

    # every query failed → total wipeout → non-zero exit, but output is written
    rc = agent_batch.main([str(db), "-q", str(qfile), "-o", str(out)])
    assert rc == 1
    line = json.loads(out.read_text(encoding="utf-8").strip())
    assert "RuntimeError: llm exploded" in line["error"]
    assert line["query"] == "only one"


def test_missing_files_exit_code(tmp_path):
    # missing graph db
    q = tmp_path / "q.json"
    q.write_text("[]", encoding="utf-8")
    assert agent_batch.main([str(tmp_path / "nope.sqlite"), "-q", str(q)]) == 2
    # missing queries file
    db = tmp_path / "g.sqlite"
    _make_graph_db(db)
    assert agent_batch.main([str(db), "-q", str(tmp_path / "nope.json")]) == 2
