"""MCP edit tool tests: knowledge_update, knowledge_unlink, knowledge_update_edge."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("aiosqlite")
pytest.importorskip("mcp")


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

        yield mcp_server

        if mcp_server._backend is not None:
            await mcp_server._backend.close()
        mcp_server._graph = None
        mcp_server._backend = None


class TestKnowledgeUpdate:
    async def test_partial_field_update(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        added = await m.knowledge_add(title="orig", content="orig content", tags="a,b")
        node_id = added["node_id"]

        result = await m.knowledge_update(node_id=node_id, title="renamed")
        assert result["success"] is True
        assert result["title"] == "renamed"
        # tags untouched
        assert set(result["tags"]) == {"a", "b"}

    async def test_properties_patch_merges(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        added = await m.knowledge_add(title="t", content="c")
        node_id = added["node_id"]

        # seed properties via replace
        await m.knowledge_update(node_id=node_id, properties_replace={"k1": "v1", "k2": "v2"})
        # patch one key
        result = await m.knowledge_update(
            node_id=node_id, properties_patch={"k2": "v2-new", "k3": "v3"}
        )
        assert result["success"] is True
        assert result["properties"] == {"k1": "v1", "k2": "v2-new", "k3": "v3"}

    async def test_properties_both_modes_rejected(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        added = await m.knowledge_add(title="t", content="c")
        result = await m.knowledge_update(
            node_id=added["node_id"],
            properties_patch={"a": "1"},
            properties_replace={"b": "2"},
        )
        assert result["success"] is False

    async def test_missing_node(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        result = await m.knowledge_update(node_id="nonexistent", title="x")
        assert result["success"] is False


class TestKnowledgeUpdateTagModes:
    async def test_tags_add(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        added = await m.knowledge_add(title="t", content="c", tags="a,b")
        node_id = added["node_id"]
        result = await m.knowledge_update(node_id=node_id, tags_add=["c", "a"])
        assert result["success"] is True
        assert set(result["tags"]) == {"a", "b", "c"}

    async def test_tags_remove(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        added = await m.knowledge_add(title="t", content="c", tags="a,b,c")
        node_id = added["node_id"]
        result = await m.knowledge_update(node_id=node_id, tags_remove=["b"])
        assert set(result["tags"]) == {"a", "c"}

    async def test_tags_mode_conflict_rejected(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        added = await m.knowledge_add(title="t", content="c")
        result = await m.knowledge_update(node_id=added["node_id"], tags=["x"], tags_add=["y"])
        assert result["success"] is False


class TestKnowledgeMergeNodes:
    async def test_merges_and_repoints(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        keep = await m.knowledge_add(title="keep", content="x", tags="a")
        drop = await m.knowledge_add(title="drop", content="y", tags="b")
        other = await m.knowledge_add(title="other", content="z")
        await m.knowledge_link(drop["node_id"], other["node_id"], kind="related")

        result = await m.knowledge_merge_nodes(keep_id=keep["node_id"], drop_id=drop["node_id"])
        assert result["success"] is True
        assert set(result["tags"]) == {"a", "b"}

    async def test_missing_id_fails(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        keep = await m.knowledge_add(title="k", content="x")
        result = await m.knowledge_merge_nodes(keep_id=keep["node_id"], drop_id="nonexistent")
        assert result["success"] is False

    async def test_same_id_fails(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        n = await m.knowledge_add(title="t", content="c")
        result = await m.knowledge_merge_nodes(keep_id=n["node_id"], drop_id=n["node_id"])
        assert result["success"] is False


class TestKnowledgeUnlink:
    async def test_unlink_all(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        a = await m.knowledge_add(title="A", content="x")
        b = await m.knowledge_add(title="B", content="y")
        await m.knowledge_link(a["node_id"], b["node_id"], kind="related")
        await m.knowledge_link(a["node_id"], b["node_id"], kind="caused")

        result = await m.knowledge_unlink(a["node_id"], b["node_id"])
        assert result["success"] is True
        assert result["removed"] == 2

    async def test_unlink_by_kind(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        a = await m.knowledge_add(title="A", content="x")
        b = await m.knowledge_add(title="B", content="y")
        await m.knowledge_link(a["node_id"], b["node_id"], kind="related")
        await m.knowledge_link(a["node_id"], b["node_id"], kind="caused")

        result = await m.knowledge_unlink(a["node_id"], b["node_id"], kind="caused")
        assert result["removed"] == 1


class TestKnowledgeUpdateEdge:
    async def test_update_weight(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        a = await m.knowledge_add(title="A", content="x")
        b = await m.knowledge_add(title="B", content="y")
        await m.knowledge_link(a["node_id"], b["node_id"], kind="related", weight=1.0)

        result = await m.knowledge_update_edge(a["node_id"], b["node_id"], new_weight=4.5)
        assert result["success"] is True
        assert result["updated"] == 1

    async def test_update_kind(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        a = await m.knowledge_add(title="A", content="x")
        b = await m.knowledge_add(title="B", content="y")
        await m.knowledge_link(a["node_id"], b["node_id"], kind="related")

        result = await m.knowledge_update_edge(a["node_id"], b["node_id"], new_kind="caused")
        assert result["updated"] == 1

    async def test_noop_returns_failure(self, fresh_mcp_graph):
        m = fresh_mcp_graph
        a = await m.knowledge_add(title="A", content="x")
        b = await m.knowledge_add(title="B", content="y")
        await m.knowledge_link(a["node_id"], b["node_id"])

        result = await m.knowledge_update_edge(a["node_id"], b["node_id"])
        assert result["success"] is False
