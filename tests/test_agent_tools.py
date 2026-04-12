"""End-to-end tests for agent_tools — the LLM-facing tool layer.

Every test drives the tool against a small in-memory graph seeded by
hand. The goal is to prove:

- Tools return the documented payload shape.
- Session state (seen nodes, budget) is updated on every call.
- Budget exhaustion triggers the short-circuit.
- Filters work (category, kind, year).
- Error paths return ok=False with a stable error code.
"""

from __future__ import annotations

import pytest

from synaptic.agent_tools import (
    ToolResult,
    count_tool,
    expand_tool,
    follow_tool,
    get_document_tool,
    list_categories_tool,
    search_exact_tool,
    search_tool,
)
from synaptic.backends.memory import MemoryBackend
from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
)
from synaptic.search_session import SearchSession

# --- Shared fixture ---
#
# Two categories, two documents per category, two chunks per document.
# Chunk content is deliberately keyword-rich so FTS has something to
# match. One chunk embeds "E217" so the exact-match tool has a target.


async def _seed_graph(backend: MemoryBackend) -> None:
    def _mk(
        id_: str,
        kind: NodeKind,
        title: str,
        content: str,
        *,
        tags: list[str],
        category: str = "",
        doc_id: str = "",
        year: str = "",
        chunk_index: str = "",
    ) -> Node:
        props: dict[str, str] = {}
        if category:
            props["category"] = category
        if doc_id:
            props["doc_id"] = doc_id
        if year:
            props["year"] = year
        if chunk_index:
            props["chunk_index"] = chunk_index
        return Node(
            id=id_,
            kind=kind,
            title=title,
            content=content,
            tags=tags,
            properties=props,
            level=ConsolidationLevel.L0_RAW,
        )

    # Categories
    await backend.save_node(_mk("cat_rule", NodeKind.CONCEPT, "규정 및 지침",
                                "규정 및 지침", tags=["category"]))
    await backend.save_node(_mk("cat_ops", NodeKind.CONCEPT, "운영계획",
                                "운영계획", tags=["category"]))

    # Rule documents
    await backend.save_node(_mk("doc_r1", NodeKind.ENTITY, "규정 문서 A",
                                "규정 준수 의무",
                                tags=["document"], category="규정 및 지침",
                                doc_id="doc_r1", year="2024"))
    await backend.save_node(_mk("doc_r2", NodeKind.ENTITY, "규정 문서 B",
                                "규정 예외 조항",
                                tags=["document"], category="규정 및 지침",
                                doc_id="doc_r2", year="2023"))

    # Ops documents
    await backend.save_node(_mk("doc_o1", NodeKind.ENTITY, "운영 문서 A",
                                "경마 운영계획",
                                tags=["document"], category="운영계획",
                                doc_id="doc_o1", year="2024"))

    # Chunks
    await backend.save_node(_mk("chunk_r1a", NodeKind.CHUNK, "규정 문서 A #0",
                                "규정 준수 의무 사항 E217 코드가 적용된다",
                                tags=["chunk"], category="규정 및 지침",
                                doc_id="doc_r1", chunk_index="0"))
    await backend.save_node(_mk("chunk_r1b", NodeKind.CHUNK, "규정 문서 A #1",
                                "규정 위반 시 제재 조치",
                                tags=["chunk"], category="규정 및 지침",
                                doc_id="doc_r1", chunk_index="1"))
    await backend.save_node(_mk("chunk_r2a", NodeKind.CHUNK, "규정 문서 B #0",
                                "규정 예외 적용 기준 해설",
                                tags=["chunk"], category="규정 및 지침",
                                doc_id="doc_r2", chunk_index="0"))
    await backend.save_node(_mk("chunk_o1a", NodeKind.CHUNK, "운영 문서 A #0",
                                "경마 운영계획 수립 절차",
                                tags=["chunk"], category="운영계획",
                                doc_id="doc_o1", chunk_index="0"))

    # Edges
    async def _edge(eid: str, src: str, dst: str, kind: EdgeKind):
        await backend.save_edge(
            Edge(id=eid, source_id=src, target_id=dst, kind=kind, weight=1.0)
        )

    await _edge("po_r1", "doc_r1", "cat_rule", EdgeKind.PART_OF)
    await _edge("po_r2", "doc_r2", "cat_rule", EdgeKind.PART_OF)
    await _edge("po_o1", "doc_o1", "cat_ops", EdgeKind.PART_OF)

    await _edge("co_r1a", "doc_r1", "chunk_r1a", EdgeKind.CONTAINS)
    await _edge("co_r1b", "doc_r1", "chunk_r1b", EdgeKind.CONTAINS)
    await _edge("co_r2a", "doc_r2", "chunk_r2a", EdgeKind.CONTAINS)
    await _edge("co_o1a", "doc_o1", "chunk_o1a", EdgeKind.CONTAINS)

    await _edge("nx_r1", "chunk_r1a", "chunk_r1b", EdgeKind.NEXT_CHUNK)


async def _fresh_backend() -> MemoryBackend:
    backend = MemoryBackend()
    await backend.connect()
    await _seed_graph(backend)
    return backend


# --- search_tool ---


@pytest.mark.asyncio
class TestSearchTool:
    async def test_search_returns_evidence(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await search_tool(backend, session, "규정 준수")
        assert result.tool == "search"
        assert result.ok is True
        assert "evidence" in result.data
        assert "anchors" in result.data
        assert session.tool_calls_used == 1

    async def test_search_records_query(self):
        backend = await _fresh_backend()
        session = SearchSession()
        await search_tool(backend, session, "규정 준수")
        assert "규정 준수" in session.queries_tried

    async def test_search_marks_seen(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await search_tool(backend, session, "규정 준수")
        for item in result.data["evidence"]:
            assert session.has_seen(item["id"])

    async def test_search_excludes_seen_on_second_call(self):
        backend = await _fresh_backend()
        session = SearchSession()
        first = await search_tool(backend, session, "규정")
        first_ids = {e["id"] for e in first.data["evidence"]}
        second = await search_tool(backend, session, "규정", exclude_seen=True)
        second_ids = {e["id"] for e in second.data["evidence"]}
        # No overlap between first and second pass
        assert first_ids.isdisjoint(second_ids)

    async def test_search_budget_enforcement(self):
        backend = await _fresh_backend()
        session = SearchSession(budget_tool_calls=1)
        result1 = await search_tool(backend, session, "규정")
        assert result1.ok is True
        result2 = await search_tool(backend, session, "운영")
        assert result2.ok is False
        assert result2.error == "budget_exceeded"

    async def test_search_category_filter(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await search_tool(backend, session, "규정 준수", category="규정 및 지침")
        for item in result.data["evidence"]:
            assert "규정" in item["category"]

    async def test_search_hints_on_empty(self):
        backend = await _fresh_backend()
        session = SearchSession()
        # Query with zero hits
        result = await search_tool(backend, session, "nonexistent-xyz-query")
        assert result.ok is True
        assert result.data["evidence"] == []
        assert len(result.hints) > 0


# --- expand_tool ---


@pytest.mark.asyncio
class TestExpandTool:
    async def test_expand_returns_neighbours(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await expand_tool(backend, session, "doc_r1")
        assert result.ok is True
        assert result.data["seed"]["id"] == "doc_r1"
        # Should have pulled chunks r1a / r1b
        neighbour_ids = {n["id"] for n in result.data["neighbours"]}
        assert "chunk_r1a" in neighbour_ids or "chunk_r1b" in neighbour_ids

    async def test_expand_unknown_node_returns_error(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await expand_tool(backend, session, "nonexistent")
        assert result.ok is False
        assert "node_not_found" in (result.error or "")

    async def test_expand_budget_enforcement(self):
        backend = await _fresh_backend()
        session = SearchSession(budget_tool_calls=0)
        result = await expand_tool(backend, session, "doc_r1")
        assert result.ok is False
        assert result.error == "budget_exceeded"


# --- get_document_tool ---


@pytest.mark.asyncio
class TestGetDocumentTool:
    async def test_get_document_by_doc_id(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await get_document_tool(backend, session, "doc_r1")
        assert result.ok is True
        assert result.data["chunk_count"] == 2
        # Chunks should be in index order
        indices = [c["index"] for c in result.data["chunks"]]
        assert indices == ["0", "1"]

    async def test_get_document_not_found(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await get_document_tool(backend, session, "nonexistent")
        assert result.ok is False
        assert "document_not_found" in (result.error or "")

    async def test_get_document_marks_chunks_seen(self):
        backend = await _fresh_backend()
        session = SearchSession()
        await get_document_tool(backend, session, "doc_r1")
        assert session.has_seen("chunk_r1a")
        assert session.has_seen("chunk_r1b")


# --- list_categories_tool ---


@pytest.mark.asyncio
class TestListCategoriesTool:
    async def test_lists_both_categories(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await list_categories_tool(backend, session)
        assert result.ok is True
        labels = {c["label"] for c in result.data["categories"]}
        assert "규정 및 지침" in labels
        assert "운영계획" in labels

    async def test_categories_have_document_counts(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await list_categories_tool(backend, session)
        rule_cat = next(
            c for c in result.data["categories"]
            if c["label"] == "규정 및 지침"
        )
        # Two rule documents in the fixture
        assert rule_cat["document_count"] == 2


# --- count_tool ---


@pytest.mark.asyncio
class TestCountTool:
    async def test_count_all_chunks(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await count_tool(backend, session, kind=NodeKind.CHUNK)
        assert result.ok is True
        # Four chunks total in the fixture
        assert result.data["count"] == 4

    async def test_count_by_category(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await count_tool(
            backend, session, kind=NodeKind.CHUNK, category="규정 및 지침"
        )
        # Three rule chunks
        assert result.data["count"] == 3

    async def test_count_by_year(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await count_tool(
            backend, session, kind=NodeKind.ENTITY, year=2024
        )
        # Two 2024 documents (doc_r1, doc_o1)
        assert result.data["count"] == 2


# --- search_exact_tool ---


@pytest.mark.asyncio
class TestSearchExactTool:
    async def test_finds_identifier(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await search_exact_tool(backend, session, "E217")
        assert result.ok is True
        assert result.data["count"] >= 1
        # chunk_r1a has "E217" verbatim
        ids = {m["id"] for m in result.data["matches"]}
        assert "chunk_r1a" in ids

    async def test_empty_identifier_errors(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await search_exact_tool(backend, session, "")
        assert result.ok is False
        assert result.error == "empty_identifier"

    async def test_no_match_returns_empty_list(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await search_exact_tool(backend, session, "NONEXISTENT-ID-999")
        assert result.ok is True
        assert result.data["count"] == 0


# --- follow_tool ---


@pytest.mark.asyncio
class TestFollowTool:
    async def test_follow_contains_returns_chunks(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await follow_tool(backend, session, "doc_r1", "contains")
        assert result.ok is True
        ids = {n["id"] for n in result.data["neighbours"]}
        assert {"chunk_r1a", "chunk_r1b"}.issubset(ids)

    async def test_follow_part_of_returns_category(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await follow_tool(backend, session, "doc_r1", "part_of")
        assert result.ok is True
        ids = {n["id"] for n in result.data["neighbours"]}
        assert "cat_rule" in ids

    async def test_unknown_edge_kind_errors(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await follow_tool(backend, session, "doc_r1", "bogus")
        assert result.ok is False
        assert "unknown_edge_kind" in (result.error or "")


# --- ToolResult shape ---


class TestToolResultShape:
    def test_to_dict_has_all_fields(self):
        result = ToolResult(tool="t", ok=True, data={"x": 1})
        d = result.to_dict()
        assert d["tool"] == "t"
        assert d["ok"] is True
        assert d["data"] == {"x": 1}
        assert d["hints"] == []
        assert d["session"] == {}
        assert d["error"] is None
