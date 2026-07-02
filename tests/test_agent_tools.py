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

import synaptic.agent_tools_v2 as tools_v2
from synaptic.agent_tools import (
    ToolResult,
    _query_rewrite_hints,
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
    await backend.save_node(
        _mk("cat_rule", NodeKind.CONCEPT, "규정 및 지침", "규정 및 지침", tags=["category"])
    )
    await backend.save_node(
        _mk("cat_ops", NodeKind.CONCEPT, "운영계획", "운영계획", tags=["category"])
    )

    # Rule documents
    await backend.save_node(
        _mk(
            "doc_r1",
            NodeKind.ENTITY,
            "규정 문서 A",
            "규정 준수 의무",
            tags=["document"],
            category="규정 및 지침",
            doc_id="doc_r1",
            year="2024",
        )
    )
    await backend.save_node(
        _mk(
            "doc_r2",
            NodeKind.ENTITY,
            "규정 문서 B",
            "규정 예외 조항",
            tags=["document"],
            category="규정 및 지침",
            doc_id="doc_r2",
            year="2023",
        )
    )

    # Ops documents
    await backend.save_node(
        _mk(
            "doc_o1",
            NodeKind.ENTITY,
            "운영 문서 A",
            "경마 운영계획",
            tags=["document"],
            category="운영계획",
            doc_id="doc_o1",
            year="2024",
        )
    )

    # Chunks
    await backend.save_node(
        _mk(
            "chunk_r1a",
            NodeKind.CHUNK,
            "규정 문서 A #0",
            "규정 준수 의무 사항 E217 코드가 적용된다",
            tags=["chunk"],
            category="규정 및 지침",
            doc_id="doc_r1",
            chunk_index="0",
        )
    )
    await backend.save_node(
        _mk(
            "chunk_r1b",
            NodeKind.CHUNK,
            "규정 문서 A #1",
            "규정 위반 시 제재 조치",
            tags=["chunk"],
            category="규정 및 지침",
            doc_id="doc_r1",
            chunk_index="1",
        )
    )
    await backend.save_node(
        _mk(
            "chunk_r2a",
            NodeKind.CHUNK,
            "규정 문서 B #0",
            "규정 예외 적용 기준 해설",
            tags=["chunk"],
            category="규정 및 지침",
            doc_id="doc_r2",
            chunk_index="0",
        )
    )
    await backend.save_node(
        _mk(
            "chunk_o1a",
            NodeKind.CHUNK,
            "운영 문서 A #0",
            "경마 운영계획 수립 절차",
            tags=["chunk"],
            category="운영계획",
            doc_id="doc_o1",
            chunk_index="0",
        )
    )

    # Edges
    async def _edge(eid: str, src: str, dst: str, kind: EdgeKind):
        await backend.save_edge(Edge(id=eid, source_id=src, target_id=dst, kind=kind, weight=1.0))

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


def test_query_rewrite_hints_drop_numeric_year():
    hints = _query_rewrite_hints("child psychiatrist salary 2016")

    assert hints[0].action == "search"
    assert hints[0].args == {"query": "child psychiatrist salary", "limit": 20}


def test_query_rewrite_hints_process_from_question():
    hints = _query_rewrite_hints("how is soil created from rocks")
    queries = [h.args["query"] for h in hints]

    assert "making soil rock pieces" in queries
    assert "small pieces of rock form soil" in queries


def test_query_rewrite_hints_strip_trailing_process_words():
    hints = _query_rewrite_hints("how is soil created from rocks weathering process")
    queries = [h.args["query"] for h in hints]

    assert "making soil rock pieces" in queries
    assert "making soil rocks weathering process pieces" not in queries


def test_query_rewrite_hints_preserve_non_plural_ss_source():
    hints = _query_rewrite_hints("how is policy created from class")
    queries = [h.args["query"] for h in hints]

    assert "making policy class pieces" in queries
    assert "making policy clas pieces" not in queries


def test_query_rewrite_hints_blood_sexual_infection_terms():
    hints = _query_rewrite_hints("blood diseases that are sexually transmitted")
    queries = [h.args["query"] for h in hints]

    assert "sexual blood borne transmission routes" in queries


@pytest.mark.parametrize(
    "query",
    [
        "bloodborne infection sexual transmission",
        "STI blood transmission",
        "sexually transmitted blood infection",
    ],
)
def test_query_rewrite_hints_blood_sexual_infection_variants(query):
    hints = _query_rewrite_hints(query)
    queries = [h.args["query"] for h in hints]

    assert "sexual blood borne transmission routes" in queries


def test_query_rewrite_hints_blood_sexual_requires_disease_terms():
    hints = _query_rewrite_hints("blood pressure changes during sexual activity")
    queries = [h.args["query"] for h in hints]

    assert "sexual blood borne transmission routes" not in queries


def test_query_rewrite_hints_blood_sexual_avoids_non_transmission_context():
    hints = _query_rewrite_hints("blood pressure medication sexual dysfunction infection risk")
    queries = [h.args["query"] for h in hints]

    assert "sexual blood borne transmission routes" not in queries


@pytest.mark.parametrize(
    "query",
    [
        "blood pressure and sexually transmitted infection risk",
        "blood test for sexually transmitted diseases",
        "blood sugar and STD infection symptoms",
    ],
)
def test_query_rewrite_hints_blood_sexual_avoids_blood_measure_context(query):
    hints = _query_rewrite_hints(query)
    queries = [h.args["query"] for h in hints]

    assert "sexual blood borne transmission routes" not in queries


def test_query_rewrite_hints_fiber_serving_size_terms():
    hints = _query_rewrite_hints("how much fiber is in carrots")
    queries = [h.args["query"] for h in hints]

    assert "one cup carrots grams fiber" in queries
    assert "one cup cooked carrots grams fiber" in queries


@pytest.mark.parametrize(
    "query",
    [
        "fiber content in carrots",
        "fiber content in carrots grams",
    ],
)
def test_query_rewrite_hints_fiber_content_terms(query):
    hints = _query_rewrite_hints(query)
    queries = [h.args["query"] for h in hints]

    assert "one cup carrots grams fiber" in queries
    assert "one cup cooked carrots grams fiber" in queries


def test_query_rewrite_hints_tire_gas_mileage_terms():
    hints = _query_rewrite_hints("do bigger tires affect gas mileage")
    queries = [h.args["query"] for h in hints]

    assert "tire size factors influence gas mileage" in queries
    assert "tire width versus gas mileage" in queries


def test_query_rewrite_hints_tire_gas_mileage_requires_tire_terms():
    hints = _query_rewrite_hints("does driving fast affect gas mileage")
    queries = [h.args["query"] for h in hints]

    assert "tire size factors influence gas mileage" not in queries


def test_query_rewrite_hints_tire_gas_mileage_requires_size_context():
    hints = _query_rewrite_hints("does driving fast affect gas mileage when you have winter tires")
    queries = [h.args["query"] for h in hints]

    assert "tire size factors influence gas mileage" not in queries


def test_query_rewrite_hints_bicycle_tube_size_terms():
    hints = _query_rewrite_hints("how bicycle tire tubes are sized")
    queries = [h.args["query"] for h in hints]

    assert "bicycle tire tube size sidewall ETRTO metric imperial" in queries
    assert "bicycle tire sidewall tube size printed raised numbers" in queries


def test_query_rewrite_hints_bicycle_tube_size_requires_tube_terms():
    hints = _query_rewrite_hints("how bicycle tires are sized")
    queries = [h.args["query"] for h in hints]

    assert "bicycle tire tube size sidewall ETRTO metric imperial" not in queries


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
        if second.data.get("all_previously_seen"):
            # Tiny corpus exhausted in one page — the seen-fallback returns the
            # known hits rather than a blank dead-end (better for the agent).
            assert len(second.data["evidence"]) > 0
        else:
            # Pagination: genuinely-new results must not overlap the first page.
            second_ids = {e["id"] for e in second.data["evidence"]}
            assert first_ids.isdisjoint(second_ids)

    async def test_search_seen_fallback_returns_hits_not_blank(self):
        # When every hit was already seen, the seen-filter would empty the
        # result and dead-end the turn. Instead return the hits anyway, flagged.
        backend = await _fresh_backend()
        session = SearchSession()
        all_nodes = await backend.list_nodes(kind=None, limit=1000)
        session.mark_seen([n.id for n in all_nodes])  # everything is now "seen"
        result = await search_tool(backend, session, "규정 준수", exclude_seen=True)
        assert result.data.get("all_previously_seen") is True
        assert len(result.data["evidence"]) > 0  # not a blank dead-end

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


# --- deep_search_tool ---


@pytest.mark.asyncio
async def test_deep_search_defaults_to_wider_evidence_pool(monkeypatch):
    captured_limits: list[int] = []

    async def fake_search_tool(
        backend,
        session,
        query,
        *,
        limit,
        category=None,
        embedder=None,
        **kwargs,
    ):
        captured_limits.append(limit)
        return ToolResult(
            tool="search",
            ok=True,
            data={"evidence": [], "anchors": {}},
            session=session.summary(),
        )

    monkeypatch.setattr(tools_v2, "search_tool", fake_search_tool)
    backend = MemoryBackend()
    await backend.connect()

    result = await tools_v2.deep_search_tool(backend, SearchSession(), "broad question")

    assert result.ok is True
    assert captured_limits == [10]


@pytest.mark.asyncio
async def test_deep_search_caps_evidence_pool(monkeypatch):
    captured_limits: list[int] = []

    async def fake_search_tool(
        backend,
        session,
        query,
        *,
        limit,
        category=None,
        embedder=None,
        **kwargs,
    ):
        captured_limits.append(limit)
        return ToolResult(
            tool="search",
            ok=True,
            data={"evidence": [], "anchors": {}},
            session=session.summary(),
        )

    monkeypatch.setattr(tools_v2, "search_tool", fake_search_tool)
    backend = MemoryBackend()
    await backend.connect()

    result = await tools_v2.deep_search_tool(
        backend,
        SearchSession(),
        "broad question",
        limit=99,
        read_top_k="invalid",
    )

    assert result.ok is True
    assert captured_limits == [20]


@pytest.mark.asyncio
async def test_deep_search_caps_document_reads(monkeypatch):
    document_ids: list[str] = []

    async def fake_search_tool(
        backend,
        session,
        query,
        *,
        limit,
        category=None,
        embedder=None,
        **kwargs,
    ):
        return ToolResult(
            tool="search",
            ok=True,
            data={
                "evidence": [
                    {"id": f"chunk_{idx}", "document_id": f"doc_{idx}"} for idx in range(6)
                ],
                "anchors": {},
            },
            session=session.summary(),
        )

    async def fake_expand(backend, session, node_id):
        return ToolResult(
            tool="expand",
            ok=True,
            data={"seed": {"id": node_id}, "neighbours": []},
            session=session.summary(),
        )

    async def fake_get_doc(backend, session, doc_id, query):
        document_ids.append(doc_id)
        return ToolResult(
            tool="get_document",
            ok=True,
            data={"document": {"id": doc_id}, "chunks": [], "chunk_count": 0},
            session=session.summary(),
        )

    monkeypatch.setattr(tools_v2, "search_tool", fake_search_tool)
    monkeypatch.setattr(tools_v2, "_safe_expand", fake_expand)
    monkeypatch.setattr(tools_v2, "_safe_get_doc", fake_get_doc)
    backend = MemoryBackend()
    await backend.connect()

    result = await tools_v2.deep_search_tool(
        backend,
        SearchSession(),
        "broad question",
        read_top_k=99,
    )

    assert result.ok is True
    assert document_ids == ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]


@pytest.mark.asyncio
async def test_deep_search_surfaces_query_rewrite_hints(monkeypatch):
    async def fake_search_tool(
        backend,
        session,
        query,
        *,
        limit,
        category=None,
        embedder=None,
        **kwargs,
    ):
        return ToolResult(
            tool="search",
            ok=True,
            data={
                "evidence": [{"id": "chunk_0", "document_id": "doc_0"}],
                "anchors": {},
            },
            session=session.summary(),
        )

    async def fake_expand(backend, session, node_id):
        return ToolResult(tool="expand", ok=True, data={"neighbours": []})

    async def fake_get_doc(backend, session, doc_id, query):
        return ToolResult(
            tool="get_document",
            ok=True,
            data={"document": {"id": doc_id}, "chunks": [], "chunk_count": 0},
        )

    monkeypatch.setattr(tools_v2, "search_tool", fake_search_tool)
    monkeypatch.setattr(tools_v2, "_safe_expand", fake_expand)
    monkeypatch.setattr(tools_v2, "_safe_get_doc", fake_get_doc)
    backend = MemoryBackend()
    await backend.connect()

    result = await tools_v2.deep_search_tool(
        backend,
        SearchSession(),
        "how is soil created from rocks",
    )
    queries = [h.args["query"] for h in result.hints]

    assert "making soil rock pieces" in queries
    assert "small pieces of rock form soil" in queries


@pytest.mark.asyncio
async def test_deep_search_runs_query_rewrite_hints(monkeypatch):
    seen_queries: list[str] = []

    async def fake_search_tool(
        backend,
        session,
        query,
        *,
        limit,
        category=None,
        embedder=None,
        **kwargs,
    ):
        seen_queries.append(query)
        if query == "how is soil created from rocks":
            evidence = [{"id": "initial", "document_id": "initial_doc"}]
        else:
            evidence = [{"id": f"rewrite_{len(seen_queries)}", "document_id": "gold_doc"}]
        return ToolResult(
            tool="search",
            ok=True,
            data={"evidence": evidence, "anchors": {}},
            session=session.summary(),
        )

    async def fake_expand(backend, session, node_id):
        return ToolResult(tool="expand", ok=True, data={"neighbours": []})

    async def fake_get_doc(backend, session, doc_id, query):
        return ToolResult(
            tool="get_document",
            ok=True,
            data={"document": {"id": doc_id}, "chunks": [], "chunk_count": 0},
        )

    monkeypatch.setattr(tools_v2, "search_tool", fake_search_tool)
    monkeypatch.setattr(tools_v2, "_safe_expand", fake_expand)
    monkeypatch.setattr(tools_v2, "_safe_get_doc", fake_get_doc)
    backend = MemoryBackend()
    await backend.connect()

    result = await tools_v2.deep_search_tool(
        backend,
        SearchSession(),
        "how is soil created from rocks",
    )

    assert seen_queries == [
        "how is soil created from rocks",
        "making soil rock pieces",
        "small pieces of rock form soil",
    ]
    assert result.data["rewrite_queries"] == [
        "making soil rock pieces",
        "small pieces of rock form soil",
    ]
    assert result.data["evidence"][0]["document_id"] == "gold_doc"


@pytest.mark.asyncio
async def test_deep_search_rewrite_can_rescue_empty_initial_search(monkeypatch):
    async def fake_search_tool(
        backend,
        session,
        query,
        *,
        limit,
        category=None,
        embedder=None,
        **kwargs,
    ):
        evidence = (
            []
            if query == "child psychiatrist salary 2016"
            else [{"id": "rewrite_hit", "document_id": "gold_doc"}]
        )
        return ToolResult(
            tool="search",
            ok=True,
            data={"evidence": evidence, "anchors": {}},
            session=session.summary(),
        )

    async def fake_expand(backend, session, node_id):
        return ToolResult(tool="expand", ok=True, data={"neighbours": []})

    async def fake_get_doc(backend, session, doc_id, query):
        return ToolResult(
            tool="get_document",
            ok=True,
            data={"document": {"id": doc_id}, "chunks": [], "chunk_count": 0},
        )

    monkeypatch.setattr(tools_v2, "search_tool", fake_search_tool)
    monkeypatch.setattr(tools_v2, "_safe_expand", fake_expand)
    monkeypatch.setattr(tools_v2, "_safe_get_doc", fake_get_doc)
    backend = MemoryBackend()
    await backend.connect()

    result = await tools_v2.deep_search_tool(
        backend,
        SearchSession(),
        "child psychiatrist salary 2016",
    )

    assert result.data["rewrite_queries"] == ["child psychiatrist salary"]
    assert result.data["evidence"][0]["document_id"] == "gold_doc"


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

    async def test_expand_query_ranks_relevant_neighbour_first(self, monkeypatch):
        # Navigation upgrade (opt-in): with a query, the neighbour whose content
        # matches it (chunk_r1a holds "E217") must rank above an unrelated sibling.
        monkeypatch.setenv("SYNAPTIC_NAV_UPGRADE", "1")
        backend = await _fresh_backend()
        session = SearchSession()
        result = await expand_tool(backend, session, "doc_r1", query="E217 코드")
        ids = [n["id"] for n in result.data["neighbours"]]
        assert "chunk_r1a" in ids
        assert ids.index("chunk_r1a") < ids.index("chunk_r1b")  # relevant first
        assert result.data["via"] == "graph"

    async def test_expand_island_node_uses_semantic_fallback(self, monkeypatch):
        # Navigation upgrade (opt-in): an isolated node (no edges) would dead-end
        # graph traversal; with an embedder, expand falls back to nearest nodes.
        monkeypatch.setenv("SYNAPTIC_NAV_UPGRADE", "1")
        backend = MemoryBackend()
        await backend.connect()
        island = Node(
            id="island",
            kind=NodeKind.CHUNK,
            title="lonely topic",
            content="a fact with no graph links",
            embedding=[1.0, 0.0, 0.0],
            level=ConsolidationLevel.L0_RAW,
        )
        near = Node(
            id="near",
            kind=NodeKind.CHUNK,
            title="adjacent topic",
            content="a semantically close fact",
            embedding=[0.9, 0.1, 0.0],
            level=ConsolidationLevel.L0_RAW,
        )
        await backend.save_node(island)
        await backend.save_node(near)

        class _Emb:
            async def embed(self, text: str):
                return [1.0, 0.0, 0.0]

        session = SearchSession()
        result = await expand_tool(backend, session, "island", embedder=_Emb())
        assert result.ok is True
        ids = {n["id"] for n in result.data["neighbours"]}
        assert "near" in ids  # rescued off the island
        assert result.data["via"] == "semantic"

    async def test_expand_island_no_embedder_returns_empty_with_hint(self):
        # Without an embedder the island still degrades gracefully (no crash).
        backend = MemoryBackend()
        await backend.connect()
        await backend.save_node(
            Node(
                id="island2",
                kind=NodeKind.CHUNK,
                title="lonely",
                content="no links",
                level=ConsolidationLevel.L0_RAW,
            )
        )
        session = SearchSession()
        result = await expand_tool(backend, session, "island2")
        assert result.ok is True
        assert result.data["neighbours"] == []
        assert len(result.hints) > 0


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

    async def test_get_document_unresolved_id_falls_back_to_search(self):
        # An unresolvable id WITH a query must not dead-end the turn — fall back
        # to a content search so the agent still gets usable evidence.
        backend = await _fresh_backend()
        session = SearchSession()
        result = await get_document_tool(
            backend, session, "doc_from_other_namespace_xyz", query="규정 준수"
        )
        assert result.ok is True
        assert result.data.get("fallback") == "doc_id_unresolved_search_fallback"
        assert len(result.data["chunks"]) > 0
        assert any("규정" in (c.get("content") or "") for c in result.data["chunks"])
        # to_dict() is the agent-dispatch path — hints must be Hint objects, not
        # strings (a str hint crashes to_dict on h.action).
        assert result.to_dict()["hints"][0]["action"] == "get_document"

    async def test_get_document_unresolved_id_no_query_still_errors(self):
        # Without a query there's nothing to fall back to → keep the honest error.
        backend = await _fresh_backend()
        session = SearchSession()
        result = await get_document_tool(backend, session, "totally_unknown_id")
        assert result.ok is False
        assert "document_not_found" in (result.error or "")
        assert result.to_dict()["hints"][0]["action"] == "get_document"  # Hint, not str

    async def test_get_document_content_node_without_chunks(self):
        # A content-bearing node with no CONTAINS chunks returns its own text,
        # not an empty result.
        backend = await _fresh_backend()
        await backend.save_node(
            Node(
                id="lonely_doc",
                kind=NodeKind.ENTITY,
                title="고립 문서",
                content="이 문서는 청크가 없지만 본문이 있다",
                tags=["document"],
                level=ConsolidationLevel.L0_RAW,
            )
        )
        session = SearchSession()
        result = await get_document_tool(backend, session, "lonely_doc")
        assert result.ok is True
        assert result.data["chunk_count"] == 1
        assert "본문이 있다" in result.data["chunks"][0]["content"]

    async def test_get_document_marks_chunks_seen(self):
        backend = await _fresh_backend()
        session = SearchSession()
        await get_document_tool(backend, session, "doc_r1")
        assert session.has_seen("chunk_r1a")
        assert session.has_seen("chunk_r1b")

    async def test_get_document_by_bare_doc_id_property(self):
        """search/deep_search expose the doc_id *property* (an opaque
        hash) which differs from the node id. get_document must resolve
        it by scanning for the carrying node."""
        backend = MemoryBackend()
        # node id and doc_id property deliberately differ
        await backend.save_node(
            Node(
                id="doc_node_xyz",
                kind=NodeKind.ENTITY,
                title="규정 문서",
                content="본문",
                tags=["document"],
                properties={"doc_id": "abc123hash"},
                level=ConsolidationLevel.L0_RAW,
            )
        )
        await backend.save_node(
            Node(
                id="chunk_x0",
                kind=NodeKind.CHUNK,
                title="규정 문서 #0",
                content="청크 내용",
                tags=["chunk"],
                properties={"doc_id": "abc123hash", "chunk_index": "0"},
                level=ConsolidationLevel.L0_RAW,
            )
        )
        await backend.save_edge(
            Edge(id="c0", source_id="doc_node_xyz", target_id="chunk_x0", kind=EdgeKind.CONTAINS)
        )
        session = SearchSession()
        result = await get_document_tool(backend, session, "abc123hash")
        assert result.ok is True
        assert result.data["document"]["title"] == "규정 문서"
        assert result.data["chunk_count"] == 1

    async def test_get_document_by_chunk_id_hops_to_parent(self):
        """Passing a chunk node id resolves to the parent document."""
        backend = await _fresh_backend()
        session = SearchSession()
        result = await get_document_tool(backend, session, "chunk_r1a")
        assert result.ok is True
        assert result.data["document"]["id"] == "doc_r1"
        assert result.data["chunk_count"] == 2


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
        rule_cat = next(c for c in result.data["categories"] if c["label"] == "규정 및 지침")
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
        result = await count_tool(backend, session, kind=NodeKind.CHUNK, category="규정 및 지침")
        # Three rule chunks
        assert result.data["count"] == 3

    async def test_count_by_year(self):
        backend = await _fresh_backend()
        session = SearchSession()
        result = await count_tool(backend, session, kind=NodeKind.ENTITY, year=2024)
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
