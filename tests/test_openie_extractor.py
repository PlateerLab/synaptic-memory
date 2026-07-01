"""Tests for the opt-in LLM OpenIE extractor (v0.30 P0)."""

from __future__ import annotations

import asyncio
import json

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.entity_extractor_openie import (
    ChainedEntityExtractor,
    LLMOpenIEExtractor,
    OpenIELinker,
    OpenIESelectionPolicy,
    purge_openie_artifacts,
)
from synaptic.extensions.entity_ids import deterministic_entity_id
from synaptic.graph import SynapticGraph
from synaptic.models import Edge, EdgeKind, MemoryEventKind, Node, NodeKind


class _FakeLLM:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(self.payload)


class _FakeRawLLM:
    def __init__(self, raw: str):
        self.raw = raw
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.raw


class _FakeExtractor:
    def __init__(self, ids: list[str] | None = None):
        self.ids = ids or []
        self.calls: list[tuple[str, str, str]] = []

    async def extract_and_link(self, graph, node_id: str, title: str, content: str):
        self.calls.append((node_id, title, content))
        return self.ids or [f"hub_{node_id}"]


class _StagedExtractor:
    def __init__(self, delays: dict[str, float]):
        self.delays = delays
        self.active = 0
        self.max_active = 0
        self.events: list[tuple[str, str]] = []

    async def extract_for_linking(self, node_id: str, title: str, content: str):
        self.events.append(("extract_start", node_id))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delays.get(node_id, 0.0))
            return [f"hub_{node_id}"]
        finally:
            self.active -= 1
            self.events.append(("extract_finish", node_id))

    async def link_result(self, graph, node_id: str, result):
        self.events.append(("link", node_id))
        return result


class _BrokenLLM:
    async def generate(self, **kwargs):
        raise RuntimeError("llm down")


async def _graph_with_chunk() -> SynapticGraph:
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    await backend.save_node(
        Node(
            id="chunk_1",
            kind=NodeKind.CHUNK,
            title="Roadmap note",
            content="Acme depends on the Roadmap. Acme also touches Legacy.",
        )
    )
    return graph


@pytest.mark.asyncio
async def test_chained_entity_extractor_runs_in_order_and_deduplicates_ids():
    first = _FakeExtractor(["ent_a", "ent_b"])
    second = _FakeExtractor(["ent_b", "ent_c"])
    graph = await _graph_with_chunk()

    ids = await ChainedEntityExtractor(first, second).extract_and_link(
        graph,
        "chunk_1",
        "Title",
        "Content",
    )

    assert ids == ["ent_a", "ent_b", "ent_c"]
    assert [call[0] for call in first.calls] == ["chunk_1"]
    assert [call[0] for call in second.calls] == ["chunk_1"]


@pytest.mark.asyncio
async def test_openie_extracts_entities_triples_and_uses_provider_controls():
    payload = {
        "entities": [
            {
                "canonical": "Acme",
                "type": "org",
                "aliases": ["ACME Inc."],
                "confidence": 0.95,
            },
            {"canonical": "Low Confidence", "type": "concept", "confidence": 0.2},
        ],
        "triples": [
            {
                "subject": "ACME Inc.",
                "predicate": "depends_on",
                "object": "Roadmap",
                "confidence": 0.9,
            },
            {
                "subject": "Acme",
                "predicate": "unknown_predicate",
                "object": "Legacy",
                "confidence": 0.75,
            },
            {
                "subject": "Acme",
                "predicate": "related",
                "object": "Low Confidence",
                "confidence": 0.1,
            },
        ],
    }
    llm = _FakeLLM(payload)
    extractor = LLMOpenIEExtractor(llm, seed=123)
    graph = await _graph_with_chunk()

    hub_ids = await extractor.extract_and_link(
        graph,
        "chunk_1",
        "Roadmap note",
        "Acme depends on the Roadmap. Acme also touches Legacy.",
    )

    assert deterministic_entity_id("Acme") in hub_ids
    assert deterministic_entity_id("Roadmap") in hub_ids
    assert deterministic_entity_id("Legacy") in hub_ids
    assert deterministic_entity_id("Low Confidence") not in hub_ids

    call = llm.calls[0]
    assert call["temperature"] == 0.0
    assert call["seed"] == 123
    assert call["max_tokens"] == 1024
    assert call["response_schema"]["type"] == "object"

    acme = await graph.backend.get_node(deterministic_entity_id("Acme"))
    assert acme is not None
    assert "_openie" in acme.tags
    assert "_type:org" in acme.tags

    acme_edges = await graph.backend.get_edges(acme.id, direction="outgoing")
    assert any(e.kind == EdgeKind.DEPENDS_ON for e in acme_edges)
    assert any(e.kind == EdgeKind.RELATED for e in acme_edges)
    assert all(e.weight >= 0.5 for e in acme_edges)
    depends_edges = [e for e in acme_edges if e.kind == EdgeKind.DEPENDS_ON]
    assert depends_edges
    assert depends_edges[0].properties["source_chunk_id"] == "chunk_1"
    assert depends_edges[0].properties["extractor"] == "LLMOpenIEExtractor"
    assert depends_edges[0].properties["prompt_version"].startswith("openie-")
    assert depends_edges[0].properties["is_openie"] == "true"
    assert float(depends_edges[0].properties["confidence"]) == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_openie_extractor_allows_larger_output_budget():
    llm = _FakeLLM({"entities": [{"canonical": "Acme"}], "triples": []})
    extractor = LLMOpenIEExtractor(llm, max_output_tokens=4096)
    graph = await _graph_with_chunk()

    await extractor.extract_and_link(graph, "chunk_1", "T", "Acme")

    assert llm.calls[0]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_openie_cache_avoids_second_llm_call():
    llm = _FakeLLM({"entities": [{"canonical": "Acme"}], "triples": []})
    extractor = LLMOpenIEExtractor(llm)
    graph = await _graph_with_chunk()

    assert extractor.cache_stats() == {"hits": 0, "misses": 0, "entries": 0}
    assert extractor.has_cached_for_linking("T", "Acme") is False
    await extractor.extract_and_link(graph, "chunk_1", "T", "Acme")
    assert extractor.cache_stats() == {"hits": 0, "misses": 1, "entries": 1}
    assert extractor.has_cached_for_linking("T", "Acme") is True
    assert extractor.has_cached_for_linking("T", "Other") is False
    await extractor.extract_and_link(graph, "chunk_1", "T", "Acme")

    assert len(llm.calls) == 1
    assert extractor.cache_stats() == {"hits": 1, "misses": 1, "entries": 1}


@pytest.mark.asyncio
async def test_openie_parser_salvages_complete_items_from_truncated_response():
    raw = """{
      "entities": [
        {"canonical": "Acme", "type": "org", "confidence": 0.95},
        {"canonical": "Roadmap", "type": "concept", "confidence": 0.95}
      ],
      "triples": [
        {"subject": "Acme", "predicate": "depends_on", "object": "Roadmap", "confidence": 0.9},
        {"subject":
    """
    extractor = LLMOpenIEExtractor(_FakeRawLLM(raw), fail_open=False)
    graph = await _graph_with_chunk()

    ids = await extractor.extract_and_link(graph, "chunk_1", "T", "Acme depends on Roadmap")

    assert deterministic_entity_id("Acme") in ids
    acme_edges = await graph.backend.get_edges(
        deterministic_entity_id("Acme"), direction="outgoing"
    )
    assert any(edge.kind == EdgeKind.DEPENDS_ON for edge in acme_edges)


@pytest.mark.asyncio
async def test_openie_parser_salvages_items_when_commas_are_missing():
    raw = """{
      "entities": [
        {"canonical": "Acme", "type": "org"}
        {"canonical": "Roadmap", "type": "concept"}
      ],
      "triples": [
        {"subject": "Acme", "predicate": "related", "object": "Roadmap"}
      ]
    }"""
    result = await LLMOpenIEExtractor(_FakeRawLLM(raw), fail_open=False).extract(
        "Acme related to Roadmap",
        title="T",
    )

    assert [entity.canonical for entity in result.entities] == ["Acme", "Roadmap"]
    assert len(result.triples) == 1


@pytest.mark.asyncio
async def test_openie_cache_does_not_store_unparseable_response(tmp_path):
    cache_path = tmp_path / "openie_cache.jsonl"
    extractor = LLMOpenIEExtractor(
        _FakeRawLLM("not json at all"),
        cache_path=cache_path,
        fail_open=False,
    )

    with pytest.raises(ValueError, match="not a JSON object"):
        await extractor.extract("Acme", title="T")

    assert extractor.cache_stats() == {"hits": 0, "misses": 1, "entries": 0}
    assert extractor.has_cached_for_linking("T", "Acme") is False
    assert not cache_path.exists()


@pytest.mark.asyncio
async def test_openie_fail_open_default_returns_empty_on_llm_error():
    graph = await _graph_with_chunk()

    ids = await LLMOpenIEExtractor(_BrokenLLM()).extract_and_link(
        graph,
        "chunk_1",
        "T",
        "Acme",
    )

    assert ids == []


@pytest.mark.asyncio
async def test_openie_fail_closed_raises_for_eval_failure_accounting():
    graph = await _graph_with_chunk()

    with pytest.raises(RuntimeError, match="llm down"):
        await LLMOpenIEExtractor(_BrokenLLM(), fail_open=False).extract_and_link(
            graph,
            "chunk_1",
            "T",
            "Acme",
        )


@pytest.mark.asyncio
async def test_openie_purge_removes_semantic_layer_and_keeps_structural_nodes():
    llm = _FakeLLM(
        {
            "entities": [{"canonical": "Acme"}],
            "triples": [{"subject": "Acme", "predicate": "related", "object": "Roadmap"}],
        }
    )
    extractor = LLMOpenIEExtractor(llm)
    graph = await _graph_with_chunk()

    await extractor.extract_and_link(graph, "chunk_1", "T", "Acme related to Roadmap")
    assert await graph.backend.get_node(deterministic_entity_id("Acme")) is not None
    assert await graph.backend.get_node("chunk_1") is not None

    deleted = await purge_openie_artifacts(graph.backend)

    assert deleted > 0
    assert await graph.backend.get_node("chunk_1") is not None
    assert await graph.backend.get_node(deterministic_entity_id("Acme")) is None
    chunk_edges = await graph.backend.get_edges("chunk_1", direction="outgoing")
    assert not any(e.id.startswith("openie_") for e in chunk_edges)


@pytest.mark.asyncio
async def test_openie_does_not_overwrite_existing_non_openie_edge():
    llm = _FakeLLM(
        {
            "entities": [{"canonical": "Acme"}, {"canonical": "Roadmap"}],
            "triples": [
                {"subject": "Acme", "predicate": "related", "object": "Roadmap", "confidence": 0.6}
            ],
        }
    )
    graph = await _graph_with_chunk()
    acme_id = deterministic_entity_id("Acme")
    roadmap_id = deterministic_entity_id("Roadmap")
    await graph.backend.save_node(Node(id=acme_id, kind=NodeKind.ENTITY, title="Acme"))
    await graph.backend.save_node(Node(id=roadmap_id, kind=NodeKind.ENTITY, title="Roadmap"))
    await graph.backend.save_edge(
        Edge(
            id="structural_fk",
            source_id=acme_id,
            target_id=roadmap_id,
            kind=EdgeKind.RELATED,
            weight=1.0,
        )
    )

    await LLMOpenIEExtractor(llm).extract_and_link(graph, "chunk_1", "T", "Acme Roadmap")

    edges = await graph.backend.get_edges(acme_id, direction="outgoing")
    related = [e for e in edges if e.target_id == roadmap_id and e.kind == EdgeKind.RELATED]
    assert len(related) == 1
    assert related[0].id == "structural_fk"
    assert related[0].weight == 1.0


@pytest.mark.asyncio
async def test_openie_does_not_mutate_existing_non_openie_hub():
    llm = _FakeLLM({"entities": [{"canonical": "Acme", "type": "org"}], "triples": []})
    graph = await _graph_with_chunk()
    acme_id = deterministic_entity_id("Acme")
    await graph.backend.save_node(
        Node(
            id=acme_id,
            kind=NodeKind.ENTITY,
            title="Acme",
            tags=["_phrase"],
            properties={"df": "3"},
        )
    )

    await LLMOpenIEExtractor(llm).extract_and_link(graph, "chunk_1", "T", "Acme")

    acme = await graph.backend.get_node(acme_id)
    assert acme is not None
    assert acme.tags == ["_phrase"]
    assert acme.properties == {"df": "3"}


@pytest.mark.asyncio
async def test_openie_linker_prefilters_chunks_by_candidate_entities():
    backend = MemoryBackend()
    await backend.save_node(
        Node(
            id="chunk_a",
            kind=NodeKind.CHUNK,
            title="Roadmap",
            content="Acme depends on Roadmap.",
        )
    )
    await backend.save_node(
        Node(id="chunk_b", kind=NodeKind.CHUNK, title="Doc", content="Acme only.")
    )
    await backend.save_node(Node(id="chunk_c", kind=NodeKind.CHUNK, title="Doc", content=""))
    extractor = _FakeExtractor()

    stats = await OpenIELinker(
        extractor,
        selection_policy=OpenIESelectionPolicy(
            min_candidate_entities=2,
            max_candidate_df_ratio=1.0,
        ),
    ).link(backend)

    assert stats.gated is False
    assert stats.chunks_scanned == 3
    assert stats.chunks_selected == 1
    assert stats.entity_nodes_touched == 1
    assert [call[0] for call in extractor.calls] == ["chunk_a"]


@pytest.mark.asyncio
async def test_openie_linker_records_semantic_event_and_stamps_provenance():
    backend = MemoryBackend()
    await backend.save_node(
        Node(
            id="chunk_a",
            kind=NodeKind.CHUNK,
            title="Roadmap",
            content="Acme depends on Roadmap.",
        )
    )
    llm = _FakeLLM(
        {
            "entities": [
                {"canonical": "Acme", "type": "org", "confidence": 0.95},
                {"canonical": "Roadmap", "type": "concept", "confidence": 0.95},
            ],
            "triples": [
                {
                    "subject": "Acme",
                    "predicate": "depends_on",
                    "object": "Roadmap",
                    "confidence": 0.9,
                }
            ],
        }
    )
    llm.model = "unit-openie-model"
    extractor = LLMOpenIEExtractor(llm, max_output_tokens=2048, max_triples_per_chunk=7)

    stats = await OpenIELinker(extractor).link(backend)

    assert stats.extraction_failures == 0
    assert stats.edge_ids
    events = await backend.list_memory_events(kind=MemoryEventKind.SEMANTIC_EXTRACT)
    assert len(events) == 1
    assert events[0].edge_ids == stats.edge_ids
    assert deterministic_entity_id("Acme") in events[0].node_ids
    assert events[0].properties["linker"] == "OpenIELinker"
    assert events[0].properties["extractor"] == "LLMOpenIEExtractor"
    assert events[0].properties["model"] == "unit-openie-model"
    assert events[0].properties["prompt_version"].startswith("openie-")
    assert events[0].properties["max_output_tokens"] == "2048"
    assert events[0].properties["max_triples_per_chunk"] == "7"

    edges = await backend.get_edges(deterministic_entity_id("Acme"), direction="outgoing")
    relation_edges = [edge for edge in edges if edge.kind == EdgeKind.DEPENDS_ON]
    assert relation_edges
    assert relation_edges[0].properties["source_event_id"] == events[0].id
    assert relation_edges[0].properties["model"] == "unit-openie-model"


@pytest.mark.asyncio
async def test_openie_linker_stages_concurrent_extraction_but_links_in_chunk_order():
    backend = MemoryBackend()
    for node_id in ("chunk_b", "chunk_a", "chunk_c"):
        await backend.save_node(
            Node(
                id=node_id,
                kind=NodeKind.CHUNK,
                title=node_id,
                content="Acme depends on Roadmap.",
            )
        )
    extractor = _StagedExtractor({"chunk_a": 0.03, "chunk_b": 0.01, "chunk_c": 0.0})

    stats = await OpenIELinker(extractor, max_concurrency=2).link(backend)

    assert stats.extraction_failures == 0
    assert stats.entity_nodes_touched == 3
    assert extractor.max_active == 2
    assert [node_id for event, node_id in extractor.events if event == "link"] == [
        "chunk_a",
        "chunk_b",
        "chunk_c",
    ]


@pytest.mark.asyncio
async def test_openie_linker_sampling_zero_gates_without_llm_calls():
    backend = MemoryBackend()
    await backend.save_node(
        Node(
            id="chunk_a",
            kind=NodeKind.CHUNK,
            title="Roadmap",
            content="Acme depends on Roadmap.",
        )
    )
    extractor = _FakeExtractor()

    stats = await OpenIELinker(
        extractor,
        selection_policy=OpenIESelectionPolicy(
            min_candidate_entities=2,
            sample_rate=0.0,
        ),
    ).link(backend)

    assert stats.gated is True
    assert stats.gate_reason == "selector chose no chunks"
    assert stats.chunks_selected == 0
    assert extractor.calls == []
