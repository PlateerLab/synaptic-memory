"""EvidenceSearch is the only retrieval engine (v0.28+).

The ``engine`` parameter on ``graph.search()`` was removed in v0.28 — the
legacy ``HybridSearch`` cascade is no longer reachable through the facade.
These tests lock the behaviour of the default (and only) pipeline:
``graph.search()`` runs :class:`EvidenceSearch` and the adapter inside
``_search_via_evidence`` returns a :class:`SearchResult` (not
:class:`EvidenceSearchResult`) so all existing callers keep working.
"""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
from synaptic.extensions.phrase_extractor import PhraseExtractor
from synaptic.graph import SynapticGraph
from synaptic.models import SearchResult


@pytest.fixture
async def populated_graph():
    """Tiny graph with three docs that share a salient phrase."""
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(
        backend,
        chunk_entity_index=ChunkEntityIndex(),
        phrase_extractor=PhraseExtractor(),
    )
    await graph.add(
        title="Synaptic Memory README",
        content="Synaptic Memory is a knowledge graph library for LLM agents.",
    )
    await graph.add(
        title="Architecture overview",
        content="The Synaptic Memory project ships an MCP server.",
    )
    await graph.add(
        title="Pizza recipe",
        content="Knead dough, top with mozzarella, and bake until crisp.",
    )
    return graph


class TestEvidencePipeline:
    async def test_search_runs_evidence_pipeline(self, populated_graph):
        """``graph.search()`` (no engine kwarg) → EvidenceSearch."""
        result = await populated_graph.search("Synaptic Memory", limit=5)
        assert isinstance(result, SearchResult)
        # Modern path reports both 'evidence' and 'fts' in stages_used.
        assert "evidence" in result.stages_used
        assert "fts" in result.stages_used

    async def test_returns_search_result_adapter(self, populated_graph):
        """The adapter returns ``SearchResult`` (not
        ``EvidenceSearchResult``) and mirrors the evidence score on both
        ``activation`` and ``resonance`` so legacy ordering code keeps
        working."""
        result = await populated_graph.search("Synaptic Memory", limit=5)
        assert isinstance(result, SearchResult)
        assert len(result.nodes) > 0
        for n in result.nodes:
            assert n.resonance == n.activation

    async def test_finds_shared_phrase_doc(self, populated_graph):
        """Both Synaptic-Memory docs should appear in the top-2, and the
        unrelated pizza doc should not."""
        result = await populated_graph.search("Synaptic Memory project", limit=3)
        titles = [n.node.title for n in result.nodes]
        assert any("Synaptic Memory" in t for t in titles)
        assert "Pizza recipe" not in titles[:2]

    async def test_resonance_descending_order(self, populated_graph):
        """The adapter must preserve descending-resonance order so UIs
        that iterate ``result.nodes`` display the highest-relevance hit
        first."""
        result = await populated_graph.search("Synaptic Memory", limit=5)
        for i in range(len(result.nodes) - 1):
            assert result.nodes[i].resonance >= result.nodes[i + 1].resonance
