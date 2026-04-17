"""Phase C — opt-in EvidenceSearch via ``graph.search(engine='evidence')``.

These tests cover the deprecation pattern introduced in v0.15.0:

- The legacy ``HybridSearch`` path is the default and unchanged
  (covered by the existing ``tests/test_search.py``).
- New code can opt into the modern :class:`EvidenceSearch` pipeline
  by passing ``engine="evidence"`` to ``graph.search()`` without
  having to instantiate the searcher itself.
- The adapter inside ``_search_via_evidence`` returns a
  :class:`SearchResult` (not :class:`EvidenceSearchResult`) so all
  existing callers keep working.
- Unknown engine names raise ``ValueError``.

When the default flips to ``"evidence"`` in v0.16.0 these tests
will move to ``test_search.py`` (the legacy ones get rewritten to
opt back in via ``engine="legacy"``). Until then they sit in their
own file so the migration boundary is obvious in code review.
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


class TestEngineParam:
    async def test_default_engine_is_evidence(self, populated_graph):
        """v0.16.0+: no ``engine`` kwarg → EvidenceSearch pipeline."""
        result = await populated_graph.search("Synaptic Memory", limit=5)
        assert isinstance(result, SearchResult)
        # Modern path reports both 'evidence' and 'fts' in stages_used.
        assert "evidence" in result.stages_used
        assert "fts" in result.stages_used

    async def test_explicit_evidence_matches_default(self, populated_graph):
        """``engine='evidence'`` must produce the same result as the
        default — they are the same pipeline from v0.16.0 onward."""
        a = await populated_graph.search("Synaptic Memory", limit=5)
        b = await populated_graph.search("Synaptic Memory", limit=5, engine="evidence")
        assert [n.node.id for n in a.nodes] == [n.node.id for n in b.nodes]
        assert a.stages_used == b.stages_used

    async def test_legacy_engine_still_reachable(self, populated_graph, recwarn):
        """``engine='legacy'`` is deprecated but still works until v0.17.0."""
        result = await populated_graph.search("Synaptic Memory", limit=5, engine="legacy")
        assert isinstance(result, SearchResult)
        assert "fts" in result.stages_used
        # Should emit a DeprecationWarning.
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "engine='legacy'" in str(w.message)
            for w in recwarn
        )

    async def test_evidence_engine_returns_search_result(self, populated_graph):
        """The opt-in path goes through the modern pipeline and the
        adapter returns ``SearchResult``, not
        ``EvidenceSearchResult``. All 67+ legacy callers must keep
        working when they flip to the new engine."""
        result = await populated_graph.search("Synaptic Memory", limit=5, engine="evidence")
        assert isinstance(result, SearchResult)
        assert len(result.nodes) > 0
        # Modern path's stages signature
        assert "evidence" in result.stages_used
        assert "fts" in result.stages_used  # always reported
        # Each ActivatedNode in the adapter carries the evidence
        # score on both `activation` and `resonance` so legacy
        # ordering code (sort by resonance) keeps working.
        for n in result.nodes:
            assert n.resonance == n.activation

    async def test_evidence_engine_finds_shared_phrase_doc(self, populated_graph):
        """Both Synaptic-Memory docs should appear in the top-2,
        and the unrelated pizza doc should not."""
        result = await populated_graph.search("Synaptic Memory project", limit=3, engine="evidence")
        titles = [n.node.title for n in result.nodes]
        assert any("Synaptic Memory" in t for t in titles)
        assert "Pizza recipe" not in titles[:2]

    async def test_unknown_engine_raises(self, populated_graph):
        with pytest.raises(ValueError, match="Unknown search engine"):
            await populated_graph.search("anything", engine="quantum")

    async def test_evidence_engine_resonance_ordering(self, populated_graph):
        """The adapter must preserve descending-resonance order so
        UIs that iterate `result.nodes` keep displaying the
        highest-relevance hit first (same contract as the legacy
        path)."""
        result = await populated_graph.search("Synaptic Memory", limit=5, engine="evidence")
        for i in range(len(result.nodes) - 1):
            assert result.nodes[i].resonance >= result.nodes[i + 1].resonance
