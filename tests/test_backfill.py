"""Backfill tests — repair the v0.14.x silent-failure modes.

Two recovery paths:

1. **Embedding backfill** — graphs ingested without an embedder
   stored ``Node.embedding=[]``. ``backfill(embeddings=True)``
   walks every node and fills the missing vectors so HNSW search
   becomes usable on the existing data without re-ingesting.

2. **Phrase-hub backfill** — graphs ingested without a
   ``phrase_extractor`` (the default for the MCP server before
   v0.14.3) have no cross-document bridges. ``backfill(phrases=True)``
   re-runs the extractor on text-bearing nodes that have no
   outgoing CONTAINS edge, creating the missing hubs in place.

Both paths are tested for: success on stale data, idempotency on
healthy data, and graceful no-op when the relevant component is
not wired into the graph.
"""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
from synaptic.extensions.phrase_extractor import PhraseExtractor
from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind, NodeKind


class FakeEmbedder:
    """Deterministic 4-dim embedder for tests.

    Returns ``[len(text)/100, num_words/10, ord(text[0])/256, 1.0]``
    so each text gets a unique-but-deterministic vector. Real
    cosine values are irrelevant here — we only check that
    ``Node.embedding`` was populated.
    """

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            t = t or " "
            out.append(
                [
                    len(t) / 100.0,
                    len(t.split()) / 10.0,
                    ord(t[0]) / 256.0,
                    1.0,
                ]
            )
        return out


# ---------------------------------------------------------------------------
# Embedding backfill
# ---------------------------------------------------------------------------


class TestEmbeddingBackfill:
    @pytest.fixture
    async def graph_no_embedder(self):
        """Graph that ingested 3 nodes WITHOUT an embedder wired up.

        Mirrors the v0.14.x silent-failure scenario: every node
        has ``embedding=[]`` and there is no way for vector search
        to find anything until backfill runs.
        """
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(backend)  # no embedder
        await graph.add(title="alpha", content="first document")
        await graph.add(title="bravo", content="second document")
        await graph.add(title="charlie", content="third document")
        return graph

    async def test_no_op_without_embedder(self, graph_no_embedder):
        """Backfill is a no-op when the graph has no embedder —
        we cannot fabricate embeddings out of thin air."""
        result = await graph_no_embedder.backfill(embeddings=True, phrases=False)
        assert result.embeddings_filled == 0
        assert result.scanned == 0  # early-exit before listing nodes

    async def test_fills_missing_embeddings(self, graph_no_embedder):
        """After wiring an embedder + calling backfill, every node
        should have a non-empty embedding."""
        # Inject an embedder into the existing graph
        graph_no_embedder._embedder = FakeEmbedder()

        result = await graph_no_embedder.backfill(embeddings=True, phrases=False)

        assert result.embeddings_filled == 3
        assert result.scanned == 3
        # Verify on the backend
        nodes = await graph_no_embedder.backend.list_nodes(limit=100)
        for n in nodes:
            assert n.embedding, f"node {n.id} still has empty embedding"
            assert len(n.embedding) == 4  # FakeEmbedder dim

    async def test_idempotent_on_healthy_graph(self, graph_no_embedder):
        """Running backfill twice in a row should be a no-op the
        second time — every node already has an embedding."""
        graph_no_embedder._embedder = FakeEmbedder()
        first = await graph_no_embedder.backfill(embeddings=True, phrases=False)
        second = await graph_no_embedder.backfill(embeddings=True, phrases=False)

        assert first.embeddings_filled == 3
        assert second.embeddings_filled == 0
        # `scanned` still increments even when nothing was filled
        assert second.scanned == 3

    async def test_skips_text_less_nodes(self):
        """A node with no title and no content cannot be embedded
        — record it as skipped, don't crash."""
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(backend, embedder=FakeEmbedder())
        # graph.add() refuses empty content, so build the node
        # via the store directly.
        await graph._store.add_node(
            title="",
            content="",
            kind=NodeKind.CONCEPT,
        )
        result = await graph.backfill(embeddings=True, phrases=False)
        assert result.skipped_no_text == 1
        assert result.embeddings_filled == 0


# ---------------------------------------------------------------------------
# Phrase-hub backfill
# ---------------------------------------------------------------------------


class TestPhraseBackfill:
    @pytest.fixture
    async def graph_no_extractor(self):
        """Graph that ingested 2 documents WITHOUT a phrase extractor.

        Mirrors the v0.14.0~v0.14.2 MCP scenario: the chunks are
        there, the chunk_entity_index is there, but no CONTAINS
        edges to phrase hubs ever got created so cross-doc PPR is
        dead in the water.
        """
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(
            backend,
            chunk_entity_index=ChunkEntityIndex(),
            # phrase_extractor intentionally omitted
        )
        await graph.add(
            title="Project README",
            content="Synaptic Memory is a knowledge graph library for LLM agents.",
        )
        await graph.add(
            title="Architecture overview",
            content="The Synaptic Memory project ships an MCP server with 35 tools.",
        )
        return graph

    async def test_no_op_without_extractor(self, graph_no_extractor):
        result = await graph_no_extractor.backfill(embeddings=False, phrases=True)
        assert result.phrases_linked == 0

    async def test_creates_bridge_after_wiring_extractor(self, graph_no_extractor):
        """Inject a PhraseExtractor and run backfill → both
        documents should now share at least one phrase hub via
        CONTAINS edges."""
        graph_no_extractor._phrase_extractor = PhraseExtractor()

        result = await graph_no_extractor.backfill(embeddings=False, phrases=True)

        assert result.phrases_linked > 0

        # Verify the bridge actually exists
        backend = graph_no_extractor.backend
        nodes = await backend.list_nodes(limit=100)
        doc_nodes = [n for n in nodes if not (n.tags and "_phrase" in n.tags)]
        assert len(doc_nodes) == 2

        targets_per_doc = []
        for d in doc_nodes:
            edges = await backend.get_edges(d.id, direction="outgoing")
            targets = {e.target_id for e in edges if e.kind == EdgeKind.CONTAINS}
            targets_per_doc.append(targets)

        shared = targets_per_doc[0] & targets_per_doc[1]
        assert shared, "expected at least one phrase hub bridging the two docs"

    async def test_idempotent_on_healthy_graph(self, graph_no_extractor):
        """A node that already has CONTAINS edges should be skipped
        on the second pass — no duplicate hubs."""
        graph_no_extractor._phrase_extractor = PhraseExtractor()
        first = await graph_no_extractor.backfill(embeddings=False, phrases=True)
        second = await graph_no_extractor.backfill(embeddings=False, phrases=True)

        assert first.phrases_linked > 0
        assert second.phrases_linked == 0  # all nodes now have CONTAINS edges

    async def test_skips_phrase_hub_nodes(self):
        """Phrase hubs themselves (tagged ``_phrase``) must not be
        re-extracted — that would create infinite hubs of hubs."""
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(
            backend,
            phrase_extractor=PhraseExtractor(),
            chunk_entity_index=ChunkEntityIndex(),
        )
        # Ingest one doc — extractor runs at add() time and creates hubs.
        await graph.add(
            title="Synaptic Memory introduction",
            content="A knowledge graph library called Synaptic Memory.",
        )
        before_hubs = [
            n for n in await backend.list_nodes(limit=100) if n.tags and "_phrase" in n.tags
        ]

        result = await graph.backfill(embeddings=False, phrases=True)

        after_hubs = [
            n for n in await backend.list_nodes(limit=100) if n.tags and "_phrase" in n.tags
        ]
        assert len(after_hubs) == len(before_hubs), (
            "backfill created duplicate phrase hubs — should have skipped them"
        )
        # The original document node already had CONTAINS edges, so
        # the phrase pass should skip it too.
        assert result.phrases_linked == 0


# ---------------------------------------------------------------------------
# Combined (default) backfill
# ---------------------------------------------------------------------------


class TestCombinedBackfill:
    async def test_default_repairs_both(self):
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(backend)  # neither embedder nor extractor
        await graph.add(title="alpha", content="Synaptic Memory rocks")
        await graph.add(title="bravo", content="Synaptic Memory is great")

        # Inject both components and run the default backfill
        graph._embedder = FakeEmbedder()
        graph._phrase_extractor = PhraseExtractor()

        result = await graph.backfill()  # defaults: embeddings + phrases

        assert result.embeddings_filled == 2
        assert result.phrases_linked > 0
        assert result.errors == []
        assert result.elapsed_ms > 0

    async def test_max_nodes_limit_is_respected(self):
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(backend, embedder=FakeEmbedder())
        for i in range(10):
            await graph.add(title=f"node-{i}", content=f"content {i}")

        result = await graph.backfill(embeddings=True, phrases=False, max_nodes=4)
        # We only scanned at most 4 nodes
        assert result.scanned <= 4
        assert result.embeddings_filled <= 4
