"""L02 — true RRF fusion of the FTS + vector seed rank lists (opt-in).

The default "cascade" pins every vector-only seed at a flat 0.08 below the
FTS floor. fusion_mode="rrf" RRF-fuses the two rank lists and max-combines
into the seed scores, lifting a vector-only seed by its rank. These tests
lock the config plumbing and the mechanism (a strong vector-only match
scores higher under rrf than under the cascade cap).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.evidence_search import EvidenceSearch
from synaptic.models import ConsolidationLevel, Node, NodeKind


@dataclass
class _SlotEmbedder:
    """One-hot slot embedder: cosine 1 iff the two strings share a slot token."""

    slots: dict[str, int]
    dim: int

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.split():
            i = self.slots.get(tok.lower())
            if i is not None:
                vec[i] = 1.0
        return vec

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


def test_fusion_mode_default_param_and_env(monkeypatch):
    assert EvidenceSearch(backend=MemoryBackend())._fusion_mode == "cascade"
    assert EvidenceSearch(backend=MemoryBackend(), fusion_mode="rrf")._fusion_mode == "rrf"
    monkeypatch.setenv("SYNAPTIC_FUSION_MODE", "rrf")
    assert EvidenceSearch(backend=MemoryBackend())._fusion_mode == "rrf"  # env wins


async def _make_corpus(backend, emb):
    # vec_only: matches the query by EMBEDDING (slot 'topic') but shares no
    # surface tokens with the query. lex: a weak lexical hit. distractor: none.
    nodes = [
        Node(
            id="vec_only",
            kind=NodeKind.CHUNK,
            title="dvec",
            content="zzz qqq surface filler",
            level=ConsolidationLevel.L0_RAW,
            embedding=await emb.embed("topic"),
        ),
        Node(
            id="lex",
            kind=NodeKind.CHUNK,
            title="dlex",
            content="keywords appear in this passage",
            level=ConsolidationLevel.L0_RAW,
            embedding=await emb.embed("nomatch"),
        ),
        Node(
            id="distractor",
            kind=NodeKind.CHUNK,
            title="ddist",
            content="completely different content entirely",
            level=ConsolidationLevel.L0_RAW,
            embedding=await emb.embed("nomatch"),
        ),
    ]
    for n in nodes:
        await backend.save_node(n)


@pytest.mark.asyncio
async def test_rrf_lifts_vector_only_seed_above_cascade():
    slots = {"topic": 0, "keywords": 1, "nomatch": 2}
    emb = _SlotEmbedder(slots=slots, dim=3)

    async def _score(mode: str) -> float:
        backend = MemoryBackend()
        await backend.connect()
        await _make_corpus(backend, emb)
        searcher = EvidenceSearch(backend=backend, embedder=emb, fusion_mode=mode)
        # Query embeds to the 'topic' slot (→ vec_only) and lexically hits
        # 'keywords' (→ lex). vec_only is therefore a vector-ONLY seed.
        q_emb = await emb.embed("topic")
        res = await searcher.search("keywords topic", k=5, query_embedding=q_emb)
        hit = next((s for s in res.scored if s.node.id == "vec_only"), None)
        return hit.total if hit else 0.0

    cascade_total = await _score("cascade")
    rrf_total = await _score("rrf")
    # The vector-only seed must score strictly higher once it is RRF-fused
    # instead of pinned at the flat 0.08 cascade floor.
    assert rrf_total > cascade_total
