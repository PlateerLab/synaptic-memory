"""v0.28 — coverage-ADAPTIVE retrieval mode (SYNAPTIC_ADAPTIVE / adaptive=True).

One anchor-coverage signal gates the three per-corpus levers per query so they
STACK safely: real-BM25 lexical sharpening (L01) only on high-coverage queries,
vector RRF fusion (L02) only on low-coverage (paraphrase) queries, and the L05
semantic tilt (self-gating). Default off. These tests lock the coverage signal,
the flag wiring, and that both gate branches run cleanly.
"""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.evidence_search import EvidenceSearch
from synaptic.extensions.query_anchor import QueryAnchors
from synaptic.models import ConsolidationLevel, Node, NodeKind

_cov = EvidenceSearch._anchor_coverage


def _a(keywords):
    return QueryAnchors(query="q", keywords=keywords, entities=[])


def _n(title, content):
    return Node(title=title, content=content)


def test_anchor_coverage_fraction():
    assert _cov(_a(["alpha", "beta"]), [_n("alpha doc", "beta here")]) == 1.0
    assert _cov(_a(["alpha", "zzz"]), [_n("alpha", "x")]) == 0.5
    assert _cov(_a(["zzz"]), [_n("a", "b")]) == 0.0
    assert _cov(_a([]), [_n("a", "b")]) is None  # no anchor terms → no signal
    assert _cov(_a(["a"]), []) is None  # no docs → no signal


def test_adaptive_flag_default_param_env(monkeypatch):
    assert EvidenceSearch(backend=MemoryBackend())._adaptive is False
    assert EvidenceSearch(backend=MemoryBackend(), adaptive=True)._adaptive is True
    monkeypatch.setenv("SYNAPTIC_ADAPTIVE", "1")
    assert EvidenceSearch(backend=MemoryBackend())._adaptive is True


@pytest.mark.asyncio
async def test_adaptive_search_runs_both_branches():
    b = SqliteGraphBackend(":memory:")
    await b.connect()
    for i in range(6):
        await b.save_node(
            Node(
                id=f"d{i}",
                kind=NodeKind.CHUNK,
                title=f"alpha topic {i}",
                content=f"alpha beta gamma content {i}",
                level=ConsolidationLevel.L0_RAW,
            )
        )
    es = EvidenceSearch(backend=b, adaptive=True)
    # High-coverage query (terms surface in the docs) → real-BM25 (L01) branch.
    assert len((await es.search("alpha beta gamma", k=3)).scored) > 0
    # Low-coverage query (terms absent) → fusion/tilt branch; runs cleanly even
    # with no embedder (no vector seeds to fuse).
    assert (await es.search("zzz qqq unrelated phrasing", k=3)) is not None
    await b.close()
