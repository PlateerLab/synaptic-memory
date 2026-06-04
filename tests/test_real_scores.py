"""L01 — real normalized BM25 into the lexical axis (opt-in SYNAPTIC_REAL_SCORES).

Locks the backend scoring contract (``search_fts(with_scores=True)`` →
``[(node, rel∈[0,1])]``, FTS5 band above LIKE band, order preserved) and the
EvidenceSearch capability detection + graceful fallback on backends that do
not implement the kwarg.
"""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.backends.sqlite import _lexical_relevance
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.evidence_search import EvidenceSearch
from synaptic.models import ConsolidationLevel, Node, NodeKind


def _chunk(nid: str, title: str, content: str) -> Node:
    return Node(
        id=nid, kind=NodeKind.CHUNK, title=title, content=content, level=ConsolidationLevel.L0_RAW
    )


def test_lexical_relevance_separates_fts_and_like_bands():
    a, b, c = Node(id="a"), Node(id="b"), Node(id="c")
    # FTS5 raws are negative (lower=better); the LIKE fallback uses ~10000.
    # Input is sorted ascending (best first), as search_fts produces it.
    out = _lexical_relevance([(a, -5.0), (b, -2.0), (c, 9998.0)])
    s = {n.id: v for n, v in out}
    assert s["a"] == 1.0  # strongest FTS5 hit → band ceiling
    assert 0.35 <= s["b"] < 1.0  # weaker FTS5 hit stays in the upper band
    assert s["c"] <= 0.30  # LIKE-fallback hit sits in the lower band
    assert s["a"] > s["b"] > s["c"]  # rank order preserved
    assert all(0.0 <= v <= 1.0 for v in s.values())


def test_lexical_relevance_single_and_empty():
    assert _lexical_relevance([]) == []
    single = _lexical_relevance([(Node(id="x"), -3.0)])
    assert single[0][1] == 1.0  # single FTS5 hit → ceiling (no spread)


@pytest.mark.asyncio
async def test_sqlite_search_fts_with_scores_contract():
    b = SqliteGraphBackend(":memory:")
    await b.connect()
    await b.save_node(_chunk("d1", "alpha beta", "alpha beta gamma content"))
    await b.save_node(_chunk("d2", "beta", "beta only here"))
    await b.save_node(_chunk("d3", "unrelated", "nothing matches the query"))

    scored = await b.search_fts("alpha beta", limit=10, with_scores=True)
    assert scored and isinstance(scored[0], tuple)
    vals = [s for _, s in scored]
    assert all(0.0 <= s <= 1.0 for s in vals)
    assert vals == sorted(vals, reverse=True)  # best first

    plain = await b.search_fts("alpha beta", limit=10)  # default → list[Node]
    assert all(hasattr(n, "id") for n in plain)
    assert {n.id for n, _ in scored} == {n.id for n in plain}
    await b.close()


@pytest.mark.asyncio
async def test_evidence_real_scores_flag_engages_on_sqlite(monkeypatch):
    b = SqliteGraphBackend(":memory:")
    await b.connect()
    for i in range(5):
        await b.save_node(_chunk(f"d{i}", f"doc {i} alpha", f"alpha content number {i}"))

    monkeypatch.setenv("SYNAPTIC_REAL_SCORES", "1")
    es_on = EvidenceSearch(backend=b)
    assert es_on._real_scores_enabled is True
    res = await es_on.search("alpha", k=3)
    assert len(res.scored) > 0  # real-score path produces candidates, no crash

    monkeypatch.delenv("SYNAPTIC_REAL_SCORES", raising=False)
    es_off = EvidenceSearch(backend=b)
    assert es_off._real_scores_enabled is False  # default off
    await b.close()


@pytest.mark.asyncio
async def test_real_scores_falls_back_when_backend_lacks_kwarg(monkeypatch):
    # MemoryBackend.search_fts has no `with_scores` param → detection must
    # turn the flag off and the pipeline must run via the rank ramp.
    b = MemoryBackend()
    await b.connect()
    await b.save_node(_chunk("m1", "alpha", "alpha content"))

    monkeypatch.setenv("SYNAPTIC_REAL_SCORES", "1")
    es = EvidenceSearch(backend=b)
    assert es._real_scores_enabled is False  # backend can't be scored
    res = await es.search("alpha", k=3)  # must not raise
    assert res is not None
    await b.close()
