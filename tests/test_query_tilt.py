"""L05 — per-query lexical↔semantic reranker-weight tilt from anchor coverage.

Default off (`SYNAPTIC_QUERY_TILT`). When on, a query whose anchor terms do
NOT surface in the retrieved docs (a paraphrase) tilts the reranker toward the
semantic axis; a lexically-confident query keeps the default weights. The
signal is paraphrase-AWARE (surface overlap is low by definition for a
paraphrase) — the property the v0.26 pseudo-self-retrieval calibration lacked.
"""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.evidence_search import EvidenceSearch
from synaptic.extensions.query_anchor import QueryAnchors
from synaptic.models import ConsolidationLevel, Node, NodeKind

_tilt = EvidenceSearch._tilt_weights


def _anchors(keywords, entities=None):
    return QueryAnchors(query="q", keywords=keywords, entities=entities or [])


def _node(title, content):
    return Node(title=title, content=content)


def test_high_coverage_no_tilt():
    # every query term surfaces in the docs → lexical-confident → no tilt
    assert _tilt(_anchors(["alpha", "beta"]), [_node("alpha", "beta content")]) is None


def test_zero_coverage_full_tilt_to_semantic_lead():
    w = _tilt(_anchors(["zzz", "qqq", "www"]), [_node("alpha", "beta gamma")])
    assert w is not None
    assert w.lexical == pytest.approx(0.30) and w.semantic == pytest.approx(0.40)
    assert w.graph == 0.20 and w.structural == 0.10
    # weights still sum to 1.0
    assert (w.lexical + w.semantic + w.graph + w.structural) == pytest.approx(1.0)


def test_partial_coverage_interpolates():
    # 1 of 2 terms hit → coverage 0.5 → partial tilt strictly between presets
    w = _tilt(_anchors(["alpha", "zzz"]), [_node("alpha", "content")])
    assert w is not None
    assert 0.30 < w.lexical < 0.45
    assert 0.25 < w.semantic < 0.40


def test_no_signal_returns_none():
    assert _tilt(_anchors([]), [_node("a", "b")]) is None
    assert _tilt(_anchors(["a"]), []) is None


def test_flag_default_off_and_env(monkeypatch):
    assert EvidenceSearch(backend=MemoryBackend())._query_tilt_enabled is False
    monkeypatch.setenv("SYNAPTIC_QUERY_TILT", "1")
    assert EvidenceSearch(backend=MemoryBackend())._query_tilt_enabled is True


@pytest.mark.asyncio
async def test_tilt_on_search_runs(monkeypatch):
    # End-to-end: tilt enabled, a low-overlap query still searches cleanly and
    # the per-call weights override threads through rerank() without error.
    monkeypatch.setenv("SYNAPTIC_QUERY_TILT", "1")
    b = MemoryBackend()
    await b.connect()
    for i in range(4):
        await b.save_node(
            Node(
                id=f"d{i}",
                kind=NodeKind.CHUNK,
                title=f"document {i}",
                content=f"alpha beta gamma content number {i}",
                level=ConsolidationLevel.L0_RAW,
            )
        )
    es = EvidenceSearch(backend=b)
    res = await es.search("zzz qqq unrelated phrasing", k=3)
    assert res is not None  # tilt path executed, no crash
    await b.close()
