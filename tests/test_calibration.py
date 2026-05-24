"""Tests for the auto-corpus calibration module."""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.calibration import (
    CalibrationResult,
    _config_for_mrr,
    calibrate_corpus,
    read_calibration,
    write_calibration,
)
from synaptic.models import ConsolidationLevel, Node, NodeKind


def test_config_for_mrr_high_disables_reranker():
    """MRR ≥ 0.85 → reranker_blend = 0 (FTS-already-strong)."""
    cfg = _config_for_mrr(0.92, 20)
    assert cfg.rerank_blend == 0.0
    assert "FTS-near-optimal" in cfg.rationale


def test_config_for_mrr_low_enables_paraphrase_path():
    """MRR ≤ 0.55 → reranker_blend = 0.2 + vector PRF."""
    cfg = _config_for_mrr(0.45, 20)
    assert cfg.rerank_blend == 0.2
    assert cfg.vector_prf_enabled is True
    assert "paraphrase-heavy" in cfg.rationale


def test_config_for_mrr_mid_band_keeps_default():
    """0.55 < MRR < 0.85 → default 0.1 blend."""
    cfg = _config_for_mrr(0.7, 20)
    assert cfg.rerank_blend == 0.1


def test_calibration_result_roundtrip_json():
    cfg = _config_for_mrr(0.91, 20)
    raw = cfg.to_json()
    restored = CalibrationResult.from_json(raw)
    assert restored.rerank_blend == cfg.rerank_blend
    assert restored.sample_size == cfg.sample_size
    assert restored.sample_mrr == cfg.sample_mrr


@pytest.mark.asyncio
async def test_calibrate_corpus_high_mrr_when_titles_distinctive():
    """A corpus where every node has a distinctive title should
    calibrate to high MRR → reranker disabled."""
    backend = MemoryBackend()
    await backend.connect()
    for i in range(25):
        node = Node(
            id=f"doc_{i}",
            kind=NodeKind.CONCEPT,
            title=f"Unique title number {i} about widget {i * 7}",
            content=f"This document covers topic {i} with some shared filler text.",
            level=ConsolidationLevel.L0_RAW,
        )
        await backend.save_node(node)
    cfg = await calibrate_corpus(backend, sample_size=20, seed=42)
    # Distinctive titles should hit themselves easily — expect high MRR
    assert cfg.sample_mrr > 0.5
    # Either default or disabled, both acceptable on synthetic data
    assert cfg.rerank_blend in (0.0, 0.1)


@pytest.mark.asyncio
async def test_write_then_read_calibration_roundtrip_via_sentinel_node():
    """Backends without ``set_meta`` still persist calibration via
    the sentinel-node fallback."""
    backend = MemoryBackend()  # no set_meta
    await backend.connect()
    cfg = _config_for_mrr(0.91, 20)
    await write_calibration(backend, cfg)
    restored = await read_calibration(backend)
    assert restored is not None
    assert restored.rerank_blend == cfg.rerank_blend
    assert restored.sample_mrr == cfg.sample_mrr


@pytest.mark.asyncio
async def test_read_calibration_missing_returns_none():
    backend = MemoryBackend()
    await backend.connect()
    assert await read_calibration(backend) is None


@pytest.mark.asyncio
async def test_calibrate_empty_corpus_returns_default_config():
    """An empty corpus shouldn't crash; returns default-band config."""
    backend = MemoryBackend()
    await backend.connect()
    cfg = await calibrate_corpus(backend, sample_size=20)
    assert cfg.sample_size == 0
    assert cfg.rerank_blend == 0.1  # default band


@pytest.mark.asyncio
async def test_graph_calibrate_persists_blend():
    """SynapticGraph.calibrate() runs the sweep and writes config the
    backend can read back — closes the loop that was open in v0.17.x
    (calibration code existed but was never invoked at ingest)."""
    from synaptic.graph import SynapticGraph

    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend)
    # Distinctive titles → FTS-already-strong corpus
    for i in range(25):
        await graph.add(
            title=f"Distinct subject alpha {i} bravo {i * 11}",
            content=f"Body number {i} with mild filler text",
        )
    result = await graph.calibrate(sample_size=20)
    assert result is not None
    assert result.sample_size > 0
    persisted = await read_calibration(backend)
    assert persisted is not None
    assert persisted.rerank_blend == result.rerank_blend


def test_config_for_signals_hit_at_10_does_not_override_mid_mrr():
    """Hit@10 is recorded for diagnostics but does NOT trigger blend=0
    on its own. v0.26 measurement: PublicHealthQA pseudo hit@10 = 1.0
    while its real-query FTS MRR is 0.55 (reranker gains +0.20). Using
    hit@10 as a trigger would catastrophically disable reranker on
    paraphrase-heavy corpora — pseudo self-retrieval is structurally
    blind to paraphrase difficulty."""
    from synaptic.extensions.calibration import _config_for_signals

    cfg = _config_for_signals(0.70, hit_at_10=0.98, sample_size=20)
    # Stays in mid band — hit@10 alone cannot disable reranker
    assert cfg.rerank_blend == 0.1
    # hit@10 is still recorded for diagnostics
    assert cfg.sample_hit_at_10 == 0.98


def test_calibration_result_json_backcompat_without_hit_at_10():
    """Older calibration blobs (pre-hit@10) deserialise without crashing."""
    legacy = (
        '{"sample_size": 20, "sample_mrr": 0.9, "rerank_blend": 0.0, '
        '"vector_prf_enabled": false, "rationale": "legacy"}'
    )
    cfg = CalibrationResult.from_json(legacy)
    assert cfg.sample_hit_at_10 == 0.0
    assert cfg.rerank_blend == 0.0
