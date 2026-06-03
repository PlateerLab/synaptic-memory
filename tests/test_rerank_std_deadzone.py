"""v0.28 — per-query cross-encoder std deadzone (PLAN-v0.18 §Q3).

The v0.17.1 adaptive blend scales the reranker contribution by
``min(1, std/3)``. On retrieval-style corpora (AutoRAG) the cross-encoder
emits a near-flat logit cluster (std ≈ 0.3–0.53) that carries no
discriminative signal, yet the ramp still leaves a small residual blend
(≈ 0.018 at std 0.53) which displaces the FTS top-1 and costs ~−0.10 MRR.

``rerank_std_deadzone`` is an opt-in hard floor: std below it zeroes the
blend outright. Default ``0.0`` never triggers, so shipped behaviour is
byte-for-byte the v0.17.1 ramp. These tests lock the decision logic
(``EvidenceSearch._rerank_discriminator``) and the constructor/env wiring.
"""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.evidence_search import EvidenceSearch

_disc = EvidenceSearch._rerank_discriminator


def test_single_score_passes_through():
    # Fewer than two scores carries no spread information → full blend.
    assert _disc([1.0], 0.0) == 1.0
    assert _disc([], 5.0) == 1.0


def test_wide_spread_full_blend():
    # std = |0-10|/2 = 5 ≥ 3 → capped at 1.0 regardless of deadzone.
    assert _disc([0.0, 10.0], 0.0) == 1.0
    assert _disc([0.0, 10.0], 2.0) == 1.0


def test_flat_cluster_is_zero_without_deadzone():
    # std = 0 → the ramp already yields 0; deadzone is irrelevant here.
    assert _disc([5.0, 5.0, 5.0], 0.0) == 0.0


def test_ramp_value_for_medium_std():
    # [0, 4] → mean 2, std 2 → min(1, 2/3) = 0.6667 (no deadzone).
    assert _disc([0.0, 4.0], 0.0) == pytest.approx(2.0 / 3.0)


def test_deadzone_zeroes_low_std_residual():
    # AutoRAG-like low spread: [5, 5.5, 4.5] → std ≈ 0.408.
    scores = [5.0, 5.5, 4.5]
    ramp = _disc(scores, 0.0)
    # The v0.17.1 ramp leaves a small but NON-zero residual — the exact
    # contribution that costs MRR on retrieval-style corpora.
    assert 0.0 < ramp < 0.2
    # A deadzone above the observed std zeroes it outright.
    assert _disc(scores, 1.0) == 0.0


def test_deadzone_below_std_leaves_ramp_untouched():
    # std ≈ 0.408; a deadzone *below* it must not change the ramp value.
    scores = [5.0, 5.5, 4.5]
    assert _disc(scores, 0.2) == pytest.approx(_disc(scores, 0.0))


def test_medium_std_survives_deadzone():
    # std = 2 with deadzone 1.0 → above the floor, ramp value preserved.
    assert _disc([0.0, 4.0], 1.0) == pytest.approx(2.0 / 3.0)


def test_default_deadzone_is_zero():
    s = EvidenceSearch(backend=MemoryBackend())
    assert s._rerank_std_deadzone == 0.0


def test_constructor_arg_sets_deadzone():
    s = EvidenceSearch(backend=MemoryBackend(), rerank_std_deadzone=1.5)
    assert s._rerank_std_deadzone == 1.5


def test_env_override_sets_deadzone(monkeypatch):
    monkeypatch.setenv("SYNAPTIC_RERANK_STD_DEADZONE", "2.5")
    s = EvidenceSearch(backend=MemoryBackend())
    assert s._rerank_std_deadzone == 2.5


def test_env_override_beats_constructor_arg(monkeypatch):
    # Mirrors the SYNAPTIC_PHRASE_SEED_K precedence: env wins so ablation
    # scripts can flip behaviour without threading the kwarg through.
    monkeypatch.setenv("SYNAPTIC_RERANK_STD_DEADZONE", "3.0")
    s = EvidenceSearch(backend=MemoryBackend(), rerank_std_deadzone=1.0)
    assert s._rerank_std_deadzone == 3.0


def test_env_override_ignores_garbage(monkeypatch):
    monkeypatch.setenv("SYNAPTIC_RERANK_STD_DEADZONE", "not-a-float")
    s = EvidenceSearch(backend=MemoryBackend(), rerank_std_deadzone=1.0)
    assert s._rerank_std_deadzone == 1.0
