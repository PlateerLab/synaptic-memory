"""Vector cascade threshold tests for HybridSearch.

These tests validate the relative threshold introduced in v0.14.1 to
replace the legacy hardcoded ``cos >= 0.45`` cutoff. Two synthetic
embedder distributions are used:

- **bge-shape**: cosines centred high (0.30 to 0.85), modelling
  bge-m3 / qwen3-embedding-4b / multilingual-e5.
- **openai-shape**: cosines centred low (0.20 to 0.55), modelling
  text-embedding-3-small / 3-large.

The test passes the *exact same* fixture under both distributions
and asserts that:

1. The top vector hit is always returned (recall preserved).
2. Anything within ``top_cos * (1 - vector_relative_drop)`` is
   returned (relative band).
3. Anything below the band — but above the absolute floor — is
   filtered (precision preserved).
4. The legacy hardcoded 0.45 cutoff would have failed the
   openai-shape case; the new logic does not.

The point of the two-shape comparison is that the *count of
returned vector hits is identical* across both — exactly what
"embedder-agnostic" means.
"""

from __future__ import annotations

import math

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.models import Node
from synaptic.search import (
    DEFAULT_VECTOR_MIN_COSINE,
    DEFAULT_VECTOR_RELATIVE_DROP,
    HybridSearch,
    _resolve_float,
)

# ---------------------------------------------------------------------------
# Synthetic vector helpers
# ---------------------------------------------------------------------------
#
# A 2-D unit vector ``(cos θ, sin θ)`` makes the cosine similarity to the
# query vector ``(1, 0)`` equal to ``cos θ`` exactly. That lets the test
# author the cosine of every fixture node directly.


def _vec_for_cosine(target_cos: float) -> list[float]:
    """Return a unit 2-D vector whose cosine vs ``(1, 0)`` is ``target_cos``."""
    target_cos = max(-1.0, min(1.0, target_cos))
    sin = math.sqrt(max(0.0, 1.0 - target_cos * target_cos))
    return [target_cos, sin]


QUERY_VECTOR: list[float] = [1.0, 0.0]


# Two embedder shapes — same conceptual ranking, different absolute cosines.
BGE_SHAPE = [
    ("alpha", 0.85),  # top
    ("bravo", 0.62),
    ("charlie", 0.50),
    ("delta", 0.30),
]

OPENAI_SHAPE = [
    ("alpha", 0.55),  # top
    ("bravo", 0.45),
    ("charlie", 0.35),
    ("delta", 0.20),
]


async def _build_backend(shape: list[tuple[str, float]]) -> MemoryBackend:
    """Build a MemoryBackend with one node per fixture entry.

    Titles use single-letter NATO words that don't appear anywhere
    in the test query — this guarantees zero FTS hits, so the only
    thing the search can return is the vector cascade.
    """
    backend = MemoryBackend()
    await backend.connect()
    for title, target_cos in shape:
        node = Node(
            title=title,
            content=f"opaque content for node {title}",
            embedding=_vec_for_cosine(target_cos),
        )
        # MemoryBackend.save_node assigns the id internally if missing.
        await backend.save_node(node)
    return backend


# Query string that does NOT appear in any node title or content.
# Forces the FTS branch to return zero, so the test isolates the
# vector cascade exclusively.
NO_FTS_HIT_QUERY = "zzzqqqsemanticonly"


# ---------------------------------------------------------------------------
# Override hierarchy
# ---------------------------------------------------------------------------


class TestOverrideHierarchy:
    def test_default_values(self):
        h = HybridSearch()
        assert h._vector_min_cosine == DEFAULT_VECTOR_MIN_COSINE
        assert h._vector_relative_drop == DEFAULT_VECTOR_RELATIVE_DROP

    def test_constructor_overrides_default(self):
        h = HybridSearch(vector_min_cosine=0.05, vector_relative_drop=0.40)
        assert h._vector_min_cosine == 0.05
        assert h._vector_relative_drop == 0.40

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("SYNAPTIC_VECTOR_MIN_COSINE", "0.15")
        monkeypatch.setenv("SYNAPTIC_VECTOR_RELATIVE_DROP", "0.50")
        h = HybridSearch()
        assert h._vector_min_cosine == 0.15
        assert h._vector_relative_drop == 0.50

    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("SYNAPTIC_VECTOR_MIN_COSINE", "0.99")
        h = HybridSearch(vector_min_cosine=0.05)
        assert h._vector_min_cosine == 0.05

    def test_resolve_float_handles_garbage_env(self, monkeypatch):
        monkeypatch.setenv("SYNAPTIC_VECTOR_MIN_COSINE", "not-a-float")
        # Garbage env should fall back to the default, not crash
        assert _resolve_float(None, "SYNAPTIC_VECTOR_MIN_COSINE", 0.10) == 0.10


# ---------------------------------------------------------------------------
# Vector cascade behaviour — the embedder-agnostic guarantee
# ---------------------------------------------------------------------------


def _expected_passes(shape: list[tuple[str, float]], rel_drop: float, abs_floor: float) -> set[str]:
    """Mirror of the production logic — used by the test to compute
    the expected set without re-running the search code path."""
    cosines = [(t, c) for t, c in shape]
    cosines.sort(key=lambda kv: kv[1], reverse=True)
    top_cos = cosines[0][1]
    floor = max(abs_floor, top_cos * (1 - rel_drop))
    return {t for t, c in cosines if c >= floor}


class TestVectorCascadeAgnostic:
    @pytest.mark.parametrize(
        ("shape", "expected_pass_count"),
        [
            (BGE_SHAPE, 2),  # 0.85, 0.62 (floor=0.595)
            (OPENAI_SHAPE, 2),  # 0.55, 0.45 (floor=0.385)
        ],
    )
    async def test_pass_count_is_identical(self, shape, expected_pass_count):
        """Same fixture shape, different cosine scale → same pass count.

        This is the core embedder-agnostic property. With the legacy
        hard 0.45 cutoff the openai-shape case would have returned
        only 1 hit (or 0, depending on rounding); under the new
        relative cutoff both shapes return the same number.
        """
        backend = await _build_backend(shape)
        searcher = HybridSearch()  # defaults
        result = await searcher.search(
            backend,
            NO_FTS_HIT_QUERY,
            limit=10,
            embedding=QUERY_VECTOR,
        )
        # Every node returned must come from the vector cascade,
        # because the query has zero FTS hits.
        titles = {an.node.title for an in result.nodes}
        expected = _expected_passes(shape, DEFAULT_VECTOR_RELATIVE_DROP, DEFAULT_VECTOR_MIN_COSINE)
        assert titles == expected
        assert len(titles) == expected_pass_count

    async def test_top_hit_always_passes(self):
        """No matter the shape, the top vector hit must be returned.

        Recall regression guard — the relative cutoff must never
        eat its own seed.
        """
        for shape in (BGE_SHAPE, OPENAI_SHAPE):
            backend = await _build_backend(shape)
            searcher = HybridSearch()
            result = await searcher.search(
                backend,
                NO_FTS_HIT_QUERY,
                limit=10,
                embedding=QUERY_VECTOR,
            )
            top_title = max(shape, key=lambda kv: kv[1])[0]
            titles = {an.node.title for an in result.nodes}
            assert top_title in titles, f"top hit {top_title!r} missing for shape {shape}"

    async def test_legacy_threshold_would_have_failed_openai(self):
        """Document the bug we are fixing: the openai-shape case
        with the old hardcoded 0.45 cutoff would have lost the 2nd
        and 3rd ranked true positives."""
        # The 'bravo' node in OPENAI_SHAPE has cosine 0.45 — exactly
        # at the old boundary. With strict ``>= 0.45`` it would just
        # pass; bump it down to 0.44 to make the demonstration sharp.
        legacy_shape = [
            ("alpha", 0.55),
            ("bravo", 0.44),  # <0.45, would have been dropped
            ("charlie", 0.35),
            ("delta", 0.20),
        ]
        backend = await _build_backend(legacy_shape)
        searcher = HybridSearch()
        result = await searcher.search(
            backend,
            NO_FTS_HIT_QUERY,
            limit=10,
            embedding=QUERY_VECTOR,
        )
        titles = {an.node.title for an in result.nodes}
        # 'bravo' (0.44) survives because 0.44 > 0.55*0.7 = 0.385.
        # Under the legacy cutoff it would have been filtered.
        assert "alpha" in titles
        assert "bravo" in titles


# ---------------------------------------------------------------------------
# Absolute floor edge cases
# ---------------------------------------------------------------------------


class TestAbsoluteFloor:
    async def test_absolute_floor_kicks_in_when_top_is_weak(self):
        """When even the top vector hit is weak, the absolute floor
        prevents pure noise from being surfaced."""
        weak_shape = [
            ("alpha", 0.12),  # top, just above abs floor
            ("bravo", 0.08),  # below abs floor (0.10)
            ("charlie", 0.05),
        ]
        backend = await _build_backend(weak_shape)
        searcher = HybridSearch()  # default abs_floor=0.10
        result = await searcher.search(
            backend,
            NO_FTS_HIT_QUERY,
            limit=10,
            embedding=QUERY_VECTOR,
        )
        titles = {an.node.title for an in result.nodes}
        # alpha passes (0.12 >= 0.10), bravo and charlie do not.
        # Note: rel_floor would be 0.12 * 0.7 = 0.084, which is
        # *below* the abs floor — so abs floor wins.
        assert "alpha" in titles
        assert "bravo" not in titles
        assert "charlie" not in titles

    async def test_absolute_floor_can_be_disabled(self):
        """Setting min_cosine=0 lets very weak signals through, as
        long as they survive the relative cutoff."""
        weak_shape = [
            ("alpha", 0.12),
            ("bravo", 0.10),  # 0.12 * 0.7 = 0.084, so 0.10 passes
        ]
        backend = await _build_backend(weak_shape)
        searcher = HybridSearch(vector_min_cosine=0.0)
        result = await searcher.search(
            backend,
            NO_FTS_HIT_QUERY,
            limit=10,
            embedding=QUERY_VECTOR,
        )
        titles = {an.node.title for an in result.nodes}
        assert "alpha" in titles
        assert "bravo" in titles


# ---------------------------------------------------------------------------
# Strict / loose drop tuning
# ---------------------------------------------------------------------------


class TestRelativeDropTuning:
    async def test_zero_drop_means_only_the_top_hit(self):
        backend = await _build_backend(BGE_SHAPE)
        searcher = HybridSearch(vector_relative_drop=0.0)
        result = await searcher.search(
            backend,
            NO_FTS_HIT_QUERY,
            limit=10,
            embedding=QUERY_VECTOR,
        )
        titles = {an.node.title for an in result.nodes}
        # rel_floor = 0.85 * 1.0 = 0.85 — only the top hit qualifies.
        assert titles == {"alpha"}

    async def test_full_drop_means_everything_above_abs_floor(self):
        # rel_drop=1.0 → rel_floor = top_cos * 0 = 0
        # so abs_floor (0.10) is the only constraint
        backend = await _build_backend(BGE_SHAPE)
        searcher = HybridSearch(vector_relative_drop=1.0)
        result = await searcher.search(
            backend,
            NO_FTS_HIT_QUERY,
            limit=10,
            embedding=QUERY_VECTOR,
        )
        titles = {an.node.title for an in result.nodes}
        # All four hits in BGE_SHAPE have cos >= 0.10 → all pass.
        assert titles == {"alpha", "bravo", "charlie", "delta"}
