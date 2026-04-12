"""Tests for QueryAnchorExtractor — the 3rd-gen retrieval entry point.

We use ``MemoryBackend`` for speed. Category nodes are seeded by hand
to match what ``DocumentIngester`` would produce in a real ingestion
(``NodeKind.CONCEPT`` with the ``"category"`` tag), so the extractor
exercises the same code path it will in production.
"""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.query_anchor import (
    QueryAnchorExtractor,
)
from synaptic.models import ConsolidationLevel, Node, NodeKind


async def _seed_categories(
    backend: MemoryBackend,
    labels: list[str],
) -> list[str]:
    """Create one CONCEPT node per label and return the node ids."""
    ids: list[str] = []
    for i, label in enumerate(labels):
        node = Node(
            id=f"cat_{i}",
            kind=NodeKind.CONCEPT,
            title=label,
            content=label,
            tags=["category"],
            level=ConsolidationLevel.L0_RAW,
        )
        await backend.save_node(node)
        ids.append(node.id)
    return ids


class _FakePhraseExtractor:
    """Hand-wired extractor — returns the caller-supplied phrase set."""

    def __init__(self, mapping: dict[str, set[str]]) -> None:
        self._mapping = mapping

    def extract(self, text: str) -> set[str]:
        return set(self._mapping.get(text, set()))


# --- Keyword extraction ---


@pytest.mark.asyncio
class TestKeywordExtraction:
    async def test_korean_keywords_preserved(self):
        backend = MemoryBackend()
        await backend.connect()
        extractor = QueryAnchorExtractor(backend=backend)
        anchors = await extractor.extract("경마 운영계획 인권경영")
        assert "경마" in anchors.keywords
        assert "운영계획" in anchors.keywords
        assert "인권경영" in anchors.keywords

    async def test_english_keywords_lowercased(self):
        backend = MemoryBackend()
        await backend.connect()
        extractor = QueryAnchorExtractor(backend=backend)
        anchors = await extractor.extract("Machine Learning Research")
        assert "machine" in anchors.keywords
        assert "learning" in anchors.keywords
        assert "research" in anchors.keywords

    async def test_short_tokens_dropped(self):
        backend = MemoryBackend()
        await backend.connect()
        extractor = QueryAnchorExtractor(backend=backend, min_keyword_length=3)
        anchors = await extractor.extract("A bb ccc dddd")
        # "A" and "bb" below the 3-char threshold
        assert "ccc" in anchors.keywords
        assert "dddd" in anchors.keywords
        assert "a" not in anchors.keywords
        assert "bb" not in anchors.keywords

    async def test_duplicates_removed(self):
        backend = MemoryBackend()
        await backend.connect()
        extractor = QueryAnchorExtractor(backend=backend)
        anchors = await extractor.extract("경마 경마 경마 경마산업")
        assert anchors.keywords.count("경마") == 1

    async def test_empty_query_returns_empty_anchors(self):
        backend = MemoryBackend()
        await backend.connect()
        extractor = QueryAnchorExtractor(backend=backend)
        anchors = await extractor.extract("")
        assert anchors.is_empty()
        assert anchors.query == ""

    async def test_whitespace_only_query_returns_empty(self):
        backend = MemoryBackend()
        await backend.connect()
        extractor = QueryAnchorExtractor(backend=backend)
        anchors = await extractor.extract("   \n   ")
        assert anchors.is_empty()


# --- Entity extraction ---


@pytest.mark.asyncio
class TestEntityExtraction:
    async def test_injected_phrase_extractor_used(self):
        backend = MemoryBackend()
        await backend.connect()
        phrase_extractor = _FakePhraseExtractor(
            {"경마 운영계획 인권경영": {"경마", "운영계획", "인권경영", "multi word"}}
        )
        extractor = QueryAnchorExtractor(
            backend=backend,
            phrase_extractor=phrase_extractor,
        )
        anchors = await extractor.extract("경마 운영계획 인권경영")
        assert "multi word" in anchors.entities

    async def test_no_extractor_falls_back_to_keywords(self):
        backend = MemoryBackend()
        await backend.connect()
        extractor = QueryAnchorExtractor(backend=backend)
        anchors = await extractor.extract("경마 운영계획")
        assert set(anchors.entities) == set(anchors.keywords)

    async def test_extractor_exception_falls_back_to_keywords(self):
        class _BrokenExtractor:
            def extract(self, text: str) -> set[str]:
                raise RuntimeError("boom")

        backend = MemoryBackend()
        await backend.connect()
        extractor = QueryAnchorExtractor(
            backend=backend,
            phrase_extractor=_BrokenExtractor(),
        )
        anchors = await extractor.extract("경마 운영계획")
        # Fallback kicked in — we still got keywords
        assert "경마" in anchors.entities

    async def test_longer_phrases_sorted_first(self):
        backend = MemoryBackend()
        await backend.connect()
        phrase_extractor = _FakePhraseExtractor(
            {"경마 운영계획": {"경마", "경마 운영계획", "운영"}}
        )
        extractor = QueryAnchorExtractor(
            backend=backend,
            phrase_extractor=phrase_extractor,
        )
        anchors = await extractor.extract("경마 운영계획")
        # Longer phrase comes first
        assert anchors.entities[0] == "경마 운영계획"


# --- Category matching ---


@pytest.mark.asyncio
class TestCategoryMatching:
    async def test_full_substring_match(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_categories(backend, ["규정 및 지침", "운영계획", "조사 및 평가"])

        extractor = QueryAnchorExtractor(backend=backend)
        anchors = await extractor.extract(
            "규정 및 지침 문서에서 경마 운영계획을 찾아줘"
        )
        assert "규정 및 지침" in anchors.categories
        assert "운영계획" in anchors.categories
        # The node ids match what we seeded
        assert len(anchors.category_node_ids) == 2

    async def test_keyword_overlap_fallback(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_categories(backend, ["경마산업관리", "인권경영"])

        extractor = QueryAnchorExtractor(backend=backend)
        anchors = await extractor.extract("경마 관련 문서")
        # "경마" keyword appears inside "경마산업관리"
        assert "경마산업관리" in anchors.categories

    async def test_full_substring_ranks_before_keyword_overlap(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_categories(backend, ["경마산업관리", "운영계획"])

        extractor = QueryAnchorExtractor(backend=backend)
        anchors = await extractor.extract("운영계획 경마")
        # "운영계획" is a full substring match, "경마산업관리" is keyword overlap
        assert anchors.categories[0] == "운영계획"

    async def test_no_match_returns_empty_categories(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_categories(backend, ["규정 및 지침"])

        extractor = QueryAnchorExtractor(backend=backend)
        anchors = await extractor.extract("unrelated english query")
        assert anchors.categories == []
        assert anchors.category_node_ids == []

    async def test_empty_category_list_is_fine(self):
        backend = MemoryBackend()
        await backend.connect()
        # No categories seeded
        extractor = QueryAnchorExtractor(backend=backend)
        anchors = await extractor.extract("경마 운영계획")
        assert anchors.categories == []
        # But keywords still work
        assert "경마" in anchors.keywords


# --- Category cache behaviour ---


class _CountingBackend:
    """Wrapper that counts ``list_nodes`` calls — for cache tests.

    MemoryBackend uses ``__slots__`` so we can't monkey-patch its
    methods. A thin proxy gives us the same observability without
    touching backend internals.
    """

    def __init__(self, inner: MemoryBackend) -> None:
        self._inner = inner
        self.list_calls = 0

    async def list_nodes(self, **kwargs):
        self.list_calls += 1
        return await self._inner.list_nodes(**kwargs)

    # Pass-through for anything else the extractor might call
    def __getattr__(self, name: str):
        return getattr(self._inner, name)


@pytest.mark.asyncio
class TestCategoryCache:
    async def test_cache_reuse_across_calls(self):
        inner = MemoryBackend()
        await inner.connect()
        await _seed_categories(inner, ["규정 및 지침"])
        backend = _CountingBackend(inner)

        extractor = QueryAnchorExtractor(backend=backend)
        await extractor.extract("규정 및 지침")
        await extractor.extract("규정 및 지침")
        await extractor.extract("규정 및 지침")

        assert backend.list_calls == 1

    async def test_invalidate_cache_forces_reload(self):
        inner = MemoryBackend()
        await inner.connect()
        await _seed_categories(inner, ["규정 및 지침"])
        backend = _CountingBackend(inner)

        extractor = QueryAnchorExtractor(backend=backend)
        await extractor.extract("규정 및 지침")
        extractor.invalidate_cache()
        await extractor.extract("규정 및 지침")

        assert backend.list_calls == 2


# --- Integration: full anchor shape ---


@pytest.mark.asyncio
class TestFullAnchorShape:
    async def test_combined_output(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_categories(backend, ["규정 및 지침", "운영계획"])

        phrase_extractor = _FakePhraseExtractor(
            {
                "규정 및 지침 문서에서 인권경영 운영계획": {
                    "규정",
                    "지침",
                    "인권경영",
                    "운영계획",
                }
            }
        )
        extractor = QueryAnchorExtractor(
            backend=backend,
            phrase_extractor=phrase_extractor,
        )
        anchors = await extractor.extract(
            "규정 및 지침 문서에서 인권경영 운영계획"
        )

        assert not anchors.is_empty()
        assert len(anchors.keywords) > 0
        assert "인권경영" in anchors.entities
        assert "규정 및 지침" in anchors.categories
        assert "운영계획" in anchors.categories
        assert len(anchors.category_node_ids) == 2
