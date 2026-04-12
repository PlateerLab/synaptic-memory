"""Tests for EntityLinker — DF-filtered post-processing hub creation."""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.domain_profile import DomainProfile
from synaptic.extensions.entity_linker import (
    EntityLinker,
    EntityLinkStats,
    _mention_edge_id,
    _phrase_hub_id,
)
from synaptic.extensions.phrase_extractor_ko import KoreanPhraseExtractor
from synaptic.extensions.phrase_extractor_en import EnglishPhraseExtractor
from synaptic.models import ConsolidationLevel, EdgeKind, Node, NodeKind


async def _seed_korean_chunks(backend: MemoryBackend, texts: list[str]) -> list[Node]:
    """Helper: save a batch of chunks into the backend and return them."""
    nodes: list[Node] = []
    for i, text in enumerate(texts):
        node = Node(
            id=f"chunk_{i:03d}",
            kind=NodeKind.CHUNK,
            title=f"doc #{i}",
            content=text,
            level=ConsolidationLevel.L0_RAW,
        )
        await backend.save_node(node)
        nodes.append(node)
    return nodes


# --- Deterministic ids ---


class TestDeterministicIds:
    def test_phrase_hub_id_stable(self):
        assert _phrase_hub_id("이사회") == _phrase_hub_id("이사회")
        assert _phrase_hub_id("이사회").startswith("phrase_")

    def test_different_phrases_different_ids(self):
        assert _phrase_hub_id("이사회") != _phrase_hub_id("위원회")

    def test_mention_edge_id_stable(self):
        eid1 = _mention_edge_id("chunk_001", "phrase_abc")
        eid2 = _mention_edge_id("chunk_001", "phrase_abc")
        assert eid1 == eid2
        assert eid1.startswith("mentions_")


# --- Basic link() behaviour ---


class TestEntityLinkerBasic:
    @pytest.mark.asyncio
    async def test_link_korean_hub_nodes_created(self):
        backend = MemoryBackend()
        # Small test corpus: loosen DF thresholds (defaults target larger corpora)
        profile = DomainProfile(
            name="test", locale="ko", min_df=2, max_df_ratio=0.9
        )
        extractor = KoreanPhraseExtractor(profile=profile, max_phrases_per_node=20)
        linker = EntityLinker(
            extractor=extractor, profile=profile, max_links_per_source=10
        )

        texts = [
            "이사회 산하 위원회에서 윤리경영을 논의",
            "이사회 의사록을 위원회에 공유",
            "윤리경영 기본계획 수립",
            "이사회 운영 결의",
        ]
        sources = await _seed_korean_chunks(backend, texts)

        stats = await linker.link(backend, source_kind=NodeKind.CHUNK)

        assert stats.source_nodes_scanned == len(sources)
        assert stats.raw_phrase_candidates > 0
        assert stats.kept_phrases > 0
        assert stats.phrase_nodes_created > 0
        assert stats.mentions_edges_created > 0

        # 이사회 appears in 3 sources → should survive min_df=3
        all_nodes = await backend.list_nodes(kind=NodeKind.ENTITY, limit=1000)
        phrase_nodes = [n for n in all_nodes if "_phrase" in (n.tags or [])]
        titles = {n.title for n in phrase_nodes}
        assert "이사회" in titles

    @pytest.mark.asyncio
    async def test_link_skips_low_df_phrases(self):
        backend = MemoryBackend()
        profile = DomainProfile(
            name="test", locale="ko", min_df=3, max_df_ratio=1.0
        )
        extractor = KoreanPhraseExtractor(profile=profile, max_phrases_per_node=20)
        linker = EntityLinker(extractor=extractor, profile=profile)

        # "이사회" in 3 chunks, "유일단어" in 1 chunk
        texts = [
            "이사회 운영",
            "이사회 결의",
            "이사회 공개",
            "유일단어 등장",
        ]
        await _seed_korean_chunks(backend, texts)
        stats = await linker.link(backend)

        titles = {n.title for n in await backend.list_nodes(kind=NodeKind.ENTITY, limit=100)}
        assert "이사회" in titles
        assert "유일단어" not in titles

    @pytest.mark.asyncio
    async def test_link_skips_high_df_phrases(self):
        backend = MemoryBackend()
        # max_df_ratio=0.5 → "omnipresent" in 100% of chunks gets dropped
        profile = DomainProfile(
            name="test", locale="ko", min_df=2, max_df_ratio=0.5
        )
        extractor = KoreanPhraseExtractor(profile=profile, max_phrases_per_node=20)
        linker = EntityLinker(extractor=extractor, profile=profile)

        texts = [
            "전사공통용어 이사회 운영",
            "전사공통용어 이사회 결의",
            "전사공통용어 이사회 감사",
            "전사공통용어 이사회 공개",
        ]
        await _seed_korean_chunks(backend, texts)
        await linker.link(backend)

        titles = {n.title for n in await backend.list_nodes(kind=NodeKind.ENTITY, limit=100)}
        # "전사공통용어" is in 4/4 = 100% > 50% → dropped
        assert "전사공통용어" not in titles
        # "이사회" is in 4/4 = 100% > 50% → also dropped
        assert "이사회" not in titles

    @pytest.mark.asyncio
    async def test_mentions_edges_connect_to_sources(self):
        backend = MemoryBackend()
        profile = DomainProfile(
            name="test", locale="ko", min_df=2, max_df_ratio=0.9
        )
        extractor = KoreanPhraseExtractor(profile=profile)
        linker = EntityLinker(extractor=extractor, profile=profile)

        texts = ["이사회 위원회", "이사회 감사", "위원회 구성"]
        sources = await _seed_korean_chunks(backend, texts)
        await linker.link(backend)

        # Each source should have at least one outgoing MENTIONS edge
        for src in sources:
            edges = await backend.get_edges(src.id, direction="outgoing")
            mention_edges = [e for e in edges if e.kind == EdgeKind.MENTIONS]
            assert len(mention_edges) > 0

    @pytest.mark.asyncio
    async def test_empty_backend(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile)
        linker = EntityLinker(extractor=extractor, profile=profile)

        stats = await linker.link(backend)
        assert stats.source_nodes_scanned == 0
        assert stats.phrase_nodes_created == 0
        assert stats.mentions_edges_created == 0


# --- Idempotency ---


class TestEntityLinkerIdempotent:
    @pytest.mark.asyncio
    async def test_rerun_does_not_duplicate_hubs(self):
        backend = MemoryBackend()
        profile = DomainProfile(
            name="test", locale="ko", min_df=2, max_df_ratio=0.9
        )
        extractor = KoreanPhraseExtractor(profile=profile)
        linker = EntityLinker(extractor=extractor, profile=profile)

        texts = ["이사회 위원회 운영", "이사회 위원회 감사", "이사회 위원회 공개"]
        await _seed_korean_chunks(backend, texts)

        stats1 = await linker.link(backend)
        nodes_after_first = await backend.list_nodes(kind=NodeKind.ENTITY, limit=1000)
        phrase_count_first = sum(1 for n in nodes_after_first if "_phrase" in (n.tags or []))

        stats2 = await linker.link(backend)
        nodes_after_second = await backend.list_nodes(kind=NodeKind.ENTITY, limit=1000)
        phrase_count_second = sum(1 for n in nodes_after_second if "_phrase" in (n.tags or []))

        # Deterministic ids → same hubs reused, count unchanged
        assert phrase_count_second == phrase_count_first
        assert stats1.phrase_nodes_created == stats2.phrase_nodes_created


# --- English extractor compatibility ---


class TestEntityLinkerEnglish:
    @pytest.mark.asyncio
    async def test_english_extractor_works(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_english()
        profile = DomainProfile(
            name="test_en",
            locale="en",
            min_df=2,
            max_df_ratio=0.9,
            min_phrase_len=3,
        )
        extractor = EnglishPhraseExtractor(
            min_phrase_length=3, max_phrases_per_node=10
        )
        linker = EntityLinker(extractor=extractor, profile=profile)

        texts = [
            "Paul Graham wrote about Hackers and Painters",
            "Paul Graham founded Y Combinator",
            "Y Combinator funds early stage startups",
        ]
        await _seed_korean_chunks(backend, texts)  # helper works for EN too
        stats = await linker.link(backend)

        assert stats.phrase_nodes_created > 0
        titles = {n.title for n in await backend.list_nodes(kind=NodeKind.ENTITY, limit=100)}
        # "Paul Graham" appears in 2 chunks → should survive min_df=2
        assert "Paul Graham" in titles


# --- Cap enforcement ---


class TestMaxLinksPerSource:
    @pytest.mark.asyncio
    async def test_cap_respected(self):
        backend = MemoryBackend()
        profile = DomainProfile(
            name="test", locale="ko", min_df=2, max_df_ratio=1.0
        )
        extractor = KoreanPhraseExtractor(profile=profile, max_phrases_per_node=50)
        linker = EntityLinker(
            extractor=extractor, profile=profile, max_links_per_source=3
        )

        # Many entities per source
        dense = "이사회 위원회 윤리경영 온실가스 지속가능경영 지역사회 전략기획 구성원"
        await _seed_korean_chunks(backend, [dense, dense, dense])
        await linker.link(backend)

        sources = await backend.list_nodes(kind=NodeKind.CHUNK, limit=10)
        for src in sources:
            edges = await backend.get_edges(src.id, direction="outgoing")
            mention_edges = [e for e in edges if e.kind == EdgeKind.MENTIONS]
            assert len(mention_edges) <= 3
