"""Tests for KoreanPhraseExtractor — zero-dep Korean noun-phrase extraction."""

from __future__ import annotations

import re

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.domain_profile import DomainProfile
from synaptic.extensions.phrase_extractor_ko import (
    KoreanPhraseExtractor,
    _strip_particle,
)
from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind, NodeKind

# ---------- Pure extraction (no graph) ----------


class TestExtractPure:
    def test_single_compound_nouns(self):
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile)
        phrases = extractor.extract("이사회 산하 위원회에서 윤리경영을 논의")
        assert "이사회" in phrases
        assert "위원회" in phrases
        assert "윤리경영" in phrases

    def test_bigram_compound(self):
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile)
        phrases = extractor.extract("온실가스 감축 계획을 수립합니다")
        assert "온실가스 감축" in phrases

    def test_stopwords_excluded(self):
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile)
        phrases = extractor.extract("경우에 따라 관련 사항")
        # "경우", "관련", "사항" are all in generic_korean stopwords
        assert "경우" not in phrases
        assert "관련" not in phrases
        assert "사항" not in phrases

    def test_domain_stopwords_excluded(self):
        profile = DomainProfile(
            name="test",
            locale="ko",
            stopwords_extra=frozenset({"분류번호", "진단항목"}),
        )
        extractor = KoreanPhraseExtractor(profile=profile)
        phrases = extractor.extract("분류번호 검토 후 진단항목 점검")
        assert "분류번호" not in phrases
        assert "진단항목" not in phrases
        assert "점검" not in phrases or len("점검") >= 3  # 점검 is 2 chars, below min_len

    def test_metadata_strip_applied(self):
        profile = DomainProfile(
            name="test",
            locale="ko",
            metadata_strip_patterns=(
                re.compile(r"<Document-Metadata>.*?</Document-Metadata>", re.DOTALL),
            ),
        )
        extractor = KoreanPhraseExtractor(profile=profile)
        text = (
            "<Document-Metadata>\n작성자: 홍길동\n</Document-Metadata>\n"
            "본문에 인권영향평가 체크리스트가 있습니다"
        )
        phrases = extractor.extract(text)
        # "작성자", "홍길동" should be stripped with the metadata block
        assert "작성자" not in phrases
        assert "홍길동" not in phrases
        # Body phrases survive
        assert "인권영향평가" in phrases or "체크리스트" in phrases

    def test_particle_not_stripped_from_short_stems(self):
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile)
        # 회계연도 should NOT lose "도" even though 도 isn't stripped anyway
        # — verifying the exclusion of 도/과 from particle regex
        phrases = extractor.extract("회계연도 기준으로 산정")
        assert "회계연도" in phrases

    def test_particle_stripped_from_long_stems(self):
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile)
        # "윤리경영의" → strip "의" to get "윤리경영"
        phrases = extractor.extract("윤리경영의 기본계획")
        assert "윤리경영" in phrases

    def test_respects_min_phrase_len(self):
        profile = DomainProfile(name="t", locale="ko", min_phrase_len=5)
        extractor = KoreanPhraseExtractor(profile=profile)
        phrases = extractor.extract("이사회 윤리경영 운영")
        assert "이사회" not in phrases  # 3 chars < 5
        assert "윤리경영" not in phrases  # 4 chars < 5

    def test_respects_max_phrase_len(self):
        profile = DomainProfile(name="t", locale="ko", max_phrase_len=5)
        extractor = KoreanPhraseExtractor(profile=profile)
        phrases = extractor.extract("이사회 지속가능경영보고서를 발간")
        assert "이사회" in phrases
        # 지속가능경영보고서 = 9 chars > 5 → excluded
        assert "지속가능경영보고서" not in phrases

    def test_empty_text(self):
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile)
        assert extractor.extract("") == set()
        assert extractor.extract("   ") == set()

    def test_ignores_non_korean(self):
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile)
        phrases = extractor.extract("The quick brown fox jumps over 123")
        assert phrases == set()

    def test_nfc_normalization(self):
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile)
        # NFD-encoded 이사회 (decomposed Hangul) should still extract
        import unicodedata
        nfd_text = unicodedata.normalize("NFD", "이사회 회의")
        phrases = extractor.extract(nfd_text)
        assert "이사회" in phrases

    def test_entity_hint_patterns(self):
        profile = DomainProfile(
            name="t",
            locale="ko",
            entity_hint_patterns=(re.compile(r"\(주\)([가-힣]+)"),),
        )
        extractor = KoreanPhraseExtractor(profile=profile)
        phrases = extractor.extract("(주)플래티어와 협력")
        assert "플래티어" in phrases


# ---------- Particle stripper ----------


class TestStripParticle:
    def test_strip_genitive(self):
        assert _strip_particle("윤리경영의") == "윤리경영"

    def test_strip_object(self):
        assert _strip_particle("계획을") == "계획"

    def test_keep_do_and_gwa(self):
        # 도/과 are NOT stripped — they're in legitimate compound nouns
        assert _strip_particle("회계연도") == "회계연도"
        assert _strip_particle("진단결과") == "진단결과"

    def test_short_stem_not_stripped(self):
        # "상의" → stripping "의" leaves 1 char, below min_stem_len=3
        assert _strip_particle("상의", min_stem_len=3) == "상의"

    def test_no_particle_untouched(self):
        assert _strip_particle("이사회") == "이사회"


# ---------- Locale enforcement ----------


class TestLocaleEnforcement:
    def test_rejects_english_only_profile(self):
        profile = DomainProfile.generic_english()
        with pytest.raises(ValueError, match="locale 'ko' or 'multi'"):
            KoreanPhraseExtractor(profile=profile)

    def test_accepts_korean_profile(self):
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile)
        assert extractor.profile is profile

    def test_accepts_multi_profile(self):
        profile = DomainProfile(name="multi", locale="multi")
        extractor = KoreanPhraseExtractor(profile=profile)
        assert extractor.profile is profile


# ---------- Graph integration ----------


class TestExtractAndLink:
    @pytest.mark.asyncio
    async def test_creates_phrase_nodes_and_mentions_edges(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile, max_phrases_per_node=15)
        graph = SynapticGraph(backend, phrase_extractor=extractor)

        await graph.add(
            title="이사회 운영",
            content="이사회 산하 위원회에서 윤리경영 기본계획을 수립한다",
            kind=NodeKind.CONCEPT,
        )

        # Should have created the passage node + several phrase nodes
        all_nodes = await backend.list_nodes()
        phrase_nodes = [n for n in all_nodes if "_phrase" in (n.tags or [])]
        assert len(phrase_nodes) > 0

        # With a generous cap all key entities should surface as either
        # standalone single-noun phrases or inside bigrams. Treat the
        # union of all phrase titles as the search space.
        all_titles_text = " ".join(n.title for n in phrase_nodes)
        for key in ("이사회", "위원회", "윤리경영"):
            assert key in all_titles_text, f"'{key}' missing from phrases: {all_titles_text}"

    @pytest.mark.asyncio
    async def test_reuses_existing_phrase_nodes_across_passages(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile, max_phrases_per_node=5)
        graph = SynapticGraph(backend, phrase_extractor=extractor)

        await graph.add(
            title="문서 1",
            content="이사회 산하 위원회에서 안건 심의",
            kind=NodeKind.CONCEPT,
        )
        nodes_after_first = await backend.list_nodes()
        phrase_count_after_first = sum(
            1 for n in nodes_after_first if "_phrase" in (n.tags or [])
        )

        await graph.add(
            title="문서 2",
            content="이사회 의사록을 위원회에 공유",
            kind=NodeKind.CONCEPT,
        )
        nodes_after_second = await backend.list_nodes()
        phrase_count_after_second = sum(
            1 for n in nodes_after_second if "_phrase" in (n.tags or [])
        )

        # Shared phrases (이사회, 위원회) must NOT be duplicated
        # Allow a few new phrase nodes (의사록) but growth should be
        # small relative to total phrase count.
        growth = phrase_count_after_second - phrase_count_after_first
        assert growth <= 3, (
            f"Phrase nodes should be reused; saw {phrase_count_after_first} "
            f"→ {phrase_count_after_second}"
        )

    @pytest.mark.asyncio
    async def test_max_phrases_per_node_cap(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()
        extractor = KoreanPhraseExtractor(profile=profile, max_phrases_per_node=2)
        graph = SynapticGraph(backend, phrase_extractor=extractor)

        passage = await graph.add(
            title="많은 엔티티",
            content=(
                "이사회 위원회 윤리경영 온실가스 말산업 경주마 사업장 구성원"
            ),
            kind=NodeKind.CONCEPT,
        )
        edges = await backend.get_edges(passage.id, direction="outgoing")
        mention_edges = [e for e in edges if e.kind == EdgeKind.MENTIONS]
        assert len(mention_edges) <= 2
