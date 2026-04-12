"""Tests for DomainProfile — domain/locale configuration injection."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from synaptic.extensions.domain_profile import (
    DomainProfile,
    locale_default_stopwords,
)
from synaptic.models import NodeKind


class TestFactoryConstructors:
    def test_generic_korean(self):
        profile = DomainProfile.generic_korean()
        assert profile.name == "generic_ko"
        assert profile.locale == "ko"
        assert profile.stopwords_extra == frozenset()
        assert profile.ontology_hints == {}

    def test_generic_english(self):
        profile = DomainProfile.generic_english()
        assert profile.name == "generic_en"
        assert profile.locale == "en"

    def test_generic_korean_custom_name(self):
        profile = DomainProfile.generic_korean(name="my_ko")
        assert profile.name == "my_ko"
        assert profile.locale == "ko"

    def test_default_locale_is_multi(self):
        profile = DomainProfile(name="test")
        assert profile.locale == "multi"


class TestLocaleDefaults:
    def test_korean_stopwords_include_particles(self):
        stops = locale_default_stopwords("ko")
        assert "경우" in stops
        assert "있다" in stops
        assert "년도" in stops

    def test_english_stopwords_include_articles(self):
        stops = locale_default_stopwords("en")
        assert "the" in stops
        assert "and" in stops
        assert "is" in stops

    def test_unknown_locale_returns_empty(self):
        assert locale_default_stopwords("klingon") == frozenset()

    def test_multi_locale_returns_empty(self):
        assert locale_default_stopwords("multi") == frozenset()


class TestStopwordsUnion:
    def test_effective_stopwords_combines_locale_and_extra(self):
        profile = DomainProfile(
            name="test",
            locale="ko",
            stopwords_extra=frozenset({"분류번호", "진단항목"}),
        )
        effective = profile.stopwords()
        # Locale defaults must be present
        assert "경우" in effective
        # Extra domain stopwords must be present
        assert "분류번호" in effective
        assert "진단항목" in effective

    def test_effective_stopwords_no_locale_defaults(self):
        profile = DomainProfile(
            name="test",
            locale="multi",
            stopwords_extra=frozenset({"custom"}),
        )
        effective = profile.stopwords()
        assert effective == frozenset({"custom"})


class TestTomlLoader:
    def _write_toml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "profile.toml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_load_minimal(self, tmp_path):
        path = self._write_toml(
            tmp_path,
            """
            name = "minimal"
            locale = "ko"
            """,
        )
        profile = DomainProfile.load(path)
        assert profile.name == "minimal"
        assert profile.locale == "ko"
        assert profile.stopwords_extra == frozenset()

    def test_load_full(self, tmp_path):
        path = self._write_toml(
            tmp_path,
            """
            name = "krra"
            locale = "ko"
            stopwords_extra = ["분류번호", "진단항목"]
            metadata_strip_patterns = ["<Document-Metadata>.*?</Document-Metadata>"]
            reference_patterns = ["(.+?)에 따라", "(.+?)에 의거"]
            entity_hint_patterns = ['\\\\(([주사재])\\\\)([가-힣]+)']
            min_df = 5
            max_df_ratio = 0.25
            min_phrase_len = 4
            max_phrase_len = 15

            [ontology_hints]
            "규정 및 지침" = "rule"
            "운영계획" = "DECISION"
            """,
        )
        profile = DomainProfile.load(path)
        assert profile.name == "krra"
        assert profile.locale == "ko"
        assert "분류번호" in profile.stopwords_extra
        assert len(profile.metadata_strip_patterns) == 1
        assert len(profile.reference_patterns) == 2
        assert len(profile.entity_hint_patterns) == 1
        assert profile.min_df == 5
        assert profile.max_df_ratio == 0.25
        assert profile.min_phrase_len == 4
        assert profile.max_phrase_len == 15
        # ontology_hints — both lowercase "rule" and uppercase "DECISION" work
        assert profile.ontology_hints["규정 및 지침"] == NodeKind.RULE
        assert profile.ontology_hints["운영계획"] == NodeKind.DECISION

    def test_load_missing_name_raises(self, tmp_path):
        path = self._write_toml(tmp_path, 'locale = "ko"')
        with pytest.raises(ValueError, match="'name' is required"):
            DomainProfile.load(path)

    def test_load_empty_name_raises(self, tmp_path):
        path = self._write_toml(
            tmp_path,
            """
            name = ""
            locale = "ko"
            """,
        )
        with pytest.raises(ValueError, match="'name' is required"):
            DomainProfile.load(path)

    def test_load_invalid_nodekind_raises(self, tmp_path):
        path = self._write_toml(
            tmp_path,
            """
            name = "test"
            locale = "ko"

            [ontology_hints]
            "category" = "NOT_A_KIND"
            """,
        )
        with pytest.raises(ValueError, match="unknown NodeKind"):
            DomainProfile.load(path)

    def test_load_invalid_regex_raises(self, tmp_path):
        path = self._write_toml(
            tmp_path,
            """
            name = "test"
            locale = "ko"
            metadata_strip_patterns = ["[unclosed"]
            """,
        )
        with pytest.raises(ValueError, match="invalid regex"):
            DomainProfile.load(path)

    def test_load_unknown_keys_ignored(self, tmp_path):
        """Forward-compat: unknown keys should not break loading."""
        path = self._write_toml(
            tmp_path,
            """
            name = "test"
            locale = "ko"
            future_feature = "whatever"
            """,
        )
        profile = DomainProfile.load(path)
        assert profile.name == "test"


class TestCompiledPatterns:
    def test_metadata_pattern_matches_multiline_block(self):
        profile = DomainProfile(
            name="test",
            metadata_strip_patterns=(
                re.compile(r"<Document-Metadata>.*?</Document-Metadata>", re.DOTALL),
            ),
        )
        text = "<Document-Metadata>\n작성자: KRA\n</Document-Metadata>\n본문"
        stripped = profile.metadata_strip_patterns[0].sub("", text)
        assert stripped.strip() == "본문"

    def test_reference_pattern_captures_target(self):
        profile = DomainProfile(
            name="test",
            reference_patterns=(re.compile(r"(.+?)에 따라"),),
        )
        text = "예산집행지침에 따라 집행"
        m = profile.reference_patterns[0].search(text)
        assert m is not None
        assert m.group(1) == "예산집행지침"
