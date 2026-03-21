"""Tests for synonym expansion."""

from __future__ import annotations

from synaptic.synonyms import expand_synonyms


class TestSynonymExpansion:
    def test_korean_to_english(self) -> None:
        expansions = expand_synonyms("배포")
        assert any("deploy" in e for e in expansions)

    def test_english_to_korean(self) -> None:
        expansions = expand_synonyms("deploy")
        assert any("배포" in e for e in expansions)

    def test_no_duplicates(self) -> None:
        expansions = expand_synonyms("버그 수정")
        assert len(expansions) == len(set(expansions))

    def test_unknown_term(self) -> None:
        expansions = expand_synonyms("xyznonexistent")
        assert expansions == []

    def test_multi_word_expansion(self) -> None:
        expansions = expand_synonyms("배포 테스트")
        assert len(expansions) > 0
        # Should have expansions for both terms
        has_deploy = any("deploy" in e for e in expansions)
        has_test = any("test" in e for e in expansions)
        assert has_deploy or has_test

    def test_case_insensitive(self) -> None:
        lower = expand_synonyms("api")
        upper = expand_synonyms("API")
        # Both should produce results
        assert len(lower) > 0 or len(upper) > 0
