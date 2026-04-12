"""Tests for ProfileGenerator — rule-based and LLM paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synaptic.extensions.domain_profile import DomainProfile
from synaptic.extensions.ontology_classifier import (
    DEFAULT_NODE_KIND_DESCRIPTIONS,
    OntologyClassifier,
)
from synaptic.extensions.profile_generator import (
    ProfileGenerator,
    _extract_json_object,
    _unique_preserve_order,
    detect_locale,
    suggest_stopwords_by_frequency,
)
from synaptic.models import NodeKind

# --- Locale detection ---


class TestDetectLocale:
    def test_korean_corpus_detected(self):
        samples = [
            "인권영향평가 결과보고서입니다",
            "한국마사회 경영계획 수립",
            "운영계획 및 예산집행 현황",
        ]
        assert detect_locale(samples) == "ko"

    def test_english_corpus_detected(self):
        samples = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning research paper abstract",
            "Clinical trial results for new medication",
        ]
        assert detect_locale(samples) == "en"

    def test_japanese_corpus_detected(self):
        samples = [
            "これは日本語のテストです",
            "ひらがなとカタカナが混ざっています",
        ]
        assert detect_locale(samples) == "ja"

    def test_empty_returns_multi(self):
        assert detect_locale([]) == "multi"
        assert detect_locale([""]) == "multi"

    def test_mixed_returns_multi(self):
        # 40% ko / 40% en / rest digits — neither crosses 50% threshold
        samples = ["한국어텍스트 English text 1234 5678"]
        assert detect_locale(samples) == "multi"


# --- Rule-based stopword suggestion ---


class TestSuggestStopwordsByFrequency:
    def test_extracts_high_df_korean_tokens(self):
        samples = [
            "분류번호 001 문서제목 인권영향평가",
            "분류번호 002 문서제목 운영계획",
            "분류번호 003 문서제목 예산집행",
        ]
        stopwords = suggest_stopwords_by_frequency(samples, "ko")
        assert "분류번호" in stopwords
        assert "문서제목" in stopwords

    def test_ignores_low_df_tokens(self):
        samples = [
            "unique content one specific term apple",
            "unique content two specific term banana",
            "unique content three specific term cherry",
        ]
        stopwords = suggest_stopwords_by_frequency(samples, "en")
        # "unique", "content", "specific", "term" appear in every doc
        assert "unique" in stopwords
        # "apple" appears only once → not a stopword
        assert "apple" not in stopwords

    def test_empty_samples_returns_empty(self):
        assert suggest_stopwords_by_frequency([], "ko") == []


# --- JSON extraction helper ---


class TestUniquePreserveOrder:
    def test_preserves_first_seen_order(self):
        assert _unique_preserve_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]

    def test_strips_and_drops_empties(self):
        assert _unique_preserve_order(["  x  ", "", None, "y"]) == ["x", "y"]

    def test_empty_input_returns_empty(self):
        assert _unique_preserve_order([]) == []


class TestExtractJsonObject:
    def test_plain_json_returned_as_is(self):
        s = '{"a": 1, "b": 2}'
        assert _extract_json_object(s) == s

    def test_strips_markdown_fences(self):
        s = '```json\n{"a": 1}\n```'
        assert _extract_json_object(s) == '{"a": 1}'

    def test_strips_bare_fences(self):
        s = '```\n{"a": 1}\n```'
        assert _extract_json_object(s) == '{"a": 1}'

    def test_carves_json_out_of_prose(self):
        s = 'Here is your profile:\n{"locale": "ko", "stopwords_extra": []}'
        out = _extract_json_object(s)
        assert out == '{"locale": "ko", "stopwords_extra": []}'

    def test_handles_nested_braces(self):
        s = 'prefix {"a": {"nested": 1}, "b": 2} suffix'
        out = _extract_json_object(s)
        assert out == '{"a": {"nested": 1}, "b": 2}'

    def test_empty_input(self):
        assert _extract_json_object("") == ""


# --- Rule-based generation path (no LLM) ---


@pytest.mark.asyncio
class TestRuleBasedGeneration:
    async def test_korean_corpus_produces_ko_profile(self):
        gen = ProfileGenerator(llm=None)
        samples = [
            "한국마사회 인권영향평가 결과보고서",
            "한국마사회 운영계획 수립 문서",
            "한국마사회 예산집행 현황 보고",
        ]
        profile = await gen.generate(name="test_ko", samples=samples)
        assert profile.name == "test_ko"
        assert profile.locale == "ko"
        assert "한국마사회" in profile.stopwords_extra  # DF=3/3
        # No LLM → no ontology hints, no patterns
        assert profile.ontology_hints == {}
        assert profile.metadata_strip_patterns == ()

    async def test_english_corpus_produces_en_profile(self):
        gen = ProfileGenerator(llm=None)
        samples = [
            "Research paper section one methodology",
            "Research paper section two results",
            "Research paper section three discussion",
        ]
        profile = await gen.generate(name="test_en", samples=samples)
        assert profile.locale == "en"
        assert "research" in profile.stopwords_extra
        assert "paper" in profile.stopwords_extra

    async def test_empty_samples_returns_generic_profile(self):
        gen = ProfileGenerator(llm=None)
        profile = await gen.generate(name="empty", samples=[])
        assert profile.name == "empty"
        assert profile.locale == "multi"
        assert profile.stopwords_extra == frozenset()


# --- LLM path (mocked) ---


class _MockLLM:
    """Test double — returns a canned JSON response without network calls."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        return self._response


@pytest.mark.asyncio
class TestLLMGeneration:
    async def test_llm_response_merged_into_profile(self):
        llm_response = json.dumps(
            {
                "locale": "ko",
                "stopwords_extra": ["사업명", "부서장"],
                "ontology_hints": {
                    "규정 및 지침": "RULE",
                    "운영계획": "DECISION",
                },
                "metadata_strip_patterns": ["<header>.*?</header>"],
                "reference_patterns": ["(.+?)에 따라"],
                "entity_hint_patterns": ["\\(주\\)[가-힣]+"],
                "rationale": "Korean compliance docs",
            }
        )
        gen = ProfileGenerator(llm=_MockLLM(llm_response))
        profile = await gen.generate(
            name="test_llm",
            samples=["한국마사회 규정 문서 분류번호 001"] * 5,
            categories=["규정 및 지침", "운영계획"],
        )
        assert profile.locale == "ko"
        # Rule-based + LLM stopwords merged
        assert "사업명" in profile.stopwords_extra
        assert "부서장" in profile.stopwords_extra
        # Ontology hints from LLM
        assert profile.ontology_hints["규정 및 지침"] == NodeKind.RULE
        assert profile.ontology_hints["운영계획"] == NodeKind.DECISION
        # Patterns compiled
        assert len(profile.metadata_strip_patterns) == 1
        assert len(profile.reference_patterns) == 1
        assert len(profile.entity_hint_patterns) == 1

    async def test_malformed_llm_json_falls_back_to_rule_based(self):
        gen = ProfileGenerator(llm=_MockLLM("this is not json"))
        profile = await gen.generate(
            name="test_bad",
            samples=["한국어 샘플 문서"] * 3,
        )
        # Still a valid profile — LLM malformation must not crash
        assert profile.name == "test_bad"
        assert profile.locale == "ko"
        assert profile.ontology_hints == {}

    async def test_invalid_regex_dropped(self):
        llm_response = json.dumps(
            {
                "locale": "en",
                "reference_patterns": ["valid.*pattern", "[unclosed"],
            }
        )
        gen = ProfileGenerator(llm=_MockLLM(llm_response))
        profile = await gen.generate(name="test_regex", samples=["text"] * 3)
        assert len(profile.reference_patterns) == 1
        assert profile.reference_patterns[0].pattern == "valid.*pattern"

    async def test_unknown_node_kind_dropped(self):
        llm_response = json.dumps(
            {
                "locale": "ko",
                "ontology_hints": {
                    "valid_cat": "RULE",
                    "invalid_cat": "NOT_A_KIND",
                },
            }
        )
        gen = ProfileGenerator(llm=_MockLLM(llm_response))
        profile = await gen.generate(name="test_kind", samples=["샘플"] * 3)
        assert "valid_cat" in profile.ontology_hints
        assert "invalid_cat" not in profile.ontology_hints

    async def test_llm_exception_downgrades_gracefully(self):
        class _FailingLLM:
            async def generate(self, **kwargs):
                raise RuntimeError("connection refused")

        gen = ProfileGenerator(llm=_FailingLLM())
        profile = await gen.generate(name="test_fail", samples=["샘플"] * 3)
        # Rule-based tier still produced a profile
        assert profile.name == "test_fail"
        assert profile.locale == "ko"


# --- Classifier tier integration ---


class _StubEmbedder:
    """Toy embedder — assigns each known label its own basis vector.

    Mirrors :class:`_FakeEmbedder` in ``test_ontology_classifier`` but is
    local to this module so the profile-generator tests can evolve
    independently. The production code doesn't care which embedder is
    injected — it only relies on the ``EmbeddingProvider`` protocol.
    """

    def __init__(self, label_to_kind: dict[str, NodeKind]) -> None:
        self._label_to_kind = label_to_kind
        self._kinds = list(DEFAULT_NODE_KIND_DESCRIPTIONS.keys())

    def _one_hot(self, kind: NodeKind) -> list[float]:
        vec = [0.0] * len(self._kinds)
        vec[self._kinds.index(kind)] = 1.0
        return vec

    async def embed(self, text: str) -> list[float]:
        for kind, desc in DEFAULT_NODE_KIND_DESCRIPTIONS.items():
            if text == desc:
                return self._one_hot(kind)
        if text in self._label_to_kind:
            return self._one_hot(self._label_to_kind[text])
        return [0.0] * len(self._kinds)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.mark.asyncio
class TestClassifierTier:
    async def test_classifier_fills_ontology_hints_without_llm(self):
        embedder = _StubEmbedder(
            {
                "규정 및 지침": NodeKind.RULE,
                "운영계획": NodeKind.DECISION,
                "조사 및 평가": NodeKind.OBSERVATION,
            }
        )
        classifier = OntologyClassifier(embedder=embedder)
        gen = ProfileGenerator(classifier=classifier, llm=None)

        profile = await gen.generate(
            name="krra_classifier",
            samples=["한국 공공기관 규정 문서 샘플"] * 4,
            categories=[
                "규정 및 지침",
                "운영계획",
                "규정 및 지침",
                "조사 및 평가",
            ],
        )
        assert profile.ontology_hints == {
            "규정 및 지침": NodeKind.RULE,
            "운영계획": NodeKind.DECISION,
            "조사 및 평가": NodeKind.OBSERVATION,
        }

    async def test_classifier_unknown_labels_are_skipped(self):
        embedder = _StubEmbedder({"잘 아는 카테고리": NodeKind.RULE})
        classifier = OntologyClassifier(embedder=embedder)
        gen = ProfileGenerator(classifier=classifier, llm=None)

        profile = await gen.generate(
            name="mixed",
            samples=["샘플 문서"] * 3,
            categories=["잘 아는 카테고리", "전혀 모르는 것"],
        )
        assert profile.ontology_hints == {"잘 아는 카테고리": NodeKind.RULE}

    async def test_classifier_and_llm_merge_preserves_classifier_hints(self):
        embedder = _StubEmbedder({"규정 및 지침": NodeKind.RULE})
        classifier = OntologyClassifier(embedder=embedder)

        # LLM disagrees — wants to relabel "규정 및 지침" as DECISION and
        # adds a second label "신규 카테고리" that the classifier missed.
        import json
        llm_payload = json.dumps(
            {
                "ontology_hints": {
                    "규정 및 지침": "DECISION",
                    "신규 카테고리": "CONCEPT",
                },
            }
        )
        gen = ProfileGenerator(
            classifier=classifier,
            llm=_MockLLM(llm_payload),
        )
        profile = await gen.generate(
            name="merge_test",
            samples=["샘플 문서"] * 3,
            categories=["규정 및 지침", "신규 카테고리"],
        )
        # Classifier win on "규정 및 지침" — RULE, not DECISION
        assert profile.ontology_hints["규정 및 지침"] == NodeKind.RULE
        # LLM fills in the category classifier didn't know
        assert profile.ontology_hints["신규 카테고리"] == NodeKind.CONCEPT

    async def test_no_categories_means_no_classifier_calls(self):
        class _FailIfCalled:
            async def embed(self, text):
                raise AssertionError("embedder must not run without categories")

            async def embed_batch(self, texts):
                raise AssertionError("embedder must not run without categories")

        classifier = OntologyClassifier(embedder=_FailIfCalled())
        gen = ProfileGenerator(classifier=classifier, llm=None)

        profile = await gen.generate(name="no_cats", samples=["샘플"] * 3)
        assert profile.ontology_hints == {}


# --- Round-trip via save/load ---


@pytest.mark.asyncio
class TestGeneratedProfileRoundTrip:
    async def test_rule_based_profile_survives_toml_roundtrip(self, tmp_path: Path):
        gen = ProfileGenerator(llm=None)
        profile = await gen.generate(
            name="roundtrip_test",
            samples=["한국마사회 예산편성 지침 문서"] * 4,
        )
        out = tmp_path / "roundtrip.toml"
        profile.save(out)
        reloaded = DomainProfile.load(out)
        assert reloaded.name == profile.name
        assert reloaded.locale == profile.locale
        assert reloaded.stopwords_extra == profile.stopwords_extra

    async def test_llm_profile_survives_toml_roundtrip(self, tmp_path: Path):
        llm_response = json.dumps(
            {
                "locale": "ko",
                "stopwords_extra": ["header", "footer"],
                "ontology_hints": {"rules": "RULE"},
                "reference_patterns": ["(.+?)에 따라"],
            }
        )
        gen = ProfileGenerator(llm=_MockLLM(llm_response))
        profile = await gen.generate(name="llm_roundtrip", samples=["샘플"] * 3)
        out = tmp_path / "llm_roundtrip.toml"
        profile.save(out)
        reloaded = DomainProfile.load(out)
        assert reloaded.ontology_hints == profile.ontology_hints
        assert len(reloaded.reference_patterns) == 1
