"""Tests for extensions — tagger, rewriter, embedder."""

from __future__ import annotations

from synaptic.extensions.embedder import MockEmbeddingProvider
from synaptic.extensions.rewriter import StaticQueryRewriter
from synaptic.extensions.tagger_regex import RegexTagExtractor


class TestRegexTagExtractor:
    def test_extract_tech_terms(self) -> None:
        extractor = RegexTagExtractor()
        tags = extractor.extract("Deploy the API to production with CI/CD pipeline")
        assert "deploy" in tags
        assert "api" in tags

    def test_extract_korean(self) -> None:
        extractor = RegexTagExtractor()
        tags = extractor.extract("배포 자동화 및 성능 최적화")
        assert "deploy" in tags
        assert "performance" in tags

    def test_no_match(self) -> None:
        extractor = RegexTagExtractor()
        tags = extractor.extract("hello world nothing special")
        assert tags == []

    def test_multiple_matches(self) -> None:
        extractor = RegexTagExtractor()
        tags = extractor.extract("Fix security bug in API auth and run test")
        assert "security" in tags
        assert "bug" in tags
        assert "api" in tags
        assert "test" in tags

    def test_custom_patterns(self) -> None:
        import re  # noqa: PLC0415

        extra = [("custom", re.compile(r"\bcustom_tag\b"))]
        extractor = RegexTagExtractor(extra_patterns=extra)
        tags = extractor.extract("This has a custom_tag in it")
        assert "custom" in tags


class TestStaticQueryRewriter:
    async def test_known_query(self) -> None:
        rewriter = StaticQueryRewriter({"배포": ["deploy", "릴리즈"]})
        result = await rewriter.rewrite("배포")
        assert result == ["deploy", "릴리즈"]

    async def test_unknown_query(self) -> None:
        rewriter = StaticQueryRewriter()
        result = await rewriter.rewrite("unknown")
        assert result == []


class TestMockEmbeddingProvider:
    async def test_embed(self) -> None:
        provider = MockEmbeddingProvider(dim=4)
        vec = await provider.embed("hello")
        assert len(vec) == 4
        assert all(0.0 <= v <= 1.0 for v in vec)

    async def test_deterministic(self) -> None:
        provider = MockEmbeddingProvider(dim=4)
        v1 = await provider.embed("hello")
        v2 = await provider.embed("hello")
        assert v1 == v2

    async def test_different_texts(self) -> None:
        provider = MockEmbeddingProvider(dim=4)
        v1 = await provider.embed("hello")
        v2 = await provider.embed("world")
        assert v1 != v2

    async def test_batch(self) -> None:
        provider = MockEmbeddingProvider(dim=4)
        vecs = await provider.embed_batch(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(len(v) == 4 for v in vecs)
