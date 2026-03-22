"""PhraseExtractor 단위 테스트 — HippoRAG2 dual-node KG phrase extraction."""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.phrase_extractor import PhraseExtractor, _is_meaningful, _normalize_phrase
from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind, NodeKind


# ---------------------------------------------------------------------------
# Helper normalization / meaningful tests
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalize_strips_whitespace(self) -> None:
        assert _normalize_phrase("  hello  ") == "hello"

    def test_normalize_nfc(self) -> None:
        # NFD composed 'é' → NFC
        assert _normalize_phrase("café") == "café"

    def test_meaningful_rejects_single_char(self) -> None:
        assert not _is_meaningful("A")

    def test_meaningful_rejects_digits_only(self) -> None:
        assert not _is_meaningful("12345")

    def test_meaningful_rejects_stop_words(self) -> None:
        assert not _is_meaningful("the and or")

    def test_meaningful_accepts_proper_noun(self) -> None:
        assert _is_meaningful("Germany")

    def test_meaningful_accepts_mixed(self) -> None:
        assert _is_meaningful("the Berlin")


# ---------------------------------------------------------------------------
# Phrase extraction (regex-based)
# ---------------------------------------------------------------------------


class TestPhraseExtraction:
    def setup_method(self) -> None:
        self.extractor = PhraseExtractor(max_phrases_per_node=10)

    def test_proper_noun_multi_word(self) -> None:
        phrases = self.extractor._extract_phrases(
            "Overview",
            "Lomonosov Moscow State University was founded in 1755.",
        )
        # Multi-word proper nouns or individual proper nouns should be extracted
        lowered = [p.lower() for p in phrases]
        assert any("lomonosov" in p for p in lowered) or any("moscow" in p for p in lowered)

    def test_single_proper_noun(self) -> None:
        phrases = self.extractor._extract_phrases(
            "City",
            "Bonn is a city in Germany on the Rhine river.",
        )
        assert "Bonn" in phrases
        assert "Germany" in phrases
        assert "Rhine" in phrases

    def test_abbreviation(self) -> None:
        phrases = self.extractor._extract_phrases(
            "Tech",
            "The Application Programming Interface (API) is widely used.",
        )
        assert "API" in phrases

    def test_korean_quoted(self) -> None:
        phrases = self.extractor._extract_phrases(
            "회사",
            "「플래티어」는 이커머스 솔루션 기업이다.",
        )
        assert "플래티어" in phrases

    def test_korean_parens(self) -> None:
        phrases = self.extractor._extract_phrases(
            "재단",
            "(주)플래티어와 (재)한국재단이 협약을 맺었다.",
        )
        # Regex captures the word after (주)/(재) — may include trailing chars
        lowered = [p.lower() for p in phrases]
        assert any("플래티어" in p for p in lowered)
        assert any("한국재단" in p for p in lowered)

    def test_year_extraction_filtered_by_meaningful(self) -> None:
        """Years (digits only) are filtered out by _is_meaningful — this is intentional."""
        extractor = PhraseExtractor(max_phrases_per_node=20)
        phrases = extractor._extract_phrases(
            "History",
            "The university was established in 1755 and expanded in 2024.",
        )
        # Pure digit years are excluded by _is_meaningful (digits-only check)
        assert "1755" not in phrases
        assert "2024" not in phrases

    def test_title_included(self) -> None:
        phrases = self.extractor._extract_phrases(
            "Berlin",
            "Berlin is the capital of Germany.",
        )
        assert "Berlin" in phrases

    def test_deduplication(self) -> None:
        phrases = self.extractor._extract_phrases(
            "Berlin",
            "Berlin is great. Berlin is the capital. berlin is cool.",
        )
        count = sum(1 for p in phrases if p.lower() == "berlin")
        assert count == 1

    def test_max_phrases_limit(self) -> None:
        extractor = PhraseExtractor(max_phrases_per_node=3)
        phrases = extractor._extract_phrases(
            "Universities",
            "Cambridge University, Oxford University, Harvard University, MIT, Stanford University.",
        )
        assert len(phrases) <= 3

    def test_min_phrase_length(self) -> None:
        extractor = PhraseExtractor(min_phrase_length=4)
        phrases = extractor._extract_phrases(
            "Test",
            "The API and LLM are important. (IO) is too short.",
        )
        # "IO" is 2 chars, should be excluded
        assert "IO" not in phrases

    def test_stop_words_excluded(self) -> None:
        phrases = self.extractor._extract_phrases(
            "The",
            "The quick brown fox.",
        )
        # "The" alone is a stop word
        lowered = [p.lower() for p in phrases]
        assert "the" not in lowered

    def test_empty_content(self) -> None:
        phrases = self.extractor._extract_phrases("", "")
        assert phrases == []


# ---------------------------------------------------------------------------
# Graph integration (extract_and_link)
# ---------------------------------------------------------------------------


class TestExtractAndLink:
    @pytest.fixture
    async def graph(self) -> SynapticGraph:
        backend = MemoryBackend()
        await backend.connect()
        return SynapticGraph(backend)

    @pytest.mark.asyncio
    async def test_creates_phrase_nodes(self, graph: SynapticGraph) -> None:
        """Phrase extraction creates ENTITY nodes with _phrase tag."""
        extractor = PhraseExtractor(max_phrases_per_node=5)
        node = await graph.add("Bonn", "Bonn is a city in Germany.", kind=NodeKind.CONCEPT)

        phrase_ids = await extractor.extract_and_link(graph, node.id, "Bonn", "Bonn is a city in Germany.")

        assert len(phrase_ids) > 0
        for pid in phrase_ids:
            phrase_node = await graph.backend.get_node(pid)
            assert phrase_node is not None
            assert phrase_node.kind == NodeKind.ENTITY
            assert "_phrase" in phrase_node.tags

    @pytest.mark.asyncio
    async def test_contains_edge_created(self, graph: SynapticGraph) -> None:
        """CONTAINS edge links passage to phrase."""
        extractor = PhraseExtractor(max_phrases_per_node=5)
        node = await graph.add("Berlin", "Berlin is the capital.", kind=NodeKind.CONCEPT)

        phrase_ids = await extractor.extract_and_link(graph, node.id, "Berlin", "Berlin is the capital.")

        edges = await graph.backend.get_edges(node.id)
        contains_edges = [e for e in edges if e.kind == EdgeKind.CONTAINS]
        assert len(contains_edges) > 0
        assert all(e.weight == 0.8 for e in contains_edges)

    @pytest.mark.asyncio
    async def test_phrase_reuse_across_documents(self, graph: SynapticGraph) -> None:
        """Same phrase from different documents reuses the same phrase node."""
        extractor = PhraseExtractor(max_phrases_per_node=5)

        node1 = await graph.add("Doc1", "Germany is in Europe.", kind=NodeKind.CONCEPT)
        ids1 = await extractor.extract_and_link(graph, node1.id, "Doc1", "Germany is in Europe.")

        node2 = await graph.add("Doc2", "Germany has 83 million people.", kind=NodeKind.CONCEPT)
        ids2 = await extractor.extract_and_link(graph, node2.id, "Doc2", "Germany has 83 million people.")

        # "Germany" phrase node should be the same
        germany_ids_1 = [pid for pid in ids1 if (await graph.backend.get_node(pid)).title == "Germany"]
        germany_ids_2 = [pid for pid in ids2 if (await graph.backend.get_node(pid)).title == "Germany"]

        if germany_ids_1 and germany_ids_2:
            assert germany_ids_1[0] == germany_ids_2[0], "Same phrase should reuse same node"

    @pytest.mark.asyncio
    async def test_bridge_via_shared_phrase(self, graph: SynapticGraph) -> None:
        """Two documents sharing a phrase are indirectly connected via phrase node."""
        extractor = PhraseExtractor(max_phrases_per_node=5)

        node1 = await graph.add("University of Bonn", "University of Bonn is in Germany.", kind=NodeKind.CONCEPT)
        await extractor.extract_and_link(graph, node1.id, "University of Bonn", "University of Bonn is in Germany.")

        node2 = await graph.add("Bonn City", "Bonn is a city on the Rhine.", kind=NodeKind.CONCEPT)
        await extractor.extract_and_link(graph, node2.id, "Bonn City", "Bonn is a city on the Rhine.")

        # Both should have edges to the shared "Bonn" phrase node
        edges1 = await graph.backend.get_edges(node1.id)
        edges2 = await graph.backend.get_edges(node2.id)

        targets1 = {e.target_id for e in edges1 if e.kind == EdgeKind.CONTAINS}
        targets2 = {e.target_id for e in edges2 if e.kind == EdgeKind.CONTAINS}

        # They should share at least one phrase node (bridge)
        shared = targets1 & targets2
        assert len(shared) > 0, "Documents should share phrase nodes as bridges"

    @pytest.mark.asyncio
    async def test_empty_content_no_phrases(self, graph: SynapticGraph) -> None:
        """Empty content produces no phrase nodes."""
        extractor = PhraseExtractor(max_phrases_per_node=5)
        node = await graph.add("", "a b c", kind=NodeKind.CONCEPT)

        phrase_ids = await extractor.extract_and_link(graph, node.id, "", "a b c")
        assert phrase_ids == []

    @pytest.mark.asyncio
    async def test_phrase_nodes_are_created_with_empty_content(self, graph: SynapticGraph) -> None:
        """Phrase nodes have empty content to minimize FTS noise."""
        extractor = PhraseExtractor(max_phrases_per_node=5)

        node = await graph.add(
            "Munich Overview",
            "Munich is a city in Germany known for Oktoberfest.",
            kind=NodeKind.CONCEPT,
        )
        phrase_ids = await extractor.extract_and_link(
            graph, node.id,
            "Munich Overview",
            "Munich is a city in Germany known for Oktoberfest.",
        )

        for pid in phrase_ids:
            phrase_node = await graph.backend.get_node(pid)
            assert phrase_node.content == "", "Phrase nodes should have empty content"
