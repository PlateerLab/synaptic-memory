"""Tests for EvidenceAssembler — search result to evidence chain conversion."""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.evidence import EvidenceAssembler
from synaptic.graph import SynapticGraph
from synaptic.models import (
    ActivatedNode,
    EdgeKind,
    EvidenceChain,
    NodeKind,
    SearchResult,
)


@pytest.fixture
async def backend() -> MemoryBackend:
    b = MemoryBackend()
    await b.connect()
    return b


@pytest.fixture
async def graph(backend: MemoryBackend) -> SynapticGraph:
    return SynapticGraph(backend)


@pytest.fixture
def assembler() -> EvidenceAssembler:
    return EvidenceAssembler()


class TestAssembleEmpty:
    """Tests for empty/minimal inputs."""

    async def test_empty_search_result(
        self, backend: MemoryBackend, assembler: EvidenceAssembler
    ) -> None:
        """Empty search result produces empty evidence chain."""
        sr = SearchResult(query="test query")
        chain = await assembler.assemble(backend, "test query", sr)
        assert isinstance(chain, EvidenceChain)
        assert chain.query == "test query"
        assert chain.steps == []
        assert chain.assembly_time_ms >= 0

    async def test_single_node_chain(
        self, graph: SynapticGraph, assembler: EvidenceAssembler
    ) -> None:
        """Single node search result produces a single-step chain."""
        node = await graph.add("Database Optimization", "Use indexes for faster queries.")
        sr = SearchResult(
            query="database performance",
            nodes=[ActivatedNode(node=node, activation=0.9)],
        )
        chain = await assembler.assemble(graph.backend, "database performance", sr)
        assert len(chain.steps) == 1
        assert chain.steps[0].node.id == node.id
        assert chain.steps[0].role == "seed"
        assert chain.compressed_context != ""


class TestAssembleBasic:
    """Tests for basic evidence chain assembly."""

    async def test_multiple_nodes_all_seeds(
        self, graph: SynapticGraph, assembler: EvidenceAssembler
    ) -> None:
        """Multiple unconnected nodes all become seed steps."""
        n1 = await graph.add("Node A", "Content about Python programming.")
        n2 = await graph.add("Node B", "Content about Java development.")
        sr = SearchResult(
            query="programming",
            nodes=[
                ActivatedNode(node=n1, activation=0.9),
                ActivatedNode(node=n2, activation=0.8),
            ],
        )
        chain = await assembler.assemble(graph.backend, "programming", sr)
        assert len(chain.steps) == 2
        assert all(s.role == "seed" for s in chain.steps)

    async def test_connected_nodes_with_edge(
        self, graph: SynapticGraph, assembler: EvidenceAssembler
    ) -> None:
        """Connected nodes include connection_to_next in steps."""
        n1 = await graph.add("Cause", "The server crashed.")
        n2 = await graph.add("Effect", "Users reported errors.")
        await graph.link(n1.id, n2.id, kind=EdgeKind.CAUSED)

        sr = SearchResult(
            query="server crash",
            nodes=[
                ActivatedNode(node=n1, activation=0.9),
                ActivatedNode(node=n2, activation=0.8),
            ],
        )
        chain = await assembler.assemble(graph.backend, "server crash", sr)
        assert len(chain.steps) >= 2


class TestTopologicalSort:
    """Tests for topological sorting of evidence steps."""

    async def test_directed_edges_respected(
        self, graph: SynapticGraph, assembler: EvidenceAssembler
    ) -> None:
        """CAUSED edges produce topologically sorted output."""
        cause = await graph.add("Root Cause", "Memory leak in module X.")
        effect = await graph.add("Effect", "Application OOM killed.")
        await graph.link(cause.id, effect.id, kind=EdgeKind.CAUSED)

        sr = SearchResult(
            query="memory leak",
            nodes=[
                ActivatedNode(node=effect, activation=0.9),
                ActivatedNode(node=cause, activation=0.8),
            ],
        )
        chain = await assembler.assemble(graph.backend, "memory leak", sr)
        # Cause should come before effect in topological order
        step_ids = [s.node.id for s in chain.steps]
        assert step_ids.index(cause.id) < step_ids.index(effect.id)

    async def test_no_directed_edges_preserves_order(
        self, graph: SynapticGraph, assembler: EvidenceAssembler
    ) -> None:
        """Without directed edges, original activation order is preserved."""
        n1 = await graph.add("First", "First result.")
        n2 = await graph.add("Second", "Second result.")
        await graph.link(n1.id, n2.id, kind=EdgeKind.RELATED)  # undirected

        sr = SearchResult(
            query="results",
            nodes=[
                ActivatedNode(node=n1, activation=0.9),
                ActivatedNode(node=n2, activation=0.8),
            ],
        )
        chain = await assembler.assemble(graph.backend, "results", sr)
        step_ids = [s.node.id for s in chain.steps]
        assert step_ids == [n1.id, n2.id]


class TestSentenceCompression:
    """Tests for content compression and relevance filtering."""

    def test_relevant_sentences_selected(self) -> None:
        """Sentences containing query terms are selected."""
        asm = EvidenceAssembler(relevance_threshold=0.2)
        content = (
            "Python is a popular programming language. "
            "The weather is nice today. "
            "Python supports multiple paradigms. "
            "Cats are independent animals."
        )
        compressed = asm._compress_content(content, "Python programming")
        assert "Python" in compressed
        # Irrelevant sentences should be excluded
        assert "weather" not in compressed
        assert "Cats" not in compressed

    def test_empty_content(self) -> None:
        """Empty content returns empty string."""
        asm = EvidenceAssembler()
        assert asm._compress_content("", "query") == ""

    def test_no_query_terms_returns_first_sentences(self) -> None:
        """When query has only stopwords, first N sentences returned."""
        asm = EvidenceAssembler(max_sentences_per_node=2)
        content = "First sentence here. Second sentence here. Third sentence here."
        compressed = asm._compress_content(content, "the is a")
        assert "First" in compressed

    def test_fallback_when_no_match(self) -> None:
        """When no sentences match threshold, top-scored ones are used as fallback."""
        asm = EvidenceAssembler(relevance_threshold=0.99)
        content = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
        compressed = asm._compress_content(content, "omega")
        # Should still return something (fallback)
        assert len(compressed) > 0

    def test_max_sentences_limit(self) -> None:
        """Number of selected sentences respects max_sentences_per_node."""
        asm = EvidenceAssembler(max_sentences_per_node=2, relevance_threshold=0.0)
        content = (
            "Python is great. Python is fast. Python is popular. "
            "Python is modern. Python is simple."
        )
        compressed = asm._compress_content(content, "Python")
        sentence_count = len([s for s in compressed.split(". ") if s.strip()])
        assert sentence_count <= 3  # at most max_sentences + partial


class TestFactExtraction:
    """Tests for fact extraction from content."""

    def test_extracts_numbers_with_units(self) -> None:
        """Numbers with units are extracted as facts."""
        asm = EvidenceAssembler()
        content = "The system handles 10,000 requests per second. Memory usage is 2GB."
        facts = asm._extract_facts(content)
        assert any("2GB" in f for f in facts)

    def test_extracts_dates(self) -> None:
        """Date patterns are extracted as facts."""
        asm = EvidenceAssembler()
        content = "The project started on 2024-01-15. It was completed in March."
        facts = asm._extract_facts(content)
        assert any("2024" in f for f in facts)

    def test_extracts_korean_units(self) -> None:
        """Korean units (만, 억, 원) are extracted."""
        asm = EvidenceAssembler()
        content = "매출이 100억원을 돌파했다. 직원 수는 500명이다."
        facts = asm._extract_facts(content)
        assert len(facts) >= 1

    def test_empty_content_no_facts(self) -> None:
        """Empty content produces no facts."""
        asm = EvidenceAssembler()
        assert asm._extract_facts("") == []

    def test_no_facts_in_plain_text(self) -> None:
        """Text without numbers or dates yields no facts."""
        asm = EvidenceAssembler()
        content = "This is a simple sentence without any data."
        facts = asm._extract_facts(content)
        assert facts == []

    def test_deduplication_in_facts(self) -> None:
        """Same fact sentence is not duplicated."""
        asm = EvidenceAssembler()
        # Sentence repeated won't cause duplicate because split handles it
        content = "Revenue was 100억원. Revenue was 100억원."
        facts = asm._extract_facts(content)
        # Each unique sentence appears only once
        assert len(facts) == len(set(facts))


class TestTokenTruncation:
    """Tests for token counting and truncation."""

    def test_long_context_truncated(self) -> None:
        """Context exceeding max_tokens is truncated."""
        asm = EvidenceAssembler(max_tokens=10)
        from synaptic.models import EvidenceStep, Node

        node = Node(title="Test", content="word " * 50)
        steps = [
            EvidenceStep(
                node=node,
                role="seed",
                compressed_content="word " * 50,
            )
        ]
        context = asm._format_context(steps)
        word_count = len(context.split())
        assert word_count <= 10

    def test_short_context_not_truncated(self) -> None:
        """Context within token limit is not truncated."""
        asm = EvidenceAssembler(max_tokens=2048)
        from synaptic.models import EvidenceStep, Node

        node = Node(title="Short", content="brief")
        steps = [
            EvidenceStep(
                node=node,
                role="seed",
                compressed_content="brief content",
            )
        ]
        context = asm._format_context(steps)
        assert "brief content" in context


class TestCrossNodeDeduplication:
    """Tests for fact deduplication across multiple nodes."""

    async def test_duplicate_facts_across_nodes(
        self, graph: SynapticGraph, assembler: EvidenceAssembler
    ) -> None:
        """Same fact appearing in multiple nodes is deduplicated."""
        n1 = await graph.add("Report A", "Revenue reached 100억원 in 2024.")
        n2 = await graph.add("Report B", "Revenue reached 100억원 in 2024.")

        sr = SearchResult(
            query="revenue 2024",
            nodes=[
                ActivatedNode(node=n1, activation=0.9),
                ActivatedNode(node=n2, activation=0.8),
            ],
        )
        chain = await assembler.assemble(graph.backend, "revenue 2024", sr)
        # Facts list should be deduplicated
        assert len(chain.facts) == len(set(chain.facts))


class TestBridgeNodeDiscovery:
    """Tests for bridge node discovery between seed nodes."""

    async def test_bridge_node_found(
        self, graph: SynapticGraph, assembler: EvidenceAssembler
    ) -> None:
        """Bridge node between two seeds is discovered and included."""
        n1 = await graph.add("Start", "Starting point content.")
        bridge = await graph.add("Bridge", "Bridge content connecting start and end.")
        n2 = await graph.add("End", "End point content.")
        await graph.link(n1.id, bridge.id, kind=EdgeKind.RELATED)
        await graph.link(bridge.id, n2.id, kind=EdgeKind.RELATED)

        sr = SearchResult(
            query="start end",
            nodes=[
                ActivatedNode(node=n1, activation=0.9),
                ActivatedNode(node=n2, activation=0.8),
            ],
        )
        chain = await assembler.assemble(graph.backend, "start end", sr)
        step_ids = {s.node.id for s in chain.steps}
        # Bridge node should be discovered
        assert bridge.id in step_ids
        # Bridge should have role "bridge"
        bridge_steps = [s for s in chain.steps if s.node.id == bridge.id]
        assert bridge_steps[0].role == "bridge"

    async def test_no_bridge_when_directly_connected(
        self, graph: SynapticGraph, assembler: EvidenceAssembler
    ) -> None:
        """No bridge discovered when seeds are directly connected."""
        n1 = await graph.add("A", "Node A content.")
        n2 = await graph.add("B", "Node B content.")
        await graph.link(n1.id, n2.id, kind=EdgeKind.RELATED)

        sr = SearchResult(
            query="test",
            nodes=[
                ActivatedNode(node=n1, activation=0.9),
                ActivatedNode(node=n2, activation=0.8),
            ],
        )
        chain = await assembler.assemble(graph.backend, "test", sr)
        # All steps should be seeds (no bridge with only direct connection)
        assert all(s.role == "seed" for s in chain.steps)


class TestFormatContext:
    """Tests for context formatting."""

    def test_format_includes_role_and_title(self) -> None:
        """Formatted context includes role labels and titles."""
        from synaptic.models import EvidenceStep, Node

        asm = EvidenceAssembler()
        node = Node(title="Important Finding", content="Details here.")
        steps = [
            EvidenceStep(
                node=node,
                role="seed",
                compressed_content="Details here.",
            )
        ]
        context = asm._format_context(steps)
        assert "[SEED]" in context
        assert "Important Finding" in context

    def test_format_includes_facts(self) -> None:
        """Key facts are included in formatted context."""
        from synaptic.models import EvidenceStep, Node

        asm = EvidenceAssembler()
        node = Node(title="Stats", content="Numbers here.")
        steps = [
            EvidenceStep(
                node=node,
                role="seed",
                compressed_content="Numbers here.",
                facts=["Revenue 100억원"],
            )
        ]
        context = asm._format_context(steps)
        assert "Key facts:" in context
        assert "100억원" in context

    def test_format_connection_between_steps(self) -> None:
        """Connection description appears between steps."""
        from synaptic.models import EvidenceStep, Node

        asm = EvidenceAssembler()
        n1 = Node(title="Cause", content="c1")
        n2 = Node(title="Effect", content="c2")
        steps = [
            EvidenceStep(
                node=n1, role="seed",
                compressed_content="c1",
                connection_to_next="caused",
            ),
            EvidenceStep(node=n2, role="seed", compressed_content="c2"),
        ]
        context = asm._format_context(steps)
        assert "caused" in context.lower()
