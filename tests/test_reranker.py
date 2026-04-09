"""Tests for Reranker — NoOp and LLM reranking."""

from synaptic import NodeKind, SynapticGraph
from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.reranker import LLMReranker, NoOpReranker
from synaptic.models import ActivatedNode, Node


def _make_activated(title: str, resonance: float) -> ActivatedNode:
    return ActivatedNode(
        node=Node(title=title, content=f"content about {title}", kind=NodeKind.CONCEPT),
        activation=0.5,
        resonance=resonance,
    )


class TestNoOpReranker:
    async def test_passthrough(self):
        reranker = NoOpReranker()
        candidates = [
            _make_activated("A", 0.9),
            _make_activated("B", 0.7),
            _make_activated("C", 0.5),
        ]

        result = await reranker.rerank("query", candidates, top_k=10)
        assert len(result) == 3
        assert result[0].node.title == "A"

    async def test_top_k_limit(self):
        reranker = NoOpReranker()
        candidates = [_make_activated(f"N{i}", 0.5) for i in range(10)]

        result = await reranker.rerank("query", candidates, top_k=3)
        assert len(result) == 3

    async def test_empty(self):
        reranker = NoOpReranker()
        result = await reranker.rerank("query", [], top_k=5)
        assert result == []


class TestLLMReranker:
    """Test LLM reranker with a mock LLM provider."""

    class MockLLM:
        """Returns JSON scores that swap the order of candidates."""

        async def generate(self, *, system: str, user: str, max_tokens: int) -> str:
            # Always rank index 1 highest, index 0 lowest
            return '[{"index": 0, "score": 2}, {"index": 1, "score": 9}, {"index": 2, "score": 5}]'

    class BrokenLLM:
        """Returns invalid JSON."""

        async def generate(self, *, system: str, user: str, max_tokens: int) -> str:
            return "not json at all"

    async def test_rerank_reorders(self):
        reranker = LLMReranker(self.MockLLM(), max_candidates=3)
        candidates = [
            _make_activated("A", 0.9),  # index 0 → LLM score 2
            _make_activated("B", 0.3),  # index 1 → LLM score 9
            _make_activated("C", 0.5),  # index 2 → LLM score 5
        ]

        result = await reranker.rerank("test query", candidates, top_k=3)
        # B should now be first (LLM scored it 9)
        assert result[0].node.title == "B"

    async def test_rerank_blends_scores(self):
        reranker = LLMReranker(self.MockLLM(), max_candidates=3)
        candidates = [_make_activated("A", 0.5)]

        result = await reranker.rerank("query", candidates, top_k=1)
        # Score should be blended: 0.6 * (2/10) + 0.4 * 0.5 = 0.32
        assert 0.0 < result[0].resonance < 1.0

    async def test_broken_llm_fallback(self):
        reranker = LLMReranker(self.BrokenLLM(), max_candidates=3)
        candidates = [
            _make_activated("A", 0.9),
            _make_activated("B", 0.7),
        ]

        result = await reranker.rerank("query", candidates, top_k=2)
        # Should fall back to original order
        assert len(result) == 2
        assert result[0].node.title == "A"

    async def test_max_candidates_limits_llm_calls(self):
        reranker = LLMReranker(self.MockLLM(), max_candidates=2)
        candidates = [
            _make_activated("A", 0.9),
            _make_activated("B", 0.7),
            _make_activated("C", 0.5),
            _make_activated("D", 0.3),
        ]

        result = await reranker.rerank("query", candidates, top_k=4)
        # All 4 should be returned, but only first 2 were reranked
        assert len(result) == 4

    async def test_empty_candidates(self):
        reranker = LLMReranker(self.MockLLM())
        result = await reranker.rerank("query", [], top_k=5)
        assert result == []


class TestRerankerIntegration:
    async def test_search_with_noop_reranker(self):
        graph = SynapticGraph(
            MemoryBackend(),
            reranker=NoOpReranker(),
        )
        await graph.add("Test Node", "content about databases")
        result = await graph.search("databases")
        assert "rerank" in result.stages_used

    async def test_search_without_reranker(self):
        graph = SynapticGraph.memory()
        await graph.add("Test Node", "content about databases")
        result = await graph.search("databases")
        assert "rerank" not in result.stages_used

    async def test_search_with_llm_reranker(self):

        class SimpleLLM:
            async def generate(self, *, system, user, max_tokens):
                return '[{"index": 0, "score": 8}]'

        graph = SynapticGraph(
            MemoryBackend(),
            reranker=LLMReranker(SimpleLLM(), max_candidates=5),
        )
        await graph.add("PostgreSQL", "relational database system")
        await graph.add("Redis", "in-memory cache")

        result = await graph.search("database")
        assert "rerank" in result.stages_used
