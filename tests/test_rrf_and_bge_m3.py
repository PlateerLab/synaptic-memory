"""Tests for RRF fusion, BGE-M3 provider, and ColBERT reranker."""

from synaptic.models import ActivatedNode, HybridEmbedding, Node, NodeKind
from synaptic.search import _rrf_fusion

# --- RRF Fusion ---


class TestRRFFusion:
    def test_single_ranking(self):
        ranking = {"a": 0.9, "b": 0.7, "c": 0.5}
        result = _rrf_fusion(ranking)
        # a is rank 0, b is rank 1, c is rank 2 (k=60)
        assert result["a"] > result["b"] > result["c"]

    def test_two_rankings_boost_overlap(self):
        r1 = {"a": 0.9, "b": 0.7}
        r2 = {"a": 0.8, "c": 0.6}
        result = _rrf_fusion(r1, r2)
        # "a" appears in both rankings → should have highest RRF score
        assert result["a"] > result["b"]
        assert result["a"] > result["c"]

    def test_disjoint_rankings(self):
        r1 = {"a": 0.9, "b": 0.7}
        r2 = {"c": 0.8, "d": 0.6}
        result = _rrf_fusion(r1, r2)
        assert set(result.keys()) == {"a", "b", "c", "d"}

    def test_empty_ranking(self):
        result = _rrf_fusion({})
        assert result == {}

    def test_custom_k(self):
        r1 = {"a": 0.9, "b": 0.7}
        result_k1 = _rrf_fusion(r1, k=1)
        result_k60 = _rrf_fusion(r1, k=60)
        # With k=1, difference between ranks is larger
        diff_k1 = result_k1["a"] - result_k1["b"]
        diff_k60 = result_k60["a"] - result_k60["b"]
        assert diff_k1 > diff_k60

    def test_many_rankings(self):
        r1 = {"a": 0.9, "b": 0.5}
        r2 = {"a": 0.8, "c": 0.5}
        r3 = {"a": 0.7, "d": 0.5}
        result = _rrf_fusion(r1, r2, r3)
        # "a" appears in all 3 → highest score
        assert result["a"] > result["b"]
        assert result["a"] > result["c"]
        assert result["a"] > result["d"]


# --- HybridEmbedding ---


class TestHybridEmbedding:
    def test_create_default(self):
        h = HybridEmbedding()
        assert h.dense == []
        assert h.sparse == {}
        assert h.colbert is None

    def test_create_with_values(self):
        h = HybridEmbedding(
            dense=[0.1, 0.2],
            sparse={100: 0.5, 200: 0.8},
            colbert=[[0.1, 0.2], [0.3, 0.4]],
        )
        assert len(h.dense) == 2
        assert h.sparse[100] == 0.5
        assert len(h.colbert) == 2


# --- MockBGEM3Provider ---


class TestMockBGEM3:
    async def test_embed_returns_dense(self):
        from synaptic.extensions.embedder_bge_m3 import MockBGEM3Provider

        provider = MockBGEM3Provider(dim=4)
        dense = await provider.embed("test text")
        assert len(dense) == 4
        assert all(isinstance(v, float) for v in dense)

    async def test_embed_hybrid_returns_all_components(self):
        from synaptic.extensions.embedder_bge_m3 import MockBGEM3Provider

        provider = MockBGEM3Provider(dim=4)
        h = await provider.embed_hybrid("test text with words")
        assert len(h.dense) == 4
        assert len(h.sparse) > 0
        assert h.colbert is not None
        assert len(h.colbert) > 0

    async def test_embed_batch(self):
        from synaptic.extensions.embedder_bge_m3 import MockBGEM3Provider

        provider = MockBGEM3Provider(dim=4)
        results = await provider.embed_batch(["text1", "text2"])
        assert len(results) == 2
        assert len(results[0]) == 4

    async def test_embed_batch_hybrid(self):
        from synaptic.extensions.embedder_bge_m3 import MockBGEM3Provider

        provider = MockBGEM3Provider(dim=4)
        results = await provider.embed_batch_hybrid(["text1", "text2"])
        assert len(results) == 2
        assert all(isinstance(h, HybridEmbedding) for h in results)

    async def test_deterministic(self):
        from synaptic.extensions.embedder_bge_m3 import MockBGEM3Provider

        provider = MockBGEM3Provider(dim=4)
        h1 = await provider.embed_hybrid("same text")
        h2 = await provider.embed_hybrid("same text")
        assert h1.dense == h2.dense


# --- ColBERT Reranker ---


class TestColBERTReranker:
    def _make_activated(self, title: str, resonance: float) -> ActivatedNode:
        return ActivatedNode(
            node=Node(title=title, kind=NodeKind.CONCEPT),
            activation=0.5,
            resonance=resonance,
        )

    def test_rerank_with_colbert(self):
        from synaptic.extensions.reranker_colbert import ColBERTReranker

        reranker = ColBERTReranker()

        query_colbert = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        # Doc A: perfect match on first query token
        doc_a_colbert = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        # Doc B: perfect match on both query tokens
        doc_b_colbert = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        a = self._make_activated("A", 0.8)
        b = self._make_activated("B", 0.5)

        result = reranker.rerank(
            query_colbert,
            [(a, doc_a_colbert), (b, doc_b_colbert)],
            top_k=2,
        )

        # B should rank higher (matches both query tokens)
        assert result[0].node.title == "B"

    def test_rerank_empty(self):
        from synaptic.extensions.reranker_colbert import ColBERTReranker

        reranker = ColBERTReranker()
        result = reranker.rerank([], [], top_k=5)
        assert result == []

    def test_rerank_no_colbert_preserves_order(self):
        from synaptic.extensions.reranker_colbert import ColBERTReranker

        reranker = ColBERTReranker()
        a = self._make_activated("A", 0.9)
        b = self._make_activated("B", 0.5)

        result = reranker.rerank(
            [[1.0, 0.0]],
            [(a, []), (b, [])],  # no ColBERT vectors
            top_k=2,
        )

        # Should preserve original resonance order
        assert result[0].node.title == "A"

    def test_rerank_top_k_limit(self):
        from synaptic.extensions.reranker_colbert import ColBERTReranker

        reranker = ColBERTReranker()
        candidates = [(self._make_activated(f"Node{i}", 0.5), [[0.1, 0.2]]) for i in range(10)]

        result = reranker.rerank([[0.1, 0.2]], candidates, top_k=3)
        assert len(result) == 3
