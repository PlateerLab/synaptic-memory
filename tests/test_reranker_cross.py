"""Tests for the URL-backed cross-encoder rerankers (reranker_cross)."""

import pytest

from synaptic.extensions.reranker_cross import (
    OllamaReranker,
    TEIReranker,
    VLLMReranker,
    reranker_from_url,
)


class TestVLLMReranker:
    def test_bare_host_url(self):
        r = VLLMReranker("http://localhost:8000")
        assert r._url == "http://localhost:8000/rerank"

    def test_openai_style_url(self):
        # vLLM also serves /v1/rerank — a /v1 root must be preserved.
        r = VLLMReranker("http://localhost:8000/v1")
        assert r._url == "http://localhost:8000/v1/rerank"

    def test_trailing_slash_stripped(self):
        assert VLLMReranker("http://h:8000/")._url == "http://h:8000/rerank"

    async def test_empty_documents_short_circuit(self):
        # No network call when there is nothing to score.
        scores = await VLLMReranker("http://unreachable:9").rerank("q", [])
        assert scores == []


class TestTEIReranker:
    async def test_empty_documents_short_circuit(self):
        scores = await TEIReranker("http://unreachable:9").rerank("q", [])
        assert scores == []

    async def test_chunks_large_batches_and_preserves_order(self):
        class _FakeTEI(TEIReranker):
            __slots__ = ("chunks",)

            def __init__(self) -> None:
                super().__init__("http://unused", max_batch_size=2)
                self.chunks: list[list[str]] = []

            async def _rerank_chunk(self, query: str, documents: list[str]) -> list[float]:
                self.chunks.append(list(documents))
                base = float(len(self.chunks) * 10)
                return [base + i for i in range(len(documents))]

        reranker = _FakeTEI()

        scores = await reranker.rerank("q", ["a", "b", "c", "d", "e"])

        assert reranker.chunks == [["a", "b"], ["c", "d"], ["e"]]
        assert scores == [10.0, 11.0, 20.0, 21.0, 30.0]


class TestRerankerFromUrl:
    def test_dispatch_vllm(self):
        assert isinstance(reranker_from_url("http://h:8000", backend="vllm"), VLLMReranker)

    def test_dispatch_ollama(self):
        r = reranker_from_url("http://h:11434", backend="ollama")
        assert isinstance(r, OllamaReranker)

    def test_dispatch_tei(self):
        assert isinstance(reranker_from_url("http://h:8080", backend="tei"), TEIReranker)

    def test_backend_name_case_insensitive(self):
        assert isinstance(reranker_from_url("http://h", backend="VLLM"), VLLMReranker)

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="unknown reranker backend"):
            reranker_from_url("http://h", backend="cohere")

    def test_model_threaded_through(self):
        r = reranker_from_url("http://h:8000", backend="vllm", model="my-model")
        assert r._model == "my-model"
