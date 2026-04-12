"""Cross-encoder reranker — BYO protocol for semantic reranking.

After the initial retrieval (FTS + vector) produces a candidate pool,
a cross-encoder scores each (query, document) pair jointly. This is
the single highest-impact quality improvement in modern IR because:

- BM25 / cosine score query and document INDEPENDENTLY
- Cross-encoder sees the query AND document TOGETHER → understands
  semantic equivalence like "말 복지" ↔ "재활힐링승마"

The protocol is BYO (bring your own): the library never ships a
model. Users inject any reranker that implements ``RerankerProtocol``:

- ``OllamaReranker`` → Ollama ``/api/rerank`` endpoint
- ``TEIReranker`` → HuggingFace TEI ``/rerank``
- ``CohereReranker`` → Cohere Rerank API
- ``MockReranker`` → for tests

Example::

    from synaptic.extensions.reranker_cross import OllamaReranker

    reranker = OllamaReranker(
        base_url="http://gpu-server:11434",
        model="bge-reranker-v2-m3",
    )
    # Inject into EvidenceSearch
    searcher = EvidenceSearch(
        backend=backend,
        embedder=embedder,
        reranker=reranker,
    )
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger("reranker-cross")


class RerankerProtocol(Protocol):
    """Score (query, document) pairs jointly.

    Returns a list of relevance scores in the same order as the input
    documents. Higher = more relevant. Scale is model-dependent — the
    caller normalises before using.
    """

    async def rerank(
        self, query: str, documents: list[str]
    ) -> list[float]: ...


class OllamaReranker:
    """Ollama reranker via /api/rerank (Ollama 0.5+).

    Supported models: bge-reranker-v2-m3, jina-reranker-v2, etc.

    Usage::

        reranker = OllamaReranker(
            base_url="http://localhost:11434",
            model="bge-reranker-v2-m3",
        )
        scores = await reranker.rerank("query", ["doc1", "doc2"])
    """

    __slots__ = ("_base_url", "_model", "_timeout")

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        model: str = "bge-reranker-v2-m3",
        timeout: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def rerank(
        self, query: str, documents: list[str]
    ) -> list[float]:
        import aiohttp

        url = f"{self._base_url}/api/rerank"
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("rerank failed (%d): %s", resp.status, body[:200])
                    return [0.0] * len(documents)
                data = await resp.json()

        # Ollama returns: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        results = data.get("results", [])
        scores = [0.0] * len(documents)
        for r in results:
            idx = r.get("index", 0)
            if 0 <= idx < len(scores):
                scores[idx] = float(r.get("relevance_score", 0.0))
        return scores


class TEIReranker:
    """HuggingFace Text Embeddings Inference (TEI) reranker.

    Usage::

        reranker = TEIReranker(base_url="http://gpu-server:8080")
    """

    __slots__ = ("_base_url", "_timeout")

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        timeout: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def rerank(
        self, query: str, documents: list[str]
    ) -> list[float]:
        import aiohttp

        url = f"{self._base_url}/rerank"
        payload = {
            "query": query,
            "texts": documents,
        }
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("TEI rerank failed (%d): %s", resp.status, body[:200])
                    return [0.0] * len(documents)
                data = await resp.json()

        scores = [0.0] * len(documents)
        for r in data:
            idx = r.get("index", 0)
            if 0 <= idx < len(scores):
                scores[idx] = float(r.get("score", 0.0))
        return scores


class MockReranker:
    """Test double — returns scores proportional to document length."""

    async def rerank(
        self, query: str, documents: list[str]
    ) -> list[float]:
        if not documents:
            return []
        max_len = max(len(d) for d in documents) or 1
        return [len(d) / max_len for d in documents]
