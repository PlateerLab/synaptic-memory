"""Embedding providers — generate vector embeddings for nodes.

Supports any OpenAI-compatible endpoint:
- OpenAI API
- vLLM (--served-model-name)
- llama.cpp (server --embedding)
- Ollama (/api/embeddings)
- TEI (Text Embeddings Inference)
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Generate embedding vectors from text."""

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class _EmbedFromBatchMixin:
    """Mixin that provides a default embed() implementation by delegating to embed_batch()."""

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])  # type: ignore[attr-defined]
        return results[0]


class MockEmbeddingProvider:
    """Mock embedding provider for testing. Returns deterministic vectors."""

    __slots__ = ("_dim",)

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(self._dim)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class OpenAIEmbeddingProvider(_EmbedFromBatchMixin):
    """OpenAI-compatible embedding provider.

    Works with any server implementing the /v1/embeddings endpoint:
    - OpenAI: api_base="https://api.openai.com/v1", model="text-embedding-3-small"
    - vLLM:   api_base="http://localhost:8000/v1", model="BAAI/bge-m3"
    - llama.cpp: api_base="http://localhost:8080/v1", model="default"
    - Ollama: api_base="http://localhost:11434/v1", model="nomic-embed-text"
    - TEI:    api_base="http://localhost:8080/v1", model="default"

    Uses aiohttp (zero extra deps — already pulled in by miniopy-async).
    """

    __slots__ = ("_api_base", "_api_key", "_model", "_timeout")

    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        *,
        api_key: str = "",
        model: str = "default",
        timeout: int = 60,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import aiohttp

        url = f"{self._api_base}/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {"model": self._model, "input": texts}

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    msg = f"Embedding API error {resp.status}: {body[:200]}"
                    raise RuntimeError(msg)
                data = await resp.json()

        embeddings: list[list[float]] = []
        for item in sorted(data["data"], key=lambda x: x["index"]):  # type: ignore[no-any-return]
            embeddings.append(item["embedding"])  # type: ignore[index]
        return embeddings


class OllamaEmbeddingProvider(_EmbedFromBatchMixin):
    """Ollama native embedding endpoint (/api/embed).

    For Ollama servers that don't expose /v1/embeddings.

    Usage:
        provider = OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="nomic-embed-text",
        )
    """

    __slots__ = ("_base_url", "_model", "_timeout")

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        model: str = "nomic-embed-text",
        timeout: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import aiohttp

        url = f"{self._base_url}/api/embed"
        payload = {"model": self._model, "input": texts}

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    msg = f"Ollama embed error {resp.status}: {body[:200]}"
                    raise RuntimeError(msg)
                data = await resp.json()

        return data["embeddings"]  # type: ignore[no-any-return]
