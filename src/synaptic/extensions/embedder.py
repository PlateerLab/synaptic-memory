"""Embedding providers — generate vector embeddings for nodes."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Generate embedding vectors from text."""

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class MockEmbeddingProvider:
    """Mock embedding provider for testing. Returns deterministic vectors."""

    __slots__ = ("_dim",)

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        # Deterministic: hash text into a vector
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(self._dim)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class OpenAIEmbeddingProvider:
    """OpenAI-compatible embedding provider (works with OpenAI, vLLM, Ollama).

    Usage:
        provider = OpenAIEmbeddingProvider(
            api_base="https://api.openai.com/v1",
            api_key="sk-...",
            model="text-embedding-3-small",
        )
    """

    __slots__ = ("_api_base", "_api_key", "_model")

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "text-embedding-3-small",
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import httpx  # type: ignore[import-untyped]  # noqa: PLC0415

        url = f"{self._api_base}/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {"model": self._model, "input": texts}

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        embeddings: list[list[float]] = []
        for item in sorted(data["data"], key=lambda x: x["index"]):  # type: ignore[no-any-return]
            embeddings.append(item["embedding"])  # type: ignore[index]
        return embeddings
