"""BGE-M3 embedding provider — dense + sparse + ColBERT in one model.

BGE-M3 generates three types of embeddings simultaneously:
  - Dense: semantic similarity (standard embedding vector)
  - Sparse: lexical matching (BM25-equivalent token weights)
  - ColBERT: per-token embeddings for late-interaction reranking

Works with any server exposing BGE-M3:
  - vLLM: --served-model-name BAAI/bge-m3
  - FlagEmbedding server
  - TEI (Text Embeddings Inference)

For backward compatibility, embed() returns dense vector only.
embed_hybrid() returns the full HybridEmbedding.

Requires: pip install synaptic-memory[embedding]  (aiohttp)
"""

from __future__ import annotations

import logging

from synaptic.models import HybridEmbedding

logger = logging.getLogger("embedder-bge-m3")


class BGEM3EmbeddingProvider:
    """BGE-M3 provider: dense + sparse + ColBERT from a single model.

    Example::

        embedder = BGEM3EmbeddingProvider(
            api_base="http://gpu-server:8080/v1",
            model="BAAI/bge-m3",
        )
        # Dense only (backward compat with EmbeddingProvider protocol)
        dense = await embedder.embed("query text")

        # Full hybrid
        hybrid = await embedder.embed_hybrid("query text")
        # hybrid.dense, hybrid.sparse, hybrid.colbert
    """

    __slots__ = ("_api_base", "_api_key", "_model", "_timeout")

    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        *,
        api_key: str = "",
        model: str = "BAAI/bge-m3",
        timeout: int = 60,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    # --- EmbeddingProvider protocol (backward compat) ---

    async def embed(self, text: str) -> list[float]:
        """Dense vector only — compatible with EmbeddingProvider protocol."""
        hybrid = await self.embed_hybrid(text)
        return hybrid.dense

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch dense vectors — compatible with EmbeddingProvider protocol."""
        hybrids = await self.embed_batch_hybrid(texts)
        return [h.dense for h in hybrids]

    # --- Hybrid API ---

    async def embed_hybrid(self, text: str) -> HybridEmbedding:
        """Full hybrid embedding: dense + sparse + ColBERT."""
        results = await self.embed_batch_hybrid([text])
        return results[0]

    async def embed_batch_hybrid(self, texts: list[str]) -> list[HybridEmbedding]:
        """Batch hybrid embedding."""
        import aiohttp

        url = f"{self._api_base}/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # Request with return_sparse=true for BGE-M3 compatible servers
        payload: dict[str, object] = {
            "input": texts,
            "model": self._model,
        }

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    msg = f"BGE-M3 embedding failed ({resp.status}): {body[:200]}"
                    raise RuntimeError(msg)

                data = await resp.json()

        results: list[HybridEmbedding] = []
        for item in data.get("data", []):
            dense = item.get("embedding", [])
            # Sparse and ColBERT may not be available from all servers
            sparse_data = item.get("sparse_embedding", {})
            sparse: dict[int, float] = {}
            if isinstance(sparse_data, dict):
                # Format: {"indices": [1, 5, ...], "values": [0.3, 0.8, ...]}
                indices = sparse_data.get("indices", [])
                values = sparse_data.get("values", [])
                sparse = dict(zip(indices, values))
            elif isinstance(sparse_data, list):
                # Format: [(token_id, weight), ...]
                sparse = {int(k): float(v) for k, v in sparse_data}

            colbert = item.get("colbert_embedding")

            results.append(
                HybridEmbedding(
                    dense=dense,
                    sparse=sparse,
                    colbert=colbert,
                )
            )

        return results


class MockBGEM3Provider:
    """Mock BGE-M3 provider for testing. Generates deterministic hybrid embeddings."""

    __slots__ = ("_dim",)

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        h = await self.embed_hybrid(text)
        return h.dense

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    async def embed_hybrid(self, text: str) -> HybridEmbedding:
        h = hash(text) & 0xFFFFFFFF
        dense = [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(self._dim)]

        # Generate sparse: a few token IDs with weights
        sparse = {
            (h + i) % 30000: ((h >> (i * 4)) & 0xF) / 15.0 for i in range(min(5, len(text.split())))
        }

        # ColBERT: per-token vectors (simplified)
        tokens = text.split()[:8]
        colbert = [
            [((hash(t) >> (j * 8)) & 0xFF) / 255.0 for j in range(self._dim)] for t in tokens
        ]

        return HybridEmbedding(dense=dense, sparse=sparse, colbert=colbert)

    async def embed_batch_hybrid(self, texts: list[str]) -> list[HybridEmbedding]:
        return [await self.embed_hybrid(t) for t in texts]
