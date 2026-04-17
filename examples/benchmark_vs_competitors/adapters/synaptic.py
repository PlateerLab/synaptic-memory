"""Synaptic adapter — FTS-only by default (no LLM, no embedder).

This is the same configuration as ``examples/benchmark_allganize.py``
— the FTS-only floor that anyone can reproduce on a laptop in under
two seconds.

To benchmark the full pipeline (embedder + cross-encoder), pass
``embed_url`` / ``reranker_url`` to the constructor.
"""

from __future__ import annotations

from examples.benchmark_vs_competitors.adapters.base import Adapter
from examples.benchmark_vs_competitors.protocol import Corpus
from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph


class SynapticAdapter(Adapter):
    name = "synaptic-fts"

    def __init__(
        self,
        *,
        embed_url: str | None = None,
        embed_model: str = "qwen3-embedding:4b",
        reranker_url: str | None = None,
    ) -> None:
        self._embed_url = embed_url
        self._embed_model = embed_model
        self._reranker_url = reranker_url
        self._backend: MemoryBackend | None = None
        self._graph: SynapticGraph | None = None
        self._searcher = None
        if embed_url or reranker_url:
            self.name = "synaptic-full"

    async def build(self, corpus: Corpus) -> None:
        self._backend = MemoryBackend()
        await self._backend.connect()
        self._graph = SynapticGraph(self._backend)

        for doc in corpus.docs:
            if not doc.text and not doc.title:
                continue
            await self._graph.add(
                title=doc.title or doc.doc_id,
                content=doc.text,
                properties={"doc_id": doc.doc_id},
            )

        if self._embed_url or self._reranker_url:
            from synaptic.extensions.evidence_search import EvidenceSearch

            embedder = None
            if self._embed_url:
                from synaptic.extensions.embedder import OpenAIEmbeddingProvider

                embedder = OpenAIEmbeddingProvider(
                    api_base=self._embed_url, model=self._embed_model
                )
            reranker = None
            if self._reranker_url:
                from synaptic.extensions.reranker_cross import TEIReranker

                reranker = TEIReranker(base_url=self._reranker_url)
            self._searcher = EvidenceSearch(
                backend=self._backend, embedder=embedder, reranker=reranker
            )

    async def search(self, query: str, k: int = 10) -> list[str]:
        assert self._graph is not None, "call build() first"

        retrieved: list[str] = []

        if self._searcher is not None:
            result = await self._searcher.search(query, k=k * 2, fts_seed_limit=30)
            for ev in result.evidence:
                did = ev.document_id or (ev.node.properties or {}).get("doc_id", "")
                if did and did not in retrieved:
                    retrieved.append(did)
        else:
            result = await self._graph.search(query, limit=k * 2)
            for hit in result.nodes:
                did = (hit.node.properties or {}).get("doc_id", "")
                if did and did not in retrieved:
                    retrieved.append(did)

        return retrieved[:k]

    async def close(self) -> None:
        if self._backend is not None:
            close = getattr(self._backend, "close", None)
            if close is not None:
                await close()
