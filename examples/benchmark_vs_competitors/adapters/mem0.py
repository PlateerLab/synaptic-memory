"""Mem0 adapter.

Install::

    pip install mem0ai

Mem0 is primarily a conversational memory layer, not a document
retrieval engine — so this adapter is an intentional stretch of its
API. We feed each corpus document in as a memory with
``metadata={"doc_id": ...}`` under a shared benchmark user_id, then
query across that user's memories.

LLM provider (default OpenAI, override via env)::

    export OPENAI_API_KEY=sk-...
    # or — Mem0 supports Anthropic via LiteLLM:
    export ANTHROPIC_API_KEY=sk-ant-...
    export LLM_PROVIDER=anthropic   # read by this adapter

Mem0 makes LLM calls both at ingest (memory extraction /
consolidation) AND at query time (fact rewriting), so comparisons
here reflect total system latency, not just retrieval.
"""

from __future__ import annotations

import os
import uuid

from examples.benchmark_vs_competitors.adapters.base import Adapter
from examples.benchmark_vs_competitors.protocol import Corpus


def _build_config() -> dict:
    """Route Mem0 through whichever LLM provider env vars are set."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "anthropic":
        return {
            "llm": {
                "provider": "anthropic",
                "config": {"model": "claude-3-5-haiku-20241022"},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"},
            },
        }
    return {
        "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"},
        },
    }


class Mem0Adapter(Adapter):
    name = "mem0"

    def __init__(self) -> None:
        try:
            from mem0 import Memory  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Mem0Adapter needs mem0ai. Install: pip install mem0ai"
            ) from exc
        self._Memory = Memory
        self._client = None
        self._user_id = f"synbench_{uuid.uuid4().hex[:8]}"

    async def build(self, corpus: Corpus) -> None:
        # Mem0's Memory class is sync — we accept the blocking calls
        # because the comparison measures end-to-end wall time, which
        # is what a production user would see anyway.
        self._client = self._Memory.from_config(_build_config())

        for doc in corpus.docs:
            text = f"{doc.title}\n\n{doc.text}" if doc.title else doc.text
            if not text.strip():
                continue
            # Mem0 defaults to LLM-driven fact extraction. infer=False
            # disables that so we can index raw corpus documents
            # without paying for a fact-extraction call per document.
            # Many Mem0 users turn this off for document-style data.
            self._client.add(
                text,
                user_id=self._user_id,
                metadata={"doc_id": doc.doc_id},
                infer=False,
            )

    async def search(self, query: str, k: int = 10) -> list[str]:
        assert self._client is not None, "call build() first"
        results = self._client.search(query=query, user_id=self._user_id, limit=k)

        # Mem0 returns either a dict {"results": [...]} or a bare list
        # depending on version — be defensive.
        items = results.get("results", results) if isinstance(results, dict) else results

        retrieved: list[str] = []
        for item in items:
            meta = item.get("metadata") or {}
            did = meta.get("doc_id")
            if did and did not in retrieved:
                retrieved.append(did)
        return retrieved[:k]

    async def close(self) -> None:
        # Clean up synthetic user to avoid polluting the default store.
        if self._client is not None:
            try:
                self._client.delete_all(user_id=self._user_id)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
