"""HyDE — Hypothetical Document Embeddings.

Wraps an existing :class:`EmbeddingProvider` so that ``embed`` (used
by :class:`EvidenceSearch` for the query embedding) is replaced by:

  1. LLM generates a short hypothetical answer to the query.
  2. The wrapped embedder embeds that answer.
  3. The result is used as the query embedding.

The corpus side (``embed_batch`` for ingest) is delegated unchanged
to the wrapped embedder. Only the query path goes through the LLM.

Why this exists
---------------
Conversational / paraphrase-heavy queries ("어떤 게 좋아요?",
"tell me about ...") are keyword-poor and surface-form-distant from
the documents that answer them. Direct query→document embedding
cosine misses badly. HyDE bridges by transforming the query into
the *answer space* — a hypothetical answer is in the same surface
form as real documents, so dense retrieval matches better.

Original paper: Gao et al. 2022 ("Precise Zero-Shot Dense Retrieval
without Relevance Labels"). Reported strong gains on TREC-COVID,
NF-Corpus, FiQA. v0.17.x measurements suggest similar potential on
KRRA Conv (single-shot 0.166), assort Conv (0.472), X2BEE Conv (0.164)
where colloquial Korean queries are the dominant pattern.

Trade-off
---------
+1 LLM call per query (~1-3 s on Qwen3.5-27B via vLLM). Production
batching makes this acceptable; for single-query latency-sensitive
paths, fall back to the wrapped embedder directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synaptic.extensions.embedder import EmbeddingProvider
    from synaptic.extensions.llm_provider import LLMProvider

logger = logging.getLogger("hyde-embedder")


_HYDE_PROMPT = """You generate plausible short answers to retrieval queries.

Write a 1-3 sentence factual passage that would directly answer the
query, as if extracted from a real document. Use the query's language
(Korean for Korean queries, English for English). Do not say "I don't
know" or refuse — write the most plausible answer you can.

Output ONLY the passage text, no preamble, no JSON, no quotes."""


class HyDEEmbedder:
    """Drop-in :class:`EmbeddingProvider` that uses HyDE for queries.

    Args:
        llm: Any LLM provider — generates the hypothetical answer.
        embedder: The underlying dense embedder (e.g. bge-m3) used for
            both the corpus side and the embed-the-hypothetical step.
        max_tokens: Cap on the LLM's hypothetical answer length.
        fallback_on_error: If True, fall back to embedding the raw
            query when the LLM call fails. Default True (production
            safety). Set False for diagnostics.
    """

    __slots__ = ("_embedder", "_fallback", "_llm", "_max_tokens")

    def __init__(
        self,
        *,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        max_tokens: int = 200,
        fallback_on_error: bool = True,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._max_tokens = max_tokens
        self._fallback = fallback_on_error

    async def embed(self, text: str) -> list[float]:
        """Generate a hypothetical answer with the LLM, then embed it."""
        cleaned = (text or "").strip()
        if not cleaned:
            return await self._embedder.embed(text)

        try:
            hypothetical = await self._llm.generate(
                system=_HYDE_PROMPT,
                user=f"Query: {cleaned}",
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            if not self._fallback:
                raise
            logger.warning("HyDE LLM call failed (%s) — falling back to raw query embed", exc)
            return await self._embedder.embed(text)

        hypothetical = (hypothetical or "").strip()
        if not hypothetical:
            return await self._embedder.embed(text)

        # Combine query + hypothetical: gives the embedder both the
        # surface form (anchors named entities) and the answer-space
        # context. The original HyDE paper used hypothetical-only;
        # follow-ups (HyDE++ etc.) report the concatenation is more
        # robust on shorter queries where hypothetical drift is risky.
        combined = f"{cleaned}\n{hypothetical}"
        return await self._embedder.embed(combined)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Corpus-side embedding goes straight through — no HyDE during ingest."""
        return await self._embedder.embed_batch(texts)
