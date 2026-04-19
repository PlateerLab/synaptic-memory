"""LLM-as-reranker — listwise rerank via any OpenAI-compatible LLM.

Drop-in replacement for ``bge-reranker-v2-m3`` (or any ``RerankerProtocol``
implementation). The LLM reads the query and the top-N candidate
documents in a single call and returns relevance scores.

Why this exists
---------------
Cross-encoder rerankers (bge-reranker-v2-m3) are sentence-pair models
trained on long-form paraphrase. On corpora where the candidates are
short structured rows or near-duplicates (AutoRAG FAQs, X2BEE Hard,
KRRA Conv) they produce near-uniform logits that override FTS's
near-optimal ranking — measured −15 % to −34 % regression vs FTS-only
before v0.17.1's adaptive blend.

An LLM reranker reasons about the query AND the candidate together. On
corpora where bge-reranker can't discriminate, the LLM can use world
knowledge / context to identify the actually relevant candidate. The
trade-off is latency (one LLM call per query, ~1-3 s on Qwen3.5-27B
via vLLM) vs the millisecond cross-encoder inference.

Listwise vs pointwise
---------------------
Listwise (default): the LLM sees all N candidates at once and assigns
each a relevance score. This lets the LLM compare candidates against
each other, which generally produces more discriminative scores than
pointwise scoring (RankZephyr 2024, RankT5 2024, PRP 2024).

The contract still matches ``RerankerProtocol``: returns a list of
scores in the same order as ``documents``. ``EvidenceSearch`` blends
those scores with its hybrid signal exactly as it does for
bge-reranker.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider

logger = logging.getLogger("llm-reranker")


_LISTWISE_PROMPT = """You are a precise relevance judge for a search system.

Score each document's relevance to the query on a 0.0-10.0 scale:
  10.0 = directly answers the query
   7.0 = strongly related, contains key information
   4.0 = tangentially related, partial information
   1.0 = same domain but not relevant
   0.0 = unrelated

Be discriminative — don't cluster scores. The most relevant document
should score noticeably higher than the next.

Return strict JSON: {"scores": [s_0, s_1, ..., s_{N-1}]}, one score per document
in the order given."""


class LLMReranker:
    """Listwise LLM reranker. Implements ``RerankerProtocol``.

    Args:
        llm: Any :class:`~synaptic.extensions.llm_provider.LLMProvider`.
            vLLM / Ollama / OpenAI / Anthropic all work.
        max_documents: Hard cap on documents per call (LLM context limit).
            Documents beyond this get score 0.0; the caller already
            ranked them lower so they remain at the bottom.
        max_doc_chars: Truncate each document body to this many chars
            before sending to the LLM. Keeps the prompt under typical
            context windows (16k vLLM default).
    """

    __slots__ = ("_llm", "_max_doc_chars", "_max_documents")

    def __init__(
        self,
        *,
        llm: LLMProvider,
        max_documents: int = 20,
        max_doc_chars: int = 400,
    ) -> None:
        self._llm = llm
        self._max_documents = max_documents
        self._max_doc_chars = max_doc_chars

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        # LLM sees only the first ``max_documents`` candidates; tail gets 0.
        head = documents[: self._max_documents]
        tail_count = len(documents) - len(head)

        # Build the user prompt: numbered candidates, query at the bottom
        # (recency bias on Qwen and most decoder LLMs makes the query
        # placement matter — last position improves reading).
        lines = ["Documents to score:"]
        for i, doc in enumerate(head):
            body = (doc or "").strip().replace("\n", " ")[: self._max_doc_chars]
            lines.append(f"[{i}] {body}")
        lines.append("")
        lines.append(f"Query: {query}")
        user = "\n".join(lines)

        try:
            response = await self._llm.generate(
                system=_LISTWISE_PROMPT,
                user=user,
                max_tokens=400,
            )
        except Exception as exc:
            logger.warning("LLM rerank request failed: %s", exc)
            # Degrade: return uniform 0.0 → caller falls back to
            # whatever scoring it was holding.
            return [0.0] * len(documents)

        scores = self._parse_scores(response, len(head))
        if scores is None:
            logger.warning("LLM rerank parse failed: %r", response[:200])
            return [0.0] * len(documents)

        # Pad tail with 0 to match input length
        return scores + [0.0] * tail_count

    def _parse_scores(self, response: str, expected_n: int) -> list[float] | None:
        """Parse ``{"scores": [...]}`` JSON. Returns ``None`` on failure
        so the caller can fall back gracefully."""
        # Strip code fences if the LLM wrapped its JSON in ```...```
        stripped = response.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.MULTILINE)

        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        scores = obj.get("scores")
        if not isinstance(scores, list):
            return None

        out: list[float] = []
        for s in scores:
            try:
                out.append(float(s))
            except (TypeError, ValueError):
                out.append(0.0)
        # Pad / truncate to expected length so downstream blend doesn't
        # drift if the LLM gave fewer/more scores than asked.
        if len(out) < expected_n:
            out.extend([0.0] * (expected_n - len(out)))
        elif len(out) > expected_n:
            out = out[:expected_n]
        return out
