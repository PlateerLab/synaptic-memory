"""LLM-driven query decomposer for multi-hop / chain questions.

Complements :class:`QueryDecomposer` (rule-based compound splitter in
``query_decomposer.py``):

- ``QueryDecomposer`` handles **compound** questions — ``A와 B 비교``,
  ``X and Y``, explicit temporal ranges. Zero LLM cost, zero latency.
- ``LLMChainDecomposer`` handles **chain** / multi-hop questions —
  ``Who founded the company that distributed the film UHF?`` — where
  the split points are semantic, not syntactic, so rules can't see them.

Both satisfy :class:`synaptic.protocols.QueryDecomposer`, so the
facade wires either transparently.

Example::

    from synaptic.extensions.llm_provider import OpenAILLMProvider
    from synaptic.extensions.query_decomposer_llm import LLMChainDecomposer

    llm = OpenAILLMProvider(api_base="http://localhost:8012/v1",
                            model="Qwen3.5-27b")
    decomposer = LLMChainDecomposer(llm=llm)
    subs = await decomposer.decompose(
        "Who founded the company that distributed the film UHF?"
    )
    # → ["Who distributed the film UHF?",
    #    "Who founded the company that distributed the film UHF?"]

The output layout is ``{"sub_queries": [...]}`` — a JSON object rather
than a bare array — because most OpenAI-compatible ``response_format:
json_object`` servers (vLLM included) reject top-level arrays.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider

logger = logging.getLogger("llm-chain-decomposer")

_SYSTEM_PROMPT = """You decompose multi-hop questions into atomic sub-questions for retrieval.

For chain questions (e.g. "X of Y of Z"), produce 2-3 sub-questions that each probe one entity or relation mentioned in the question.
For single-hop atomic questions, return one element matching the original.
Keep sub-questions short and retrieval-friendly (no pronouns, use explicit entity names from the question).
Works for English, Korean, or mixed input — respond in the same language as the question.

Return strict JSON: {"sub_queries": ["...", ...]}"""


class LLMChainDecomposer:
    """Decomposes multi-hop questions via an LLM.

    Args:
        llm: Any :class:`~synaptic.extensions.llm_provider.LLMProvider`.
            vLLM / Ollama / OpenAI / Anthropic all work.
        max_subs: Hard cap on sub-queries returned. Prevents runaway
            outputs from inflating downstream FTS cost.
        max_tokens: Budget for the LLM response. 256 is plenty for the
            JSON payload; most chain decompositions are <50 tokens.
    """

    __slots__ = ("_llm", "_max_subs", "_max_tokens")

    def __init__(
        self,
        *,
        llm: LLMProvider,
        max_subs: int = 4,
        max_tokens: int = 256,
    ) -> None:
        self._llm = llm
        self._max_subs = max_subs
        self._max_tokens = max_tokens

    async def decompose(self, query: str) -> list[str]:
        query = (query or "").strip()
        if not query:
            return []

        try:
            response = await self._llm.generate(
                system=_SYSTEM_PROMPT,
                user=f"Question: {query}",
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            logger.warning("LLM decomposition request failed: %s", exc)
            return [query]

        try:
            obj = json.loads(response)
        except json.JSONDecodeError as exc:
            logger.warning("LLM response is not JSON (%s): %r", exc, response[:200])
            return [query]

        if not isinstance(obj, dict):
            return [query]
        subs = obj.get("sub_queries")
        if not isinstance(subs, list):
            return [query]

        cleaned: list[str] = []
        seen: set[str] = set()
        for s in subs:
            if not isinstance(s, str):
                continue
            s = s.strip()
            if not s or s in seen:
                continue
            cleaned.append(s)
            seen.add(s)
            if len(cleaned) >= self._max_subs:
                break

        return cleaned if cleaned else [query]
