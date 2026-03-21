"""LLM-based query rewriter — expands search queries via language model."""

from __future__ import annotations

from typing import Protocol


class LLMChatFn(Protocol):
    """Minimal LLM chat interface for query rewriting."""

    async def __call__(self, *, system: str, user: str, max_tokens: int) -> str: ...


class LLMQueryRewriter:
    """Rewrites search queries using an LLM for better recall.

    Generates 2-3 alternative phrasings of the query.
    Uses economy-tier models (e.g. Haiku) for cost efficiency.
    """

    __slots__ = ("_chat_fn",)

    def __init__(self, chat_fn: LLMChatFn) -> None:
        self._chat_fn = chat_fn

    async def rewrite(self, query: str) -> list[str]:
        """Rewrite query into 2-3 alternative forms."""
        if not query.strip():
            return []

        system = (
            "You are a search query expander. Given a search query, "
            "generate 2-3 alternative phrasings that could match relevant documents. "
            "Include both Korean and English variants if applicable. "
            "Return one query per line, nothing else."
        )
        try:
            response = await self._chat_fn(
                system=system,
                user=f"Query: {query}",
                max_tokens=256,
            )
            lines = [
                line.strip().lstrip("- ·•0123456789.") for line in response.strip().splitlines()
            ]
            return [line for line in lines if line and line != query][:3]
        except Exception:
            return []


class StaticQueryRewriter:
    """Static query rewriter for testing — returns predefined expansions."""

    __slots__ = ("_expansions",)

    def __init__(self, expansions: dict[str, list[str]] | None = None) -> None:
        self._expansions = expansions or {}

    async def rewrite(self, query: str) -> list[str]:
        return self._expansions.get(query, [])
