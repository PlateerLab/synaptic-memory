"""Adapter contract — implement these three methods for each system."""

from __future__ import annotations

from abc import ABC, abstractmethod

from examples.benchmark_vs_competitors.protocol import Corpus


class Adapter(ABC):
    """Minimal common interface.

    Implementations should:

    1. **build()** — ingest the whole corpus. May call LLMs, spin up
       services, etc. The wall-clock time spent here is reported as
       "Build" in the comparison table.
    2. **search()** — run a single query and return a list of
       candidate doc_ids ordered by relevance. ``k`` is the number of
       hits the harness wants; adapters may over-fetch internally for
       their own reranking logic.
    3. **close()** — release external resources (connections,
       sockets, tempdirs).
    """

    name: str = "unknown"

    @abstractmethod
    async def build(self, corpus: Corpus) -> None:
        """Ingest the full corpus into the system."""

    @abstractmethod
    async def search(self, query: str, k: int = 10) -> list[str]:
        """Return top-``k`` doc_ids for ``query``."""

    async def close(self) -> None:
        """Release resources. Default no-op."""
        return None
