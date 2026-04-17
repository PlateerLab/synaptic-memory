"""Cognee adapter.

Install::

    pip install cognee

Cognee's three-stage pipeline (Extract → Cognify → Load) always
involves an LLM at ingest time — ``cognify()`` runs entity
extraction, ontology generation, and graph construction. That cost
is part of what we're measuring: Cognee's value proposition is
"LLM-synthesized knowledge graph," so benchmarking it without the
LLM would be meaningless.

Required env::

    export OPENAI_API_KEY=sk-...
    # Cognee also supports Anthropic / local models — see Cognee docs
"""

from __future__ import annotations

from examples.benchmark_vs_competitors.adapters.base import Adapter
from examples.benchmark_vs_competitors.protocol import Corpus


class CogneeAdapter(Adapter):
    name = "cognee"

    def __init__(self, dataset: str = "synbench") -> None:
        try:
            import cognee  # type: ignore  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CogneeAdapter needs cognee. Install: pip install cognee"
            ) from exc
        self._dataset = dataset
        # Store a {doc_id: preview} map so we can recover doc_ids from
        # Cognee's search results, which return text snippets.
        self._doc_id_by_title: dict[str, str] = {}

    async def build(self, corpus: Corpus) -> None:
        import cognee  # type: ignore

        # Cognee stores everything under a "dataset" namespace. Prune
        # prior runs so a fresh ingest is reproducible.
        try:
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)
        except Exception:  # noqa: BLE001 - best-effort
            pass

        for doc in corpus.docs:
            title = doc.title or doc.doc_id
            text = f"{title}\n\n{doc.text}" if doc.text else title
            if not text.strip():
                continue
            self._doc_id_by_title[title] = doc.doc_id
            await cognee.add(text, dataset_name=self._dataset)

        # LLM-powered knowledge-graph construction.
        await cognee.cognify([self._dataset])

    async def search(self, query: str, k: int = 10) -> list[str]:
        import cognee  # type: ignore
        from cognee.modules.search.types import SearchType  # type: ignore

        results = await cognee.search(
            query_text=query,
            query_type=SearchType.INSIGHTS,
            datasets=[self._dataset],
        )

        # Cognee returns a list of dicts or strings depending on
        # SearchType. We scan each result for a known title.
        retrieved: list[str] = []
        for item in results[: k * 3]:
            text = item if isinstance(item, str) else str(item)
            for title, doc_id in self._doc_id_by_title.items():
                if title and title in text and doc_id not in retrieved:
                    retrieved.append(doc_id)
                    break
        return retrieved[:k]
