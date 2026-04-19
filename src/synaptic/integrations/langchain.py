"""LangChain adapter — expose Synaptic Memory as a ``BaseRetriever``.

Install with::

    pip install "synaptic-memory[langchain]"

Use with::

    from synaptic import SynapticGraph
    from synaptic.integrations.langchain import SynapticRetriever

    graph = await SynapticGraph.from_data("./docs/")
    retriever = SynapticRetriever(graph=graph, k=5, engine="evidence")

    # Works anywhere LangChain expects a retriever
    docs = await retriever.ainvoke("my question")

The adapter is async-native: each hit becomes a ``Document`` whose
``page_content`` is the node's content (or title if the content is
empty) and whose ``metadata`` carries the node id, original title,
retrieval score, and any structured properties (doc_id, category, ...).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:  # pragma: no cover - optional extra
    raise ImportError(
        "SynapticRetriever requires langchain-core. "
        "Install with: pip install 'synaptic-memory[langchain]'"
    ) from exc

if TYPE_CHECKING:
    from synaptic.graph import SynapticGraph


class SynapticRetriever(BaseRetriever):
    """LangChain retriever backed by a :class:`SynapticGraph`.

    Attributes:
        graph: A ready-to-search :class:`synaptic.SynapticGraph`
            instance. Build it with ``await SynapticGraph.from_data(...)``
            or ``await SynapticGraph.from_database(...)`` before
            constructing the retriever.
        k: Top-k hits to return (default 5).
        engine: Which retrieval engine to use — ``"evidence"`` for the
            hybrid pipeline (BM25 + HNSW + PPR + cross-encoder + MMR)
            or ``"legacy"`` for the 3-stage FTS cascade. ``"evidence"``
            is recommended for production; legacy is the v0.15.x
            default and will be removed in v0.17.0.
    """

    graph: Any
    """SynapticGraph instance — declared Any to avoid pydantic validation
    on the internal dataclass."""

    k: int = 5
    engine: str = "evidence"

    model_config: ClassVar[dict] = {"arbitrary_types_allowed": True}

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        graph: SynapticGraph = self.graph
        result = await graph.search(query, limit=self.k, engine=self.engine)

        docs: list[Document] = []
        for hit in result.nodes[: self.k]:
            node = hit.node
            properties = dict(node.properties) if node.properties else {}
            metadata: dict[str, Any] = {
                "node_id": node.id,
                "title": node.title,
                "kind": str(node.kind),
                "score": hit.activation,
                **properties,
            }
            content = node.content or node.title or node.id
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        import asyncio

        # Prefer the async path — but support sync callers for drop-in
        # compatibility with chains that haven't moved to ainvoke yet.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            raise RuntimeError(
                "SynapticRetriever is async-native. Call `await "
                "retriever.ainvoke(query)` from async code. "
                "Synchronous `retriever.invoke(query)` is only safe "
                "outside a running event loop."
            )

        return asyncio.run(
            self._aget_relevant_documents(
                query,
                run_manager=run_manager,  # type: ignore[arg-type]
            )
        )


__all__ = ["SynapticRetriever"]
