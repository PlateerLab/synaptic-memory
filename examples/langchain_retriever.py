"""LangChain integration — use Synaptic as a drop-in retriever.

Install::

    pip install "synaptic-memory[sqlite,korean,vector,langchain]"

Run::

    python examples/langchain_retriever.py

What it shows
-------------
Build a SynapticGraph from the shipped sample CSV, wrap it in a
``SynapticRetriever``, and call it through LangChain's async
``ainvoke`` interface. The returned list of ``Document`` objects is
what any LangChain chain (retrieval QA, agent tool, etc.) expects.

No LLM is called — this is retrieval only. Chain the retriever into
your own QA chain, agent, or RAG application as usual.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from synaptic import SynapticGraph
from synaptic.integrations.langchain import SynapticRetriever

DATA_PATH = Path(__file__).parent / "data" / "products.csv"
DB_PATH = "langchain_retriever.db"


async def main() -> None:
    graph = await SynapticGraph.from_data(str(DATA_PATH), db=DB_PATH)

    try:
        retriever = SynapticRetriever(graph=graph, k=3, engine="evidence")

        for query in [
            "laptop with good battery life",
            "spicy Korean food",
            "skincare products",
        ]:
            print(f"Query: {query!r}")
            docs = await retriever.ainvoke(query)
            for i, doc in enumerate(docs, 1):
                name = doc.metadata.get("name", doc.metadata.get("title", "?"))
                score = doc.metadata.get("score", 0.0)
                print(f"  {i}. {name:<30}  score={score:.3f}")
                print(f"     metadata keys: {sorted(doc.metadata.keys())}")
            print()
    finally:
        close = getattr(graph._backend, "close", None)
        if close is not None:
            await close()


if __name__ == "__main__":
    asyncio.run(main())
