"""HippoRAG2 adapter.

Install (subject to upstream availability — HippoRAG moves quickly)::

    pip install hipporag
    # see https://github.com/OSU-NLP-Group/HippoRAG for current steps

HippoRAG2 is the academic baseline — Personalized PageRank over an
LLM-extracted entity graph, with the explicit "neurobiologically
plausible memory" framing (NeurIPS '24, ICML '25). It needs an LLM
for entity + relation extraction at index time.

This adapter is a scaffold. The HippoRAG research codebase changes
shape between releases, so the concrete ``build()`` / ``search()``
call sites are documented as TODOs rather than coded in-line — that
way the comparison harness still imports cleanly if HippoRAG isn't
installed.
"""

from __future__ import annotations

from examples.benchmark_vs_competitors.adapters.base import Adapter
from examples.benchmark_vs_competitors.protocol import Corpus


class HippoRAG2Adapter(Adapter):
    name = "hipporag2"

    def __init__(
        self,
        *,
        save_dir: str = "hipporag_bench",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        try:
            # The package is published as both "hipporag" and
            # "hipporag2" at different points in time — try both.
            try:
                from hipporag import HippoRAG  # type: ignore
            except ImportError:
                from hipporag2 import HippoRAG  # type: ignore  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HippoRAG2Adapter needs hipporag / hipporag2. "
                "See https://github.com/OSU-NLP-Group/HippoRAG"
            ) from exc

        self._save_dir = save_dir
        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._rag = None
        self._doc_id_by_index: dict[int, str] = {}

    async def build(self, corpus: Corpus) -> None:
        # Import inside build() so an empty environment still passes
        # static import checks on the rest of the harness.
        try:
            from hipporag import HippoRAG  # type: ignore
        except ImportError:
            from hipporag2 import HippoRAG  # type: ignore

        self._rag = HippoRAG(
            save_dir=self._save_dir,
            llm_model_name=self._llm_model,
            embedding_model_name=self._embedding_model,
        )

        docs: list[str] = []
        for i, doc in enumerate(corpus.docs):
            text = f"{doc.title}\n\n{doc.text}" if doc.title else doc.text
            if not text.strip():
                continue
            self._doc_id_by_index[i] = doc.doc_id
            docs.append(text)

        # HippoRAG's index() call is synchronous and does the heavy
        # LLM-based OpenIE-style extraction.
        self._rag.index(docs=docs)

    async def search(self, query: str, k: int = 10) -> list[str]:
        assert self._rag is not None, "call build() first"
        # HippoRAG returns a ranked passage list; shapes vary by
        # release. Users who actually run this should update the
        # result-parsing below to match their installed version.
        results = self._rag.rag_qa(queries=[query], num_to_retrieve=k)

        retrieved: list[str] = []
        for item in results:
            idx = item.get("idx") if isinstance(item, dict) else None
            did = self._doc_id_by_index.get(idx, "") if idx is not None else ""
            if did and did not in retrieved:
                retrieved.append(did)
        return retrieved[:k]
