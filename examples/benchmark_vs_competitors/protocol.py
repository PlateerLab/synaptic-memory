"""Common benchmark protocol — BEIR-style corpus, standard IR metrics.

Every adapter implements the :class:`Adapter` interface in
``adapters.base``. The runner feeds it the same ``Corpus`` +
``Query`` objects and scores with :func:`score_run`. No per-system
metric fudging.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    relevant: frozenset[str]


@dataclass
class Corpus:
    name: str
    docs: list[Document] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)


def load_corpus(path: Path | str, name: str | None = None) -> Corpus:
    """Load a BEIR-ish JSON corpus produced by Synaptic's eval pipeline.

    Expected keys:

    * ``corpus``: dict of ``{doc_id: {title, text}}`` OR list of dicts
    * ``queries``: dict of ``{qid: text}`` OR list of ``{qid, query, relevant_docs}``
    * ``relevant_docs`` / ``qrels``: BEIR-style qrels when queries is a dict
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    docs: list[Document] = []
    corpus_raw = data.get("corpus", data.get("documents", []))
    if isinstance(corpus_raw, dict):
        for doc_id, doc in corpus_raw.items():
            if isinstance(doc, dict):
                docs.append(
                    Document(
                        doc_id=str(doc_id),
                        title=str(doc.get("title", "")),
                        text=str(doc.get("text", "")),
                    )
                )
    elif isinstance(corpus_raw, list):
        for doc in corpus_raw:
            did = str(doc.get("doc_id") or doc.get("_id") or doc.get("id") or "")
            if not did:
                continue
            docs.append(
                Document(
                    doc_id=did,
                    title=str(doc.get("title", "")),
                    text=str(doc.get("text") or doc.get("content", "")),
                )
            )

    queries: list[Query] = []
    raw_queries = data.get("queries", [])
    qrels = data.get("relevant_docs", data.get("qrels", {}))
    if isinstance(raw_queries, dict):
        for qid, text in raw_queries.items():
            rel = qrels.get(qid, {})
            ids = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
            if ids and text:
                queries.append(Query(qid=str(qid), text=str(text), relevant=frozenset(ids)))
    elif isinstance(raw_queries, list):
        for q in raw_queries:
            qid = str(q.get("qid") or q.get("query_id") or q.get("_id") or "")
            text = str(q.get("query") or q.get("question") or "")
            rel = q.get("relevant_docs") or q.get("answer_ids") or q.get("positive_doc_ids") or []
            ids = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
            if ids and text:
                queries.append(Query(qid=qid, text=text, relevant=frozenset(ids)))

    return Corpus(name=name or path.stem, docs=docs, queries=queries)


# --- Metrics ----


def reciprocal_rank(retrieved: list[str], relevant: frozenset[str]) -> float:
    for i, did in enumerate(retrieved):
        if did in relevant:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved: list[str], relevant: frozenset[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: frozenset[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in relevant) / len(top_k)


# --- Run result ----


@dataclass
class RunResult:
    system: str
    corpus: str
    n_docs: int
    n_queries: int
    mrr: float
    recall_at_k: float
    precision_at_k: float
    hit_count: int
    build_sec: float
    search_sec: float
    k: int = 10
    error: str | None = None

    @property
    def hit_rate_str(self) -> str:
        return f"{self.hit_count}/{self.n_queries}"

    def to_row(self) -> list[str]:
        if self.error:
            return [self.system, self.corpus, "—", "—", "—", "—", "—", "ERROR: " + self.error]
        return [
            self.system,
            self.corpus,
            str(self.n_docs),
            f"{self.mrr:.3f}",
            f"{self.recall_at_k:.3f}",
            self.hit_rate_str,
            f"{self.build_sec:.1f}s",
            f"{self.search_sec:.1f}s",
        ]


HEADER = ["System", "Corpus", "Docs", "MRR", "R@10", "Hit", "Build", "Search"]


def score_run(
    system: str,
    corpus: Corpus,
    retrieved_per_query: list[list[str]],
    build_sec: float,
    search_sec: float,
    k: int = 10,
) -> RunResult:
    n = len(corpus.queries)
    mrr_total = 0.0
    recall_total = 0.0
    precision_total = 0.0
    hit_count = 0
    for query, retrieved in zip(corpus.queries, retrieved_per_query):
        rr = reciprocal_rank(retrieved[:k], query.relevant)
        mrr_total += rr
        recall_total += recall_at_k(retrieved, query.relevant, k)
        precision_total += precision_at_k(retrieved, query.relevant, k)
        if rr > 0:
            hit_count += 1
    return RunResult(
        system=system,
        corpus=corpus.name,
        n_docs=len(corpus.docs),
        n_queries=n,
        mrr=mrr_total / max(n, 1),
        recall_at_k=recall_total / max(n, 1),
        precision_at_k=precision_total / max(n, 1),
        hit_count=hit_count,
        build_sec=build_sec,
        search_sec=search_sec,
        k=k,
    )


def format_table(results: list[RunResult]) -> str:
    rows = [HEADER] + [r.to_row() for r in results]
    widths = [max(len(row[i]) for row in rows) for i in range(len(HEADER))]
    lines = []
    for r, row in enumerate(rows):
        cells = [row[i].ljust(widths[i]) for i in range(len(widths))]
        lines.append("  ".join(cells))
        if r == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


class Timer:
    """Context manager that captures elapsed wall-clock seconds."""

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0
