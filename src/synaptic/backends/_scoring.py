"""Shared BM25 + substring hybrid scoring used by in-process backends.

Memory and Kuzu backends share this scoring so search ranking stays
identical across them (parity guarantee). Any backend that can provide a
`list[Node]` candidate set may reuse these functions.

The BM25 hybrid is lifted from the v0.10.0 tuning in ``MemoryBackend``
and kept here verbatim so that swapping backends does not change IR
metrics.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from difflib import SequenceMatcher

from synaptic.models import Node


def bm25_hybrid_score(
    nodes: Iterable[Node],
    query: str,
    *,
    limit: int = 20,
) -> list[Node]:
    """Rank ``nodes`` against ``query`` using BM25 + substring hybrid.

    This mirrors ``MemoryBackend.search_fts``:
      - Okapi BM25 with k1=1.5, b=0.75 and a 3x title boost
      - Substring component (corpus-size independent)
      - Bigram bonus, tag bonus, ``_search_keywords`` bonus
      - Query term coverage bonus
      - Adaptive BM25/substring weight based on corpus size
    """
    query_lower = query.lower()
    terms = [t for t in query_lower.split() if len(t) >= 1]
    if not terms:
        return []

    node_list = list(nodes)
    n_docs = len(node_list)
    if n_docs == 0:
        return []

    k1 = 1.5
    b = 0.75
    title_boost = 3.0

    doc_texts: dict[str, str] = {}
    doc_lengths: dict[str, int] = {}
    for node in node_list:
        text = f"{node.title.lower()} {node.content.lower()}"
        if node.tags:
            text += " " + " ".join(node.tags).lower()
        if node.properties:
            kw = node.properties.get("_search_keywords", "")
            if kw:
                text += " " + kw.lower()
        doc_texts[node.id] = text
        doc_lengths[node.id] = len(text.split())

    avgdl = sum(doc_lengths.values()) / n_docs if n_docs > 0 else 1.0

    doc_freq: dict[str, int] = {}
    for t in terms:
        count = 0
        for text in doc_texts.values():
            if t in text:
                count += 1
        doc_freq[t] = count

    bigrams: list[str] = []
    if len(terms) >= 2:
        for i in range(len(terms) - 1):
            bigrams.append(f"{terms[i]} {terms[i + 1]}")

    scored: list[tuple[Node, float]] = []
    for node in node_list:
        title_lower = node.title.lower()
        content_lower = node.content.lower()
        full_text = doc_texts[node.id]
        dl = doc_lengths[node.id]

        bm25_score = 0.0
        substr_score = 0.0
        matched_terms = 0

        if query_lower in title_lower:
            substr_score += len(terms) * 3.0

        for t in terms:
            tf_content = content_lower.count(t)
            tf_title = title_lower.count(t)
            if tf_content == 0 and tf_title == 0:
                continue

            df = doc_freq.get(t, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

            if tf_content > 0:
                numerator = tf_content * (k1 + 1)
                denominator = tf_content + k1 * (1 - b + b * dl / avgdl)
                bm25_score += idf * numerator / denominator
            if tf_title > 0:
                bm25_score += idf * title_boost

            if tf_title > 0:
                substr_score += 2.0
            if tf_content > 0:
                substr_score += 1.0
            matched_terms += 1

        for bg in bigrams:
            if bg in full_text:
                bm25_score += 1.5
                substr_score += 1.5

        if node.tags:
            tag_text = " ".join(node.tags).lower()
            for t in terms:
                if t in tag_text:
                    substr_score += 1.0

        if node.properties:
            search_kw = node.properties.get("_search_keywords", "").lower()
            if search_kw:
                for t in terms:
                    if t in search_kw:
                        substr_score += 1.5

        if len(terms) >= 2 and matched_terms > 0:
            coverage = matched_terms / len(terms)
            if coverage >= 0.8:
                substr_score += len(terms) * 1.5
            elif coverage >= 0.5:
                substr_score += len(terms) * 0.5

        bm25_weight = min(0.8, max(0.1, (n_docs - 500) / 5000))
        score = bm25_score * bm25_weight + substr_score * (1 - bm25_weight)

        if score > 0:
            scored.append((node, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored[:limit]]


def fuzzy_score(
    nodes: Iterable[Node],
    query: str,
    *,
    limit: int = 20,
    threshold: float = 0.4,
) -> list[Node]:
    """Fuzzy string matching across title/content/tags (``MemoryBackend`` parity)."""
    query_lower = query.lower()
    query_terms = list(dict.fromkeys(query_lower.split()))[:10]

    scored: list[tuple[Node, float]] = []
    for node in nodes:
        title_lower = node.title.lower()
        title_ratio = SequenceMatcher(None, query_lower[:200], title_lower).ratio()
        best = title_ratio

        if query_terms:
            title_words = title_lower.split()
            content_words = node.content.lower().split()[:100]
            tag_words = [t.lower() for t in (node.tags or [])]
            text_words = title_words + content_words + tag_words

            term_scores: list[float] = []
            for qt in query_terms:
                term_best = 0.0
                for tw in text_words:
                    r = SequenceMatcher(None, qt, tw).ratio()
                    if r > term_best:
                        term_best = r
                term_scores.append(term_best)
            avg_term = sum(term_scores) / len(term_scores)

            title_boost = sum(0.1 for qt in query_terms if qt in title_lower)
            best = max(best, avg_term) + title_boost

        if best >= threshold:
            scored.append((node, best))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored[:limit]]
