"""In-memory storage backend for testing."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from synaptic.models import (
    ConsolidationLevel,
    Edge,
    Node,
    NodeKind,
)


class MemoryBackend:
    """Dict-based in-memory backend. No external dependencies."""

    __slots__ = ("_edges", "_nodes")

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        self._nodes.clear()
        self._edges.clear()

    # --- Node CRUD ---

    async def save_node(self, node: Node) -> None:
        self._nodes[node.id] = node

    async def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    async def update_node(self, node: Node) -> None:
        if node.id in self._nodes:
            self._nodes[node.id] = node

    async def delete_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        # Cascade delete edges
        to_delete = [
            eid
            for eid, e in self._edges.items()
            if e.source_id == node_id or e.target_id == node_id
        ]
        for eid in to_delete:
            del self._edges[eid]

    async def list_nodes(
        self,
        *,
        kind: str | NodeKind | None = None,
        level: ConsolidationLevel | None = None,
        limit: int = 100,
    ) -> list[Node]:
        result: list[Node] = []
        for node in self._nodes.values():
            if kind is not None and node.kind != kind:
                continue
            if level is not None and node.level != level:
                continue
            result.append(node)
            if len(result) >= limit:
                break
        return result

    # --- Edge CRUD ---

    async def save_edge(self, edge: Edge) -> None:
        self._edges[edge.id] = edge

    async def get_edges(self, node_id: str, *, direction: str = "both") -> list[Edge]:
        result: list[Edge] = []
        for edge in self._edges.values():
            if direction in ("both", "outgoing") and edge.source_id == node_id:
                result.append(edge)
            elif direction in ("both", "incoming") and edge.target_id == node_id:
                result.append(edge)
        return result

    async def update_edge(self, edge: Edge) -> None:
        if edge.id in self._edges:
            self._edges[edge.id] = edge

    async def delete_edge(self, edge_id: str) -> None:
        self._edges.pop(edge_id, None)

    # --- Search ---

    async def search_fts(self, query: str, *, limit: int = 20) -> list[Node]:
        query_lower = query.lower()
        terms = [t for t in query_lower.split() if len(t) >= 1]
        if not terms:
            return []

        # --- BM25 parameters ---
        k1 = 1.5
        b = 0.75
        title_boost = 3.0

        # Pre-compute corpus statistics for BM25
        N = len(self._nodes)  # total documents
        if N == 0:
            return []

        # Document frequencies: how many docs contain each term (substring match)
        doc_freq: dict[str, int] = {}
        doc_texts: dict[str, str] = {}  # node_id → full searchable text
        doc_lengths: dict[str, int] = {}  # node_id → word count

        for node in self._nodes.values():
            text = f"{node.title.lower()} {node.content.lower()}"
            if node.tags:
                text += " " + " ".join(node.tags).lower()
            if node.properties:
                kw = node.properties.get("_search_keywords", "")
                if kw:
                    text += " " + kw.lower()
            doc_texts[node.id] = text
            doc_lengths[node.id] = len(text.split())

        avgdl = sum(doc_lengths.values()) / N if N > 0 else 1.0

        for t in terms:
            count = 0
            for text in doc_texts.values():
                if t in text:
                    count += 1
            doc_freq[t] = count

        # Bigrams for phrase matching
        bigrams: list[str] = []
        if len(terms) >= 2:
            for i in range(len(terms) - 1):
                bigrams.append(f"{terms[i]} {terms[i + 1]}")

        # --- Score each document (BM25 + substring hybrid) ---
        scored: list[tuple[Node, float]] = []
        for node in self._nodes.values():
            title_lower = node.title.lower()
            content_lower = node.content.lower()
            full_text = doc_texts[node.id]
            dl = doc_lengths[node.id]

            bm25_score = 0.0
            substr_score = 0.0

            for t in terms:
                tf_content = content_lower.count(t)
                tf_title = title_lower.count(t)

                if tf_content == 0 and tf_title == 0:
                    continue

                # --- BM25 component ---
                df = doc_freq.get(t, 0)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

                if tf_content > 0:
                    numerator = tf_content * (k1 + 1)
                    denominator = tf_content + k1 * (1 - b + b * dl / avgdl)
                    bm25_score += idf * numerator / denominator

                if tf_title > 0:
                    bm25_score += idf * title_boost

                # --- Substring component (corpus-size independent) ---
                if tf_title > 0:
                    substr_score += 2.0
                if tf_content > 0:
                    substr_score += 1.0

            # Bigram bonus
            for bg in bigrams:
                if bg in full_text:
                    bm25_score += 1.5
                    substr_score += 1.5

            # Tag match
            if node.tags:
                tag_text = " ".join(node.tags).lower()
                for t in terms:
                    if t in tag_text:
                        substr_score += 1.0

            # Search keywords
            if node.properties:
                search_kw = node.properties.get("_search_keywords", "").lower()
                if search_kw:
                    for t in terms:
                        if t in search_kw:
                            substr_score += 1.5

            # Hybrid: BM25 weight increases with corpus size
            # N=100: 30% BM25 + 70% substr, N=1000+: 80% BM25 + 20% substr
            bm25_weight = min(0.8, 0.3 + 0.5 * min(1.0, N / 1000))
            score = bm25_score * bm25_weight + substr_score * (1 - bm25_weight)

            if score > 0:
                scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scored[:limit]]

    async def search_fuzzy(
        self, query: str, *, limit: int = 20, threshold: float = 0.4
    ) -> list[Node]:
        query_lower = query.lower()
        # Deduplicate and cap query terms to avoid O(n*m) explosion on long queries
        query_terms = list(dict.fromkeys(query_lower.split()))[:10]
        scored: list[tuple[Node, float]] = []
        for node in self._nodes.values():
            title_lower = node.title.lower()
            # Compare against title (short text → fair ratio)
            title_ratio = SequenceMatcher(None, query_lower[:200], title_lower).ratio()
            best = title_ratio

            # Per-term fuzzy: match each query term against title words + content sample
            if query_terms:
                title_words = title_lower.split()
                # Content: first 100 words for broader coverage
                content_words = node.content.lower().split()[:100]
                # Tag words too
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

                # Title term match bonus: boost when term is exactly in title
                title_boost = sum(0.1 for qt in query_terms if qt in title_lower)
                best = max(best, avg_term) + title_boost

            if best >= threshold:
                scored.append((node, best))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scored[:limit]]

    async def search_vector(self, embedding: list[float], *, limit: int = 20) -> list[Node]:
        if not embedding:
            return []
        scored: list[tuple[Node, float]] = []
        for node in self._nodes.values():
            if not node.embedding:
                continue
            sim = _cosine_similarity(embedding, node.embedding)
            scored.append((node, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scored[:limit]]

    # --- Graph traversal ---

    async def get_neighbors(self, node_id: str, *, depth: int = 1) -> list[tuple[Node, Edge]]:
        result: list[tuple[Node, Edge]] = []
        visited: set[str] = {node_id}
        frontier: set[str] = {node_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                for edge in self._edges.values():
                    neighbor_id: str | None = None
                    if edge.source_id == nid and edge.target_id not in visited:
                        neighbor_id = edge.target_id
                    elif edge.target_id == nid and edge.source_id not in visited:
                        neighbor_id = edge.source_id

                    if neighbor_id is not None:
                        neighbor = self._nodes.get(neighbor_id)
                        if neighbor is not None:
                            result.append((neighbor, edge))
                            visited.add(neighbor_id)
                            next_frontier.add(neighbor_id)
            frontier = next_frontier

        return result

    # --- Batch ---

    async def save_nodes_batch(self, nodes: Sequence[Node]) -> None:
        for node in nodes:
            self._nodes[node.id] = node

    async def save_edges_batch(self, edges: Sequence[Edge]) -> None:
        for edge in edges:
            self._edges[edge.id] = edge

    # --- Maintenance ---

    async def prune_edges(self, *, weight_below: float = 0.1) -> int:
        to_delete = [eid for eid, e in self._edges.items() if e.weight < weight_below]
        for eid in to_delete:
            del self._edges[eid]
        return len(to_delete)

    async def decay_vitality(self, *, factor: float = 0.95) -> int:
        count = 0
        for node in self._nodes.values():
            node.vitality *= factor
            count += 1
        return count


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
