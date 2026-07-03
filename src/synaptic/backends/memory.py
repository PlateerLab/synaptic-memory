"""In-memory storage backend for testing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from difflib import SequenceMatcher

from synaptic.indexing import IndexLagReport, IngestionJob, IngestionJobStage, IngestionJobStatus
from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    MemoryEvent,
    MemoryScope,
    MemoryScore,
    Node,
    NodeKind,
    RetrievalEvent,
    memory_scope_key,
)


class MemoryBackend:
    """Dict-based in-memory backend. No external dependencies."""

    __slots__ = (
        "_edges",
        "_index_lag_reports",
        "_ingestion_jobs",
        "_memory_events",
        "_memory_scores",
        "_nodes",
        "_retrieval_events",
    )

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._memory_events: dict[str, MemoryEvent] = {}
        self._retrieval_events: dict[str, RetrievalEvent] = {}
        self._memory_scores: dict[tuple[str, str, str], MemoryScore] = {}
        self._ingestion_jobs: dict[str, IngestionJob] = {}
        self._index_lag_reports: list[IndexLagReport] = []

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._memory_events.clear()
        self._retrieval_events.clear()
        self._memory_scores.clear()
        self._ingestion_jobs.clear()
        self._index_lag_reports.clear()

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

    async def list_nodes_by_tag(
        self,
        tag: str,
        *,
        kind: str | NodeKind | None = None,
        limit: int = 100,
    ) -> list[Node]:
        result: list[Node] = []
        for node in self._nodes.values():
            if kind is not None and node.kind != kind:
                continue
            if tag not in (node.tags or []):
                continue
            result.append(node)
            if len(result) >= limit:
                break
        return result

    async def get_nodes_batch(self, node_ids: list[str]) -> list[Node]:
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    async def count_nodes(
        self,
        *,
        kind: str | NodeKind | None = None,
        category: str | None = None,
        year: int | None = None,
    ) -> int:
        count = 0
        for node in self._nodes.values():
            if kind is not None and node.kind != kind:
                continue
            props = node.properties or {}
            if category and category.lower() not in (props.get("category") or "").lower():
                continue
            if year is not None and str(props.get("year")) != str(year):
                continue
            count += 1
        return count

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

    async def get_edges_batch(
        self, node_ids: list[str], *, direction: str = "both"
    ) -> dict[str, list[Edge]]:
        result: dict[str, list[Edge]] = {nid: [] for nid in dict.fromkeys(node_ids)}
        if not result:
            return result
        node_set = set(result)
        for edge in self._edges.values():
            if direction in ("both", "outgoing") and edge.source_id in node_set:
                result[edge.source_id].append(edge)
            if (
                direction in ("both", "incoming")
                and edge.target_id in node_set
                and not (direction == "both" and edge.target_id == edge.source_id)
            ):
                result[edge.target_id].append(edge)
        return result

    async def get_edges_batch_light(
        self, node_ids: list[str], *, direction: str = "both"
    ) -> dict[str, list[Edge]]:
        result: dict[str, list[Edge]] = {nid: [] for nid in dict.fromkeys(node_ids)}
        if not result:
            return result
        node_set = set(result)
        for edge in self._edges.values():
            if direction in ("both", "outgoing") and edge.source_id in node_set:
                result[edge.source_id].append(edge)
            if (
                direction in ("both", "incoming")
                and edge.target_id in node_set
                and not (direction == "both" and edge.target_id == edge.source_id)
            ):
                result[edge.target_id].append(edge)
        return result

    async def get_edges_batch_filtered(
        self,
        node_ids: list[str],
        *,
        direction: str = "both",
        kinds: Sequence[str | EdgeKind],
    ) -> dict[str, list[Edge]]:
        result: dict[str, list[Edge]] = {nid: [] for nid in dict.fromkeys(node_ids)}
        if not result:
            return result
        kind_set = {kind.value if isinstance(kind, EdgeKind) else str(kind) for kind in kinds}
        if not kind_set:
            return result
        node_set = set(result)
        for edge in self._edges.values():
            if edge.kind.value not in kind_set:
                continue
            if direction in ("both", "outgoing") and edge.source_id in node_set:
                result[edge.source_id].append(edge)
            if (
                direction in ("both", "incoming")
                and edge.target_id in node_set
                and not (direction == "both" and edge.target_id == edge.source_id)
            ):
                result[edge.target_id].append(edge)
        return result

    async def get_edges_batch_filtered_light(
        self,
        node_ids: list[str],
        *,
        direction: str = "both",
        kinds: Sequence[str | EdgeKind],
    ) -> dict[str, list[Edge]]:
        return await self.get_edges_batch_filtered(node_ids, direction=direction, kinds=kinds)

    async def get_edges_batch_filtered_selective_light(
        self,
        node_ids: list[str],
        *,
        direction: str = "both",
        light_kinds: Sequence[str | EdgeKind],
        full_kinds: Sequence[str | EdgeKind],
    ) -> dict[str, list[Edge]]:
        result: dict[str, list[Edge]] = {nid: [] for nid in dict.fromkeys(node_ids)}
        if not result:
            return result
        light_set = {
            kind.value if isinstance(kind, EdgeKind) else str(kind) for kind in light_kinds
        }
        full_set = {kind.value if isinstance(kind, EdgeKind) else str(kind) for kind in full_kinds}
        kind_set = light_set | full_set
        if not kind_set:
            return result
        node_set = set(result)
        for edge in self._edges.values():
            edge_kind = edge.kind.value if isinstance(edge.kind, EdgeKind) else str(edge.kind)
            if edge_kind not in kind_set:
                continue
            materialized = edge
            if edge_kind in light_set:
                materialized = Edge(
                    id=edge.id,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    kind=edge.kind,
                    weight=edge.weight,
                    properties={},
                    created_at=edge.created_at,
                )
            if direction in ("both", "outgoing") and edge.source_id in node_set:
                result[edge.source_id].append(materialized)
            if (
                direction in ("both", "incoming")
                and edge.target_id in node_set
                and not (direction == "both" and edge.target_id == edge.source_id)
            ):
                result[edge.target_id].append(materialized)
        return result

    async def has_edges_of_kind(self, kind: str | EdgeKind) -> bool:
        kind_value = kind.value if isinstance(kind, EdgeKind) else str(kind)
        return any(
            (edge.kind.value if isinstance(edge.kind, EdgeKind) else str(edge.kind)) == kind_value
            for edge in self._edges.values()
        )

    async def update_edge(self, edge: Edge) -> None:
        if edge.id in self._edges:
            self._edges[edge.id] = edge

    async def stamp_openie_edges_event(self, edge_ids: Sequence[str], event_id: str) -> int:
        updated = 0
        for edge_id in set(edge_ids):
            edge = self._edges.get(edge_id)
            if edge is None:
                continue
            props = dict(edge.properties or {})
            if props.get("source_event_id") == event_id:
                continue
            props["source_event_id"] = event_id
            edge.properties = props
            updated += 1
        return updated

    async def delete_edge(self, edge_id: str) -> None:
        self._edges.pop(edge_id, None)

    async def purge_openie_artifacts(self, *, node_limit: int = 1_000_000) -> int:
        edge_ids = [edge_id for edge_id in self._edges if edge_id.startswith("openie_")]
        node_ids = [
            node_id for node_id, node in self._nodes.items() if "_openie" in (node.tags or [])
        ][:node_limit]
        for edge_id in edge_ids:
            self._edges.pop(edge_id, None)
        for node_id in node_ids:
            self._nodes.pop(node_id, None)
        openie_nodes = set(node_ids)
        for edge_id, edge in list(self._edges.items()):
            if edge.source_id in openie_nodes or edge.target_id in openie_nodes:
                del self._edges[edge_id]
        return len(edge_ids) + len(node_ids)

    # --- Memory operating layer ---

    async def save_memory_event(self, event: MemoryEvent) -> None:
        self._memory_events[event.id] = event

    async def save_memory_events_batch(self, events: Sequence[MemoryEvent]) -> None:
        for event in events:
            self._memory_events[event.id] = event

    async def list_memory_events(
        self,
        *,
        kind: str | None = None,
        scope: MemoryScope | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        scope_key = memory_scope_key(scope) if scope is not None else None
        out: list[MemoryEvent] = []
        for event in sorted(self._memory_events.values(), key=lambda e: e.created_at, reverse=True):
            if kind is not None and str(event.kind) != str(kind):
                continue
            if scope_key is not None and memory_scope_key(event.scope) != scope_key:
                continue
            if since is not None and event.created_at < since:
                continue
            out.append(event)
            if len(out) >= limit:
                break
        return out

    async def save_retrieval_event(self, event: RetrievalEvent) -> None:
        self._retrieval_events[event.id] = event

    async def get_retrieval_event(self, event_id: str) -> RetrievalEvent | None:
        return self._retrieval_events.get(event_id)

    async def list_retrieval_events(
        self,
        *,
        scope: MemoryScope | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[RetrievalEvent]:
        scope_key = memory_scope_key(scope) if scope is not None else None
        out: list[RetrievalEvent] = []
        for event in sorted(
            self._retrieval_events.values(), key=lambda e: e.created_at, reverse=True
        ):
            if scope_key is not None and memory_scope_key(event.scope) != scope_key:
                continue
            if since is not None and event.created_at < since:
                continue
            out.append(event)
            if len(out) >= limit:
                break
        return out

    async def save_memory_score(self, score: MemoryScore) -> None:
        self._memory_scores[(score.scope_key, score.node_id, score.edge_id)] = score

    async def get_memory_score(
        self,
        scope_key: str,
        *,
        node_id: str = "",
        edge_id: str = "",
    ) -> MemoryScore | None:
        return self._memory_scores.get((scope_key, node_id, edge_id))

    async def list_memory_scores(
        self,
        *,
        scope_key: str | None = None,
        node_ids: list[str] | None = None,
        edge_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[MemoryScore]:
        node_filter = set(node_ids or [])
        edge_filter = set(edge_ids or [])
        out: list[MemoryScore] = []
        scores = sorted(self._memory_scores.values(), key=lambda s: s.score, reverse=True)
        for score in scores:
            if scope_key is not None and score.scope_key != scope_key:
                continue
            if node_filter and score.node_id not in node_filter:
                continue
            if edge_filter and score.edge_id not in edge_filter:
                continue
            out.append(score)
            if len(out) >= limit:
                break
        return out

    async def save_ingestion_job(self, job: IngestionJob) -> None:
        self._ingestion_jobs[job.id] = job

    async def get_ingestion_job(self, job_id: str) -> IngestionJob | None:
        return self._ingestion_jobs.get(job_id)

    async def list_ingestion_jobs(
        self,
        *,
        document_id: str = "",
        stage: str | IngestionJobStage | None = None,
        status: str | IngestionJobStatus | None = None,
        limit: int = 100,
    ) -> list[IngestionJob]:
        out: list[IngestionJob] = []
        jobs = sorted(self._ingestion_jobs.values(), key=lambda job: job.updated_at, reverse=True)
        for job in jobs:
            if document_id and job.document_id != document_id:
                continue
            if stage is not None and str(job.stage) != str(stage):
                continue
            if status is not None and str(job.status) != str(status):
                continue
            out.append(job)
            if len(out) >= limit:
                break
        return out

    async def save_index_lag_report(self, report: IndexLagReport) -> None:
        self._index_lag_reports.append(report)

    async def list_index_lag_reports(
        self,
        *,
        provider: str = "",
        since: float | None = None,
        limit: int = 100,
    ) -> list[IndexLagReport]:
        out: list[IndexLagReport] = []
        reports = sorted(
            self._index_lag_reports,
            key=lambda report: report.checked_at,
            reverse=True,
        )
        for report in reports:
            if provider and report.provider != provider:
                continue
            if since is not None and report.checked_at < since:
                continue
            out.append(report)
            if len(out) >= limit:
                break
        return out

    # --- Search ---

    async def search_fts(self, query: str, *, limit: int = 20) -> list[Node]:
        # Apply the same query-time noise-stripping used by SqliteBackend —
        # Korean question-form verbs ("설명해주세요 / 무엇인가요 / 어떻게")
        # are dropped for natural-language Korean queries (≥50 % Hangul),
        # which measurably improves MRR on the Allganize RAG-ko / RAG-Eval
        # benchmarks. English / code queries are unchanged.
        from synaptic.backends.sqlite import _normalize_korean

        normalized = _normalize_korean(query, query_mode=True)
        query_lower = normalized.lower() if normalized else query.lower()
        terms = [t for t in query_lower.split() if len(t) >= 1]
        # Re-add original tokens that carry structural signal (digits /
        # ASCII letters) — same guard as SqliteBackend.search_fts.
        seen = set(terms)
        for t in query.strip().split():
            tl = t.lower()
            if tl in seen:
                continue
            if any(c.isdigit() or ("a" <= c.lower() <= "z") for c in tl):
                terms.append(tl)
                seen.add(tl)
        if not terms:
            return []

        # --- BM25 parameters ---
        k1 = 1.5
        b = 0.75
        title_boost = 3.0

        # Pre-compute corpus statistics for BM25
        n_docs = len(self._nodes)  # total documents
        if n_docs == 0:
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

        avgdl = sum(doc_lengths.values()) / n_docs if n_docs > 0 else 1.0

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
            matched_terms = 0  # query term coverage 계산용

            # Full query in title — 강한 신호 (모든 corpus 크기에서 유효)
            if query_lower in title_lower:
                substr_score += len(terms) * 3.0

            for t in terms:
                tf_content = content_lower.count(t)
                tf_title = title_lower.count(t)

                if tf_content == 0 and tf_title == 0:
                    continue

                # --- BM25 component ---
                df = doc_freq.get(t, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

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
                matched_terms += 1

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

            # Query term coverage bonus — 쿼리 단어 대부분 매칭 시 보너스
            # coverage 80%+ → 보너스, 대규모 corpus에서 precision 향상
            if len(terms) >= 2 and matched_terms > 0:
                coverage = matched_terms / len(terms)
                if coverage >= 0.8:
                    substr_score += len(terms) * 1.5  # 높은 coverage 보상
                elif coverage >= 0.5:
                    substr_score += len(terms) * 0.5

            # Hybrid: BM25 weight increases with corpus size
            # N≤500: mostly substring, N=5000+: mostly BM25
            bm25_weight = min(0.8, max(0.1, (n_docs - 500) / 5000))
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
