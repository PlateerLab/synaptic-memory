"""Scale-out indexing contracts.

These types define the optional boundary between the retrieval core and
large external indexes. They do not change the default StorageBackend path;
instead they make the 10TB route explicit: providers return scored candidate
ids, and the normal retrieval pipeline decides how to hydrate, expand, rerank,
and record them.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from typing import Protocol
from uuid import uuid4

from synaptic.models import MemoryScope

__all__ = [
    "CandidateProvider",
    "CandidateScoreSource",
    "CandidateSearchRequest",
    "CandidateSearchResult",
    "IndexFilter",
    "IndexHealthBackend",
    "IndexLagReport",
    "IndexRouter",
    "InProcessIndexRouter",
    "IngestionJob",
    "IngestionJobStage",
    "IngestionJobStatus",
    "IngestionJobStore",
    "OpenSearchCandidateProvider",
    "QdrantCandidateProvider",
    "ScoredCandidate",
    "StorageCandidateProvider",
    "unique_candidates",
]


def _new_id() -> str:
    return uuid4().hex[:16]


def _str_dict() -> dict[str, str]:
    return {}


def _object_dict() -> dict[str, object]:
    return {}


def _int_dict() -> dict[str, int]:
    return {}


def _float_dict() -> dict[str, float]:
    return {}


def _scope() -> MemoryScope:
    return MemoryScope()


class CandidateScoreSource(StrEnum):
    """Origin of a candidate score before router-level fusion."""

    LEXICAL = "lexical"
    VECTOR = "vector"
    GRAPH = "graph"
    MEMORY = "memory"
    RERANK = "rerank"
    ROUTER = "router"


class IngestionJobStage(StrEnum):
    """Durable ingestion/indexing stages for large document corpora."""

    DISCOVER = "discover"
    PARSE = "parse"
    OCR = "ocr"
    CHUNK = "chunk"
    EMBED = "embed"
    LEXICAL_INDEX = "lexical_index"
    VECTOR_INDEX = "vector_index"
    GRAPH_INDEX = "graph_index"
    SEMANTIC_EXTRACT = "semantic_extract"
    LEDGER_COMMIT = "ledger_commit"


class IngestionJobStatus(StrEnum):
    """State of a durable ingestion job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class IndexFilter:
    """Provider-side filters that must be applied before candidate materialization.

    Empty strings and empty lists mean "no filter". Providers must not interpret
    an empty string as a literal workspace/user/session/domain value.
    """

    workspace_id: str = ""
    user_id: str = ""
    session_id: str = ""
    domain: str = ""
    acl_policy_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    mime_types: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_after: float | None = None
    created_before: float | None = None
    updated_after: float | None = None
    updated_before: float | None = None
    properties: dict[str, str] = field(default_factory=_str_dict)


@dataclass(slots=True)
class CandidateSearchRequest:
    """A routed candidate search request.

    `query` is the user-visible search text. `query_variants` contains bounded
    deterministic rewrites or agent-proposed follow-up targets. Providers may
    search variants independently but must preserve `query_variant` in returned
    candidates so diagnostics can explain which route found each hit.
    """

    query: str
    query_variants: list[str] = field(default_factory=list)
    limit: int = 20
    filters: IndexFilter = field(default_factory=IndexFilter)
    scope: MemoryScope = field(default_factory=_scope)
    embedding: list[float] | None = None
    providers: list[str] = field(default_factory=list)
    properties: dict[str, str] = field(default_factory=_str_dict)


@dataclass(slots=True)
class ScoredCandidate:
    """Scored candidate id returned by an external index provider."""

    node_id: str
    document_id: str = ""
    score: float = 0.0
    score_source: CandidateScoreSource = CandidateScoreSource.ROUTER
    rank: int = 0
    query_variant: str = ""
    provider: str = ""
    index_generation: str = ""
    metadata: dict[str, object] = field(default_factory=_object_dict)


@dataclass(slots=True)
class CandidateSearchResult:
    """Provider/router candidate response before node hydration."""

    candidates: list[ScoredCandidate] = field(default_factory=list)
    provider_counts: dict[str, int] = field(default_factory=_int_dict)
    score_ranges: dict[str, float] = field(default_factory=_float_dict)
    diagnostics: dict[str, object] = field(default_factory=_object_dict)
    index_generation: str = ""


@dataclass(slots=True)
class IndexLagReport:
    """Operational freshness report for an index or provider."""

    provider: str = ""
    status: str = "unknown"
    index_generation: str = ""
    pending_documents: int = 0
    pending_chunks: int = 0
    failed_jobs: int = 0
    lag_seconds: float = 0.0
    p95_index_latency_seconds: float = 0.0
    properties: dict[str, object] = field(default_factory=_object_dict)
    checked_at: float = field(default_factory=time)


@dataclass(slots=True)
class IngestionJob:
    """Durable unit of indexing work for incremental corpus updates."""

    id: str = field(default_factory=_new_id)
    document_id: str = ""
    version: str = ""
    stage: IngestionJobStage = IngestionJobStage.DISCOVER
    status: IngestionJobStatus = IngestionJobStatus.PENDING
    attempt: int = 0
    error: str = ""
    properties: dict[str, str] = field(default_factory=_str_dict)
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)


class CandidateProvider(Protocol):
    """External index provider that returns candidate ids, not hydrated nodes."""

    @property
    def name(self) -> str: ...

    async def search_candidates(
        self,
        request: CandidateSearchRequest,
    ) -> CandidateSearchResult: ...

    async def index_health(self) -> IndexLagReport: ...


class IndexRouter(Protocol):
    """Coordinates multiple providers and returns fused candidate ids."""

    async def search_candidates(
        self,
        request: CandidateSearchRequest,
    ) -> CandidateSearchResult: ...

    async def index_health(self) -> list[IndexLagReport]: ...


class IngestionJobStore(Protocol):
    """Durable ingestion queue/state store."""

    async def save_ingestion_job(self, job: IngestionJob) -> None: ...
    async def get_ingestion_job(self, job_id: str) -> IngestionJob | None: ...
    async def list_ingestion_jobs(
        self,
        *,
        document_id: str = "",
        stage: str | IngestionJobStage | None = None,
        status: str | IngestionJobStatus | None = None,
        limit: int = 100,
    ) -> list[IngestionJob]: ...


class IndexHealthBackend(Protocol):
    """Aggregated freshness/lag reporting across index providers."""

    async def save_index_lag_report(self, report: IndexLagReport) -> None: ...
    async def list_index_lag_reports(
        self,
        *,
        provider: str = "",
        since: float | None = None,
        limit: int = 100,
    ) -> list[IndexLagReport]: ...


class StorageCandidateProvider:
    """Compatibility provider backed by the existing StorageBackend.

    This is the Phase-2 bridge: it proves the router boundary without requiring
    OpenSearch/Qdrant. External providers should apply filters natively before
    scoring; this compatibility provider over-fetches and post-filters because
    legacy StorageBackend search methods do not accept a structured filter.
    """

    __slots__ = ("_backend", "_max_overfetch", "_name", "_overfetch_factor")

    def __init__(
        self,
        backend: object,
        *,
        name: str = "storage_fts",
        overfetch_factor: int = 5,
        max_overfetch: int = 1000,
    ) -> None:
        self._backend = backend
        self._name = name
        self._overfetch_factor = max(1, int(overfetch_factor))
        self._max_overfetch = max(1, int(max_overfetch))

    @property
    def name(self) -> str:
        return self._name

    async def search_candidates(
        self,
        request: CandidateSearchRequest,
    ) -> CandidateSearchResult:
        limit = max(0, int(request.limit))
        if limit == 0:
            return CandidateSearchResult(index_generation="storage")

        queries = _unique_queries(request.query, request.query_variants)
        if not queries:
            return CandidateSearchResult(index_generation="storage")

        search_limit = limit
        if _has_filters(request.filters):
            search_limit = min(self._max_overfetch, max(limit, limit * self._overfetch_factor))

        candidates: list[ScoredCandidate] = []
        for query_variant in queries:
            rows = await self._search_fts(
                query_variant,
                limit=search_limit,
                include_embedding=request.embedding is not None,
            )
            for rank, row in enumerate(rows):
                node, score = _node_and_score(row, rank)
                if not _matches_filter(node, request.filters):
                    continue
                candidates.append(
                    ScoredCandidate(
                        node_id=str(getattr(node, "id", "")),
                        document_id=_document_id(node),
                        score=score,
                        score_source=CandidateScoreSource.LEXICAL,
                        rank=rank + 1,
                        query_variant=query_variant,
                        provider=self.name,
                        index_generation="storage",
                        metadata=_candidate_metadata(node),
                    )
                )

        return CandidateSearchResult(
            candidates=candidates,
            provider_counts={self.name: len(candidates)},
            score_ranges=_provider_score_max(candidates),
            diagnostics={
                "query_variant_count": len(queries),
                "filtered": _has_filters(request.filters),
                "search_limit": search_limit,
            },
            index_generation="storage",
        )

    async def index_health(self) -> IndexLagReport:
        return IndexLagReport(provider=self.name, status="ok", index_generation="storage")

    async def _search_fts(
        self,
        query: str,
        *,
        limit: int,
        include_embedding: bool,
    ) -> list[object]:
        search_fts = self._backend.search_fts
        kwargs: dict[str, object] = {"limit": limit}
        try:
            params = inspect.signature(search_fts).parameters
        except (TypeError, ValueError):
            params = {}
        if "with_scores" in params:
            kwargs["with_scores"] = True
        if "include_embedding" in params:
            kwargs["include_embedding"] = include_embedding
        return list(await search_fts(query, **kwargs))


class OpenSearchCandidateProvider:
    """OpenSearch/Elasticsearch lexical provider.

    The provider accepts an already-created async or sync client so importing
    synaptic does not require OpenSearch dependencies. It applies structured
    `IndexFilter` clauses in the index query before materialization and returns
    only scored candidate ids plus compact metadata.
    """

    __slots__ = (
        "_client",
        "_field_map",
        "_index",
        "_metadata_fields",
        "_name",
        "_query_fields",
    )

    def __init__(
        self,
        client: object,
        *,
        index: str,
        name: str = "opensearch",
        query_fields: Sequence[str] = ("title^3", "content", "tags", "properties.search_keywords"),
        metadata_fields: Sequence[str] = (
            "node_id",
            "document_id",
            "doc_id",
            "title",
            "kind",
            "source",
            "page",
            "chunk_id",
            "category",
            "domain",
            "language",
        ),
        field_map: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._index = index
        self._name = name
        self._query_fields = list(query_fields)
        self._metadata_fields = list(metadata_fields)
        self._field_map = {
            "node_id": "node_id",
            "document_id": "document_id",
            "doc_id": "doc_id",
            "workspace_id": "workspace_id",
            "user_id": "user_id",
            "session_id": "session_id",
            "domain": "domain",
            "acl_policy_id": "acl_policy_id",
            "source_id": "source_id",
            "source": "source",
            "language": "language",
            "mime_type": "mime_type",
            "tags": "tags",
            "created_at": "created_at",
            "updated_at": "updated_at",
            **(field_map or {}),
        }

    @property
    def name(self) -> str:
        return self._name

    async def search_candidates(
        self,
        request: CandidateSearchRequest,
    ) -> CandidateSearchResult:
        limit = max(0, int(request.limit))
        if limit == 0:
            return CandidateSearchResult(index_generation=self._index)
        queries = _unique_queries(request.query, request.query_variants)
        if not queries:
            return CandidateSearchResult(index_generation=self._index)

        raw_hits: list[tuple[dict[str, object], str, int]] = []
        for query_variant in queries:
            body = self._query_body(query_variant, request.filters, limit=limit)
            response = await _maybe_await(
                self._client.search(
                    index=self._index,
                    body=body,
                    size=limit,
                    _source=self._metadata_fields,
                )
            )
            for rank, hit in enumerate(_search_hits(response)):
                raw_hits.append((hit, query_variant, rank + 1))

        max_raw_score = max(
            (float(hit.get("_score") or 0.0) for hit, _query_variant, _rank in raw_hits),
            default=0.0,
        )
        candidates: list[ScoredCandidate] = []
        for hit, query_variant, rank in raw_hits:
            source = _hit_source(hit)
            node_id = _source_value(source, "node_id") or str(hit.get("_id", ""))
            if not node_id:
                continue
            raw_score = float(hit.get("_score") or 0.0)
            score = (raw_score / max_raw_score) if max_raw_score > 0 else _rank_score(rank - 1)
            candidates.append(
                ScoredCandidate(
                    node_id=node_id,
                    document_id=_source_value(source, "document_id", "doc_id"),
                    score=_clamp_score(score),
                    score_source=CandidateScoreSource.LEXICAL,
                    rank=rank,
                    query_variant=query_variant,
                    provider=self.name,
                    index_generation=_source_value(source, "index_generation") or self._index,
                    metadata=_source_metadata(source, raw_score=raw_score),
                )
            )

        return CandidateSearchResult(
            candidates=unique_candidates(candidates, limit=limit, prefer_highest_score=True),
            provider_counts={self.name: len(candidates)},
            score_ranges=_provider_score_max(candidates),
            diagnostics={
                "query_variant_count": len(queries),
                "raw_candidate_count": len(candidates),
                "filtered": _has_filters(request.filters),
            },
            index_generation=self._index,
        )

    async def index_health(self) -> IndexLagReport:
        cluster = getattr(self._client, "cluster", None)
        health = getattr(cluster, "health", None) if cluster is not None else None
        if callable(health):
            try:
                response = await _maybe_await(health(index=self._index))
                payload = _response_dict(response)
                return IndexLagReport(
                    provider=self.name,
                    status=str(payload.get("status") or "unknown"),
                    index_generation=self._index,
                    properties={"index": self._index},
                )
            except Exception as exc:
                return IndexLagReport(
                    provider=self.name,
                    status="error",
                    index_generation=self._index,
                    properties={"error": type(exc).__name__, "index": self._index},
                )
        return IndexLagReport(
            provider=self.name,
            status="unknown",
            index_generation=self._index,
            properties={"index": self._index},
        )

    def _query_body(self, query: str, filters: IndexFilter, *, limit: int) -> dict[str, object]:
        filter_clauses = _opensearch_filter_clauses(filters, self._field_map)
        return {
            "size": limit,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": self._query_fields,
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": filter_clauses,
                }
            },
        }


class QdrantCandidateProvider:
    """Qdrant vector provider.

    The provider accepts an injected async or sync Qdrant-compatible client and
    converts `IndexFilter` into payload filters. It returns scored candidate ids
    only; graph/metadata hydration stays in the normal retrieval pipeline.
    """

    __slots__ = (
        "_client",
        "_collection",
        "_field_map",
        "_name",
        "_payload_fields",
        "_using",
    )

    def __init__(
        self,
        client: object,
        *,
        collection: str,
        name: str = "qdrant",
        payload_fields: Sequence[str] = (
            "node_id",
            "document_id",
            "doc_id",
            "title",
            "kind",
            "source",
            "page",
            "chunk_id",
            "category",
            "domain",
            "language",
        ),
        field_map: dict[str, str] | None = None,
        using: str | None = None,
    ) -> None:
        self._client = client
        self._collection = collection
        self._name = name
        self._payload_fields = list(payload_fields)
        self._using = using
        self._field_map = {
            "node_id": "node_id",
            "document_id": "document_id",
            "doc_id": "doc_id",
            "workspace_id": "workspace_id",
            "user_id": "user_id",
            "session_id": "session_id",
            "domain": "domain",
            "acl_policy_id": "acl_policy_id",
            "source_id": "source_id",
            "source": "source",
            "language": "language",
            "mime_type": "mime_type",
            "tags": "tags",
            "created_at": "created_at",
            "updated_at": "updated_at",
            **(field_map or {}),
        }

    @property
    def name(self) -> str:
        return self._name

    async def search_candidates(
        self,
        request: CandidateSearchRequest,
    ) -> CandidateSearchResult:
        limit = max(0, int(request.limit))
        if limit == 0:
            return CandidateSearchResult(index_generation=self._collection)
        if not request.embedding:
            return CandidateSearchResult(
                diagnostics={"missing_embedding": True},
                index_generation=self._collection,
            )

        payload_filter = _qdrant_filter(request.filters, self._field_map)
        response = await self._search(request.embedding, limit=limit, payload_filter=payload_filter)
        raw_points = _qdrant_points(response)
        max_raw_score = max((_qdrant_score(point) for point in raw_points), default=0.0)
        candidates: list[ScoredCandidate] = []
        for rank, point in enumerate(raw_points, start=1):
            payload = _qdrant_payload(point)
            node_id = _payload_value(payload, "node_id") or _qdrant_id(point)
            if not node_id:
                continue
            raw_score = _qdrant_score(point)
            candidates.append(
                ScoredCandidate(
                    node_id=node_id,
                    document_id=_payload_value(payload, "document_id", "doc_id"),
                    score=_normalize_vector_score(raw_score, max_raw_score, rank),
                    score_source=CandidateScoreSource.VECTOR,
                    rank=rank,
                    query_variant=request.query,
                    provider=self.name,
                    index_generation=_payload_value(payload, "index_generation")
                    or self._collection,
                    metadata=_payload_metadata(payload, raw_score=raw_score),
                )
            )

        return CandidateSearchResult(
            candidates=unique_candidates(candidates, limit=limit, prefer_highest_score=True),
            provider_counts={self.name: len(candidates)},
            score_ranges=_provider_score_max(candidates),
            diagnostics={
                "raw_candidate_count": len(candidates),
                "filtered": _has_filters(request.filters),
                "payload_filter": payload_filter is not None,
            },
            index_generation=self._collection,
        )

    async def index_health(self) -> IndexLagReport:
        getter = getattr(self._client, "get_collection", None)
        if callable(getter):
            try:
                try:
                    response = await _maybe_await(getter(collection_name=self._collection))
                except TypeError:
                    response = await _maybe_await(getter(self._collection))
                payload = _response_dict(response)
                points_count = _object_int(response, payload, "points_count")
                return IndexLagReport(
                    provider=self.name,
                    status=str(_object_str(response, payload, "status") or "ok"),
                    index_generation=self._collection,
                    properties={
                        "collection": self._collection,
                        "points_count": points_count,
                    },
                )
            except Exception as exc:
                return IndexLagReport(
                    provider=self.name,
                    status="error",
                    index_generation=self._collection,
                    properties={"collection": self._collection, "error": type(exc).__name__},
                )
        return IndexLagReport(
            provider=self.name,
            status="unknown",
            index_generation=self._collection,
            properties={"collection": self._collection},
        )

    async def _search(
        self,
        embedding: list[float],
        *,
        limit: int,
        payload_filter: dict[str, object] | None,
    ) -> object:
        query_points = getattr(self._client, "query_points", None)
        if callable(query_points):
            kwargs: dict[str, object] = {
                "collection_name": self._collection,
                "query": embedding,
                "limit": limit,
                "with_payload": self._payload_fields,
            }
            if payload_filter is not None:
                kwargs["query_filter"] = payload_filter
            if self._using:
                kwargs["using"] = self._using
            return await _maybe_await(query_points(**kwargs))

        search = getattr(self._client, "search", None)
        if callable(search):
            kwargs = {
                "collection_name": self._collection,
                "query_vector": embedding,
                "limit": limit,
                "with_payload": self._payload_fields,
            }
            if payload_filter is not None:
                kwargs["query_filter"] = payload_filter
            if self._using:
                kwargs["vector_name"] = self._using
            return await _maybe_await(search(**kwargs))

        msg = "Qdrant client must expose query_points() or search()."
        raise TypeError(msg)


class InProcessIndexRouter:
    """Simple deterministic router for local providers.

    Production routers can add parallel fan-out, timeouts, and circuit breakers.
    This implementation stays small so tests can lock the retrieval contract
    before external index services are wired.
    """

    __slots__ = ("_providers",)

    def __init__(self, providers: Sequence[CandidateProvider]) -> None:
        self._providers = list(providers)

    async def search_candidates(
        self,
        request: CandidateSearchRequest,
    ) -> CandidateSearchResult:
        requested = set(request.providers)
        all_candidates: list[ScoredCandidate] = []
        provider_counts: dict[str, int] = {}
        score_ranges: dict[str, float] = {}
        diagnostics: dict[str, object] = {
            "provider_count": 0,
            "raw_candidate_count": 0,
            "deduped_candidate_count": 0,
            "failed_providers": [],
        }

        for provider in self._providers:
            if requested and provider.name not in requested:
                continue
            try:
                result = await provider.search_candidates(request)
            except Exception as exc:
                failed = diagnostics["failed_providers"]
                if isinstance(failed, list):
                    failed.append({"provider": provider.name, "error": type(exc).__name__})
                continue
            diagnostics["provider_count"] = int(diagnostics["provider_count"]) + 1
            all_candidates.extend(result.candidates)
            for name, count in result.provider_counts.items():
                provider_counts[name] = provider_counts.get(name, 0) + count
            for name, score in result.score_ranges.items():
                score_ranges[name] = max(score_ranges.get(name, 0.0), score)

        diagnostics["raw_candidate_count"] = len(all_candidates)
        deduped = unique_candidates(
            all_candidates,
            limit=max(0, int(request.limit)),
            prefer_highest_score=True,
        )
        diagnostics["deduped_candidate_count"] = len(deduped)
        return CandidateSearchResult(
            candidates=deduped,
            provider_counts=provider_counts,
            score_ranges=score_ranges,
            diagnostics=diagnostics,
            index_generation="in_process",
        )

    async def index_health(self) -> list[IndexLagReport]:
        reports: list[IndexLagReport] = []
        for provider in self._providers:
            try:
                reports.append(await provider.index_health())
            except Exception as exc:
                reports.append(
                    IndexLagReport(
                        provider=provider.name,
                        status="error",
                        properties={"error": type(exc).__name__},
                    )
                )
        return reports


def unique_candidates(
    candidates: Sequence[ScoredCandidate],
    *,
    limit: int = 20,
    prefer_highest_score: bool = False,
) -> list[ScoredCandidate]:
    """Dedupe candidates by node id.

    By default this preserves first-seen order, which lets an upstream router
    pass in an already-fused candidate list. Set `prefer_highest_score=True`
    when merging unordered provider outputs and the best duplicate score should
    win.
    """

    if prefer_highest_score:
        best_by_id: dict[str, ScoredCandidate] = {}
        first_seen_rank: dict[str, int] = {}
        for rank, candidate in enumerate(candidates):
            if candidate.node_id not in first_seen_rank:
                first_seen_rank[candidate.node_id] = rank
            current = best_by_id.get(candidate.node_id)
            if current is None or candidate.score > current.score:
                best_by_id[candidate.node_id] = candidate
        return sorted(
            best_by_id.values(),
            key=lambda candidate: (-candidate.score, first_seen_rank[candidate.node_id]),
        )[:limit]

    seen: set[str] = set()
    out: list[ScoredCandidate] = []
    for candidate in candidates:
        key = candidate.node_id
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= limit:
            break
    return out


def _unique_queries(query: str, variants: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in [query, *variants]:
        value = str(item).strip()
        if value and value not in out:
            out.append(value)
    return out


def _rank_score(rank: int) -> float:
    return max(0.10, 0.95 - max(0, rank) * 0.03)


def _node_and_score(row: object, rank: int) -> tuple[object, float]:
    if isinstance(row, tuple) and len(row) >= 2:
        node = row[0]
        try:
            score = float(row[1])
        except (TypeError, ValueError):
            score = _rank_score(rank)
        return node, _clamp_score(score) or _rank_score(rank)
    return row, _rank_score(rank)


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def _has_filters(filters: IndexFilter) -> bool:
    return any(
        (
            filters.workspace_id,
            filters.user_id,
            filters.session_id,
            filters.domain,
            filters.acl_policy_ids,
            filters.source_ids,
            filters.document_ids,
            filters.languages,
            filters.mime_types,
            filters.tags,
            filters.created_after is not None,
            filters.created_before is not None,
            filters.updated_after is not None,
            filters.updated_before is not None,
            filters.properties,
        )
    )


def _matches_filter(node: object, filters: IndexFilter) -> bool:
    if not _has_filters(filters):
        return True
    props = getattr(node, "properties", {}) or {}
    tags = list(getattr(node, "tags", []) or [])
    if filters.workspace_id and _prop(props, "workspace_id") != filters.workspace_id:
        return False
    if filters.user_id and _prop(props, "user_id") != filters.user_id:
        return False
    if filters.session_id and _prop(props, "session_id") != filters.session_id:
        return False
    if filters.domain and _prop(props, "domain") != filters.domain:
        return False
    if filters.acl_policy_ids and not _matches_any(
        _prop(props, "acl_policy_id", "acl_policy_ids"),
        filters.acl_policy_ids,
    ):
        return False
    if filters.source_ids and not _matches_any(
        _prop(props, "source_id", "source", default=str(getattr(node, "source", ""))),
        filters.source_ids,
    ):
        return False
    if filters.document_ids and not _matches_any(
        _document_id(node) or str(getattr(node, "id", "")),
        filters.document_ids,
    ):
        return False
    if filters.languages and not _matches_any(_prop(props, "language", "lang"), filters.languages):
        return False
    if filters.mime_types and not _matches_any(_prop(props, "mime_type"), filters.mime_types):
        return False
    if filters.tags and not set(filters.tags).issubset(set(tags)):
        return False
    created_at = float(getattr(node, "created_at", 0.0) or 0.0)
    updated_at = float(getattr(node, "updated_at", 0.0) or 0.0)
    if filters.created_after is not None and created_at < filters.created_after:
        return False
    if filters.created_before is not None and created_at > filters.created_before:
        return False
    if filters.updated_after is not None and updated_at < filters.updated_after:
        return False
    if filters.updated_before is not None and updated_at > filters.updated_before:
        return False
    for key, value in filters.properties.items():
        if _prop(props, key) != value:
            return False
    return True


def _prop(props: dict[str, object], *keys: str, default: str = "") -> str:
    for key in keys:
        value = props.get(key)
        if value is not None:
            return str(value)
    return default


def _matches_any(value: object, allowed: Sequence[str]) -> bool:
    allowed_set = {str(item) for item in allowed}
    if not allowed_set:
        return True
    if isinstance(value, (list, tuple, set)):
        return any(str(item) in allowed_set for item in value)
    raw = str(value)
    if raw in allowed_set:
        return True
    return any(part.strip() in allowed_set for part in raw.split(","))


def _document_id(node: object) -> str:
    props = getattr(node, "properties", {}) or {}
    return _prop(props, "document_id", "doc_id")


def _candidate_metadata(node: object) -> dict[str, object]:
    props = getattr(node, "properties", {}) or {}
    metadata: dict[str, object] = {
        "title": str(getattr(node, "title", "")),
        "kind": str(getattr(node, "kind", "")),
    }
    for key in ("source", "page", "chunk_id", "category", "domain", "language"):
        value = props.get(key)
        if value is not None:
            metadata[key] = value
    source = str(getattr(node, "source", ""))
    if source and "source" not in metadata:
        metadata["source"] = source
    return metadata


def _provider_score_max(candidates: Sequence[ScoredCandidate]) -> dict[str, float]:
    out: dict[str, float] = {}
    for candidate in candidates:
        provider = candidate.provider or "unknown"
        out[provider] = max(out.get(provider, 0.0), candidate.score)
    return out


async def _maybe_await(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


def _opensearch_filter_clauses(
    filters: IndexFilter,
    field_map: dict[str, str],
) -> list[dict[str, object]]:
    clauses: list[dict[str, object]] = []
    _term_filter(clauses, field_map, "workspace_id", filters.workspace_id)
    _term_filter(clauses, field_map, "user_id", filters.user_id)
    _term_filter(clauses, field_map, "session_id", filters.session_id)
    _term_filter(clauses, field_map, "domain", filters.domain)
    _terms_filter(clauses, field_map, "acl_policy_id", filters.acl_policy_ids)
    if filters.source_ids:
        clauses.append(
            {
                "bool": {
                    "should": [
                        {"terms": {field_map.get("source_id", "source_id"): filters.source_ids}},
                        {"terms": {field_map.get("source", "source"): filters.source_ids}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
    if filters.document_ids:
        clauses.append(
            {
                "bool": {
                    "should": [
                        {
                            "terms": {
                                field_map.get("document_id", "document_id"): filters.document_ids
                            }
                        },
                        {"terms": {field_map.get("doc_id", "doc_id"): filters.document_ids}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
    _terms_filter(clauses, field_map, "language", filters.languages)
    _terms_filter(clauses, field_map, "mime_type", filters.mime_types)
    _terms_filter(clauses, field_map, "tags", filters.tags)
    _range_filter(
        clauses,
        field_map.get("created_at", "created_at"),
        gte=filters.created_after,
        lte=filters.created_before,
    )
    _range_filter(
        clauses,
        field_map.get("updated_at", "updated_at"),
        gte=filters.updated_after,
        lte=filters.updated_before,
    )
    for key, value in filters.properties.items():
        clauses.append({"term": {field_map.get(key, f"properties.{key}"): value}})
    return clauses


def _term_filter(
    clauses: list[dict[str, object]],
    field_map: dict[str, str],
    key: str,
    value: str,
) -> None:
    if value:
        clauses.append({"term": {field_map.get(key, key): value}})


def _terms_filter(
    clauses: list[dict[str, object]],
    field_map: dict[str, str],
    key: str,
    values: Sequence[str],
) -> None:
    if values:
        clauses.append({"terms": {field_map.get(key, key): list(values)}})


def _range_filter(
    clauses: list[dict[str, object]],
    field: str,
    *,
    gte: float | None,
    lte: float | None,
) -> None:
    bounds: dict[str, float] = {}
    if gte is not None:
        bounds["gte"] = gte
    if lte is not None:
        bounds["lte"] = lte
    if bounds:
        clauses.append({"range": {field: bounds}})


def _response_dict(response: object) -> dict[str, object]:
    if isinstance(response, dict):
        return response
    body = getattr(response, "body", None)
    if isinstance(body, dict):
        return body
    return {}


def _search_hits(response: object) -> list[dict[str, object]]:
    payload = _response_dict(response)
    hits_obj = payload.get("hits", {})
    if not isinstance(hits_obj, dict):
        return []
    raw_hits = hits_obj.get("hits", [])
    if not isinstance(raw_hits, list):
        return []
    return [hit for hit in raw_hits if isinstance(hit, dict)]


def _hit_source(hit: dict[str, object]) -> dict[str, object]:
    source = hit.get("_source", {})
    return source if isinstance(source, dict) else {}


def _source_value(source: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = source.get(key)
        if value is not None:
            return str(value)
    return ""


def _source_metadata(source: dict[str, object], *, raw_score: float) -> dict[str, object]:
    metadata: dict[str, object] = {"raw_score": raw_score}
    for key in (
        "title",
        "kind",
        "source",
        "page",
        "chunk_id",
        "category",
        "domain",
        "language",
    ):
        value = source.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def _qdrant_filter(
    filters: IndexFilter,
    field_map: dict[str, str],
) -> dict[str, object] | None:
    must: list[dict[str, object]] = []
    _qdrant_match(must, field_map, "workspace_id", filters.workspace_id)
    _qdrant_match(must, field_map, "user_id", filters.user_id)
    _qdrant_match(must, field_map, "session_id", filters.session_id)
    _qdrant_match(must, field_map, "domain", filters.domain)
    _qdrant_match_any(must, field_map, "acl_policy_id", filters.acl_policy_ids)
    if filters.source_ids:
        must.append(
            {
                "should": [
                    _qdrant_match_condition(
                        field_map.get("source_id", "source_id"),
                        filters.source_ids,
                    ),
                    _qdrant_match_condition(
                        field_map.get("source", "source"),
                        filters.source_ids,
                    ),
                ]
            }
        )
    if filters.document_ids:
        must.append(
            {
                "should": [
                    _qdrant_match_condition(
                        field_map.get("document_id", "document_id"),
                        filters.document_ids,
                    ),
                    _qdrant_match_condition(
                        field_map.get("doc_id", "doc_id"),
                        filters.document_ids,
                    ),
                ]
            }
        )
    _qdrant_match_any(must, field_map, "language", filters.languages)
    _qdrant_match_any(must, field_map, "mime_type", filters.mime_types)
    _qdrant_match_any(must, field_map, "tags", filters.tags)
    _qdrant_range(
        must,
        field_map.get("created_at", "created_at"),
        gte=filters.created_after,
        lte=filters.created_before,
    )
    _qdrant_range(
        must,
        field_map.get("updated_at", "updated_at"),
        gte=filters.updated_after,
        lte=filters.updated_before,
    )
    for key, value in filters.properties.items():
        must.append({"key": field_map.get(key, f"properties.{key}"), "match": {"value": value}})
    return {"must": must} if must else None


def _qdrant_match(
    must: list[dict[str, object]],
    field_map: dict[str, str],
    key: str,
    value: str,
) -> None:
    if value:
        must.append({"key": field_map.get(key, key), "match": {"value": value}})


def _qdrant_match_any(
    must: list[dict[str, object]],
    field_map: dict[str, str],
    key: str,
    values: Sequence[str],
) -> None:
    if values:
        must.append(_qdrant_match_condition(field_map.get(key, key), values))


def _qdrant_match_condition(field: str, values: Sequence[str]) -> dict[str, object]:
    return {"key": field, "match": {"any": list(values)}}


def _qdrant_range(
    must: list[dict[str, object]],
    field: str,
    *,
    gte: float | None,
    lte: float | None,
) -> None:
    bounds: dict[str, float] = {}
    if gte is not None:
        bounds["gte"] = gte
    if lte is not None:
        bounds["lte"] = lte
    if bounds:
        must.append({"key": field, "range": bounds})


def _qdrant_points(response: object) -> list[object]:
    if isinstance(response, list):
        return response
    points = getattr(response, "points", None)
    if isinstance(points, list):
        return points
    payload = _response_dict(response)
    result = payload.get("result")
    if isinstance(result, dict):
        raw_points = result.get("points", [])
        return list(raw_points) if isinstance(raw_points, list) else []
    if isinstance(result, list):
        return result
    return []


def _qdrant_payload(point: object) -> dict[str, object]:
    if isinstance(point, dict):
        payload = point.get("payload", {})
    else:
        payload = getattr(point, "payload", {})
    return payload if isinstance(payload, dict) else {}


def _qdrant_id(point: object) -> str:
    if isinstance(point, dict):
        value = point.get("id", "")
    else:
        value = getattr(point, "id", "")
    return "" if value is None else str(value)


def _qdrant_score(point: object) -> float:
    if isinstance(point, dict):
        value = point.get("score", 0.0)
    else:
        value = getattr(point, "score", 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _payload_value(payload: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return ""


def _payload_metadata(payload: dict[str, object], *, raw_score: float) -> dict[str, object]:
    metadata: dict[str, object] = {"raw_score": raw_score}
    for key in (
        "title",
        "kind",
        "source",
        "page",
        "chunk_id",
        "category",
        "domain",
        "language",
    ):
        value = payload.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def _normalize_vector_score(raw_score: float, max_raw_score: float, rank: int) -> float:
    if raw_score <= 0:
        return _rank_score(rank - 1)
    if max_raw_score > 1.0:
        return _clamp_score(raw_score / max_raw_score)
    return _clamp_score(raw_score)


def _object_str(response: object, payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if value is None:
        value = getattr(response, key, "")
    return "" if value is None else str(value)


def _object_int(response: object, payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if value is None:
        value = getattr(response, key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
