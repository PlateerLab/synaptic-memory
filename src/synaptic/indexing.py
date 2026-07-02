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
