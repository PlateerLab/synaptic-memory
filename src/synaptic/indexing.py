"""Scale-out indexing contracts.

These types define the optional boundary between the retrieval core and
large external indexes. They do not change the default StorageBackend path;
instead they make the 10TB route explicit: providers return scored candidate
ids, and the normal retrieval pipeline decides how to hydrate, expand, rerank,
and record them.
"""

from __future__ import annotations

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
    "IngestionJob",
    "IngestionJobStage",
    "IngestionJobStatus",
    "IngestionJobStore",
    "ScoredCandidate",
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
