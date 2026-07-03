import pytest

from synaptic import (
    CandidateScoreSource,
    CandidateSearchRequest,
    CandidateSearchResult,
    IndexFilter,
    IndexLagReport,
    IngestionJob,
    IngestionJobStage,
    IngestionJobStatus,
    InProcessIndexRouter,
    Node,
    NodeKind,
    OpenSearchCandidateProvider,
    QdrantCandidateProvider,
    ScoredCandidate,
    StorageCandidateProvider,
    unique_candidates,
)
from synaptic.backends.memory import MemoryBackend


def test_index_filter_defaults_are_isolated():
    first = IndexFilter()
    second = IndexFilter()

    first.acl_policy_ids.append("acl-a")
    first.properties["source"] = "policy"

    assert second.acl_policy_ids == []
    assert second.properties == {}


def test_candidate_search_request_keeps_query_variants_and_scope():
    request = CandidateSearchRequest(
        query="how much fiber is in carrots",
        query_variants=["one cup cooked carrots grams fiber"],
        limit=5,
        filters=IndexFilter(workspace_id="workspace-a", acl_policy_ids=["acl-a"]),
    )

    assert request.query == "how much fiber is in carrots"
    assert request.query_variants == ["one cup cooked carrots grams fiber"]
    assert request.limit == 5
    assert request.filters.workspace_id == "workspace-a"
    assert request.filters.acl_policy_ids == ["acl-a"]
    assert request.scope.workspace_id == ""


def test_candidate_search_request_scope_defaults_are_isolated():
    first = CandidateSearchRequest(query="first")
    second = CandidateSearchRequest(query="second")

    first.scope.workspace_id = "workspace-a"

    assert second.scope.workspace_id == ""


def test_unique_candidates_preserves_first_seen_order():
    candidates = [
        ScoredCandidate(node_id="n1", score=0.9, provider="lexical"),
        ScoredCandidate(node_id="n2", score=0.8, provider="vector"),
        ScoredCandidate(node_id="n1", score=0.7, provider="graph"),
    ]

    deduped = unique_candidates(candidates, limit=10)

    assert [candidate.node_id for candidate in deduped] == ["n1", "n2"]
    assert deduped[0].provider == "lexical"


def test_unique_candidates_can_prefer_highest_score():
    candidates = [
        ScoredCandidate(node_id="n1", score=0.2, provider="lexical"),
        ScoredCandidate(node_id="n2", score=0.8, provider="vector"),
        ScoredCandidate(node_id="n1", score=0.9, provider="graph"),
    ]

    deduped = unique_candidates(candidates, limit=10, prefer_highest_score=True)

    assert [(candidate.node_id, candidate.provider) for candidate in deduped] == [
        ("n1", "graph"),
        ("n2", "vector"),
    ]


@pytest.mark.asyncio
async def test_candidate_provider_shape_with_fake_provider():
    class FakeProvider:
        @property
        def name(self) -> str:
            return "fake"

        async def search_candidates(
            self,
            request: CandidateSearchRequest,
        ) -> CandidateSearchResult:
            return CandidateSearchResult(
                candidates=[
                    ScoredCandidate(
                        node_id="chunk-a",
                        document_id="doc-a",
                        score=0.75,
                        score_source=CandidateScoreSource.LEXICAL,
                        rank=1,
                        query_variant=request.query,
                        provider=self.name,
                        index_generation="gen-1",
                    )
                ],
                provider_counts={self.name: 1},
                score_ranges={self.name: 0.75},
                index_generation="gen-1",
            )

        async def index_health(self) -> IndexLagReport:
            return IndexLagReport(provider=self.name, status="ok", index_generation="gen-1")

    provider = FakeProvider()
    result = await provider.search_candidates(CandidateSearchRequest(query="policy"))
    health = await provider.index_health()

    assert result.candidates[0].node_id == "chunk-a"
    assert result.candidates[0].score_source == CandidateScoreSource.LEXICAL
    assert result.provider_counts == {"fake": 1}
    assert health.status == "ok"


def test_ingestion_job_defaults_and_status_values():
    job = IngestionJob(document_id="doc-a", version="v1")

    assert job.id
    assert job.document_id == "doc-a"
    assert job.version == "v1"
    assert job.stage == IngestionJobStage.DISCOVER
    assert job.status == IngestionJobStatus.PENDING
    assert job.created_at <= job.updated_at


@pytest.mark.asyncio
async def test_storage_candidate_provider_returns_ids_variants_and_filters():
    backend = MemoryBackend()
    await backend.connect()
    await backend.save_node(
        Node(
            id="alpha-w1",
            kind=NodeKind.CHUNK,
            title="Alpha policy",
            content="alpha storage candidate",
            tags=["policy"],
            properties={"workspace_id": "w1", "doc_id": "doc-alpha"},
        )
    )
    await backend.save_node(
        Node(
            id="alpha-w2",
            kind=NodeKind.CHUNK,
            title="Alpha other workspace",
            content="alpha storage candidate",
            tags=["policy"],
            properties={"workspace_id": "w2", "doc_id": "doc-other"},
        )
    )
    await backend.save_node(
        Node(
            id="beta-w1",
            kind=NodeKind.CHUNK,
            title="Beta policy",
            content="beta storage candidate",
            tags=["policy"],
            properties={"workspace_id": "w1", "doc_id": "doc-beta"},
        )
    )

    provider = StorageCandidateProvider(backend)
    result = await provider.search_candidates(
        CandidateSearchRequest(
            query="alpha",
            query_variants=["beta"],
            limit=5,
            filters=IndexFilter(workspace_id="w1", tags=["policy"]),
        )
    )

    assert [candidate.node_id for candidate in result.candidates] == ["alpha-w1", "beta-w1"]
    assert [candidate.query_variant for candidate in result.candidates] == ["alpha", "beta"]
    assert result.candidates[0].document_id == "doc-alpha"
    assert result.provider_counts == {"storage_fts": 2}
    assert result.diagnostics["filtered"] is True


@pytest.mark.asyncio
async def test_in_process_index_router_dedupes_and_reports_diagnostics():
    class FirstProvider:
        @property
        def name(self) -> str:
            return "first"

        async def search_candidates(
            self,
            request: CandidateSearchRequest,
        ) -> CandidateSearchResult:
            return CandidateSearchResult(
                candidates=[
                    ScoredCandidate(
                        node_id="n1",
                        score=0.2,
                        provider=self.name,
                        score_source=CandidateScoreSource.LEXICAL,
                    )
                ],
                provider_counts={self.name: 1},
                score_ranges={self.name: 0.2},
            )

        async def index_health(self) -> IndexLagReport:
            return IndexLagReport(provider=self.name, status="ok")

    class SecondProvider:
        @property
        def name(self) -> str:
            return "second"

        async def search_candidates(
            self,
            request: CandidateSearchRequest,
        ) -> CandidateSearchResult:
            return CandidateSearchResult(
                candidates=[
                    ScoredCandidate(
                        node_id="n1",
                        score=0.9,
                        provider=self.name,
                        score_source=CandidateScoreSource.VECTOR,
                    ),
                    ScoredCandidate(
                        node_id="n2",
                        score=0.8,
                        provider=self.name,
                        score_source=CandidateScoreSource.VECTOR,
                    ),
                ],
                provider_counts={self.name: 2},
                score_ranges={self.name: 0.9},
            )

        async def index_health(self) -> IndexLagReport:
            return IndexLagReport(provider=self.name, status="ok")

    router = InProcessIndexRouter([FirstProvider(), SecondProvider()])
    result = await router.search_candidates(CandidateSearchRequest(query="q", limit=10))

    assert [(candidate.node_id, candidate.provider) for candidate in result.candidates] == [
        ("n1", "second"),
        ("n2", "second"),
    ]
    assert result.provider_counts == {"first": 1, "second": 2}
    assert result.diagnostics["raw_candidate_count"] == 3
    assert result.diagnostics["deduped_candidate_count"] == 2


@pytest.mark.asyncio
async def test_opensearch_candidate_provider_builds_filtered_queries_and_candidates():
    class FakeOpenSearchClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def search(self, **kwargs):
            self.calls.append(kwargs)
            body = kwargs["body"]
            query = body["query"]["bool"]["must"][0]["multi_match"]["query"]
            if query == "alpha":
                return {
                    "hits": {
                        "hits": [
                            {
                                "_id": "fallback-id",
                                "_score": 4.0,
                                "_source": {
                                    "node_id": "chunk-alpha",
                                    "document_id": "doc-alpha",
                                    "title": "Alpha",
                                    "kind": "chunk",
                                    "source": "policy",
                                },
                            },
                            {
                                "_id": "chunk-shared",
                                "_score": 2.0,
                                "_source": {
                                    "node_id": "chunk-shared",
                                    "doc_id": "doc-shared",
                                    "title": "Shared",
                                    "kind": "chunk",
                                },
                            },
                        ]
                    }
                }
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "chunk-shared",
                            "_score": 8.0,
                            "_source": {
                                "node_id": "chunk-shared",
                                "document_id": "doc-shared",
                                "title": "Shared Better",
                                "kind": "chunk",
                            },
                        }
                    ]
                }
            }

    client = FakeOpenSearchClient()
    provider = OpenSearchCandidateProvider(client, index="synaptic-chunks")

    result = await provider.search_candidates(
        CandidateSearchRequest(
            query="alpha",
            query_variants=["beta"],
            limit=5,
            filters=IndexFilter(
                workspace_id="workspace-a",
                acl_policy_ids=["acl-a", "acl-b"],
                source_ids=["policy"],
                document_ids=["doc-alpha"],
                created_after=10.0,
                properties={"department": "risk"},
            ),
        )
    )

    assert [candidate.node_id for candidate in result.candidates] == [
        "chunk-shared",
        "chunk-alpha",
    ]
    assert result.candidates[0].score == 1.0
    assert result.candidates[0].query_variant == "beta"
    assert result.candidates[1].score == 0.5
    assert result.candidates[1].document_id == "doc-alpha"
    assert result.candidates[1].metadata["source"] == "policy"
    assert result.provider_counts == {"opensearch": 3}
    assert result.diagnostics["query_variant_count"] == 2

    first_body = client.calls[0]["body"]
    filters = first_body["query"]["bool"]["filter"]
    assert {"term": {"workspace_id": "workspace-a"}} in filters
    assert {"terms": {"acl_policy_id": ["acl-a", "acl-b"]}} in filters
    assert {
        "bool": {
            "should": [
                {"terms": {"source_id": ["policy"]}},
                {"terms": {"source": ["policy"]}},
            ],
            "minimum_should_match": 1,
        }
    } in filters
    assert {"range": {"created_at": {"gte": 10.0}}} in filters
    assert {"term": {"properties.department": "risk"}} in filters
    assert {
        "bool": {
            "should": [
                {"terms": {"document_id": ["doc-alpha"]}},
                {"terms": {"doc_id": ["doc-alpha"]}},
            ],
            "minimum_should_match": 1,
        }
    } in filters


@pytest.mark.asyncio
async def test_opensearch_candidate_provider_health_uses_cluster_status():
    class _Cluster:
        async def health(self, *, index: str):
            return {"status": "green", "index": index}

    class FakeOpenSearchClient:
        cluster = _Cluster()

    provider = OpenSearchCandidateProvider(FakeOpenSearchClient(), index="synaptic-chunks")

    report = await provider.index_health()

    assert report.provider == "opensearch"
    assert report.status == "green"
    assert report.index_generation == "synaptic-chunks"


@pytest.mark.asyncio
async def test_qdrant_candidate_provider_builds_payload_filters_and_candidates():
    class _Point:
        def __init__(
            self,
            *,
            point_id: str,
            score: float,
            payload: dict[str, object],
        ) -> None:
            self.id = point_id
            self.score = score
            self.payload = payload

    class _Response:
        def __init__(self, points: list[_Point]) -> None:
            self.points = points

    class FakeQdrantClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def query_points(self, **kwargs):
            self.calls.append(kwargs)
            return _Response(
                [
                    _Point(
                        point_id="fallback-a",
                        score=0.42,
                        payload={
                            "node_id": "chunk-alpha",
                            "document_id": "doc-alpha",
                            "title": "Alpha",
                            "kind": "chunk",
                            "source": "policy",
                        },
                    ),
                    _Point(
                        point_id="fallback-b",
                        score=0.88,
                        payload={
                            "node_id": "chunk-beta",
                            "doc_id": "doc-beta",
                            "title": "Beta",
                            "kind": "chunk",
                        },
                    ),
                    _Point(
                        point_id="fallback-a",
                        score=0.96,
                        payload={
                            "node_id": "chunk-alpha",
                            "document_id": "doc-alpha",
                            "title": "Alpha Better",
                            "kind": "chunk",
                        },
                    ),
                ]
            )

    client = FakeQdrantClient()
    provider = QdrantCandidateProvider(client, collection="synaptic-vectors")

    result = await provider.search_candidates(
        CandidateSearchRequest(
            query="alpha",
            embedding=[0.1, 0.2, 0.3],
            limit=5,
            filters=IndexFilter(
                workspace_id="workspace-a",
                acl_policy_ids=["acl-a"],
                source_ids=["policy"],
                document_ids=["doc-alpha"],
                updated_before=20.0,
                properties={"department": "risk"},
            ),
        )
    )

    assert [candidate.node_id for candidate in result.candidates] == [
        "chunk-alpha",
        "chunk-beta",
    ]
    assert result.candidates[0].score == 0.96
    assert result.candidates[0].score_source == CandidateScoreSource.VECTOR
    assert result.candidates[0].document_id == "doc-alpha"
    assert result.candidates[1].document_id == "doc-beta"
    assert result.provider_counts == {"qdrant": 3}
    assert result.diagnostics["payload_filter"] is True

    call = client.calls[0]
    assert call["collection_name"] == "synaptic-vectors"
    assert call["query"] == [0.1, 0.2, 0.3]
    assert call["with_payload"] == [
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
    ]
    payload_filter = call["query_filter"]
    assert {"key": "workspace_id", "match": {"value": "workspace-a"}} in payload_filter["must"]
    assert {"key": "acl_policy_id", "match": {"any": ["acl-a"]}} in payload_filter["must"]
    assert {"key": "updated_at", "range": {"lte": 20.0}} in payload_filter["must"]
    assert {"key": "properties.department", "match": {"value": "risk"}} in payload_filter["must"]
    assert {
        "should": [
            {"key": "source_id", "match": {"any": ["policy"]}},
            {"key": "source", "match": {"any": ["policy"]}},
        ]
    } in payload_filter["must"]
    assert {
        "should": [
            {"key": "document_id", "match": {"any": ["doc-alpha"]}},
            {"key": "doc_id", "match": {"any": ["doc-alpha"]}},
        ]
    } in payload_filter["must"]


@pytest.mark.asyncio
async def test_qdrant_candidate_provider_requires_embedding():
    class FakeQdrantClient:
        async def query_points(self, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError(kwargs)

    provider = QdrantCandidateProvider(FakeQdrantClient(), collection="synaptic-vectors")

    result = await provider.search_candidates(CandidateSearchRequest(query="alpha"))

    assert result.candidates == []
    assert result.diagnostics["missing_embedding"] is True


@pytest.mark.asyncio
async def test_qdrant_candidate_provider_health_uses_collection_status():
    class _Collection:
        status = "green"
        points_count = 42

    class FakeQdrantClient:
        async def get_collection(self, *, collection_name: str):
            assert collection_name == "synaptic-vectors"
            return _Collection()

    provider = QdrantCandidateProvider(FakeQdrantClient(), collection="synaptic-vectors")

    report = await provider.index_health()

    assert report.provider == "qdrant"
    assert report.status == "green"
    assert report.index_generation == "synaptic-vectors"
    assert report.properties["points_count"] == 42
