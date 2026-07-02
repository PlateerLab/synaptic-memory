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
