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
    ScoredCandidate,
    unique_candidates,
)


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
