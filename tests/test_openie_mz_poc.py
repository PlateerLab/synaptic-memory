"""Tests for the local OpenIE PoC gate semantics."""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    MemoryScope,
    Node,
    NodeKind,
    RetrievalEvent,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "scripts" / "openie_mz_poc.py"
_SPEC = importlib.util.spec_from_file_location("openie_mz_poc", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
openie_mz_poc = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = openie_mz_poc
_SPEC.loader.exec_module(openie_mz_poc)

OpenIERunSummary = openie_mz_poc.OpenIERunSummary
RelationProbeSummary = openie_mz_poc.RelationProbeSummary
ScoreSummary = openie_mz_poc.ScoreSummary
GateSummary = openie_mz_poc.GateSummary
CacheOnlyLLMProvider = openie_mz_poc._CacheOnlyLLMProvider
LLMOpenIEExtractor = openie_mz_poc.LLMOpenIEExtractor
audit_openie_cache = openie_mz_poc.audit_openie_cache
embed_missing_openie_nodes = openie_mz_poc.embed_missing_openie_nodes
evaluate_gates = openie_mz_poc.evaluate_gates
graph_fingerprint = openie_mz_poc.graph_fingerprint
load_queries = openie_mz_poc.load_queries
load_cache_warm_rows = openie_mz_poc.load_cache_warm_rows
memory_health_for_db = openie_mz_poc.memory_health_for_db
preflight_dependencies = openie_mz_poc.preflight_dependencies
precompute_query_embeddings = openie_mz_poc.precompute_query_embeddings
profile_from_args = openie_mz_poc.profile_from_args
relation_probe_for_db = openie_mz_poc.relation_probe_for_db
warm_openie_cache = openie_mz_poc.warm_openie_cache
write_cache_warm_results = openie_mz_poc.write_cache_warm_results
write_cache_audit_results = openie_mz_poc.write_cache_audit_results
write_results = openie_mz_poc.write_results
cache_only_selector = openie_mz_poc._cache_only_selector


class _TinyEmbedder:
    async def embed_batch(self, texts):
        return [[float(i + 1)] for i, _ in enumerate(texts)]


class _RecordingEmbedder:
    calls: ClassVar[list[list[str]]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def embed_batch(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


class _JsonLLM:
    async def generate(self, **kwargs):
        return json.dumps({"entities": [], "triples": []})


class _CacheWarmProvider:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def generate(self, **kwargs):
        return json.dumps(
            {
                "entities": [
                    {"canonical": "Acme"},
                    {"canonical": "Roadmap"},
                ],
                "triples": [
                    {
                        "subject": "Acme",
                        "predicate": "depends_on",
                        "object": "Roadmap",
                        "confidence": 0.9,
                    }
                ],
            }
        )


class _FailingCacheWarmProvider:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def generate(self, **kwargs):
        user = str(kwargs.get("user") or "")
        if "Bad body" in user:
            raise RuntimeError("synthetic extraction failure")
        return json.dumps({"entities": [{"canonical": "Good"}], "triples": []})


def _score(name: str, *, n: int = 1, r5: int = 1) -> ScoreSummary:
    summary = ScoreSummary(name=name)
    summary.n = n
    summary.r5 = r5
    return summary


def _args(*, skip_openie: bool = False) -> Namespace:
    return Namespace(
        min_delta_r5=0.0,
        min_relation_expanded_lift=0,
        min_relation_evidence_lift=0,
        min_strong_relation_evidence_rate=0.0,
        min_openie_cache_coverage=0.0,
        verify_revertibility=False,
        skip_openie=skip_openie,
        llm_base_url="http://llm.local/v1",
        llm_api_key_env="OPENIE_TEST_API_KEY",
        llm_timeout=10,
        embed_base_url="http://embed.local/v1",
        embed_model="emb",
        embed_timeout=10,
        llm_model="",
        openie_cache=None,
        openie_cache_audit=False,
        openie_cache_audit_bad_output=None,
        openie_cache_compact_output=None,
        openie_cache_only=False,
        openie_model_profile="",
        openie_relation_whitelist="depends_on,related",
        openie_min_candidate_entities=2,
        openie_max_candidate_df_ratio=0.3,
        openie_sample_rate=1.0,
        openie_max_chunks=200,
        openie_max_concurrency=4,
        openie_max_output_tokens=1024,
        openie_max_triples_per_chunk=24,
        openie_cache_missing_output=None,
        openie_cache_warm_input=None,
        openie_cache_warm_limit=0,
        openie_cache_warm_total_chunks=0,
        openie_cache_warm_target_coverage=0.0,
        openie_cache_warm_dry_run=False,
        openie_cache_warm_pending_output=None,
        openie_cache_warm_failure_output=None,
        openie_seed=42,
        results=None,
    )


def test_load_queries_accepts_gold_files_and_relevant_docs(tmp_path: Path):
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        """
        {
          "queries": [
            {"query": "exact title", "gold_files": ["doc-a"]},
            {"query": "hard graph", "relevant_docs": ["doc-b", "doc-c"]},
            {"query": "not scoreable", "relevant_docs": []}
          ]
        }
        """,
        encoding="utf-8",
    )

    rows = load_queries([queries_path])

    assert [row["query"] for row in rows] == ["exact title", "hard graph"]
    assert rows[0]["gold_files"] == ["doc-a"]
    assert rows[1]["gold_files"] == ["doc-b", "doc-c"]


def test_load_cache_warm_rows_filters_invalid_lines(tmp_path: Path):
    path = tmp_path / "missing.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"title": "T1", "content": "Body 1", "doc_id": "doc-1"}),
                json.dumps({"title": "", "content": ""}),
                json.dumps({"title": "T2", "content": "Body 2"}),
            ]
        ),
        encoding="utf-8",
    )

    rows = load_cache_warm_rows(path, limit=1)

    assert rows == [{"title": "T1", "content": "Body 1", "doc_id": "doc-1"}]


def test_preflight_requires_api_key_for_remote_cache_warm(monkeypatch):
    monkeypatch.delenv("OPENIE_TEST_API_KEY", raising=False)
    args = _args()
    args.openie_cache_warm_input = Path("missing.jsonl")
    args.llm_base_url = "https://api.deepseek.com/v1"
    args.llm_model = "deepseek-v4-flash"
    args.openie_cache_warm_dry_run = False

    with pytest.raises(SystemExit, match="OPENIE_TEST_API_KEY is required"):
        preflight_dependencies(args)


def test_preflight_allows_remote_cache_warm_dry_run_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENIE_TEST_API_KEY", raising=False)
    args = _args()
    args.openie_cache_warm_input = Path("missing.jsonl")
    args.llm_base_url = "https://api.deepseek.com/v1"
    args.llm_model = "deepseek-v4-flash"
    args.openie_cache_warm_dry_run = True

    preflight_dependencies(args)


def test_preflight_allows_local_llm_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENIE_TEST_API_KEY", raising=False)
    args = _args()
    args.skip_openie = False
    args.llm_base_url = "http://localhost:8000/v1"
    args.llm_model = "Qwen3.6-27B"

    preflight_dependencies(args)


def test_audit_openie_cache_reports_bad_rows_and_duplicates(tmp_path: Path):
    cache_path = tmp_path / "openie_cache.jsonl"
    bad_output = tmp_path / "bad_rows.jsonl"
    compact_output = tmp_path / "compact.jsonl"
    results_path = tmp_path / "audit_results.json"
    good_raw = json.dumps(
        {
            "entities": [{"canonical": "Acme"}],
            "triples": [{"subject": "Acme", "predicate": "related", "object": "Roadmap"}],
        }
    )
    newer_raw = json.dumps({"entities": [{"canonical": "Newer"}], "triples": []})
    cache_path.write_text(
        "\n".join(
            [
                json.dumps({"key": "k1", "raw": good_raw}),
                json.dumps({"key": "k1", "raw": newer_raw}),
                "{not json}",
                json.dumps({"key": "bad", "raw": "not json at all"}),
                json.dumps({"key": "shape"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = _args()
    args.openie_cache = cache_path
    args.openie_cache_audit_bad_output = bad_output
    args.openie_cache_compact_output = compact_output
    args.results = results_path

    summary = audit_openie_cache(args)
    write_cache_audit_results(args, summary)

    assert summary.passed is False
    assert summary.lines_total == 5
    assert summary.valid_record_lines == 3
    assert summary.duplicate_keys == 1
    assert summary.unique_keys == 2
    assert summary.parseable_records == 2
    assert summary.unparseable_records == 1
    assert summary.invalid_json_lines == 1
    assert summary.invalid_record_lines == 1
    assert summary.compacted_records == 1
    assert summary.entities == 2
    assert summary.triples == 1
    bad_rows = [
        json.loads(line) for line in bad_output.read_text(encoding="utf-8").splitlines() if line
    ]
    assert [row["reason"] for row in bad_rows] == [
        "invalid_json_line",
        "unparseable_raw",
        "invalid_record_shape",
    ]
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["openie_cache_audit"]["passed"] is False
    assert payload["openie_cache_audit"]["bad_rows_output"] == str(bad_output)
    compact_rows = [
        json.loads(line) for line in compact_output.read_text(encoding="utf-8").splitlines() if line
    ]
    assert compact_rows == [{"key": "k1", "raw": newer_raw}]
    assert payload["openie_cache_audit"]["compact_output"] == str(compact_output)
    assert payload["openie_cache_audit"]["compacted_records"] == 1


@pytest.mark.asyncio
async def test_warm_openie_cache_writes_cache_and_summary(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(openie_mz_poc, "OpenAILLMProvider", _CacheWarmProvider)
    warm_input = tmp_path / "missing.jsonl"
    cache_path = tmp_path / "openie_cache.jsonl"
    results_path = tmp_path / "warm_results.json"
    warm_input.write_text(
        json.dumps({"title": "Warm title", "content": "Warm body"}, ensure_ascii=False),
        encoding="utf-8",
    )
    args = _args()
    args.llm_model = "deepseek-v4-flash"
    args.openie_cache = cache_path
    args.openie_cache_warm_input = warm_input
    args.results = results_path

    summary = await warm_openie_cache(args)
    write_cache_warm_results(args, summary)

    assert summary.rows_loaded == 1
    assert summary.rows_pending == 1
    assert summary.rows_deferred_limit == 0
    assert summary.rows_attempted == 1
    assert summary.rows_succeeded == 1
    assert summary.rows_skipped_cached == 0
    assert summary.coverage_projection_after_batch_rate == 0.0
    assert summary.extraction_failures == 0
    assert summary.entities == 2
    assert summary.triples == 1
    assert summary.cache_hits == 0
    assert summary.cache_misses == 1
    assert cache_path.read_text(encoding="utf-8").count("\n") == 1
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["openie_cache_warm"]["rows_succeeded"] == 1
    assert payload["openie_cache_warm"]["rows_pending"] == 1
    assert payload["openie_cache_warm"]["rows_deferred_limit"] == 0
    assert payload["openie_cache_warm"]["rows_skipped_cached"] == 0
    assert payload["openie_cache_warm"]["cache_path"] == str(cache_path)

    second = await warm_openie_cache(args)

    assert second.rows_loaded == 1
    assert second.rows_pending == 0
    assert second.rows_deferred_limit == 0
    assert second.rows_attempted == 0
    assert second.rows_succeeded == 0
    assert second.rows_skipped_cached == 1
    assert second.cache_hits == 0
    assert second.cache_misses == 0
    assert cache_path.read_text(encoding="utf-8").count("\n") == 1


@pytest.mark.asyncio
async def test_warm_openie_cache_writes_failure_manifest(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(openie_mz_poc, "OpenAILLMProvider", _FailingCacheWarmProvider)
    warm_input = tmp_path / "missing.jsonl"
    cache_path = tmp_path / "openie_cache.jsonl"
    failure_output = tmp_path / "failures.jsonl"
    warm_input.write_text(
        "\n".join(
            [
                json.dumps({"title": "Good title", "content": "Good body", "doc_id": "good"}),
                json.dumps({"title": "Bad title", "content": "Bad body", "doc_id": "bad"}),
            ]
        ),
        encoding="utf-8",
    )
    args = _args()
    args.llm_model = "deepseek-v4-flash"
    args.openie_cache = cache_path
    args.openie_cache_warm_input = warm_input
    args.openie_cache_warm_failure_output = failure_output

    summary = await warm_openie_cache(args)

    assert summary.rows_loaded == 2
    assert summary.rows_pending == 2
    assert summary.rows_attempted == 2
    assert summary.rows_succeeded == 1
    assert summary.extraction_failures == 1
    assert summary.failure_output == str(failure_output)
    assert cache_path.read_text(encoding="utf-8").count("\n") == 1
    failed_rows = [
        json.loads(line) for line in failure_output.read_text(encoding="utf-8").splitlines() if line
    ]
    assert failed_rows == [
        {
            "title": "Bad title",
            "content": "Bad body",
            "doc_id": "bad",
            "error_type": "RuntimeError",
            "error": "synthetic extraction failure",
        }
    ]


@pytest.mark.asyncio
async def test_warm_openie_cache_dry_run_counts_pending_without_llm(tmp_path: Path):
    cache_path = tmp_path / "openie_cache.jsonl"
    pending_output = tmp_path / "pending.jsonl"
    extractor = LLMOpenIEExtractor(_JsonLLM(), cache_path=cache_path)
    await extractor.extract("Cached title\nCached body", title="Cached title")
    before = cache_path.read_text(encoding="utf-8")

    warm_input = tmp_path / "missing.jsonl"
    warm_input.write_text(
        "\n".join(
            [
                json.dumps({"title": "Cached title", "content": "Cached body"}),
                json.dumps(
                    {
                        "title": "Pending title",
                        "content": "Pending body",
                        "doc_id": "doc-pending",
                    }
                ),
                json.dumps(
                    {
                        "title": "Deferred title",
                        "content": "Deferred body",
                        "doc_id": "doc-deferred",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    args = _args()
    args.llm_base_url = ""
    args.llm_model = ""
    args.openie_cache = cache_path
    args.openie_cache_warm_input = warm_input
    args.openie_cache_warm_limit = 1
    args.openie_cache_warm_total_chunks = 4
    args.openie_cache_warm_target_coverage = 1.0
    args.openie_cache_warm_dry_run = True
    args.openie_cache_warm_pending_output = pending_output

    summary = await warm_openie_cache(args)

    assert summary.dry_run is True
    assert summary.rows_loaded == 3
    assert summary.rows_pending == 1
    assert summary.rows_deferred_limit == 1
    assert summary.rows_attempted == 0
    assert summary.rows_succeeded == 0
    assert summary.rows_skipped_cached == 1
    assert summary.coverage_projection_total_chunks == 4
    assert summary.coverage_projection_existing_covered_chunks == 2
    assert summary.coverage_projection_added_chunks == 1
    assert summary.coverage_projection_after_batch_chunks == 3
    assert summary.coverage_projection_after_batch_rate == pytest.approx(0.75)
    assert summary.coverage_projection_target_rate == pytest.approx(1.0)
    assert summary.coverage_projection_target_chunks == 4
    assert summary.coverage_projection_rows_needed_for_target == 2
    assert summary.coverage_projection_batches_needed_at_limit == 2
    assert summary.coverage_projection_target_reachable is True
    assert summary.pending_output == str(pending_output)
    assert summary.cache_hits == 0
    assert summary.cache_misses == 0
    assert cache_path.read_text(encoding="utf-8") == before
    pending_rows = [
        json.loads(line) for line in pending_output.read_text(encoding="utf-8").splitlines() if line
    ]
    assert pending_rows == [
        {"title": "Pending title", "content": "Pending body", "doc_id": "doc-pending"}
    ]


def test_score_summary_clone_as_preserves_counts_with_new_name():
    summary = ScoreSummary("baseline")
    summary.add("semantic", ["doc-a"], ["doc-a"])
    summary.record_timing(10.0, {"fts": 4.0, "expand": 6.0})

    clone = summary.clone_as("openie")

    assert clone.name == "openie"
    assert clone.to_dict()["r5"] == summary.to_dict()["r5"]
    assert clone.by_type == summary.by_type
    assert clone.to_dict()["timing"] == summary.to_dict()["timing"]
    assert clone is not summary


def test_write_results_records_score_reuse_flag(tmp_path: Path):
    results_path = tmp_path / "results.json"
    args = _args(skip_openie=True)
    args.results = results_path
    args.chunks = tmp_path / "chunks.jsonl"
    args.queries = []
    args.baseline_db = tmp_path / "base.db"
    args.openie_db = tmp_path / "openie.db"
    args.max_input_chunks = 1
    args.search_limit = 10

    write_results(
        args,
        base=_score("baseline"),
        openie=_score("openie"),
        openie_run=OpenIERunSummary(skipped=True),
        gates=GateSummary(
            no_regress_r5=True,
            min_delta_r5=True,
            scored_queries_ok=True,
            openie_applied=True,
            openie_extraction_ok=True,
            revertible=True,
            delta_r5=0.0,
            required_delta_r5=0.0,
        ),
        memory_health={
            "scope_key": "global",
            "openie_artifact_count": 2,
            "signal_count": 1,
        },
        scores_reused=True,
    )

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["scores_reused"] is True
    assert payload["scores"]["baseline"]["timing"]["n"] == 0
    assert payload["openie"]["cache_checked_chunks"] == 0
    assert payload["openie"]["cache_eligible_chunks"] == 0
    assert payload["openie"]["cache_skipped_chunks"] == 0
    assert payload["openie"]["cache_coverage_rate"] == 0.0
    assert payload["openie"]["cache_hits"] == 0
    assert payload["openie"]["cache_misses"] == 0
    assert payload["openie"]["cache_hit_rate"] == 0.0
    assert payload["openie"]["cache_missing_rows"] == 0
    assert payload["openie"]["cache_missing_output"] == ""
    assert payload["memory_health"]["scope_key"] == "global"
    assert payload["memory_health"]["openie_artifact_count"] == 2
    assert payload["memory_health"]["signal_count"] == 1
    assert payload["relation_probe"]["n"] == 0


@pytest.mark.asyncio
async def test_graph_fingerprint_includes_edge_properties(tmp_path: Path):
    db_path = tmp_path / "fingerprint.db"
    backend = openie_mz_poc.SqliteGraphBackend(str(db_path))
    await backend.connect()
    try:
        await backend.save_node(Node(id="a", title="A"))
        await backend.save_node(Node(id="b", title="B"))
        await backend.save_edge(
            Edge(
                id="edge_ab",
                source_id="a",
                target_id="b",
                kind=EdgeKind.RELATED,
                properties={"source_event_id": "evt_1"},
            )
        )
    finally:
        await backend.close()

    original = graph_fingerprint(db_path)

    backend = openie_mz_poc.SqliteGraphBackend(str(db_path))
    await backend.connect()
    try:
        await backend.save_edge(
            Edge(
                id="edge_ab",
                source_id="a",
                target_id="b",
                kind=EdgeKind.RELATED,
                properties={"source_event_id": "evt_2"},
            )
        )
    finally:
        await backend.close()

    assert graph_fingerprint(db_path) != original


def test_deepseek_profile_infers_larger_default_output_budget():
    args = _args()
    args.llm_model = "deepseek-v4-flash"

    profile = profile_from_args(args, openie_enabled=True)

    assert profile.openie_model_profile == "deepseek_v4_flash"
    assert profile.openie_max_output_tokens == 4096


def test_explicit_openie_output_budget_overrides_deepseek_default():
    args = _args()
    args.llm_model = "deepseek-v4-flash"
    args.openie_max_output_tokens = 2048

    profile = profile_from_args(args, openie_enabled=True)

    assert profile.openie_model_profile == "deepseek_v4_flash"
    assert profile.openie_max_output_tokens == 2048


def test_qwen36_profile_inferred_without_deepseek_budget_bump():
    args = _args()
    args.llm_model = "/models/Qwen3.6-27B"

    profile = profile_from_args(args, openie_enabled=True)

    assert profile.openie_model_profile == "qwen36_local"
    assert profile.openie_max_output_tokens == 1024


def test_explicit_generic_profile_is_not_overridden_by_model_name():
    args = _args()
    args.llm_model = "deepseek-v4-flash"
    args.openie_model_profile = "generic_openai_compatible"

    profile = profile_from_args(args, openie_enabled=True)

    assert profile.openie_model_profile == "generic_openai_compatible"
    assert profile.openie_max_output_tokens == 1024


@pytest.mark.asyncio
async def test_cache_only_provider_never_calls_remote_model():
    provider = CacheOnlyLLMProvider(model="Qwen3.6-27B")

    assert provider._model == "Qwen3.6-27B"
    with pytest.raises(RuntimeError, match="cache miss"):
        await provider.generate(system="", user="")


@pytest.mark.asyncio
async def test_cache_only_selector_keeps_only_cached_chunks():
    extractor = LLMOpenIEExtractor(_JsonLLM())
    await extractor.extract("Cached title\nCached body", title="Cached title")
    selector = cache_only_selector(extractor, collect_missing=True)

    cached = Node(
        id="chunk_cached",
        kind=NodeKind.CHUNK,
        title="Cached title",
        content="Cached body",
        source="cached-source",
        properties={"doc_id": "doc-cached", "chunk_index": "0"},
    )
    uncached = Node(
        id="chunk_uncached",
        kind=NodeKind.CHUNK,
        title="Uncached title",
        content="Uncached body",
        source="uncached-source",
        properties={"doc_id": "doc-uncached", "chunk_index": "7"},
    )

    assert selector(cached) is True
    assert selector(uncached) is False
    assert selector.checked_chunks == 2
    assert selector.eligible_chunks == 1
    assert selector.skipped_chunks == 1
    assert selector.missing_rows == [
        {
            "node_id": "chunk_uncached",
            "title": "Uncached title",
            "source": "uncached-source",
            "doc_id": "doc-uncached",
            "chunk_index": "7",
            "content_length": len("Uncached body"),
            "content": "Uncached body",
        }
    ]


@pytest.mark.asyncio
async def test_relation_probe_records_graph_lift_for_openie_relation(tmp_path: Path):
    db_path = tmp_path / "probe.db"
    backend = openie_mz_poc.SqliteGraphBackend(str(db_path))
    await backend.connect()
    try:
        await backend.save_node(
            Node(
                id="ent_acme",
                kind=NodeKind.ENTITY,
                title="Acme",
                content="Acme depends on another plan.",
                tags=["_openie", "_openie_entity"],
            )
        )
        await backend.save_node(
            Node(
                id="ent_roadmap",
                kind=NodeKind.ENTITY,
                title="Roadmap",
                content="Release milestones.",
                tags=["_openie", "_openie_entity"],
            )
        )
        await backend.save_edge(
            Edge(
                id="openie_acme_depends_roadmap",
                source_id="ent_acme",
                target_id="ent_roadmap",
                kind=EdgeKind.DEPENDS_ON,
                weight=0.9,
                properties={"is_openie": "true", "relation": "depends_on", "confidence": "0.9"},
            )
        )
    finally:
        await backend.close()

    summary = await relation_probe_for_db(db_path, limit=5, search_k=3)

    payload = summary.to_dict()
    assert payload["n"] == 1
    assert payload["no_graph_expanded_hits"] == 0
    assert payload["graph_expanded_hits"] == 1
    assert payload["expanded_lift"] == 1
    assert payload["by_relation"]["depends_on"]["n"] == 1
    assert payload["by_relation"]["depends_on"]["expanded_lift"] == 1
    assert payload["relation_groups"]["strong"]["n"] == 1
    assert payload["relation_groups"]["strong"]["evidence_lift"] == 1
    assert payload["relation_groups"]["weak"]["n"] == 0
    assert payload["probes"][0]["target_id"] == "ent_roadmap"
    assert payload["probes"][0]["confidence"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_relation_probe_shares_fts_cache_between_ablation_searches(
    monkeypatch,
    tmp_path: Path,
):
    db_path = tmp_path / "probe_cache.db"
    original_backend = openie_mz_poc.SqliteGraphBackend
    backend = original_backend(str(db_path))
    await backend.connect()
    try:
        await backend.save_node(
            Node(
                id="ent_acme",
                kind=NodeKind.ENTITY,
                title="Acme",
                content="Acme depends on another plan.",
                tags=["_openie", "_openie_entity"],
            )
        )
        await backend.save_node(
            Node(
                id="ent_roadmap",
                kind=NodeKind.ENTITY,
                title="Roadmap",
                content="Release milestones.",
                tags=["_openie", "_openie_entity"],
            )
        )
        await backend.save_edge(
            Edge(
                id="openie_acme_depends_roadmap",
                source_id="ent_acme",
                target_id="ent_roadmap",
                kind=EdgeKind.DEPENDS_ON,
                weight=0.9,
                properties={"is_openie": "true", "relation": "depends_on"},
            )
        )
    finally:
        await backend.close()

    class CountingBackend(original_backend):
        search_fts_calls: ClassVar[int] = 0

        async def search_fts(self, query: str, *, limit: int = 20, with_scores: bool = False):
            type(self).search_fts_calls += 1
            return await super().search_fts(query, limit=limit, with_scores=with_scores)

    class FtsOnlyEvidenceSearch:
        def __init__(self, backend, graph_expansion=True):
            self.backend = backend
            self.graph_expansion = graph_expansion

        async def search(self, query: str, k: int = 10):
            await self.backend.search_fts(query, limit=k)
            return SimpleNamespace(expanded=[], evidence=[])

    monkeypatch.setattr(openie_mz_poc, "SqliteGraphBackend", CountingBackend)
    monkeypatch.setattr(openie_mz_poc, "EvidenceSearch", FtsOnlyEvidenceSearch)

    summary = await relation_probe_for_db(db_path, limit=1, search_k=3)

    assert summary.n == 1
    assert CountingBackend.search_fts_calls == 1


@pytest.mark.asyncio
async def test_relation_probe_caps_repeated_source_relation_pairs(tmp_path: Path):
    db_path = tmp_path / "probe_cap.db"
    backend = openie_mz_poc.SqliteGraphBackend(str(db_path))
    await backend.connect()
    try:
        await backend.save_node(
            Node(
                id="ent_hub",
                kind=NodeKind.ENTITY,
                title="Hub",
                content="Hub produced several artifacts.",
                tags=["_openie", "_openie_entity"],
            )
        )
        for idx in range(5):
            target_id = f"ent_target_{idx}"
            await backend.save_node(
                Node(
                    id=target_id,
                    kind=NodeKind.ENTITY,
                    title=f"Target {idx}",
                    content=f"Produced target {idx}.",
                    tags=["_openie", "_openie_entity"],
                )
            )
            await backend.save_edge(
                Edge(
                    id=f"openie_hub_produced_{idx}",
                    source_id="ent_hub",
                    target_id=target_id,
                    kind=EdgeKind.PRODUCED,
                    weight=0.9,
                    properties={
                        "is_openie": "true",
                        "relation": "produced",
                        "confidence": "0.9",
                    },
                )
            )
    finally:
        await backend.close()

    summary = await relation_probe_for_db(
        db_path,
        limit=10,
        search_k=10,
        max_per_source_relation=2,
    )

    payload = summary.to_dict()
    assert payload["n"] == 2
    assert payload["by_relation"]["produced"]["n"] == 2
    assert len({probe["edge_id"] for probe in payload["probes"]}) == 2


@pytest.mark.asyncio
async def test_eval_memory_health_is_read_only(tmp_path: Path):
    db_path = tmp_path / "health.db"
    backend = openie_mz_poc.SqliteGraphBackend(str(db_path))
    await backend.connect()
    try:
        for node_id in ("ent_source", "ent_target"):
            await backend.save_node(
                Node(
                    id=node_id,
                    kind=NodeKind.ENTITY,
                    title=node_id,
                    content=node_id,
                    tags=["_openie", "_openie_entity"],
                )
            )
        await backend.save_edge(
            Edge(
                id="custom_low_confidence_relation",
                source_id="ent_source",
                target_id="ent_target",
                kind=EdgeKind.RELATED,
                weight=0.4,
                properties={"is_openie": "true", "confidence": "0.4"},
            )
        )
        await backend.save_retrieval_event(
            RetrievalEvent(
                id="ret_memory_ranked",
                query="ranked memory",
                scope=MemoryScope(),
                properties={
                    "memory_scope_boosted_nodes": "2.000000",
                    "memory_scope_max_abs_boost": "0.100000",
                    "memory_signal_penalized_nodes": "1.000000",
                    "memory_signal_max_penalty": "0.050000",
                },
            )
        )
    finally:
        await backend.close()

    report = await memory_health_for_db(db_path)

    backend = openie_mz_poc.SqliteGraphBackend(str(db_path))
    await backend.connect()
    try:
        nodes = await backend.list_nodes(limit=100)
    finally:
        await backend.close()
    assert report["signal_count"] >= 1
    assert report["memory_boosted_retrieval_count"] == 1
    assert report["memory_penalized_retrieval_count"] == 1
    assert report["memory_boosted_node_count"] == 2
    assert report["memory_penalized_node_count"] == 1
    assert report["max_memory_scope_boost"] == pytest.approx(0.10)
    assert report["max_memory_signal_penalty"] == pytest.approx(0.05)
    assert report["openie_artifact_count"] == 3
    assert not any("_memory_signal" in (node.tags or []) for node in nodes)


@pytest.mark.asyncio
async def test_relation_probe_thresholds_participate_in_gate_result():
    args = _args()
    args.min_relation_expanded_lift = 1
    args.min_relation_evidence_lift = 1
    args.min_strong_relation_evidence_rate = 1.0
    relation_probe = RelationProbeSummary(
        n=1,
        graph_expanded_hits=1,
        graph_evidence_hits=1,
        no_graph_expanded_hits=0,
        no_graph_evidence_hits=0,
        by_relation={
            "depends_on": {
                "n": 1,
                "graph_expanded_hits": 1,
                "graph_evidence_hits": 1,
                "no_graph_expanded_hits": 0,
                "no_graph_evidence_hits": 0,
            }
        },
    )

    passing = await evaluate_gates(
        args,
        base=_score("baseline"),
        openie=_score("openie"),
        openie_run=OpenIERunSummary(
            chunks_selected=1,
            entity_nodes_touched=2,
            relation_edges_created=1,
        ),
        relation_probe=relation_probe,
    )

    assert passing.relation_probe_ok is True
    assert passing.passed is True

    args.min_relation_evidence_lift = 2
    failing = await evaluate_gates(
        args,
        base=_score("baseline"),
        openie=_score("openie"),
        openie_run=OpenIERunSummary(
            chunks_selected=1,
            entity_nodes_touched=2,
            relation_edges_created=1,
        ),
        relation_probe=relation_probe,
    )

    assert failing.relation_evidence_lift_ok is False
    assert failing.relation_probe_ok is False
    assert failing.passed is False


@pytest.mark.asyncio
async def test_cache_coverage_threshold_participates_in_gate_result():
    args = _args()
    args.min_openie_cache_coverage = 0.5

    passing = await evaluate_gates(
        args,
        base=_score("baseline"),
        openie=_score("openie"),
        openie_run=OpenIERunSummary(
            chunks_selected=1,
            entity_nodes_touched=2,
            relation_edges_created=1,
            cache_checked_chunks=4,
            cache_eligible_chunks=2,
        ),
    )

    assert passing.cache_coverage_ok is True
    assert passing.cache_coverage_rate == pytest.approx(0.5)
    assert passing.required_cache_coverage_rate == pytest.approx(0.5)
    assert passing.passed is True

    args.min_openie_cache_coverage = 0.75
    failing = await evaluate_gates(
        args,
        base=_score("baseline"),
        openie=_score("openie"),
        openie_run=OpenIERunSummary(
            chunks_selected=1,
            entity_nodes_touched=2,
            relation_edges_created=1,
            cache_checked_chunks=4,
            cache_eligible_chunks=2,
        ),
    )

    assert failing.cache_coverage_ok is False
    assert failing.cache_coverage_rate == pytest.approx(0.5)
    assert failing.passed is False


@pytest.mark.asyncio
async def test_precompute_query_embeddings_batches_unique_queries(monkeypatch):
    _RecordingEmbedder.calls = []
    monkeypatch.setattr(openie_mz_poc, "OpenAIEmbeddingProvider", _RecordingEmbedder)

    vectors = await precompute_query_embeddings(
        [{"query": "b"}, {"query": "a"}, {"query": "b"}],
        _args(),
        batch_size=2,
    )

    assert _RecordingEmbedder.calls == [["a", "b"]]
    assert set(vectors) == {"a", "b"}


@pytest.mark.asyncio
async def test_non_skipped_gate_fails_when_openie_creates_no_relation_edges():
    gates = await evaluate_gates(
        _args(),
        base=_score("baseline"),
        openie=_score("openie"),
        openie_run=OpenIERunSummary(
            skipped=False,
            chunks_scanned=1,
            chunks_selected=1,
            entity_nodes_touched=2,
            relation_edges_created=0,
            extraction_failures=0,
        ),
    )

    assert gates.no_regress_r5 is True
    assert gates.openie_applied is False
    assert gates.openie_extraction_ok is True
    assert gates.passed is False


@pytest.mark.asyncio
async def test_non_skipped_gate_passes_when_openie_touches_entities():
    gates = await evaluate_gates(
        _args(),
        base=_score("baseline"),
        openie=_score("openie"),
        openie_run=OpenIERunSummary(
            skipped=False,
            chunks_scanned=1,
            chunks_selected=1,
            entity_nodes_touched=2,
            relation_edges_created=1,
            extraction_failures=0,
        ),
    )

    assert gates.openie_applied is True
    assert gates.openie_extraction_ok is True
    assert gates.passed is True


@pytest.mark.asyncio
async def test_skipped_gate_allows_no_openie_artifacts_for_smoke_mode():
    gates = await evaluate_gates(
        _args(skip_openie=True),
        base=_score("baseline"),
        openie=_score("openie"),
        openie_run=OpenIERunSummary(skipped=True),
    )

    assert gates.openie_applied is True
    assert gates.openie_extraction_ok is True
    assert gates.passed is True


@pytest.mark.asyncio
async def test_gate_fails_when_no_queries_are_scoreable():
    gates = await evaluate_gates(
        _args(skip_openie=True),
        base=_score("baseline", n=0, r5=0),
        openie=_score("openie", n=0, r5=0),
        openie_run=OpenIERunSummary(skipped=True),
    )

    assert gates.scored_queries_ok is False
    assert gates.passed is False


@pytest.mark.asyncio
async def test_embed_missing_openie_nodes_only_updates_openie_entity_hubs():
    backend = MemoryBackend()
    await backend.connect()
    openie_entity = Node(
        id="ent_openie",
        kind=NodeKind.ENTITY,
        title="roadmap",
        level=ConsolidationLevel.L0_RAW,
        tags=["_openie", "_openie_entity"],
    )
    phrase_entity = Node(
        id="ent_phrase",
        kind=NodeKind.ENTITY,
        title="roadmap",
        level=ConsolidationLevel.L0_RAW,
        tags=["_phrase"],
    )
    already_embedded = Node(
        id="ent_openie_embedded",
        kind=NodeKind.ENTITY,
        title="done",
        level=ConsolidationLevel.L0_RAW,
        tags=["_openie", "_openie_entity"],
        embedding=[0.5],
    )
    await backend.save_node(openie_entity)
    await backend.save_node(phrase_entity)
    await backend.save_node(already_embedded)

    filled = await embed_missing_openie_nodes(backend, _TinyEmbedder(), batch_size=2)

    assert filled == 1
    assert (await backend.get_node("ent_openie")).embedding == [1.0]
    assert (await backend.get_node("ent_phrase")).embedding == []
    assert (await backend.get_node("ent_openie_embedded")).embedding == [0.5]
