"""Tests for Tier-1 benchmark runner helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "ablation" / "run_tier1_benchmarks.py"
)
SPEC = importlib.util.spec_from_file_location("run_tier1_benchmarks", RUNNER_PATH)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_msmarco_path_override_retargets_dataset(tmp_path):
    manifest = tmp_path / "msmarco_passage_5m.json"
    by_key = runner._dataset_key_map(manifest)

    assert by_key["msmarco"].path == manifest
    assert by_key["msmarco"].name == runner.DATASETS[6].name


def test_reuse_signature_excludes_search_time_components():
    signature = runner._reuse_signature(
        runner.Dataset(name="Tiny", path=Path("tiny.json"), reference="unit"),
        {"corpus": {"doc": {}}, "source": "unit"},
        corpus_limit=1,
        embedder=None,
        phrase_extractor=None,
        entity_linker_cfg=None,
    )

    assert "reranker" not in signature
    assert "decomposer" not in signature


@pytest.mark.asyncio
async def test_fts_seed_limit_rejects_non_positive():
    with pytest.raises(SystemExit, match="--fts-seed-limit must be positive"):
        await runner.amain(["--fts-seed-limit", "0"])


@pytest.mark.asyncio
async def test_diagnose_raw_fts_limit_rejects_non_positive():
    with pytest.raises(SystemExit, match="--diagnose-raw-fts-limit must be positive"):
        await runner.amain(["--diagnose-raw-fts-limit", "0"])


@pytest.mark.asyncio
async def test_sqlite_fast_build_pragmas_apply(tmp_path):
    backend = runner.SqliteGraphBackend(str(tmp_path / "fast.db"))
    await backend.connect()

    await runner._apply_sqlite_fast_build_pragmas(backend)

    async with backend._db().execute("PRAGMA synchronous") as cur:
        row = await cur.fetchone()
    await backend.close()

    assert row[0] == 0


@pytest.mark.asyncio
async def test_corpus_limit_keeps_selected_query_gold_docs(tmp_path):
    path = tmp_path / "tiny_bench.json"
    path.write_text(
        json.dumps(
            {
                "corpus": {
                    "filler_a": {"title": "Filler A", "text": "unrelated alpha"},
                    "filler_b": {"title": "Filler B", "text": "unrelated beta"},
                    "gold_doc": {"title": "Gold", "text": "needle targetterm"},
                },
                "queries": {"q1": "targetterm"},
                "qrels": {"q1": {"gold_doc": 1}},
            }
        ),
        encoding="utf-8",
    )

    report = await runner.run_one(
        runner.Dataset(name="Tiny", path=path, reference="unit"),
        subset=1,
        corpus_limit=2,
    )

    assert report.n_docs == 2
    assert report.hit_at_10 == 1
    assert report.recall_at_10 == 1.0


@pytest.mark.asyncio
async def test_jsonl_corpus_limit_keeps_selected_query_gold_docs(tmp_path):
    manifest = tmp_path / "large_bench.json"
    corpus_path = tmp_path / "large_bench.corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "_id": "filler_a",
                        "title": "Filler A",
                        "text": "unrelated alpha",
                    }
                ),
                json.dumps(
                    {
                        "_id": "filler_b",
                        "title": "Filler B",
                        "text": "unrelated beta",
                    }
                ),
                json.dumps(
                    {
                        "_id": "gold_doc",
                        "title": "Gold",
                        "text": "needle targetterm",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "name": "Tiny JSONL",
                "schema": "beir_jsonl_v1",
                "corpus_path": corpus_path.name,
                "corpus_size": 3,
                "queries": {"q1": "targetterm"},
                "qrels": {"q1": {"gold_doc": 1}},
            }
        ),
        encoding="utf-8",
    )

    report = await runner.run_one(
        runner.Dataset(name="Tiny JSONL", path=manifest, reference="unit"),
        subset=1,
        corpus_limit=2,
    )

    assert report.n_docs == 2
    assert report.hit_at_10 == 1
    assert report.recall_at_10 == 1.0


@pytest.mark.asyncio
async def test_progress_every_reports_ingest_progress(tmp_path, capsys):
    path = tmp_path / "tiny_bench.json"
    path.write_text(
        json.dumps(
            {
                "corpus": {
                    "gold_doc": {"title": "Gold", "text": "needle targetterm"},
                    "filler": {"title": "Filler", "text": "unrelated alpha"},
                },
                "queries": {"q1": "targetterm"},
                "qrels": {"q1": {"gold_doc": 1}},
            }
        ),
        encoding="utf-8",
    )

    await runner.run_one(
        runner.Dataset(name="Tiny", path=path, reference="unit"),
        subset=1,
        corpus_limit=2,
        ingest_batch=1,
        progress_every=1,
    )

    output = capsys.readouterr().out

    assert "ingest: 1/2 docs" in output
    assert "ingest: 2/2 docs" in output


@pytest.mark.asyncio
async def test_progress_every_reports_crossed_batch_boundary(tmp_path, capsys):
    path = tmp_path / "tiny_bench.json"
    path.write_text(
        json.dumps(
            {
                "corpus": {
                    "gold_doc": {"title": "Gold", "text": "needle targetterm"},
                    "filler_a": {"title": "Filler A", "text": "unrelated alpha"},
                    "filler_b": {"title": "Filler B", "text": "unrelated beta"},
                    "filler_c": {"title": "Filler C", "text": "unrelated gamma"},
                    "filler_d": {"title": "Filler D", "text": "unrelated delta"},
                },
                "queries": {"q1": "targetterm"},
                "qrels": {"q1": {"gold_doc": 1}},
            }
        ),
        encoding="utf-8",
    )

    await runner.run_one(
        runner.Dataset(name="Tiny", path=path, reference="unit"),
        subset=1,
        corpus_limit=5,
        ingest_batch=2,
        progress_every=3,
    )

    output = capsys.readouterr().out

    assert "ingest: 4/5 docs" in output
    assert "ingest: 5/5 docs" in output


@pytest.mark.asyncio
async def test_fts_seed_limit_passes_to_graph_search(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "tiny_bench.json"
    path.write_text(
        json.dumps(
            {
                "corpus": {"gold_doc": {"title": "Gold", "text": "needle targetterm"}},
                "queries": {"q1": "targetterm"},
                "qrels": {"q1": {"gold_doc": 1}},
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, int | None] = {}

    class _FakeGraph:
        def __init__(self, *args, **kwargs) -> None:
            self.backend = args[0]

        async def search(self, query: str, *, limit: int, fts_seed_limit: int | None = None):
            seen["fts_seed_limit"] = fts_seed_limit
            node = SimpleNamespace(properties={"doc_id": "gold_doc"})
            return SimpleNamespace(nodes=[SimpleNamespace(node=node)])

    monkeypatch.setattr(runner, "SynapticGraph", _FakeGraph)

    report = await runner.run_one(
        runner.Dataset(name="Tiny", path=path, reference="unit"),
        subset=1,
        corpus_limit=1,
        fts_seed_limit=77,
    )

    assert seen["fts_seed_limit"] == 77
    assert report.hit_at_10 == 1


@pytest.mark.asyncio
async def test_diagnose_raw_fts_limit_reports_pool_metrics(tmp_path):
    path = tmp_path / "tiny_bench.json"
    path.write_text(
        json.dumps(
            {
                "corpus": {
                    "gold_doc": {"title": "Gold", "text": "needle targetterm"},
                    "distractor": {"title": "Other", "text": "needle unrelated"},
                },
                "queries": {"q1": "targetterm"},
                "qrels": {"q1": {"gold_doc": 1}},
            }
        ),
        encoding="utf-8",
    )

    report = await runner.run_one(
        runner.Dataset(name="Tiny", path=path, reference="unit"),
        subset=1,
        corpus_limit=2,
        diagnose_raw_fts_limit=2,
    )

    assert report.hit_at_10 == 1
    assert report.raw_fts_pool_limit == 2
    assert report.raw_fts_hit_at_10 == 1
    assert report.raw_fts_any_at_pool == 1
    assert report.raw_fts_mrr == 1.0


@pytest.mark.asyncio
async def test_sqlite_db_reuse_skips_reingest(tmp_path, capsys):
    path = tmp_path / "tiny_bench.json"
    db_path = tmp_path / "tiny.db"
    path.write_text(
        json.dumps(
            {
                "corpus": {
                    "gold_doc": {"title": "Gold", "text": "needle targetterm"},
                    "filler": {"title": "Filler", "text": "unrelated alpha"},
                },
                "queries": {"q1": "targetterm"},
                "qrels": {"q1": {"gold_doc": 1}},
            }
        ),
        encoding="utf-8",
    )

    first = await runner.run_one(
        runner.Dataset(name="Tiny", path=path, reference="unit"),
        subset=1,
        corpus_limit=2,
        use_sqlite_graph=True,
        sqlite_db_path=db_path,
        progress_every=0,
    )
    capsys.readouterr()

    # If the second run ingested from the source file again, the gold
    # document would no longer match the query. Reuse should keep the
    # already-built SQLite index intact.
    path.write_text(
        json.dumps(
            {
                "corpus": {
                    "gold_doc": {"title": "Gold", "text": "changed text"},
                    "filler": {"title": "Filler", "text": "unrelated alpha"},
                },
                "queries": {"q1": "targetterm"},
                "qrels": {"q1": {"gold_doc": 1}},
            }
        ),
        encoding="utf-8",
    )

    second = await runner.run_one(
        runner.Dataset(name="Tiny", path=path, reference="unit"),
        subset=1,
        corpus_limit=2,
        use_sqlite_graph=True,
        sqlite_db_path=db_path,
        reuse_sqlite_db=True,
        progress_every=0,
    )
    output = capsys.readouterr().out

    assert first.hit_at_10 == 1
    assert second.hit_at_10 == 1
    assert second.n_docs == 2
    assert "reuse sqlite db:" in output
    assert db_path.with_name(f"{db_path.name}.tier1.json").exists()


def test_reuse_meta_mismatches_report_changed_signature():
    mismatches = runner._reuse_meta_mismatches(
        {"version": 1, "dataset": "Tiny", "corpus_limit": 2},
        {"version": 1, "dataset": "Tiny", "corpus_limit": 1},
    )

    assert mismatches == ["corpus_limit: 2 != 1"]


def test_threshold_violations_report_scale_regressions():
    report = runner.Report(
        name="Tiny",
        n_docs=100,
        n_queries=10,
        mrr=0.25,
        recall_at_5=0.1,
        recall_at_10=0.2,
        hit_at_10=3,
        build_sec=12.0,
        search_sec=4.0,
        reference="unit",
    )

    violations = runner._threshold_violations(
        [report],
        max_build_sec=10.0,
        max_search_sec=3.0,
        min_hit_rate_at_10=0.5,
        min_mrr=0.3,
    )

    assert violations == [
        "Tiny: build 12.0s > 10.0s",
        "Tiny: search 4.0s > 3.0s",
        "Tiny: hit@10 rate 0.300 < 0.500",
        "Tiny: MRR@10 0.250 < 0.300",
    ]


def test_emit_markdown_includes_raw_fts_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    report = runner.Report(
        name="Tiny",
        n_docs=100,
        n_queries=10,
        mrr=0.25,
        recall_at_5=0.1,
        recall_at_10=0.2,
        hit_at_10=3,
        build_sec=12.0,
        search_sec=4.0,
        reference="unit",
        raw_fts_pool_limit=50,
        raw_fts_mrr=0.4,
        raw_fts_recall_at_5=0.2,
        raw_fts_recall_at_10=0.3,
        raw_fts_hit_at_10=5,
        raw_fts_any_at_pool=8,
        raw_fts_sec=1.2,
    )

    path = runner._emit_markdown(
        [report],
        subset=10,
        embedder_label="none",
        reranker_label="none",
        diagnose_raw_fts_limit=50,
    )

    content = path.read_text(encoding="utf-8")
    assert "Raw FTS Pool Diagnostic" in content
    assert "| Tiny | 50 | 0.400 | 0.200 | 0.300 | 5/10 | 8/10 | 1.2s |" in content


def test_threshold_violations_accept_passing_report():
    report = runner.Report(
        name="Tiny",
        n_docs=100,
        n_queries=10,
        mrr=0.5,
        recall_at_5=0.1,
        recall_at_10=0.2,
        hit_at_10=8,
        build_sec=2.0,
        search_sec=1.0,
        reference="unit",
    )

    assert (
        runner._threshold_violations(
            [report],
            max_build_sec=10.0,
            max_search_sec=3.0,
            min_hit_rate_at_10=0.5,
            min_mrr=0.3,
        )
        == []
    )
