"""Tests for ``eval/dump_perquery_hits.py`` (v0.29 E1 / Track A).

Locks the contract the routing GT's ``--hits-jsonl`` axis depends on:

1. hit/rank correctness against both id_field conventions —
   ``doc_id`` (``ev.document_id or properties.doc_id``) and
   ``node_title`` — mirroring ``eval/run_all.py:run_custom_dataset``.
2. The qid convention: the query file's own ``qid``/``id`` field when
   present (must match the gt_datasets.xlsx sheet qids for the routing
   GT axis merge), zero-based ``q{i:03d}`` fallback otherwise, and
   ``qid_mode="index"`` forcing the rag_vs_agent JSONL convention.
3. Queries without ``relevant_docs`` are skipped, exactly as run_all
   drops them from the MRR denominator.
4. Determinism — same graph + queries + k → byte-identical output file.
5. The public-bench path — BEIR dict format, graph auto-build from the
   embedded corpus, ``properties.doc_id`` matching.

All tests run on small fixture graphs under tmp_path; the real-data
sanity test is skipif-guarded on the corpus sqlite being present.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.dump_perquery_hits import (
    dump_hits,
    hit_rank,
    is_public_dataset,
    parse_public_corpus,
    parse_public_queries,
    resolve_qid,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSORT_GRAPH = REPO_ROOT / "eval" / "data" / "assort_graph.sqlite"
ASSORT_QUERIES = REPO_ROOT / "eval" / "data" / "queries" / "assort.json"


# --- unit: hit_rank --------------------------------------------------


def test_hit_rank_first_relevant_position() -> None:
    assert hit_rank(["a", "b", "c"], {"b", "c"}, k=5) == 2


def test_hit_rank_miss_returns_none() -> None:
    assert hit_rank(["a", "b"], {"z"}, k=5) is None


def test_hit_rank_respects_k_cutoff() -> None:
    # gold at position 6 is NOT a hit@5 — run_all truncates to k first
    retrieved = ["r1", "r2", "r3", "r4", "r5", "gold"]
    assert hit_rank(retrieved, {"gold"}, k=5) is None
    assert hit_rank(retrieved, {"gold"}, k=6) == 6


def test_hit_rank_empty_retrieved() -> None:
    assert hit_rank([], {"gold"}, k=5) is None


# --- unit: qid convention --------------------------------------------


def test_resolve_qid_prefers_qid_field() -> None:
    assert resolve_qid({"qid": "a007"}, 0) == "a007"


def test_resolve_qid_falls_back_to_id_field() -> None:
    assert resolve_qid({"id": "x1"}, 3) == "x1"


def test_resolve_qid_index_fallback_is_zero_based() -> None:
    # the rag_vs_agent_answer.py / routing_gt finreg-loader convention
    assert resolve_qid({}, 0) == "q000"
    assert resolve_qid({}, 12) == "q012"


def test_resolve_qid_index_mode_ignores_fields() -> None:
    assert resolve_qid({"qid": "s001"}, 4, qid_mode="index") == "q004"


# --- fixture graph ----------------------------------------------------


async def _build_doc_graph(path: Path) -> None:
    """Three distinctive docs with properties.doc_id, FTS-only."""
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.graph import SynapticGraph

    backend = SqliteGraphBackend(str(path))
    await backend.connect()
    graph = SynapticGraph(backend)
    docs = [
        ("d1", "Penguin habitat", "Emperor penguins breed on Antarctic sea ice colonies."),
        ("d2", "Espresso brewing", "Espresso extraction uses nine bars of pressure."),
        ("d3", "Volcano basics", "Shield volcanoes erupt low-viscosity basaltic lava."),
    ]
    for doc_id, title, text in docs:
        await graph.add(title=title, content=text, properties={"doc_id": doc_id})
    await backend.close()


def _write_queries(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def doc_graph(tmp_path: Path) -> Path:
    graph_path = tmp_path / "fixture_graph.sqlite"
    asyncio.run(_build_doc_graph(graph_path))
    return graph_path


# --- custom path: doc_id matching --------------------------------------


def test_custom_doc_id_hit_and_miss(doc_graph: Path, tmp_path: Path) -> None:
    queries = _write_queries(
        tmp_path / "queries.json",
        {
            "id_field": "doc_id",
            "queries": [
                {"qid": "q001", "query": "emperor penguins antarctic", "relevant_docs": ["d1"]},
                {"qid": "q002", "query": "espresso pressure", "relevant_docs": ["d2"]},
                {"qid": "q003", "query": "quantum chromodynamics", "relevant_docs": ["d3"]},
            ],
        },
    )
    out = tmp_path / "hits.jsonl"
    rows = dump_hits(doc_graph, queries, k=5, out=out)

    by_qid = {r["qid"]: r for r in rows}
    assert by_qid["q001"]["hit"] is True
    assert by_qid["q001"]["rank"] == 1
    assert by_qid["q002"]["hit"] is True
    assert by_qid["q002"]["rank"] == 1
    # q003's gold is d3 but the query matches nothing about volcanoes
    assert by_qid["q003"]["hit"] is False
    assert by_qid["q003"]["rank"] is None

    # written file mirrors the returned rows, one JSON object per line
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines == rows


def test_custom_node_title_matching(doc_graph: Path, tmp_path: Path) -> None:
    queries = _write_queries(
        tmp_path / "queries.json",
        {
            "id_field": "node_title",
            "queries": [
                {
                    "qid": "q001",
                    "query": "emperor penguins antarctic",
                    "relevant_docs": ["Penguin habitat"],
                },
                # gold is a doc_id, not a title — must MISS under node_title
                {"qid": "q002", "query": "espresso pressure", "relevant_docs": ["d2"]},
            ],
        },
    )
    rows = dump_hits(doc_graph, queries, k=5, out=None)
    by_qid = {r["qid"]: r for r in rows}
    assert by_qid["q001"]["hit"] is True
    assert by_qid["q002"]["hit"] is False


def test_custom_skips_queries_without_gold(doc_graph: Path, tmp_path: Path) -> None:
    queries = _write_queries(
        tmp_path / "queries.json",
        {
            "queries": [
                {"qid": "q001", "query": "espresso pressure", "relevant_docs": []},
                {"qid": "q002", "query": "espresso pressure", "relevant_docs": ["d2"]},
                {"qid": "q003", "query": "no gold at all"},
            ],
        },
    )
    rows = dump_hits(doc_graph, queries, k=5, out=None)
    assert [r["qid"] for r in rows] == ["q002"]


def test_custom_index_qid_fallback_uses_file_position(doc_graph: Path, tmp_path: Path) -> None:
    # no qid/id fields anywhere → q{i:03d} over the file's query list,
    # so the skipped no-gold query still advances the index
    queries = _write_queries(
        tmp_path / "queries.json",
        {
            "queries": [
                {"query": "nothing relevant here"},
                {"query": "espresso pressure", "relevant_docs": ["d2"]},
            ],
        },
    )
    rows = dump_hits(doc_graph, queries, k=5, out=None)
    assert [r["qid"] for r in rows] == ["q001"]


def test_custom_qid_mode_index_forces_convention(doc_graph: Path, tmp_path: Path) -> None:
    queries = _write_queries(
        tmp_path / "queries.json",
        {
            "queries": [
                {"qid": "s001", "query": "espresso pressure", "relevant_docs": ["d2"]},
            ],
        },
    )
    rows = dump_hits(doc_graph, queries, k=5, out=None, qid_mode="index")
    assert rows[0]["qid"] == "q000"


def test_k_cutoff_changes_hit(doc_graph: Path, tmp_path: Path) -> None:
    # the query lexically matches only d3, so it must rank first; with
    # gold=d3 even k=1 hits, while gold=d1 (unmatched by the query) misses.
    queries = _write_queries(
        tmp_path / "queries.json",
        {
            "queries": [
                {"qid": "q001", "query": "basaltic lava shield", "relevant_docs": ["d3"]},
                {"qid": "q002", "query": "basaltic lava shield", "relevant_docs": ["d1"]},
            ],
        },
    )
    rows = dump_hits(doc_graph, queries, k=1, out=None)
    by_qid = {r["qid"]: r for r in rows}
    assert by_qid["q001"]["hit"] is True
    assert by_qid["q001"]["rank"] == 1
    assert by_qid["q002"]["hit"] is False


def test_missing_graph_fails_for_custom(tmp_path: Path) -> None:
    queries = _write_queries(
        tmp_path / "queries.json",
        {"queries": [{"qid": "q001", "query": "x", "relevant_docs": ["d1"]}]},
    )
    with pytest.raises(SystemExit):
        dump_hits(tmp_path / "missing.sqlite", queries, k=5, out=None)


# --- determinism -------------------------------------------------------


def test_dump_is_byte_identical_across_runs(doc_graph: Path, tmp_path: Path) -> None:
    queries = _write_queries(
        tmp_path / "queries.json",
        {
            "queries": [
                {"qid": "q001", "query": "emperor penguins antarctic", "relevant_docs": ["d1"]},
                {"qid": "q002", "query": "espresso pressure", "relevant_docs": ["d2"]},
                {"qid": "q003", "query": "shield volcano lava", "relevant_docs": ["d3"]},
            ],
        },
    )
    out1 = tmp_path / "hits1.jsonl"
    out2 = tmp_path / "hits2.jsonl"
    dump_hits(doc_graph, queries, k=5, out=out1)
    dump_hits(doc_graph, queries, k=5, out=out2)
    assert out1.read_bytes() == out2.read_bytes()


# --- public bench JSON path ---------------------------------------------


def _public_payload() -> dict:
    """A minimal BEIR-dict bench file (the AutoRAG shape)."""
    return {
        "corpus": {
            "doc - a.pdf - 1": {"title": "", "text": "Emperor penguins breed on Antarctic ice."},
            "doc - a.pdf - 2": {"title": "", "text": "Espresso extraction needs nine bars."},
            "doc - b.pdf - 1": {"title": "", "text": "Shield volcanoes erupt basaltic lava."},
        },
        "queries": {
            "0_x": "emperor penguins antarctic",
            "1_x": "nine bars espresso",
            "2_x": "completely unrelated query terms",
        },
        "qrels": {
            "0_x": {"doc - a.pdf - 1": 1},
            "1_x": {"doc - a.pdf - 2": 1},
            "2_x": {"doc - b.pdf - 1": 1},
        },
    }


def test_is_public_dataset_detection() -> None:
    assert is_public_dataset(_public_payload()) is True
    assert is_public_dataset({"queries": [], "id_field": "doc_id"}) is False


def test_parse_public_corpus_and_queries() -> None:
    payload = _public_payload()
    corpus = parse_public_corpus(payload)
    assert ("doc - a.pdf - 1", "", "Emperor penguins breed on Antarctic ice.") in corpus
    queries = parse_public_queries(payload)
    assert ("0_x", "emperor penguins antarctic", {"doc - a.pdf - 1"}) in queries
    # qrels-less queries are dropped, mirroring run_public_dataset
    payload["queries"]["3_x"] = "orphan query"
    assert all(qid != "3_x" for qid, _, _ in parse_public_queries(payload))


def test_public_builds_graph_and_scores(tmp_path: Path) -> None:
    payload = _public_payload()
    queries_path = _write_queries(tmp_path / "bench.json", payload)
    graph_path = tmp_path / "public_graph.sqlite"
    out = tmp_path / "hits.jsonl"

    rows = dump_hits(graph_path, queries_path, k=5, out=out)
    assert graph_path.exists()  # auto-built from the embedded corpus

    by_qid = {r["qid"]: r for r in rows}
    assert set(by_qid) == {"0_x", "1_x", "2_x"}
    assert by_qid["0_x"]["hit"] is True
    assert by_qid["0_x"]["rank"] == 1
    assert by_qid["1_x"]["hit"] is True
    assert by_qid["2_x"]["hit"] is False
    assert by_qid["2_x"]["rank"] is None

    # second dump against the (now existing) graph is byte-identical
    out2 = tmp_path / "hits2.jsonl"
    dump_hits(graph_path, queries_path, k=5, out=out2)
    assert out.read_bytes() == out2.read_bytes()


# --- routing_gt interop -------------------------------------------------


def test_hits_file_feeds_routing_gt_loader(doc_graph: Path, tmp_path: Path) -> None:
    """The written JSONL is directly consumable by routing_gt's hits source."""
    from eval.routing_gt import load_hits_source

    queries = _write_queries(
        tmp_path / "queries.json",
        {
            "queries": [
                {"qid": "q001", "query": "emperor penguins antarctic", "relevant_docs": ["d1"]},
                {"qid": "q002", "query": "unmatchable nonsense", "relevant_docs": ["d2"]},
            ],
        },
    )
    out = tmp_path / "hits.jsonl"
    dump_hits(doc_graph, queries, k=5, out=out)

    gt_rows = load_hits_source(out, corpus="fixture")
    assert [r["qid"] for r in gt_rows] == ["fixture:q001", "fixture:q002"]
    assert gt_rows[0]["single_shot_hit"] is True
    assert gt_rows[0]["single_shot_basis"] == "id_hit"
    assert gt_rows[0]["tier"] == "hit_only"
    assert gt_rows[1]["single_shot_hit"] is False


# --- real-data sanity (skipif-guarded) -----------------------------------


@pytest.mark.skipif(
    not (ASSORT_GRAPH.exists() and ASSORT_QUERIES.exists()),
    reason="assort corpus sqlite not present",
)
def test_real_assort_easy_direction(tmp_path: Path) -> None:
    """assort Easy is a strong FTS-only bench (MRR 0.760 baseline) — the
    deterministic hit@5 axis must point the same direction (well above
    chance), and qids must match the gt_datasets.xlsx sheet convention."""
    rows = dump_hits(ASSORT_GRAPH, ASSORT_QUERIES, k=5, out=None)
    assert len(rows) == 15
    assert rows[0]["qid"] == "q001"
    hit_rate = sum(r["hit"] for r in rows) / len(rows)
    assert hit_rate >= 0.6
