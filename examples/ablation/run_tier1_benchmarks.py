"""Tier-1 English retrieval benchmark runner.

Runs Synaptic's retrieval pipeline over standard multi-hop corpora
(HotPotQA-dev full, MuSiQue-Ans-dev, and 2WikiMultiHopQA-dev) and
large BEIR-style retrieval corpora (FiQA, TREC-COVID, SciFact). The
multi-hop sets are the datasets HippoRAG2, GraphRAG, and the broader
KG-RAG line use for head-to-head comparisons; the BEIR sets are useful
large-corpus scale checks with query/qrels ground truth. MS MARCO uses
a metadata JSON + corpus JSONL shard so 1M+ passage runs do not require
committing giant benchmark artifacts.

Two modes:

1. Default (no flags): embedder-free baseline (FTS + PPR only). Same
   pipeline as the v0.16.0 published numbers.
2. With ``--embedder-url`` and/or ``--reranker-url``: full pipeline with
   GPU-backed semantic signal. This is the configuration to compare
   against HippoRAG2 / NV-Embed-v2 head-to-head.

Prerequisite
------------
Download the datasets first::

    pip install datasets
    python examples/ablation/download_benchmarks.py

Usage
-----
::

    # Embedder-free baseline (current published numbers)
    python examples/ablation/run_tier1_benchmarks.py
    python examples/ablation/run_tier1_benchmarks.py --only hotpotqa
    python examples/ablation/run_tier1_benchmarks.py --only fiqa --subset 100
    python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 50 \\
        --corpus-limit 1000000 --use-sqlite-graph
    python examples/ablation/run_tier1_benchmarks.py --subset 200

    # Full pipeline with Ollama embedder + TEI cross-encoder
    python examples/ablation/run_tier1_benchmarks.py --subset 500 \\
        --embedder-url http://localhost:11434 \\
        --embedder-model qwen3-embedding:4b \\
        --reranker-url http://localhost:8180

The JSON input files are gitignored (``tests/benchmark/data/*.json``);
the download script is the source of truth. This runner prints a
results table and writes ``examples/ablation/diagnostics/tier1_<ts>.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from synaptic.backends.memory import MemoryBackend
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.embedder import (
    EmbeddingProvider,
    OllamaEmbeddingProvider,
)
from synaptic.extensions.reranker_cross import TEIReranker
from synaptic.graph import SynapticGraph
from synaptic.models import Node, NodeKind

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "tests" / "benchmark" / "data"
OUT_DIR = Path(__file__).parent / "diagnostics"

TOP_K = 10
REUSE_META_VERSION = 1


def _benchmark_node_id(doc_id: str) -> str:
    digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=16).hexdigest()
    return f"bench_{digest}"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sqlite_reuse_meta_path(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.name}.tier1.json")


async def _apply_sqlite_fast_build_pragmas(backend: object) -> None:
    """Relax SQLite durability for rebuildable local benchmark DBs."""
    db_getter = getattr(backend, "_db", None)
    if not callable(db_getter):
        return
    db = db_getter()
    for pragma in (
        "PRAGMA synchronous=OFF",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA cache_size=-262144",
    ):
        await db.execute(pragma)
    await db.commit()


async def _checkpoint_sqlite_backend(backend: object) -> None:
    db_getter = getattr(backend, "_db", None)
    if not callable(db_getter):
        return
    db = db_getter()
    await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    await db.commit()


def _remove_sqlite_artifacts(db_path: Path) -> None:
    for path in (
        db_path,
        db_path.with_name(f"{db_path.name}-shm"),
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}.hnsw"),
        db_path.with_name(f"{db_path.name}.hnsw.meta.json"),
        _sqlite_reuse_meta_path(db_path),
    ):
        if path.exists():
            path.unlink()


@dataclass
class Dataset:
    name: str
    path: Path
    reference: str  # what prior published number to contextualise against


DATASETS = [
    Dataset(
        name="HotPotQA dev (full)",
        path=BENCH / "hotpotqa_full.json",
        reference="HippoRAG2: 56.7 % string accuracy",
    ),
    Dataset(
        name="MuSiQue-Ans dev",
        path=BENCH / "musique_dev.json",
        reference="HippoRAG2: F1 51.9, R@5 74.7 %",
    ),
    Dataset(
        name="2WikiMultihopQA dev",
        path=BENCH / "2wiki_dev.json",
        reference="HippoRAG2: R@5 90.4 %",
    ),
    Dataset(
        name="FiQA test",
        path=BENCH / "fiqa.json",
        reference="BEIR FiQA: ~57k docs / 648 test queries",
    ),
    Dataset(
        name="TREC-COVID test",
        path=BENCH / "trec_covid.json",
        reference="BEIR TREC-COVID: ~171k docs / 50 test queries",
    ),
    Dataset(
        name="SciFact test",
        path=BENCH / "scifact.json",
        reference="BEIR SciFact: ~5k docs / 300 test queries",
    ),
    Dataset(
        name="MS MARCO passage dev",
        path=BENCH / "msmarco_passage.json",
        reference="BEIR/MS MARCO passage: ~8.8M source passages; JSONL shard",
    ),
]


CorpusItem = tuple[str, str, str]


def _dataset_key_map(msmarco_path: Path | None = None) -> dict[str, Dataset]:
    by_key = {
        "hotpotqa": DATASETS[0],
        "musique": DATASETS[1],
        "2wiki": DATASETS[2],
        "fiqa": DATASETS[3],
        "trec_covid": DATASETS[4],
        "scifact": DATASETS[5],
        "msmarco": DATASETS[6],
    }
    if msmarco_path is not None:
        base = by_key["msmarco"]
        by_key["msmarco"] = Dataset(
            name=base.name,
            path=msmarco_path,
            reference=base.reference,
        )
    return by_key


def _selected_gold_doc_ids(
    qrels: dict,
    query_items: list[tuple[str, str]],
) -> set[str]:
    gold_doc_ids: set[str] = set()
    for qid, _qtext in query_items:
        rel = qrels.get(qid, {})
        if isinstance(rel, dict):
            gold_doc_ids.update(str(doc_id) for doc_id in rel)
        else:
            gold_doc_ids.update(str(doc_id) for doc_id in rel)
    return gold_doc_ids


def _load_inline_corpus_items(
    corpus: dict,
    qrels: dict,
    query_items: list[tuple[str, str]],
    corpus_limit: int | None,
) -> list[CorpusItem]:
    items_all = [
        (
            doc_id,
            str(doc.get("title", "") or doc_id),
            str(doc.get("text", "")),
        )
        for doc_id, doc in corpus.items()
    ]
    items_all = [(d, t, x) for d, t, x in items_all if t or x]
    if corpus_limit is not None and 0 < corpus_limit < len(items_all):
        gold_doc_ids = _selected_gold_doc_ids(qrels, query_items)
        gold_items = [item for item in items_all if item[0] in gold_doc_ids]
        filler_items = [item for item in items_all if item[0] not in gold_doc_ids]
        if len(gold_items) >= corpus_limit:
            return gold_items
        return [*gold_items, *filler_items[: corpus_limit - len(gold_items)]]
    return items_all


def _load_jsonl_corpus_items(
    data: dict,
    dataset_path: Path,
    qrels: dict,
    query_items: list[tuple[str, str]],
    corpus_limit: int | None,
) -> list[CorpusItem]:
    raw_corpus_path = data.get("corpus_path")
    if not raw_corpus_path:
        raise ValueError(f"{dataset_path} is missing corpus_path")
    corpus_path = dataset_path.parent / str(raw_corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} missing. Run: "
            "python examples/ablation/download_benchmarks.py --only msmarco_passage"
        )

    downloaded_size = int(data.get("corpus_size") or 0)
    target_limit = corpus_limit if corpus_limit and corpus_limit > 0 else downloaded_size
    gold_doc_ids = _selected_gold_doc_ids(qrels, query_items)
    filler_budget = max(target_limit - len(gold_doc_ids), 0)

    gold_items: list[CorpusItem] = []
    filler_items: list[CorpusItem] = []
    seen_gold: set[str] = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("_id") or row.get("id") or "")
            if not doc_id:
                continue
            title = str(row.get("title") or doc_id)
            text = str(row.get("text") or "")
            if not (title or text):
                continue
            item = (doc_id, title, text)
            if doc_id in gold_doc_ids:
                if doc_id not in seen_gold:
                    gold_items.append(item)
                    seen_gold.add(doc_id)
            elif len(filler_items) < filler_budget:
                filler_items.append(item)

            if len(filler_items) >= filler_budget and seen_gold >= gold_doc_ids:
                break

    if len(gold_items) >= target_limit:
        return gold_items
    return [*gold_items, *filler_items[: target_limit - len(gold_items)]]


def _load_corpus_items(
    data: dict,
    dataset_path: Path,
    qrels: dict,
    query_items: list[tuple[str, str]],
    corpus_limit: int | None,
) -> list[CorpusItem]:
    if data.get("schema") == "beir_jsonl_v1":
        return _load_jsonl_corpus_items(data, dataset_path, qrels, query_items, corpus_limit)
    return _load_inline_corpus_items(data["corpus"], qrels, query_items, corpus_limit)


def _object_signature(obj: object | None) -> str:
    if obj is None:
        return "none"
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def _reuse_signature(
    ds: Dataset,
    data: dict,
    *,
    corpus_limit: int | None,
    embedder: object | None,
    phrase_extractor: object | None,
    entity_linker_cfg: tuple[int, float] | None,
) -> dict[str, object]:
    if data.get("schema") == "beir_jsonl_v1":
        corpus_size = int(data.get("corpus_size") or 0)
    else:
        corpus_size = len(data.get("corpus", {}))
    return {
        "version": REUSE_META_VERSION,
        "dataset": ds.name,
        "dataset_path": _display_path(ds.path),
        "schema": str(data.get("schema") or "inline_json_v1"),
        "source": str(data.get("source") or ""),
        "corpus_path": str(data.get("corpus_path") or ""),
        "manifest_corpus_size": corpus_size,
        "corpus_limit": int(corpus_limit or 0),
        "benchmark_node_id": "blake2b16:bench_",
        "embedder": _object_signature(embedder),
        "phrase_extractor": _object_signature(phrase_extractor),
        "entity_linker_cfg": list(entity_linker_cfg) if entity_linker_cfg else [],
    }


def _reuse_meta_mismatches(
    existing: dict,
    expected: dict[str, object],
) -> list[str]:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual = existing.get(key)
        if actual != expected_value:
            mismatches.append(f"{key}: {actual!r} != {expected_value!r}")
    return mismatches


def _write_reuse_meta(
    db_path: Path,
    expected: dict[str, object],
    *,
    node_count: int,
) -> None:
    meta = {
        **expected,
        "node_count": node_count,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    _sqlite_reuse_meta_path(db_path).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_reuse_meta(db_path: Path) -> dict:
    path = _sqlite_reuse_meta_path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; build the benchmark DB once without --reuse-sqlite-db"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, did in enumerate(retrieved):
        if did in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def _doc_ids_from_nodes(nodes: list[object]) -> list[str]:
    retrieved: list[str] = []
    for node in nodes:
        did = (getattr(node, "properties", {}) or {}).get("doc_id", "")
        if did and did not in retrieved:
            retrieved.append(str(did))
    return retrieved


async def _raw_fts_doc_ids(backend: object, query: str, *, limit: int) -> list[str]:
    try:
        nodes = await backend.search_fts(query, limit=limit, include_embedding=False)  # type: ignore[attr-defined]
    except TypeError:
        nodes = await backend.search_fts(query, limit=limit)  # type: ignore[attr-defined]
    return _doc_ids_from_nodes(nodes)


@dataclass
class Report:
    name: str
    n_docs: int
    n_queries: int
    mrr: float
    recall_at_5: float
    recall_at_10: float
    hit_at_10: int
    build_sec: float
    search_sec: float
    reference: str
    raw_fts_pool_limit: int = 0
    raw_fts_mrr: float = 0.0
    raw_fts_recall_at_5: float = 0.0
    raw_fts_recall_at_10: float = 0.0
    raw_fts_hit_at_10: int = 0
    raw_fts_any_at_pool: int = 0
    raw_fts_sec: float = 0.0


async def run_one(
    ds: Dataset,
    subset: int | None,
    *,
    embedder: EmbeddingProvider | None = None,
    reranker: object | None = None,
    decomposer: object | None = None,
    phrase_extractor: object | None = None,
    entity_linker_cfg: tuple[int, float] | None = None,
    use_sqlite_graph: bool = False,
    embed_batch: int = 256,
    ingest_batch: int = 20000,
    corpus_limit: int | None = None,
    progress_every: int = 100000,
    fts_seed_limit: int | None = None,
    diagnose_raw_fts_limit: int | None = None,
    sqlite_db_path: Path | None = None,
    reuse_sqlite_db: bool = False,
    overwrite_sqlite_db: bool = False,
    sqlite_fast_build: bool = False,
) -> Report:
    if sqlite_db_path is not None and not use_sqlite_graph:
        raise ValueError("--sqlite-db-path requires --use-sqlite-graph")
    if reuse_sqlite_db and sqlite_db_path is None:
        raise ValueError("--reuse-sqlite-db requires --sqlite-db-path")
    if overwrite_sqlite_db and sqlite_db_path is None:
        raise ValueError("--overwrite-sqlite-db requires --sqlite-db-path")
    if reuse_sqlite_db and overwrite_sqlite_db:
        raise ValueError("--reuse-sqlite-db and --overwrite-sqlite-db are mutually exclusive")
    if not ds.path.exists():
        raise FileNotFoundError(
            f"{ds.path} missing. Run:  python examples/ablation/download_benchmarks.py"
        )
    with open(ds.path, encoding="utf-8") as f:
        data = json.load(f)

    queries_all = data["queries"]
    qrels = data["qrels"]

    query_items = list(queries_all.items())
    if subset is not None and subset < len(query_items):
        query_items = query_items[:subset]

    reuse_signature = _reuse_signature(
        ds,
        data,
        corpus_limit=corpus_limit,
        embedder=embedder,
        phrase_extractor=phrase_extractor,
        entity_linker_cfg=entity_linker_cfg,
    )

    # Build the graph once for the whole dataset.
    t_build = time.perf_counter()
    if use_sqlite_graph:
        if sqlite_db_path is not None:
            if overwrite_sqlite_db:
                _remove_sqlite_artifacts(sqlite_db_path)
            elif sqlite_db_path.exists() and not reuse_sqlite_db:
                raise FileExistsError(
                    f"{sqlite_db_path} exists; pass --reuse-sqlite-db to reuse it "
                    "or --overwrite-sqlite-db to rebuild it"
                )
            sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
            backend = SqliteGraphBackend(str(sqlite_db_path))
        else:
            tmp_db = tempfile.NamedTemporaryFile(
                prefix=f"tier1_{ds.name.replace(' ', '_')}_",
                suffix=".db",
                delete=False,
            )
            tmp_db.close()
            backend = SqliteGraphBackend(tmp_db.name)
    else:
        backend = MemoryBackend()
    await backend.connect()
    if use_sqlite_graph and sqlite_fast_build and not reuse_sqlite_db:
        await _apply_sqlite_fast_build_pragmas(backend)
    graph = SynapticGraph(
        backend,
        embedder=embedder,
        reranker=reranker,
        query_decomposer=decomposer,
        phrase_extractor=phrase_extractor,
    )

    reused_sqlite = False
    items: list[CorpusItem] = []
    if reuse_sqlite_db:
        assert sqlite_db_path is not None
        existing_meta = _read_reuse_meta(sqlite_db_path)
        mismatches = _reuse_meta_mismatches(existing_meta, reuse_signature)
        if mismatches:
            joined = "; ".join(mismatches)
            raise ValueError(f"{sqlite_db_path} reuse metadata mismatch: {joined}")
        n_docs = int(existing_meta.get("node_count") or await backend.count_nodes())
        print(
            f"  reuse sqlite db: {_display_path(sqlite_db_path)} ({n_docs:,} docs)",
            flush=True,
        )
        reused_sqlite = True
    else:
        # Pre-compute embeddings in large batches (GPU-friendly).
        # ``graph.add()`` accepts an ``embedding`` arg; if we pass it we
        # avoid the per-node single embed call that bottlenecks at batch=1.
        items = _load_corpus_items(data, ds.path, qrels, query_items, corpus_limit)
        n_docs = len(items)

    total_items = len(items)

    next_progress = progress_every

    def maybe_print_progress(done: int, start: float) -> None:
        nonlocal next_progress
        if progress_every <= 0:
            return
        if done < total_items and done < next_progress:
            return
        elapsed = time.perf_counter() - start
        print(
            f"  ingest: {done:,}/{total_items:,} docs ({elapsed:.1f}s)",
            flush=True,
        )
        while next_progress <= done:
            next_progress += progress_every

    embeddings: list[list[float] | None] = [None] * len(items)
    if not reused_sqlite and embedder is not None:
        embed_inputs = [f"{title}\n{(text or '')[:1500]}" for _doc_id, title, text in items]
        for i in range(0, len(embed_inputs), embed_batch):
            chunk = embed_inputs[i : i + embed_batch]
            vecs = await embedder.embed_batch(chunk)
            for j, v in enumerate(vecs):
                embeddings[i + j] = v if v else None

    save_nodes_batch = getattr(backend, "save_nodes_batch", None)
    t_ingest = time.perf_counter()
    if reused_sqlite:
        pass
    elif phrase_extractor is None and callable(save_nodes_batch):
        for i in range(0, len(items), ingest_batch):
            done = min(i + ingest_batch, total_items)
            batch = [
                Node(
                    id=_benchmark_node_id(doc_id),
                    kind=NodeKind.CONCEPT,
                    title=title,
                    content=text,
                    properties={"doc_id": doc_id},
                    embedding=emb or [],
                )
                for (doc_id, title, text), emb in zip(
                    items[i : i + ingest_batch],
                    embeddings[i : i + ingest_batch],
                )
            ]
            await save_nodes_batch(batch)
            maybe_print_progress(done, t_ingest)
    else:
        for idx, ((doc_id, title, text), emb) in enumerate(zip(items, embeddings), start=1):
            await graph.add(
                title=title,
                content=text,
                properties={"doc_id": doc_id},
                embedding=emb,
                record_memory_event=False,
            )
            maybe_print_progress(idx, t_ingest)

    # Post-hoc DF-filtered entity linking (opt-in via --entity-linker).
    # Runs AFTER ingest because the DF filter needs global corpus
    # statistics. Typically 5-20× cheaper than inline phrase-extractor
    # because it uses batch writes and skips per-node re-hash.
    if entity_linker_cfg is not None and not reused_sqlite:
        from synaptic.extensions.domain_profile import DomainProfile
        from synaptic.extensions.entity_linker import EntityLinker
        from synaptic.extensions.phrase_extractor import PhraseExtractor

        min_df, max_df_ratio = entity_linker_cfg
        profile = DomainProfile(
            name=f"{ds.name}-tier1",
            locale="multi",
            min_df=min_df,
            max_df_ratio=max_df_ratio,
        )
        linker = EntityLinker(
            extractor=PhraseExtractor(),
            profile=profile,
            max_links_per_source=15,
        )
        from synaptic.models import NodeKind as _NK

        stats = await linker.link(backend, source_kind=_NK.CONCEPT, embedder=embedder)
        print(
            f"  entity-linker: {stats.phrase_nodes_created} hubs / "
            f"{stats.mentions_edges_created} MENTIONS "
            f"(min_df={min_df}, max_df={max_df_ratio:.3f}, "
            f"{stats.elapsed_seconds:.1f}s)"
        )
        if stats.top_phrases_by_df:
            top5 = ", ".join(f"{p}({d})" for p, d in stats.top_phrases_by_df[:5])
            print(f"    top-DF: {top5}")

    if sqlite_db_path is not None and not reused_sqlite:
        _write_reuse_meta(sqlite_db_path, reuse_signature, node_count=n_docs)
    if sqlite_fast_build and not reused_sqlite:
        await _checkpoint_sqlite_backend(backend)
    build_sec = time.perf_counter() - t_build

    mrr_total = 0.0
    r5_total = 0.0
    r10_total = 0.0
    hit10 = 0
    raw_fts_mrr_total = 0.0
    raw_fts_r5_total = 0.0
    raw_fts_r10_total = 0.0
    raw_fts_hit10 = 0
    raw_fts_any_pool = 0
    raw_fts_sec = 0.0

    search_sec = 0.0
    for qid, qtext in query_items:
        rel = qrels.get(qid, {})
        relevant = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
        if not relevant:
            continue
        if diagnose_raw_fts_limit:
            t_raw = time.perf_counter()
            raw_retrieved = await _raw_fts_doc_ids(
                backend,
                str(qtext),
                limit=diagnose_raw_fts_limit,
            )
            raw_fts_sec += time.perf_counter() - t_raw
            raw_rr = _reciprocal_rank(raw_retrieved[:TOP_K], relevant)
            raw_fts_mrr_total += raw_rr
            raw_fts_r5_total += _recall_at_k(raw_retrieved, relevant, 5)
            raw_fts_r10_total += _recall_at_k(raw_retrieved, relevant, TOP_K)
            if raw_rr > 0:
                raw_fts_hit10 += 1
            if any(did in relevant for did in raw_retrieved):
                raw_fts_any_pool += 1

        t_one_search = time.perf_counter()
        result = await graph.search(
            str(qtext),
            limit=TOP_K * 2,
            fts_seed_limit=fts_seed_limit,
        )
        retrieved = _doc_ids_from_nodes([hit.node for hit in result.nodes])
        rr = _reciprocal_rank(retrieved[:TOP_K], relevant)
        mrr_total += rr
        r5_total += _recall_at_k(retrieved, relevant, 5)
        r10_total += _recall_at_k(retrieved, relevant, TOP_K)
        if rr > 0:
            hit10 += 1
        search_sec += time.perf_counter() - t_one_search

    n = max(len(query_items), 1)
    report = Report(
        name=ds.name,
        n_docs=n_docs,
        n_queries=len(query_items),
        mrr=mrr_total / n,
        recall_at_5=r5_total / n,
        recall_at_10=r10_total / n,
        hit_at_10=hit10,
        build_sec=build_sec,
        search_sec=search_sec,
        reference=ds.reference,
        raw_fts_pool_limit=int(diagnose_raw_fts_limit or 0),
        raw_fts_mrr=raw_fts_mrr_total / n if diagnose_raw_fts_limit else 0.0,
        raw_fts_recall_at_5=raw_fts_r5_total / n if diagnose_raw_fts_limit else 0.0,
        raw_fts_recall_at_10=raw_fts_r10_total / n if diagnose_raw_fts_limit else 0.0,
        raw_fts_hit_at_10=raw_fts_hit10,
        raw_fts_any_at_pool=raw_fts_any_pool,
        raw_fts_sec=raw_fts_sec,
    )
    close = getattr(backend, "close", None)
    if callable(close):
        await close()
    return report


def _emit_markdown(
    reports: list[Report],
    subset: int | None,
    *,
    embedder_label: str,
    reranker_label: str,
    decomposer_label: str = "none",
    phrase_extractor_label: str = "none",
    entity_linker_label: str = "none",
    corpus_limit: int | None = None,
    ingest_batch: int = 20000,
    progress_every: int = 100000,
    fts_seed_limit: int | None = None,
    diagnose_raw_fts_limit: int | None = None,
    sqlite_db_path: Path | None = None,
    reuse_sqlite_db: bool = False,
    sqlite_fast_build: bool = False,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"tier1_{stamp}.md"
    lines = [
        "# Tier-1 English retrieval benchmark — Synaptic",
        "",
        f"- Run at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Subset: {subset if subset else 'full'}",
        f"- Corpus limit: {corpus_limit if corpus_limit else 'full'}",
        f"- FTS seed limit: {fts_seed_limit if fts_seed_limit else 'default'}",
        f"- Raw FTS diagnostic limit: {diagnose_raw_fts_limit if diagnose_raw_fts_limit else 'disabled'}",
        f"- Ingest batch: {ingest_batch}",
        f"- Progress every: {progress_every if progress_every > 0 else 'disabled'}",
        f"- SQLite DB path: {_display_path(sqlite_db_path) if sqlite_db_path else 'temporary'}",
        f"- SQLite DB reuse: {'yes' if reuse_sqlite_db else 'no'}",
        f"- SQLite fast build: {'yes' if sqlite_fast_build else 'no'}",
        "- SQLite FTS AND-first threshold: "
        f"{os.environ.get('SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD', '').strip() or '0'}",
        "- SQLite FTS lexical rerank pool: "
        f"{os.environ.get('SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL', '').strip() or '0'}",
        "- Cross-rerank top N: "
        f"{os.environ.get('SYNAPTIC_CROSS_RERANK_TOP_N', '').strip() or '20'}",
        f"- Embedder: {embedder_label}",
        f"- Reranker: {reranker_label}",
        f"- Decomposer: {decomposer_label}",
        f"- Phrase hub (inline): {phrase_extractor_label}",
        f"- Entity linker (post-hoc): {entity_linker_label}",
        "- Engine: `graph.search()` default (EvidenceSearch)",
        "",
        "| Dataset | Docs | Queries | MRR@10 | R@5 | R@10 | Hit@10 | Build | Search |",
        "|---------|-----:|--------:|-------:|----:|-----:|-------:|------:|-------:|",
    ]
    for r in reports:
        lines.append(
            f"| {r.name} | {r.n_docs} | {r.n_queries} | "
            f"{r.mrr:.3f} | {r.recall_at_5:.3f} | {r.recall_at_10:.3f} | "
            f"{r.hit_at_10}/{r.n_queries} | {r.build_sec:.1f}s | {r.search_sec:.1f}s |"
        )
    raw_reports = [r for r in reports if r.raw_fts_pool_limit > 0]
    if raw_reports:
        lines.extend(
            [
                "",
                "## Raw FTS Pool Diagnostic",
                "",
                "| Dataset | Pool | MRR@10 | R@5 | R@10 | Hit@10 | Any@Pool | Raw FTS Time |",
                "|---------|-----:|-------:|----:|-----:|-------:|---------:|-------------:|",
            ]
        )
        for r in raw_reports:
            lines.append(
                f"| {r.name} | {r.raw_fts_pool_limit} | {r.raw_fts_mrr:.3f} | "
                f"{r.raw_fts_recall_at_5:.3f} | {r.raw_fts_recall_at_10:.3f} | "
                f"{r.raw_fts_hit_at_10}/{r.n_queries} | "
                f"{r.raw_fts_any_at_pool}/{r.n_queries} | {r.raw_fts_sec:.1f}s |"
            )
    lines.append("")
    lines.append("## Context")
    lines.append("")
    for r in reports:
        lines.append(f"- **{r.name}** — published baseline: {r.reference}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _threshold_violations(
    reports: list[Report],
    *,
    max_build_sec: float | None = None,
    max_search_sec: float | None = None,
    min_hit_rate_at_10: float | None = None,
    min_mrr: float | None = None,
) -> list[str]:
    violations: list[str] = []
    for report in reports:
        hit_rate = report.hit_at_10 / max(report.n_queries, 1)
        if max_build_sec is not None and report.build_sec > max_build_sec:
            violations.append(
                f"{report.name}: build {report.build_sec:.1f}s > {max_build_sec:.1f}s"
            )
        if max_search_sec is not None and report.search_sec > max_search_sec:
            violations.append(
                f"{report.name}: search {report.search_sec:.1f}s > {max_search_sec:.1f}s"
            )
        if min_hit_rate_at_10 is not None and hit_rate < min_hit_rate_at_10:
            violations.append(
                f"{report.name}: hit@10 rate {hit_rate:.3f} < {min_hit_rate_at_10:.3f}"
            )
        if min_mrr is not None and report.mrr < min_mrr:
            violations.append(f"{report.name}: MRR@10 {report.mrr:.3f} < {min_mrr:.3f}")
    return violations


async def amain(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        default=",".join(["hotpotqa", "musique", "2wiki"]),
        help=(
            "comma-separated dataset keys "
            "(hotpotqa | musique | 2wiki | fiqa | trec_covid | scifact | msmarco)"
        ),
    )
    p.add_argument("--subset", type=int, default=None)
    p.add_argument(
        "--embedder-url",
        default=None,
        help="Ollama base URL (e.g. http://localhost:11434). If unset, "
        "runs FTS-only (embedder-free baseline).",
    )
    p.add_argument(
        "--embedder-model",
        default="qwen3-embedding:4b",
        help="Ollama embedding model name (default: qwen3-embedding:4b).",
    )
    p.add_argument(
        "--embedder-backend",
        choices=("ollama", "openai"),
        default="ollama",
        help="Embedder API style. 'openai' for any /v1/embeddings server "
        "(vLLM, TEI-OpenAI, OpenAI), 'ollama' (default) for /api/embed.",
    )
    p.add_argument(
        "--reranker-url",
        default=None,
        help="TEI reranker base URL (e.g. http://localhost:8180). "
        "If unset, no cross-encoder reranking.",
    )
    p.add_argument(
        "--local-bge",
        action="store_true",
        help="Load BAAI/bge-m3 + bge-reranker-v2-m3 directly via "
        "transformers (no external endpoint). Requires torch + GPU.",
    )
    p.add_argument(
        "--local-bge-device",
        default="cuda:0",
        help="GPU device for --local-bge (default: cuda:0).",
    )
    p.add_argument(
        "--use-sqlite-graph",
        action="store_true",
        help="Use SqliteGraphBackend (usearch HNSW) instead of MemoryBackend. "
        "Required for fast vector search at corpus sizes > 5k.",
    )
    p.add_argument(
        "--sqlite-db-path",
        type=Path,
        default=None,
        help=(
            "Persistent SQLite DB path for SqliteGraphBackend. Use with "
            "--use-sqlite-graph to keep a built large-corpus index."
        ),
    )
    p.add_argument(
        "--msmarco-path",
        type=Path,
        default=None,
        help=(
            "Override the MS MARCO manifest path, e.g. "
            "tests/benchmark/data/msmarco_passage_5m.json for side-by-side large tiers."
        ),
    )
    p.add_argument(
        "--reuse-sqlite-db",
        action="store_true",
        help=(
            "Reuse an existing --sqlite-db-path and skip corpus ingest after "
            "validating its tier1 sidecar metadata."
        ),
    )
    p.add_argument(
        "--overwrite-sqlite-db",
        action="store_true",
        help=(
            "Delete an existing --sqlite-db-path and its tier1 sidecar before "
            "building. Mutually exclusive with --reuse-sqlite-db."
        ),
    )
    p.add_argument(
        "--sqlite-fast-build",
        action="store_true",
        help=(
            "Use relaxed SQLite durability PRAGMAs for rebuildable benchmark DBs. "
            "Only affects non-reuse SqliteGraphBackend builds."
        ),
    )
    p.add_argument(
        "--embed-batch",
        type=int,
        default=64,
        help="Pre-compute corpus embeddings in batches of this size "
        "(default: 64 - safe under 6 GB free VRAM). Bump to 128-256 "
        "if more headroom.",
    )
    p.add_argument(
        "--ingest-batch",
        type=int,
        default=20000,
        help="Batch size for benchmark corpus node writes (default: 20000).",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help=(
            "Print ingest progress every N docs during build (default: 100000; set 0 to disable)."
        ),
    )
    p.add_argument(
        "--corpus-limit",
        type=int,
        default=None,
        help=(
            "Index at most this many docs for staged scale smoke. The selected "
            "queries' gold docs are kept first, then distractors are filled in."
        ),
    )
    p.add_argument(
        "--fts-seed-limit",
        type=int,
        default=None,
        help=(
            "Override graph.search FTS seed-pool size. Useful with cross-encoder "
            "rerankers that can promote relevant docs from a wider lexical pool."
        ),
    )
    p.add_argument(
        "--diagnose-raw-fts-limit",
        type=int,
        default=None,
        help=(
            "Also measure backend.search_fts(query, limit=N) as a raw candidate-pool "
            "diagnostic. Reports official top-10 metrics plus Any@N without changing "
            "graph.search() results."
        ),
    )
    p.add_argument(
        "--max-build-sec",
        type=float,
        default=None,
        help="Fail if any dataset build takes longer than this many seconds.",
    )
    p.add_argument(
        "--max-search-sec",
        type=float,
        default=None,
        help="Fail if any dataset search phase takes longer than this many seconds.",
    )
    p.add_argument(
        "--min-hit-rate-at-10",
        type=float,
        default=None,
        help="Fail if any dataset Hit@10 / queries is below this value.",
    )
    p.add_argument(
        "--min-mrr",
        type=float,
        default=None,
        help="Fail if any dataset MRR@10 is below this value.",
    )
    p.add_argument(
        "--llm-decomposer-url",
        default=None,
        help="OpenAI-compatible endpoint for LLMChainDecomposer "
        "(e.g. http://localhost:8012/v1 for a local vLLM). "
        "If unset, no decomposition.",
    )
    p.add_argument(
        "--llm-decomposer-model",
        default="Qwen3.5-27b",
        help="Model name served at --llm-decomposer-url.",
    )
    p.add_argument(
        "--rule-decomposer",
        action="store_true",
        help="Use the rule-based QueryDecomposer (compound KO/EN splitter). "
        "Overridden by --llm-decomposer-url if both are set.",
    )
    p.add_argument(
        "--phrase-extractor",
        action="store_true",
        help="Build the phrase hub INLINE at ingestion (no DF filter). "
        "Cheap path but super-hubs can poison PPR on large English corpora.",
    )
    p.add_argument(
        "--entity-linker",
        action="store_true",
        help="Run EntityLinker AFTER ingest to build a DF-filtered phrase hub. "
        "Preferred over --phrase-extractor for large English corpora: batch "
        "writes + DF filter kill super-hubs.",
    )
    p.add_argument(
        "--entity-min-df",
        type=int,
        default=2,
        help="EntityLinker: minimum distinct-source DF for a phrase to survive "
        "(default 2 — keeps 2-hop bridge candidates, drops hapax).",
    )
    p.add_argument(
        "--phrase-seed-k",
        type=int,
        default=0,
        help="v0.27 query→phrase dense seed top-K (default: 0 — "
        "disabled). MuSiQue 100q ablation showed 0 net contribution "
        "with this set to 5 (top-DF generic phrases dominate cosine "
        "match; reranker rejects). Keep available as an opt-in for "
        "future paraphrase-aware phrase selection.",
    )
    p.add_argument(
        "--entity-max-df-ratio",
        type=float,
        default=0.02,
        help="EntityLinker: max df / corpus_size (default 0.02 = 2%%). "
        "Tighter than DomainProfile default 0.3 so generic adjectives "
        "('American', 'French') don't form super-hubs.",
    )
    args = p.parse_args(argv)
    if args.sqlite_db_path is not None and not args.use_sqlite_graph:
        raise SystemExit("--sqlite-db-path requires --use-sqlite-graph")
    if args.reuse_sqlite_db and args.sqlite_db_path is None:
        raise SystemExit("--reuse-sqlite-db requires --sqlite-db-path")
    if args.overwrite_sqlite_db and args.sqlite_db_path is None:
        raise SystemExit("--overwrite-sqlite-db requires --sqlite-db-path")
    if args.reuse_sqlite_db and args.overwrite_sqlite_db:
        raise SystemExit("--reuse-sqlite-db and --overwrite-sqlite-db are mutually exclusive")
    if args.sqlite_fast_build and not args.use_sqlite_graph:
        raise SystemExit("--sqlite-fast-build requires --use-sqlite-graph")
    if args.fts_seed_limit is not None and args.fts_seed_limit <= 0:
        raise SystemExit("--fts-seed-limit must be positive")
    if args.diagnose_raw_fts_limit is not None and args.diagnose_raw_fts_limit <= 0:
        raise SystemExit("--diagnose-raw-fts-limit must be positive")

    embedder: EmbeddingProvider | None = None
    embedder_label = "none (FTS-only baseline)"
    reranker: object | None = None
    reranker_label = "none"
    decomposer: object | None = None
    decomposer_label = "none"

    if args.llm_decomposer_url:
        from synaptic.extensions.llm_provider import OpenAILLMProvider
        from synaptic.extensions.query_decomposer_llm import LLMChainDecomposer

        llm = OpenAILLMProvider(
            api_base=args.llm_decomposer_url,
            model=args.llm_decomposer_model,
            timeout=60,
        )
        decomposer = LLMChainDecomposer(llm=llm)
        decomposer_label = (
            f"LLMChainDecomposer ({args.llm_decomposer_model} @ {args.llm_decomposer_url})"
        )
    elif args.rule_decomposer:
        from synaptic.extensions.query_decomposer import QueryDecomposer

        decomposer = QueryDecomposer()
        decomposer_label = "rule-based QueryDecomposer"

    phrase_extractor_obj: object | None = None
    phrase_extractor_label = "none"
    if args.phrase_extractor:
        from synaptic.extensions.phrase_extractor import PhraseExtractor

        phrase_extractor_obj = PhraseExtractor()
        phrase_extractor_label = "EnglishPhraseExtractor (inline)"

    entity_linker_cfg: tuple[int, float] | None = None
    entity_linker_label = "none"
    if args.entity_linker:
        entity_linker_cfg = (args.entity_min_df, args.entity_max_df_ratio)
        entity_linker_label = (
            f"EntityLinker (min_df={args.entity_min_df}, "
            f"max_df_ratio={args.entity_max_df_ratio:.3f})"
        )

    if args.local_bge:
        from local_bge import LocalBgeM3Embedder, LocalBgeRerankerV2

        print(f"Loading bge-m3 + bge-reranker-v2-m3 on {args.local_bge_device} ...")
        embedder = LocalBgeM3Embedder(device=args.local_bge_device)
        reranker = LocalBgeRerankerV2(device=args.local_bge_device)
        embedder_label = f"local BAAI/bge-m3 ({args.local_bge_device})"
        reranker_label = f"local BAAI/bge-reranker-v2-m3 ({args.local_bge_device})"
    else:
        if args.embedder_url:
            if args.embedder_backend == "openai":
                from synaptic.extensions.embedder import OpenAIEmbeddingProvider

                embedder = OpenAIEmbeddingProvider(
                    api_base=args.embedder_url,
                    model=args.embedder_model,
                )
                embedder_label = f"OpenAI-compat {args.embedder_model} @ {args.embedder_url}"
            else:
                embedder = OllamaEmbeddingProvider(
                    base_url=args.embedder_url,
                    model=args.embedder_model,
                )
                embedder_label = f"Ollama {args.embedder_model} @ {args.embedder_url}"
        if args.reranker_url:
            reranker = TEIReranker(base_url=args.reranker_url)
            reranker_label = f"TEI cross-encoder @ {args.reranker_url}"

    by_key = _dataset_key_map(args.msmarco_path)
    selected = []
    for raw_key in args.only.split(","):
        key = raw_key.strip()
        if not key:
            continue
        if key not in by_key:
            raise SystemExit(f"Unknown dataset key: {key}; available: {', '.join(by_key)}")
        selected.append(by_key[key])

    mode = "full pipeline" if embedder or reranker else "embedder-free"
    backend_label = "SqliteGraphBackend (HNSW)" if args.use_sqlite_graph else "MemoryBackend"
    print(f"Tier-1 English retrieval benchmarks — Synaptic {mode}")
    print(f"  backend:  {backend_label}")
    print(f"  embedder: {embedder_label}")
    print(f"  reranker: {reranker_label}")
    print(f"  decomposer: {decomposer_label}")
    print(f"  phrase hub: {phrase_extractor_label}")
    print(f"  entity linker: {entity_linker_label}")
    print(f"  corpus limit: {args.corpus_limit if args.corpus_limit else 'full'}")
    print(f"  FTS seed limit: {args.fts_seed_limit if args.fts_seed_limit else 'default'}")
    print(
        "  raw FTS diagnostic limit: "
        f"{args.diagnose_raw_fts_limit if args.diagnose_raw_fts_limit else 'disabled'}"
    )
    print(f"  ingest batch: {args.ingest_batch}")
    print(f"  progress every: {args.progress_every if args.progress_every > 0 else 'disabled'}")
    print(
        f"  sqlite db: {_display_path(args.sqlite_db_path) if args.sqlite_db_path else 'temporary'}"
    )
    if args.sqlite_db_path:
        print(
            "  sqlite reuse: "
            f"{'yes' if args.reuse_sqlite_db else 'no'}"
            f"{' (overwrite)' if args.overwrite_sqlite_db else ''}"
        )
    if args.use_sqlite_graph:
        print(f"  sqlite fast build: {'yes' if args.sqlite_fast_build else 'no'}")
        print(
            "  sqlite FTS AND-first threshold: "
            f"{os.environ.get('SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD', '').strip() or '0'}"
        )
        print(
            "  sqlite FTS lexical rerank pool: "
            f"{os.environ.get('SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL', '').strip() or '0'}"
        )
    print(
        f"  cross-rerank top N: {os.environ.get('SYNAPTIC_CROSS_RERANK_TOP_N', '').strip() or '20'}"
    )
    if embedder is not None:
        print(f"  embed batch: {args.embed_batch}")
    print()
    header = f"{'Dataset':<24} {'Docs':>7} {'Qs':>6} {'MRR@10':>8} {'R@5':>7} {'R@10':>7} {'Hit':>10} {'Build':>7} {'Search':>8}"
    print(header)
    print("-" * len(header))

    reports: list[Report] = []
    for ds in selected:
        try:
            r = await run_one(
                ds,
                args.subset,
                embedder=embedder,
                reranker=reranker,
                decomposer=decomposer,
                phrase_extractor=phrase_extractor_obj,
                entity_linker_cfg=entity_linker_cfg,
                use_sqlite_graph=args.use_sqlite_graph,
                embed_batch=args.embed_batch,
                ingest_batch=args.ingest_batch,
                corpus_limit=args.corpus_limit,
                progress_every=args.progress_every,
                fts_seed_limit=args.fts_seed_limit,
                diagnose_raw_fts_limit=args.diagnose_raw_fts_limit,
                sqlite_db_path=args.sqlite_db_path,
                reuse_sqlite_db=args.reuse_sqlite_db,
                overwrite_sqlite_db=args.overwrite_sqlite_db,
                sqlite_fast_build=args.sqlite_fast_build,
            )
        except FileNotFoundError as e:
            print(f"{ds.name:<24}  SKIP — {e}")
            continue
        reports.append(r)
        print(
            f"{r.name:<24} {r.n_docs:>7} {r.n_queries:>6} "
            f"{r.mrr:>8.3f} {r.recall_at_5:>7.3f} {r.recall_at_10:>7.3f} "
            f"{r.hit_at_10:>5}/{r.n_queries:<4} {r.build_sec:>6.1f}s {r.search_sec:>7.1f}s"
        )
        if r.raw_fts_pool_limit > 0:
            print(
                f"  raw FTS@{r.raw_fts_pool_limit}: "
                f"MRR@10={r.raw_fts_mrr:.3f} "
                f"R@5={r.raw_fts_recall_at_5:.3f} "
                f"R@10={r.raw_fts_recall_at_10:.3f} "
                f"Hit@10={r.raw_fts_hit_at_10}/{r.n_queries} "
                f"Any@{r.raw_fts_pool_limit}={r.raw_fts_any_at_pool}/{r.n_queries} "
                f"({r.raw_fts_sec:.1f}s)"
            )

    if reports:
        out = _emit_markdown(
            reports,
            args.subset,
            embedder_label=embedder_label,
            reranker_label=reranker_label,
            decomposer_label=decomposer_label,
            phrase_extractor_label=phrase_extractor_label,
            entity_linker_label=entity_linker_label,
            corpus_limit=args.corpus_limit,
            ingest_batch=args.ingest_batch,
            progress_every=args.progress_every,
            fts_seed_limit=args.fts_seed_limit,
            diagnose_raw_fts_limit=args.diagnose_raw_fts_limit,
            sqlite_db_path=args.sqlite_db_path,
            reuse_sqlite_db=args.reuse_sqlite_db,
            sqlite_fast_build=args.sqlite_fast_build,
        )
        print()
        print(f"Markdown report → {out.relative_to(REPO_ROOT)}")
    violations = _threshold_violations(
        reports,
        max_build_sec=args.max_build_sec,
        max_search_sec=args.max_search_sec,
        min_hit_rate_at_10=args.min_hit_rate_at_10,
        min_mrr=args.min_mrr,
    )
    if violations:
        print()
        print("Threshold violations:")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    return 0


def main() -> None:
    import sys as _sys

    _sys.exit(asyncio.run(amain(_sys.argv[1:])))


if __name__ == "__main__":
    main()
