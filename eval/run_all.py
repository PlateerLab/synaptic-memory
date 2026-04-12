"""Unified QA benchmark runner — run after every development cycle.

Runs all evaluation datasets (custom + public) through the synaptic
pipeline and produces a regression-aware comparison table.

Usage::

    # Full run (all datasets)
    uv run python eval/run_all.py

    # Quick run (custom only, skip large public datasets)
    uv run python eval/run_all.py --quick

    # Compare against last baseline
    uv run python eval/run_all.py --compare eval/results/baseline.json

Output::

    ┌──────────────────┬────────┬───────┬───────┬───────┬──────────┐
    │ Dataset          │ Corpus │  MRR  │ P@10  │ R@10  │ Status   │
    ├──────────────────┼────────┼───────┼───────┼───────┼──────────┤
    │ KRRA Easy        │ 19,720 │ 0.967 │ 0.496 │ 0.914 │ ✅       │
    │ KRRA Hard        │ 19,720 │ 0.507 │ 0.157 │ 0.633 │ ✅       │
    │ assort Easy      │ 13,909 │ 0.880 │ 0.100 │ 0.933 │ ✅       │
    │ assort Hard      │ 13,909 │ 0.127 │ 0.047 │ 0.267 │ ✅       │
    │ HotPotQA-200     │  1,990 │ 0.742 │       │       │ NEW      │
    │ Ko-StrategyQA    │  9,251 │ 0.317 │       │       │ NEW      │
    │ ...              │        │       │       │       │          │
    └──────────────────┴────────┴───────┴───────┴───────┴──────────┘
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import NodeKind
from synaptic.search import HybridSearch
from tests.benchmark.metrics import BenchmarkResult

# --- Dataset registry ---

BENCHMARK_DIR = REPO_ROOT / "tests" / "benchmark" / "data"
EVAL_DIR = REPO_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"


@dataclass
class DatasetConfig:
    name: str
    path: Path
    query_path: Path | None = None  # None = queries embedded in dataset
    corpus_key: str = "corpus"
    query_key: str = "queries"
    doc_id_key: str = "doc_id"
    text_key: str = "text"
    title_key: str = "title"
    k: int = 10
    is_custom: bool = False  # custom = KRRA/assort, not public
    quick: bool = True  # include in --quick mode


# Custom datasets (KRRA, assort)
CUSTOM_DATASETS = [
    DatasetConfig(
        name="KRRA Easy",
        path=EVAL_DIR / "data" / "krra_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "krra.json",
        is_custom=True, quick=True,
    ),
    DatasetConfig(
        name="KRRA Hard",
        path=EVAL_DIR / "data" / "krra_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "krra_hard.json",
        is_custom=True, quick=True,
    ),
    DatasetConfig(
        name="assort Easy",
        path=EVAL_DIR / "data" / "assort_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "assort.json",
        is_custom=True, quick=True,
    ),
    DatasetConfig(
        name="assort Hard",
        path=EVAL_DIR / "data" / "assort_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "assort_hard.json",
        is_custom=True, quick=True,
    ),
]

# Public datasets (in-memory, from benchmark JSON)
PUBLIC_DATASETS = [
    DatasetConfig(name="HotPotQA-24", path=BENCHMARK_DIR / "hotpotqa_24.json", quick=True),
    DatasetConfig(name="HotPotQA-200", path=BENCHMARK_DIR / "hotpotqa.json", quick=False),
    DatasetConfig(name="Allganize RAG-ko", path=BENCHMARK_DIR / "allganize_rag_ko.json", quick=True),
    DatasetConfig(name="Allganize RAG-Eval", path=BENCHMARK_DIR / "allganize_rag_eval.json", quick=True),
    DatasetConfig(name="PublicHealthQA", path=BENCHMARK_DIR / "publichealthqa_ko.json", quick=True),
    DatasetConfig(name="AutoRAG", path=BENCHMARK_DIR / "autorag_retrieval.json", quick=True),
    DatasetConfig(name="KLUE-MRC", path=BENCHMARK_DIR / "klue_mrc.json", quick=False),
    DatasetConfig(name="Ko-StrategyQA", path=BENCHMARK_DIR / "ko_strategyqa.json", quick=False),
]


@dataclass
class RunResult:
    name: str
    corpus_size: int = 0
    mrr: float = 0.0
    p_at_k: float = 0.0
    r_at_k: float = 0.0
    ndcg: float = 0.0
    hit_rate: str = ""
    elapsed: float = 0.0
    error: str | None = None


# --- Custom dataset runner (SQLite graph) ---

async def run_custom_dataset(cfg: DatasetConfig) -> RunResult:
    """Run a custom dataset against its pre-built SQLite graph."""
    if not cfg.path.exists():
        return RunResult(name=cfg.name, error="graph not found")
    if not cfg.query_path or not cfg.query_path.exists():
        return RunResult(name=cfg.name, error="queries not found")

    from synaptic.backends.sqlite_graph import SqliteGraphBackend

    backend = SqliteGraphBackend(str(cfg.path))
    await backend.connect()
    graph = SynapticGraph(backend)

    with open(cfg.query_path, encoding="utf-8") as f:
        gt = json.load(f)
    queries = gt.get("queries", [])
    id_field = gt.get("id_field", "doc_id")

    bench = BenchmarkResult()
    t0 = time.time()

    for q in queries:
        qid = q.get("qid", "")
        query_text = q.get("query", "")
        relevant = set(q.get("relevant_docs", []))
        if not relevant:
            continue

        result = await graph.search(query_text, limit=50)

        if id_field == "node_title":
            retrieved = []
            for hit in result.nodes:
                title = hit.node.title
                if title and title not in retrieved:
                    retrieved.append(title)
        else:
            retrieved = []
            for hit in result.nodes:
                doc_id = (hit.node.properties or {}).get("doc_id", "")
                if doc_id and doc_id not in retrieved:
                    retrieved.append(doc_id)

        bench.add(
            query_id=qid, query=query_text,
            retrieved=retrieved[:cfg.k], relevant=relevant,
            k=cfg.k,
        )

    elapsed = time.time() - t0
    await backend.close()

    summary = bench.summary()
    total = len(queries)
    hits = sum(1 for q in bench.queries if q.get("mrr", 0) > 0)

    return RunResult(
        name=cfg.name,
        corpus_size=summary.get("total_queries", 0),
        mrr=summary.get("mrr", 0),
        p_at_k=summary.get("mean_precision", 0),
        r_at_k=summary.get("mean_recall", 0),
        ndcg=summary.get("mean_ndcg", 0),
        hit_rate=f"{hits}/{total}",
        elapsed=elapsed,
    )


# --- Public dataset runner (in-memory) ---

async def run_public_dataset(cfg: DatasetConfig) -> RunResult:
    """Run a public benchmark dataset — full pipeline: ingest → index → search.

    Uses MemoryBackend for speed (no disk I/O). The graph.add() path
    exercises the same NFC normalization, FTS indexing, and search
    pipeline as production SQLite/Kuzu backends.
    """
    if not cfg.path.exists():
        return RunResult(name=cfg.name, error="file not found")

    with open(cfg.path, encoding="utf-8") as f:
        data = json.load(f)

    raw_corpus = data.get("corpus", data.get("documents", []))
    queries = data.get("queries", [])
    if not raw_corpus or not queries:
        return RunResult(name=cfg.name, error="empty dataset")

    # Normalize corpus to list of (doc_id, title, text)
    corpus: list[tuple[str, str, str]] = []
    if isinstance(raw_corpus, dict):
        for doc_id, doc in raw_corpus.items():
            if isinstance(doc, dict):
                corpus.append((str(doc_id), str(doc.get("title", "")), str(doc.get("text", ""))))
            elif isinstance(doc, str):
                corpus.append((str(doc_id), "", doc))
    elif isinstance(raw_corpus, list):
        for doc in raw_corpus:
            if isinstance(doc, dict):
                doc_id = str(doc.get("doc_id", doc.get("_id", doc.get("id", ""))))
                corpus.append((doc_id, str(doc.get("title", "")), str(doc.get("text", doc.get("content", "")))))

    if not corpus:
        return RunResult(name=cfg.name, error="could not parse corpus")

    # Full pipeline: build graph via graph.add()
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend)

    for doc_id, title, text in corpus:
        if not text and not title:
            continue
        await graph.add(
            title=title or doc_id,
            content=text,
            properties={"doc_id": doc_id},
        )

    # Parse queries — support both list and BEIR dict format
    qrels = data.get("relevant_docs", data.get("qrels", {}))
    query_list: list[tuple[str, str, set[str]]] = []  # (qid, text, relevant_ids)

    if isinstance(queries, dict):
        # BEIR format: queries={qid: text}, relevant_docs={qid: {doc_id: score}}
        for qid, text in queries.items():
            rel = qrels.get(qid, {})
            if isinstance(rel, dict):
                relevant = set(str(k) for k in rel.keys())
            elif isinstance(rel, list):
                relevant = set(str(x) for x in rel)
            else:
                continue
            if relevant and text:
                query_list.append((str(qid), str(text), relevant))
    elif isinstance(queries, list):
        for q in queries:
            qid = str(q.get("qid", q.get("query_id", q.get("_id", ""))))
            text = str(q.get("query", q.get("question", "")))
            rel_raw = q.get("relevant_docs", q.get("answer_ids", q.get("positive_doc_ids", [])))
            if isinstance(rel_raw, dict):
                relevant = set(str(k) for k in rel_raw.keys())
            elif isinstance(rel_raw, list):
                relevant = set(str(x) for x in rel_raw)
            else:
                continue
            if relevant and text:
                query_list.append((qid, text, relevant))

    if not query_list:
        return RunResult(name=cfg.name, error="no valid queries")

    # Search
    bench = BenchmarkResult()
    t0 = time.time()

    for qid, query_text, relevant in query_list:
        result = await graph.search(query_text, limit=cfg.k * 2)
        retrieved = []
        for hit in result.nodes:
            doc_id = (hit.node.properties or {}).get("doc_id", "")
            if doc_id and doc_id not in retrieved:
                retrieved.append(doc_id)

        bench.add(
            query_id=qid, query=query_text,
            retrieved=retrieved[:cfg.k], relevant=relevant,
            k=cfg.k,
        )

    elapsed = time.time() - t0

    summary = bench.summary()
    total_q = summary.get("total_queries", 0)
    hits = sum(1 for q in bench.queries if q.get("mrr", 0) > 0)

    return RunResult(
        name=cfg.name,
        corpus_size=len(corpus),
        mrr=summary.get("mrr", 0),
        p_at_k=summary.get("mean_precision", 0),
        r_at_k=summary.get("mean_recall", 0),
        ndcg=summary.get("mean_ndcg", 0),
        hit_rate=f"{hits}/{total_q}",
        elapsed=elapsed,
    )


# --- Report ---

def print_table(results: list[RunResult], baseline: dict | None = None):
    print()
    print(f"{'Dataset':<22} {'Corpus':>7} {'MRR':>7} {'P@10':>7} {'R@10':>7} {'nDCG':>7} {'Hit':>8} {'Time':>7}  Status")
    print("-" * 95)

    for r in results:
        if r.error:
            print(f"{r.name:<22} {'':>7} {'':>7} {'':>7} {'':>7} {'':>7} {'':>8} {'':>7}  ❌ {r.error}")
            continue

        status = "✅"
        if baseline and r.name in baseline:
            prev_mrr = baseline[r.name].get("mrr", 0)
            if r.mrr < prev_mrr - 0.01:
                status = f"⚠️  {prev_mrr:.3f}→{r.mrr:.3f}"
            elif r.mrr > prev_mrr + 0.01:
                status = f"📈 {prev_mrr:.3f}→{r.mrr:.3f}"
        elif baseline:
            status = "NEW"

        print(
            f"{r.name:<22} {r.corpus_size:>7,} {r.mrr:>7.3f} {r.p_at_k:>7.3f} "
            f"{r.r_at_k:>7.3f} {r.ndcg:>7.3f} {r.hit_rate:>8} {r.elapsed:>6.1f}s  {status}"
        )
    print()


def save_results(results: list[RunResult], path: Path):
    data = {}
    for r in results:
        if r.error:
            continue
        data[r.name] = {
            "mrr": round(r.mrr, 4),
            "p_at_k": round(r.p_at_k, 4),
            "r_at_k": round(r.r_at_k, 4),
            "ndcg": round(r.ndcg, 4),
            "corpus_size": r.corpus_size,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Results saved → {path}")


# --- Main ---

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="Skip large datasets (KLUE, Ko-StrategyQA)")
    p.add_argument("--custom-only", action="store_true", help="Only run KRRA + assort")
    p.add_argument("--compare", type=Path, default=None, help="Compare against a baseline JSON")
    p.add_argument("--save", type=Path, default=RESULTS_DIR / "qa_latest.json", help="Save results to")
    return p.parse_args()


async def main():
    args = _parse_args()

    baseline = None
    if args.compare and args.compare.exists():
        with open(args.compare) as f:
            baseline = json.load(f)
        print(f"Comparing against: {args.compare}")

    datasets = list(CUSTOM_DATASETS)
    if not args.custom_only:
        for d in PUBLIC_DATASETS:
            if args.quick and not d.quick:
                continue
            datasets.append(d)

    print(f"\nRunning {len(datasets)} benchmarks...")
    results: list[RunResult] = []

    for cfg in datasets:
        print(f"  {cfg.name}...", end=" ", flush=True)
        try:
            if cfg.is_custom:
                r = await run_custom_dataset(cfg)
            else:
                r = await run_public_dataset(cfg)
            results.append(r)
            if r.error:
                print(f"❌ {r.error}")
            else:
                print(f"MRR={r.mrr:.3f} ({r.elapsed:.1f}s)")
        except Exception as exc:
            results.append(RunResult(name=cfg.name, error=str(exc)[:80]))
            print(f"❌ {exc}")

    print_table(results, baseline)
    save_results(results, args.save)

    # Regression check
    if baseline:
        regressions = []
        for r in results:
            if r.error or r.name not in baseline:
                continue
            prev = baseline[r.name]["mrr"]
            if r.mrr < prev - 0.01:
                regressions.append(f"{r.name}: {prev:.3f} → {r.mrr:.3f}")
        if regressions:
            print("⚠️  REGRESSIONS DETECTED:")
            for reg in regressions:
                print(f"  {reg}")
            return 1
        print("✅ No regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
