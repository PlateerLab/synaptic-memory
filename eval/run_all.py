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
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datetime import UTC

from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
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
        is_custom=True,
        quick=True,
    ),
    DatasetConfig(
        name="KRRA Hard",
        path=EVAL_DIR / "data" / "krra_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "krra_hard.json",
        is_custom=True,
        quick=True,
    ),
    DatasetConfig(
        name="assort Easy",
        path=EVAL_DIR / "data" / "assort_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "assort.json",
        is_custom=True,
        quick=True,
    ),
    DatasetConfig(
        name="assort Hard",
        path=EVAL_DIR / "data" / "assort_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "assort_hard.json",
        is_custom=True,
        quick=True,
    ),
    DatasetConfig(
        name="X2BEE Easy",
        path=EVAL_DIR / "data" / "x2bee_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "x2bee.json",
        is_custom=True,
        quick=True,
    ),
    DatasetConfig(
        name="X2BEE Hard",
        path=EVAL_DIR / "data" / "x2bee_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "x2bee_hard.json",
        is_custom=True,
        quick=True,
    ),
    DatasetConfig(
        name="KRRA Conv",
        path=EVAL_DIR / "data" / "krra_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "krra_conversational.json",
        is_custom=True,
        quick=True,
    ),
    DatasetConfig(
        name="assort Conv",
        path=EVAL_DIR / "data" / "assort_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "assort_conversational.json",
        is_custom=True,
        quick=True,
    ),
    DatasetConfig(
        name="X2BEE Conv",
        path=EVAL_DIR / "data" / "x2bee_graph.sqlite",
        query_path=EVAL_DIR / "data" / "queries" / "x2bee_conversational.json",
        is_custom=True,
        quick=True,
    ),
]

# Public datasets (in-memory, from benchmark JSON)
PUBLIC_DATASETS = [
    DatasetConfig(name="HotPotQA-24", path=BENCHMARK_DIR / "hotpotqa_24.json", quick=True),
    DatasetConfig(name="HotPotQA-200", path=BENCHMARK_DIR / "hotpotqa.json", quick=False),
    DatasetConfig(
        name="Allganize RAG-ko", path=BENCHMARK_DIR / "allganize_rag_ko.json", quick=True
    ),
    DatasetConfig(
        name="Allganize RAG-Eval", path=BENCHMARK_DIR / "allganize_rag_eval.json", quick=True
    ),
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


async def run_custom_dataset(
    cfg: DatasetConfig,
    embedder: object | None = None,
    reranker: object | None = None,
    use_flashrank: bool = False,
) -> RunResult:
    """Run a custom dataset against its pre-built SQLite graph.

    ``embedder`` / ``reranker`` are instantiated once in ``main()`` and
    shared across all datasets, so model weights load exactly once per
    suite run (critical for local --local-bge runs where bge-m3 load is
    ~10-15s on first call).
    """
    if not cfg.path.exists():
        return RunResult(name=cfg.name, error="graph not found")
    if not cfg.query_path or not cfg.query_path.exists():
        return RunResult(name=cfg.name, error="queries not found")

    from synaptic.backends.sqlite_graph import SqliteGraphBackend

    backend = SqliteGraphBackend(str(cfg.path))
    await backend.connect()

    with open(cfg.query_path, encoding="utf-8") as f:
        gt = json.load(f)
    queries = gt.get("queries", [])
    id_field = gt.get("id_field", "doc_id")

    from synaptic.extensions.evidence_search import EvidenceSearch

    searcher = EvidenceSearch(backend=backend, embedder=embedder, reranker=reranker)

    bench = BenchmarkResult()
    t0 = time.time()

    for q in queries:
        qid = q.get("qid", "")
        query_text = q.get("query", "")
        relevant = set(q.get("relevant_docs", []))
        if not relevant:
            continue

        result = await searcher.search(query_text, k=cfg.k * 2, fts_seed_limit=30)

        if id_field == "node_title":
            retrieved = []
            for ev in result.evidence:
                title = ev.node.title
                if title and title not in retrieved:
                    retrieved.append(title)
        else:
            retrieved = []
            for ev in result.evidence:
                doc_id = ev.document_id or (ev.node.properties or {}).get("doc_id", "")
                if doc_id and doc_id not in retrieved:
                    retrieved.append(doc_id)

        bench.add(
            query_id=qid,
            query=query_text,
            retrieved=retrieved[: cfg.k],
            relevant=relevant,
            k=cfg.k,
        )

    elapsed = time.time() - t0
    await backend.close()

    summary = bench.summary()
    total = len(queries)
    hits = sum(1 for q in bench.queries if q.get("mrr", 0) > 0)

    return RunResult(
        name=cfg.name,
        corpus_size=total,
        mrr=summary.get("mrr", 0),
        p_at_k=summary.get("mean_precision@k", 0),
        r_at_k=summary.get("mean_recall@k", 0),
        ndcg=summary.get("mean_ndcg@k", 0),
        hit_rate=f"{hits}/{total}",
        elapsed=elapsed,
    )


# --- Public dataset runner (in-memory) ---


async def run_public_dataset(
    cfg: DatasetConfig,
    embedder: object | None = None,
    reranker: object | None = None,
    entity_linker_cfg: tuple[int, float] | None = None,
) -> RunResult:
    """Run a public benchmark dataset — full pipeline: ingest → index → search.

    Uses MemoryBackend for speed (no disk I/O). Shared ``embedder``/
    ``reranker`` objects (instantiated once in ``main()``) avoid
    per-dataset model reload. When ``entity_linker_cfg = (min_df,
    max_df_ratio)`` is set, runs :class:`EntityLinker` post-hoc to build
    a DF-filtered phrase hub before search.
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
                corpus.append(
                    (
                        doc_id,
                        str(doc.get("title", "")),
                        str(doc.get("text", doc.get("content", ""))),
                    )
                )

    if not corpus:
        return RunResult(name=cfg.name, error="could not parse corpus")

    # Full pipeline: build graph via graph.add()
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend, embedder=embedder, reranker=reranker)

    # Pre-compute corpus embeddings in batches when an embedder is wired.
    # Passing ``embedding=`` to ``graph.add`` avoids the per-node single
    # embed call that bottlenecks at batch=1 (the main reason public
    # bench ingest was GPU-idle in previous runs).
    embeddings: list[list[float] | None] = [None] * len(corpus)
    if embedder is not None and hasattr(embedder, "embed_batch"):
        embed_inputs = [
            f"{title or doc_id}\n{(text or '')[:1500]}" for doc_id, title, text in corpus
        ]
        BATCH = 64
        for i in range(0, len(embed_inputs), BATCH):
            chunk = embed_inputs[i : i + BATCH]
            vecs = await embedder.embed_batch(chunk)
            for j, v in enumerate(vecs):
                embeddings[i + j] = v if v else None

    for (doc_id, title, text), emb in zip(corpus, embeddings):
        if not text and not title:
            continue
        await graph.add(
            title=title or doc_id,
            content=text,
            properties={"doc_id": doc_id},
            embedding=emb,
        )

    # Post-hoc DF-filtered entity linking.
    if entity_linker_cfg is not None:
        from synaptic.extensions.domain_profile import DomainProfile
        from synaptic.extensions.entity_linker import EntityLinker
        from synaptic.extensions.phrase_extractor import PhraseExtractor
        from synaptic.models import NodeKind as _NK

        min_df, max_df_ratio = entity_linker_cfg
        profile = DomainProfile(
            name=f"{cfg.name}-eval",
            locale="multi",
            min_df=min_df,
            max_df_ratio=max_df_ratio,
        )
        linker = EntityLinker(
            extractor=PhraseExtractor(),
            profile=profile,
            max_links_per_source=15,
        )
        await linker.link(backend, source_kind=_NK.CONCEPT)

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

    # Build searcher — EvidenceSearch when embedder available, else graph.search
    use_evidence = embedder is not None or reranker is not None
    searcher = None
    if use_evidence:
        from synaptic.extensions.evidence_search import EvidenceSearch

        searcher = EvidenceSearch(backend=backend, embedder=embedder, reranker=reranker)

    # Search
    bench = BenchmarkResult()
    t0 = time.time()

    for qid, query_text, relevant in query_list:
        if searcher:
            result = await searcher.search(query_text, k=cfg.k * 2, fts_seed_limit=30)
            retrieved = []
            for ev in result.evidence:
                doc_id = ev.document_id or (ev.node.properties or {}).get("doc_id", "")
                if doc_id and doc_id not in retrieved:
                    retrieved.append(doc_id)
        else:
            result = await graph.search(query_text, limit=cfg.k * 2)
            retrieved = []
            for hit in result.nodes:
                doc_id = (hit.node.properties or {}).get("doc_id", "")
                if doc_id and doc_id not in retrieved:
                    retrieved.append(doc_id)

        bench.add(
            query_id=qid,
            query=query_text,
            retrieved=retrieved[: cfg.k],
            relevant=relevant,
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
        p_at_k=summary.get("mean_precision@k", 0),
        r_at_k=summary.get("mean_recall@k", 0),
        ndcg=summary.get("mean_ndcg@k", 0),
        hit_rate=f"{hits}/{total_q}",
        elapsed=elapsed,
    )


# --- Multi-turn Agent Benchmark ---

AGENT_SYSTEM = """\
You are a research agent. Use the provided tools to answer the question.

## Tool selection (pick the RIGHT one first time)
- Text question → deep_search(query, category="relevant category from metadata")
- Price/date/attribute filter → filter_nodes(table, property, op, value)
- "how many per X" / TOP N → aggregate_nodes(table, group_by, metric)
- "find related records" → join_related(from_value, fk_property, target_table)
- Find by name/text → filter_nodes(table, property=name_column, op="contains", value="keyword")

## Key rules
- Use the exact table and column names from the structured data metadata below
- ALWAYS use category filter when you can identify the topic from metadata
- You can call MULTIPLE tools in ONE turn for efficiency
- Max 15 tool calls total. Be efficient.
- Respond in the same language as the question.

## Fallback when search returns 0 results
1. Try filter_nodes with op="contains" on text columns (e.g., product_name, goods_nm)
2. Try search with shorter/individual keywords from your query
3. Try search with translated terms (Korean ↔ English)

## Structured data patterns
- Node titles = table_name:pk_value (e.g., "products:12800000", "colors:1")
- Use FK relationships from metadata to chain queries across tables
- For cross-table questions: find source → join_related → target table

## Examples
Q: "말 복지 향상 프로그램"
→ deep_search(query="말 복지", category="복지 및 교육")

Q: "50만원 이상 고가 상품"
→ filter_nodes(table="pr_goods_base", property="sales_prc", op=">=", value="500000")

Q: "가장 많이 팔린 상품"
→ aggregate_nodes(table="pr_goods_sold_hist", group_by="goods_no", metric="sum")

Q: "5점 리뷰가 가장 많은 상품"
→ aggregate_nodes(table="feedback", group_by="goods_no", metric="count", where_property="score", where_op="==", where_value="5")

Q: "스마트폰 제품 찾기"
→ filter_nodes(table="pr_goods_base", property="goods_nm", op="contains", value="phone")

## Date queries — use starts_with or date_range or group_by_format
Q: "2023년 12월 판매 건수"
→ filter_nodes(table="sold_hist", property="sold_dtm", op="starts_with", value="2023-12")

Q: "2023년 여름(6-8월) 판매"
→ filter_nodes(table="sold_hist", property="sold_dtm", op="date_range", value="2023-06-01..2023-08-31")

Q: "월별 매출 추이"
→ aggregate_nodes(table="sold_hist", group_by="sold_dtm", group_by_format="YYYY-MM", metric="count")

## Multi-hop chaining — pass previous step's node_titles or group values as from_ids
Q: "판매량 1위 상품의 리뷰 평점 평균"
Step 1: aggregate_nodes(table="sold_hist", group_by="goods_no", metric="sum", metric_property="sold_qunt")
  → top groups include {"group": "G00001", "node_title": "pr_goods_base:G00001"}
Step 2: aggregate_nodes(table="feedback", group_by="score", metric="count",
                         where_property="goods_no", where_op="==", where_value="G00001")

Q: "5점 리뷰 최다 상품 중 가장 저렴한 것"
Step 1: aggregate_nodes(table="feedback", group_by="goods_no", metric="count",
                         where_property="score", where_op="==", where_value="5")
  → groups=[{node_title:"pr_goods_base:G00857"}, ...]
Step 2: filter_nodes(from_ids=["pr_goods_base:G00857","pr_goods_base:G00472"],
                      property="sales_prc", op=">=", value="0")
  → then pick the cheapest from results

Q: "iPhone과 Galaxy Book의 판매 이력"
→ join_related(from_values=["G00007","G00003"], fk_property="goods_no", target_table="pr_goods_sold_hist")

## Language fallback
- If data contains English product names, try English keywords when Korean search returns 0
- Example: "치즈" returns 0 → try "cheese" instead
"""

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "deep_search",
            "description": "Search + expand + read in ONE call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Basic text search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_nodes",
            "description": "Filter by property. Returns {total, showing, results}. Supports multi-hop chaining via from_ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name from metadata e.g. pr_goods_base",
                    },
                    "property": {"type": "string", "description": "Column name e.g. sales_prc"},
                    "op": {
                        "type": "string",
                        "description": ">=, <=, >, <, ==, !=, contains, starts_with, date_range",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value. For date_range: '2023-06-01..2023-08-31'. For starts_with: prefix like '2023-12'",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 20). Use higher for listings.",
                    },
                    "from_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: restrict to these node titles/IDs (multi-hop chaining from previous step's results)",
                    },
                },
                "required": ["property", "op", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_nodes",
            "description": "GROUP BY + COUNT/SUM/AVG/MAX/MIN with WHERE pre-filter, date bucketing, and multi-hop chaining.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name from metadata"},
                    "group_by": {"type": "string", "description": "Column to group by"},
                    "metric": {"type": "string", "enum": ["count", "sum", "avg", "max", "min"]},
                    "metric_property": {
                        "type": "string",
                        "description": "Numeric column for sum/avg/max/min",
                    },
                    "where_property": {
                        "type": "string",
                        "description": "Pre-filter column e.g. score",
                    },
                    "where_op": {
                        "type": "string",
                        "description": "==, !=, >=, <=, >, <, contains, starts_with, date_range",
                    },
                    "where_value": {"type": "string", "description": "Pre-filter value e.g. 5"},
                    "group_by_format": {
                        "type": "string",
                        "description": "Date bucket format: 'YYYY', 'YYYY-MM', 'YYYY-MM-DD'. Use for monthly/yearly aggregation on datetime columns.",
                    },
                    "limit": {"type": "integer", "description": "Max groups (default 50)"},
                    "from_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: restrict aggregation to these node titles/IDs (multi-hop chaining)",
                    },
                },
                "required": ["group_by"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "join_related",
            "description": "FK lookup — find related records. Accepts single from_value OR list of from_values for batch JOIN.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_value": {"type": "string", "description": "Single FK value e.g. G00001"},
                    "from_values": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple FK values for batch IN-clause JOIN (multi-hop chaining)",
                    },
                    "fk_property": {"type": "string", "description": "FK column e.g. goods_no"},
                    "target_table": {
                        "type": "string",
                        "description": "Target table e.g. pr_goods_sold_hist",
                    },
                    "limit": {"type": "integer", "description": "Max results (default 20)"},
                },
                "required": ["fk_property", "target_table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Read a full document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand",
            "description": "Explore graph neighbours of a node — follow edges to discover related nodes (FK-linked rows, document chunks, category siblings).",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID to expand from"},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "follow",
            "description": "Follow a specific edge type from a node. Edge types: contains, part_of, next_chunk, related, mentions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Source node ID"},
                    "edge_kind": {
                        "type": "string",
                        "description": "Edge type to follow: related, contains, part_of, etc.",
                    },
                },
                "required": ["node_id", "edge_kind"],
            },
        },
    },
]


def _extract_ids(data: dict, found_ids: set[str], known_tables: set[str] | None = None) -> None:
    """Extract ALL possible document identifiers from any tool result.

    Covers every tool's response structure:
    - evidence[].document_id, evidence[].properties.doc_id, evidence[].title
    - results[].properties.doc_id, results[].title
    - merged_evidence[].document_id
    - document_excerpts[].document.properties.doc_id
    - sub_results[].top_result.document_id
    - document.properties.doc_id (get_document)
    - chunks[].properties (get_document)
    - groups[].group (aggregate — group value may be a PK like goods_no)

    Args:
        known_tables: Set of actual table names from the graph (e.g. {"colors", "products"}).
            Used to resolve FK column stems to real table names for aggregate groups.
    """
    # Flat item lists
    for key in (
        "evidence",
        "results",
        "merged_evidence",
        "matches",
        "expanded_neighbours",
        "neighbours",
    ):
        for item in data.get(key, []):
            # Direct document_id field (from EvidenceAggregator)
            did = item.get("document_id", "")
            if did:
                found_ids.add(did)
            # properties.doc_id
            props = item.get("properties", {})
            did2 = props.get("doc_id", "")
            if did2:
                found_ids.add(did2)
            # title (for assort: "products:12800000")
            title = item.get("title", "")
            if title:
                found_ids.add(title)

    # document_excerpts (from deep_search)
    for excerpt in data.get("document_excerpts", []):
        doc = excerpt.get("document", {})
        did = doc.get("properties", {}).get("doc_id", "")
        if did:
            found_ids.add(did)
        title = doc.get("title", "")
        if title:
            found_ids.add(title)

    # sub_results (from compare_search)
    for sub in data.get("sub_results", []):
        top = sub.get("top_result")
        if isinstance(top, dict):
            did = top.get("document_id", "")
            if did:
                found_ids.add(did)
            props = top.get("properties", {})
            did2 = props.get("doc_id", "")
            if did2:
                found_ids.add(did2)

    # get_document response
    doc_data = data.get("document", {})
    if isinstance(doc_data, dict):
        did = doc_data.get("properties", {}).get("doc_id", "")
        if did:
            found_ids.add(did)

    # filter_nodes / join_related results already covered by "results" above

    # aggregate groups — group value may be a PK (e.g. goods_no "G00001")
    agg_info = data.get("aggregation", {})
    agg_table = agg_info.get("table", "")
    group_by = agg_info.get("group_by", "")
    for grp in data.get("groups", []):
        g = grp.get("group", "")
        if not g:
            continue
        # Add raw value
        found_ids.add(g)
        # If aggregate tool provided node_title (Phase 2B), use it directly
        nt = grp.get("node_title", "")
        if nt:
            found_ids.add(nt)

        # Heuristic prefix generation is only useful when the group value
        # looks like a primary key (short, identifier-like). Skip for
        # dates, long strings, spaces, or non-PK-looking values to avoid
        # flooding found_ids with noise like "pr_sold_base:2023-12-20...".
        looks_like_pk = (
            g
            and len(g) <= 30
            and " " not in g
            and "-" not in g[:5]  # not a date prefix
            and not g.startswith("20")  # reject common year-start strings
        )
        if not looks_like_pk:
            continue

        # Source table prefix
        if agg_table:
            found_ids.add(f"{agg_table}:{g}")

        # Resolve FK column → target table name
        if group_by:
            # Strip trailing _no/_id/_code to get stem
            base = group_by.rsplit("_", 1)[0] if "_" in group_by else group_by

            # Method 1: Match against known table names (most reliable)
            if known_tables:
                for tbl in known_tables:
                    # "color" matches "colors", "goods" matches "pr_goods_base"
                    tbl_lower = tbl.lower()
                    if base in tbl_lower or tbl_lower.startswith(base):
                        found_ids.add(f"{tbl}:{g}")

            # Method 2: Heuristic fallbacks (singular, plural, prefixed)
            for candidate in (
                f"{base}:{g}",
                f"{base}s:{g}",  # plural: color → colors
                f"{base}es:{g}",  # plural: address → addresses
                f"pr_{base}_base:{g}",  # Korean DB: goods → pr_goods_base
                f"pr_{base}:{g}",
            ):
                found_ids.add(candidate)


async def _agent_dispatch(name, args, backend, session, *, embedder=None):
    """Route agent tool calls to synaptic tools."""
    from synaptic.agent_tools import (
        expand_tool,
        follow_tool,
        get_document_tool,
        search_tool,
    )
    from synaptic.agent_tools_structured import (
        aggregate_nodes_tool,
        filter_nodes_tool,
        join_related_tool,
    )
    from synaptic.agent_tools_v2 import deep_search_tool

    if name == "deep_search":
        r = await deep_search_tool(
            backend,
            session,
            args.get("query", ""),
            category=args.get("category"),
            embedder=embedder,
        )
    elif name == "search":
        r = await search_tool(backend, session, args.get("query", ""), embedder=embedder)
    elif name == "expand":
        r = await expand_tool(backend, session, args.get("node_id", ""))
    elif name == "follow":
        r = await follow_tool(
            backend, session, args.get("node_id", ""), args.get("edge_kind", "related")
        )
    elif name == "filter_nodes":
        r = await filter_nodes_tool(
            backend,
            session,
            table=args.get("table", ""),
            property=args.get("property", ""),
            op=args.get("op", "contains"),
            value=args.get("value", ""),
            limit=int(args.get("limit", 20)),
            from_ids=args.get("from_ids") or None,
        )
    elif name == "aggregate_nodes":
        r = await aggregate_nodes_tool(
            backend,
            session,
            table=args.get("table", ""),
            group_by=args.get("group_by", ""),
            metric=args.get("metric", "count"),
            metric_property=args.get("metric_property", ""),
            where_property=args.get("where_property", ""),
            where_op=args.get("where_op", ""),
            where_value=args.get("where_value", ""),
            group_by_format=args.get("group_by_format", ""),
            limit=int(args.get("limit", 50)),
            from_ids=args.get("from_ids") or None,
        )
    elif name == "join_related":
        r = await join_related_tool(
            backend,
            session,
            from_value=args.get("from_value", ""),
            from_values=args.get("from_values") or None,
            fk_property=args.get("fk_property", ""),
            target_table=args.get("target_table", ""),
            limit=int(args.get("limit", 20)),
        )
    elif name == "get_document":
        r = await get_document_tool(backend, session, args["doc_id"], query=args.get("query", ""))
    else:
        return {"error": f"unknown: {name}"}
    return r.to_dict()


async def _llm_judge(
    client: Any,
    query: str,
    agent_answer: str,
    relevant_samples: list[str],
) -> bool:
    """Ask an LLM whether the agent's answer semantically satisfies the query.

    Compares the agent's final text answer against ground-truth sample node
    titles. Returns True if the answer is a plausible response to the query,
    even if it doesn't match the exact GT IDs.
    """
    if not agent_answer.strip():
        return False
    prompt = f"""Judge whether the agent answer correctly addresses the query.

Query: {query}

Expected answer domain (sample relevant items — just examples, not exhaustive):
{", ".join(relevant_samples[:5])}

Agent answer:
{agent_answer[:1500]}

Rules:
- Answer YES if the response is a reasonable, factually plausible answer.
- Answer NO only if the agent completely failed or gave a clearly wrong response.
- Do NOT require exact ID matches — the samples are just examples, many
  other valid items may exist.
- For counting / listing queries: YES if the count or category is correct.
- For filter / search queries: YES if the returned items satisfy the criteria.
- For document queries: YES if the answer discusses the right topic area
  (even if specific document IDs differ from samples).
- For recommendation queries: YES if any reasonable recommendation is given.
- For multi-hop queries: YES if the final answer is correct, regardless of
  intermediate IDs.

Reply with only YES or NO."""
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        return "YES" in (resp.choices[0].message.content or "").upper()
    except Exception:
        return False


async def run_agent_benchmark(
    cfg: DatasetConfig,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_turns: int = 3,
    embed_url: str | None = None,
    embed_model: str = "qwen3-embedding:4b",
    judge: bool = False,
    llm_base_url: str | None = None,
) -> RunResult:
    """Run multi-turn agent on a custom dataset's hard queries."""
    if not cfg.query_path or not cfg.query_path.exists():
        return RunResult(name=cfg.name + " (agent)", error="queries not found")
    if not cfg.path.exists():
        return RunResult(name=cfg.name + " (agent)", error="graph not found")

    import os

    os.environ["OPENAI_API_KEY"] = api_key or "ollama"

    from openai import AsyncOpenAI

    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.search_session import SearchSession, build_graph_context

    client = AsyncOpenAI(base_url=llm_base_url) if llm_base_url else AsyncOpenAI()
    backend = SqliteGraphBackend(str(cfg.path))
    await backend.connect()

    # Build embedder for agent search tools (same as run_custom_dataset)
    embedder = None
    if embed_url:
        from synaptic.extensions.embedder import OpenAIEmbeddingProvider

        embedder = OpenAIEmbeddingProvider(api_base=embed_url, model=embed_model)

    graph_ctx = await build_graph_context(backend)
    system = AGENT_SYSTEM + "\n\n" + graph_ctx

    # Collect known table names for _extract_ids matching
    from synaptic.models import NodeKind as _NK

    _sample = await backend.list_nodes(kind=_NK.ENTITY, limit=50_000)
    known_tables: set[str] = set()
    for _n in _sample:
        _tbl = (_n.properties or {}).get("_table_name")
        if _tbl:
            known_tables.add(_tbl)

    with open(cfg.query_path, encoding="utf-8") as f:
        gt = json.load(f)
    queries = gt.get("queries", [])
    id_field = gt.get("id_field", "doc_id")

    solved = 0
    total = 0
    total_turns = 0
    total_calls = 0
    t0 = time.time()

    for q in queries:
        query_text = q.get("query", "")
        relevant = set(q.get("relevant_docs", []))
        if not relevant or not query_text:
            continue
        total += 1

        session = SearchSession(budget_tool_calls=max_turns * 3)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query_text},
        ]

        found_ids: set[str] = set()
        turns_used = 0
        final_answer = ""

        for turn in range(max_turns):
            turns_used = turn + 1
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=AGENT_TOOLS,
                    max_tokens=2048,
                )
            except Exception as exc:
                print(f"    ⚠ API error: {exc}")
                break

            msg = resp.choices[0].message
            if msg.tool_calls:
                messages.append(msg.model_dump())
                for tc in msg.tool_calls:
                    fn = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}
                    result = await _agent_dispatch(fn, fn_args, backend, session, embedder=embedder)
                    total_calls += 1
                    # Extract ALL possible identifiers from result
                    data = result.get("data", {})
                    _extract_ids(data, found_ids, known_tables=known_tables)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, ensure_ascii=False)[:5000],
                        }
                    )
            else:
                # Agent gave a text answer — no more tool calls.
                final_answer = msg.content or ""
                break

        total_turns += turns_used
        id_hit = bool(found_ids & relevant)
        hit = id_hit
        judge_hit = False
        # LLM-as-Judge fallback: if ID matching failed but agent produced
        # an answer, check whether the answer is semantically correct.
        if not id_hit and judge and final_answer:
            judge_hit = await _llm_judge(client, query_text, final_answer, list(relevant))
            if judge_hit:
                hit = True
        if hit:
            solved += 1
        # Debug: show progress per query
        tag = "id" if id_hit else ("judge" if judge_hit else "miss")
        if not hit and found_ids:
            print(
                f"      [{q.get('qid', '')}] turns={turns_used} found={len(found_ids)} hit=False | relevant={relevant} | sample_found={list(found_ids)[:3]}"
            )
        else:
            print(
                f"      [{q.get('qid', '')}] turns={turns_used} found={len(found_ids)} hit={hit} ({tag})"
            )

    elapsed = time.time() - t0
    await backend.close()

    return RunResult(
        name=cfg.name + " (agent)",
        corpus_size=total,
        mrr=solved / total if total > 0 else 0,
        p_at_k=0,
        r_at_k=0,
        ndcg=0,
        hit_rate=f"{solved}/{total}",
        elapsed=elapsed,
    )


# --- Report ---


def print_table(results: list[RunResult], baseline: dict | None = None):
    print()
    print(
        f"{'Dataset':<22} {'Corpus':>7} {'MRR':>7} {'P@10':>7} {'R@10':>7} {'nDCG':>7} {'Hit':>8} {'Time':>7}  Status"
    )
    print("-" * 95)

    for r in results:
        if r.error:
            print(
                f"{r.name:<22} {'':>7} {'':>7} {'':>7} {'':>7} {'':>7} {'':>8} {'':>7}  ❌ {r.error}"
            )
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
    """Persist baseline metrics with a `_meta` block.

    The `_meta` block lets future readers verify quickly whether
    a saved baseline matches the corpus + code they are running
    against. Stale baselines were a real source of phantom
    "regressions" during the v0.14.x release cycle — having
    snapshot hashes inline makes the mismatch obvious.

    Underscore prefix keeps `_meta` from being mistaken for a
    dataset name by older readers that iterate `data.values()`.
    """
    import hashlib
    from datetime import datetime

    try:
        from synaptic import __version__ as _synaptic_version
    except Exception:
        _synaptic_version = "unknown"

    def _md5(p: Path) -> str:
        try:
            with p.open("rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except OSError:
            return "missing"

    # Capture corpus snapshots that the in-house bench depends
    # on. Public datasets are downloaded fresh each run, so they
    # don't need a hash.
    corpus_paths = {
        "krra_chunks_jsonl": EVAL_DIR / "data" / "parsed" / "krra" / "chunks.jsonl",
        "krra_documents_jsonl": EVAL_DIR / "data" / "parsed" / "krra" / "documents.jsonl",
        "assort_products_csv": EVAL_DIR / "data" / "raw" / "assort" / "products.csv",
    }
    corpus_md5 = {name: _md5(p) for name, p in corpus_paths.items()}

    # Same for query files — augmenting a query set is a common
    # source of "score moved" surprises.
    query_files = list((EVAL_DIR / "data" / "queries").glob("*.json"))
    query_md5 = {p.stem: _md5(p) for p in sorted(query_files)}

    data: dict = {
        "_meta": {
            "synaptic_version": _synaptic_version,
            "saved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "corpus_md5": corpus_md5,
            "query_md5": query_md5,
            "note": (
                "Baseline written by eval/run_all.py. The `_meta` block "
                "captures corpus + query file hashes so phantom regressions "
                "from drifting source data are easy to diagnose. If you see "
                "a score change after a code-only PR, compare the hashes "
                "first."
            ),
        }
    }

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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--quick", action="store_true", help="Skip large datasets (KLUE, Ko-StrategyQA)")
    p.add_argument("--custom-only", action="store_true", help="Only run KRRA + assort")
    p.add_argument("--compare", type=Path, default=None, help="Compare against a baseline JSON")
    p.add_argument(
        "--save",
        type=Path,
        default=EVAL_DIR / "baselines" / "qa_latest.json",
        help="Save results to",
    )
    p.add_argument("--embed-url", default=None, help="Embedding API URL (enables vector cascade)")
    p.add_argument("--embed-model", default="qwen3-embedding:4b")
    p.add_argument("--reranker-url", default=None, help="TEI reranker URL (enables cross-encoder)")
    p.add_argument(
        "--local-bge",
        action="store_true",
        help="Load BAAI/bge-m3 + bge-reranker-v2-m3 directly via transformers "
        "(no external endpoint). Requires torch + GPU. Overrides --embed-url / "
        "--reranker-url when set.",
    )
    p.add_argument(
        "--local-bge-device",
        default="cuda:0",
        help="GPU device for --local-bge (default: cuda:0).",
    )
    p.add_argument(
        "--entity-linker",
        action="store_true",
        help="Run post-hoc DF-filtered EntityLinker on each public benchmark "
        "corpus (min_df=2, max_df_ratio=0.02 by default). Custom SQLite "
        "graphs are left untouched.",
    )
    p.add_argument(
        "--entity-min-df",
        type=int,
        default=2,
    )
    p.add_argument(
        "--entity-max-df-ratio",
        type=float,
        default=0.02,
    )
    p.add_argument(
        "--flashrank", action="store_true", help="Use FlashRank CPU reranker (no GPU needed)"
    )
    p.add_argument(
        "--agent", action="store_true", help="Run multi-turn agent benchmark (requires OpenAI key)"
    )
    p.add_argument(
        "--agent-only",
        action="store_true",
        help="Skip single-shot retrieval, only run agent benchmark",
    )
    p.add_argument("--openai-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env)")
    p.add_argument(
        "--llm-base-url",
        default=None,
        help="Custom LLM base URL (OpenAI-compatible, e.g. http://host:11434/v1 for Ollama)",
    )
    p.add_argument("--agent-model", default="gpt-4o-mini", help="Agent LLM model")
    p.add_argument("--agent-max-turns", type=int, default=5, help="Max turns per agent query")
    p.add_argument(
        "--judge", action="store_true", help="Enable LLM-as-Judge fallback when ID matching fails"
    )
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

    if args.agent_only:
        datasets = []
        print("\n--agent-only: skipping single-shot retrieval")

    # Instantiate embedder / reranker ONCE so model weights load once per
    # suite run and stay warm across all datasets.
    shared_embedder: object | None = None
    shared_reranker: object | None = None
    embedder_label = "none"
    reranker_label = "none"
    if args.local_bge:
        import sys as _sys

        _sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent / "examples" / "ablation")
        )
        from local_bge import LocalBgeM3Embedder, LocalBgeRerankerV2

        print(f"Loading bge-m3 + bge-reranker-v2-m3 on {args.local_bge_device} ...")
        shared_embedder = LocalBgeM3Embedder(device=args.local_bge_device)
        shared_reranker = LocalBgeRerankerV2(device=args.local_bge_device)
        embedder_label = f"local BAAI/bge-m3 ({args.local_bge_device})"
        reranker_label = f"local BAAI/bge-reranker-v2-m3 ({args.local_bge_device})"
    else:
        if args.embed_url:
            from synaptic.extensions.embedder import OpenAIEmbeddingProvider

            shared_embedder = OpenAIEmbeddingProvider(
                api_base=args.embed_url, model=args.embed_model
            )
            embedder_label = f"OpenAI {args.embed_model} @ {args.embed_url}"
        if args.reranker_url:
            from synaptic.extensions.reranker_cross import TEIReranker

            shared_reranker = TEIReranker(base_url=args.reranker_url)
            reranker_label = f"TEI @ {args.reranker_url}"

    entity_linker_cfg: tuple[int, float] | None = None
    if args.entity_linker:
        entity_linker_cfg = (args.entity_min_df, args.entity_max_df_ratio)

    print(f"\nRunning {len(datasets)} benchmarks...")
    print(f"  embedder: {embedder_label}")
    print(f"  reranker: {reranker_label}")
    if entity_linker_cfg:
        print(
            f"  entity linker: min_df={entity_linker_cfg[0]}, "
            f"max_df_ratio={entity_linker_cfg[1]} (public datasets only)"
        )
    results: list[RunResult] = []

    for cfg in datasets:
        print(f"  {cfg.name}...", end=" ", flush=True)
        try:
            if cfg.is_custom:
                r = await run_custom_dataset(
                    cfg,
                    embedder=shared_embedder,
                    reranker=shared_reranker,
                    use_flashrank=args.flashrank,
                )
            else:
                r = await run_public_dataset(
                    cfg,
                    embedder=shared_embedder,
                    reranker=shared_reranker,
                    entity_linker_cfg=entity_linker_cfg,
                )
            results.append(r)
            if r.error:
                print(f"❌ {r.error}")
            else:
                print(f"MRR={r.mrr:.3f} ({r.elapsed:.1f}s)")
        except Exception as exc:
            results.append(RunResult(name=cfg.name, error=str(exc)[:80]))
            print(f"❌ {exc}")

    # Agent benchmark (optional, requires OpenAI key)
    if args.agent:
        api_key = args.openai_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key and not args.llm_base_url:
            print("  ⚠ --agent requires --openai-key/OPENAI_API_KEY env or --llm-base-url")
        else:
            agent_datasets = [d for d in CUSTOM_DATASETS if "Hard" in d.name or "Conv" in d.name]
            for cfg in agent_datasets:
                print(
                    f"  {cfg.name} (agent, max {args.agent_max_turns} turns)...",
                    end=" ",
                    flush=True,
                )
                try:
                    r = await run_agent_benchmark(
                        cfg,
                        api_key,
                        model=args.agent_model,
                        max_turns=args.agent_max_turns,
                        embed_url=args.embed_url,
                        embed_model=args.embed_model,
                        judge=args.judge,
                        llm_base_url=args.llm_base_url,
                    )
                    results.append(r)
                    if r.error:
                        print(f"❌ {r.error}")
                    else:
                        print(f"solved={r.hit_rate} ({r.elapsed:.1f}s)")
                except Exception as exc:
                    results.append(RunResult(name=cfg.name + " (agent)", error=str(exc)[:80]))
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
