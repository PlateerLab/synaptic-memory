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

async def run_custom_dataset(
    cfg: DatasetConfig,
    embed_url: str | None = None,
    embed_model: str = "qwen3-embedding:4b",
    reranker_url: str | None = None,
    use_flashrank: bool = False,
) -> RunResult:
    """Run a custom dataset against its pre-built SQLite graph.

    When embed_url is provided, uses EvidenceSearch with vector cascade.
    When reranker_url is provided, adds cross-encoder reranking.
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

    # Build searcher — with optional embedding + reranker
    embedder = None
    if embed_url:
        from synaptic.extensions.embedder import OpenAIEmbeddingProvider
        embedder = OpenAIEmbeddingProvider(api_base=embed_url, model=embed_model)

    reranker = None
    # FlashRank is English-only (ms-marco trained). For Korean datasets
    # use TEI with bge-reranker-v2-m3 instead.
    if reranker_url:
        from synaptic.extensions.reranker_cross import TEIReranker
        reranker = TEIReranker(base_url=reranker_url)

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
        corpus_size=total,
        mrr=summary.get("mrr", 0),
        p_at_k=summary.get("mean_precision@k", 0),
        r_at_k=summary.get("mean_recall@k", 0),
        ndcg=summary.get("mean_ndcg@k", 0),
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
- "how many per X" → aggregate_nodes(table, group_by)
- "find related records" → join_related(from_value, fk_property, target_table)
- Paraphrase/synonym issue → try deep_search with DIFFERENT keywords

## Key rules
- ALWAYS use category filter when you can identify the topic from metadata below
- If first search fails, REPHRASE with official/formal terms (not casual language)
- You can call MULTIPLE tools in ONE turn for efficiency
- Max 5 tool calls total. Be efficient.
- Respond in the same language as the question.

## Example
Q: "말 복지 향상 프로그램"
→ deep_search(query="말 복지", category="복지 및 교육")  ← category from metadata
"""

AGENT_TOOLS = [
    {"type": "function", "function": {"name": "deep_search",
        "description": "Search + expand + read in ONE call.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "category": {"type": "string"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search",
        "description": "Basic text search.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {"name": "filter_nodes",
        "description": "Filter by property. Operators: >=, <=, contains.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"},
            "property": {"type": "string"},
            "op": {"type": "string"},
            "value": {"type": "string"},
        }, "required": ["property", "op", "value"]}}},
    {"type": "function", "function": {"name": "aggregate_nodes",
        "description": "GROUP BY + COUNT/SUM.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"},
            "group_by": {"type": "string"},
            "metric": {"type": "string", "default": "count"},
        }, "required": ["group_by"]}}},
    {"type": "function", "function": {"name": "join_related",
        "description": "FK lookup — find related records.",
        "parameters": {"type": "object", "properties": {
            "from_value": {"type": "string"},
            "fk_property": {"type": "string"},
            "target_table": {"type": "string"},
        }, "required": ["from_value", "fk_property", "target_table"]}}},
    {"type": "function", "function": {"name": "get_document",
        "description": "Read a full document.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"},
            "query": {"type": "string"},
        }, "required": ["doc_id"]}}},
]


async def _agent_dispatch(name, args, backend, session):
    """Route agent tool calls to synaptic tools."""
    from synaptic.agent_tools import (
        get_document_tool, search_tool,
    )
    from synaptic.agent_tools_v2 import deep_search_tool
    from synaptic.agent_tools_structured import (
        aggregate_nodes_tool, filter_nodes_tool, join_related_tool,
    )

    if name == "deep_search":
        r = await deep_search_tool(backend, session, args.get("query", ""),
                                   category=args.get("category"))
    elif name == "search":
        r = await search_tool(backend, session, args.get("query", ""))
    elif name == "filter_nodes":
        r = await filter_nodes_tool(backend, session, table=args.get("table", ""),
                                     property=args["property"], op=args["op"], value=args["value"])
    elif name == "aggregate_nodes":
        r = await aggregate_nodes_tool(backend, session, table=args.get("table", ""),
                                        group_by=args["group_by"], metric=args.get("metric", "count"))
    elif name == "join_related":
        r = await join_related_tool(backend, session, from_value=args["from_value"],
                                     fk_property=args["fk_property"], target_table=args["target_table"])
    elif name == "get_document":
        r = await get_document_tool(backend, session, args["doc_id"],
                                    query=args.get("query", ""))
    else:
        return {"error": f"unknown: {name}"}
    return r.to_dict()


async def run_agent_benchmark(
    cfg: DatasetConfig,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_turns: int = 3,
) -> RunResult:
    """Run multi-turn agent on a custom dataset's hard queries."""
    if not cfg.query_path or not cfg.query_path.exists():
        return RunResult(name=cfg.name + " (agent)", error="queries not found")
    if not cfg.path.exists():
        return RunResult(name=cfg.name + " (agent)", error="graph not found")

    import os
    os.environ["OPENAI_API_KEY"] = api_key

    from openai import AsyncOpenAI
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.search_session import SearchSession, build_graph_context

    client = AsyncOpenAI()
    backend = SqliteGraphBackend(str(cfg.path))
    await backend.connect()

    graph_ctx = await build_graph_context(backend)
    system = AGENT_SYSTEM + "\n\n" + graph_ctx

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

        for turn in range(max_turns):
            turns_used = turn + 1
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=messages,
                    tools=AGENT_TOOLS, max_tokens=2048,
                )
            except Exception:
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
                    result = await _agent_dispatch(fn, fn_args, backend, session)
                    total_calls += 1
                    # Extract doc_ids from result
                    data = result.get("data", {})
                    for key in ("evidence", "results", "merged_evidence"):
                        for item in data.get(key, []):
                            props = item.get("properties", {})
                            did = props.get("doc_id", "")
                            if did:
                                found_ids.add(did)
                            title = item.get("title", "")
                            if title:
                                found_ids.add(title)
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False)[:5000],
                    })
            else:
                break

        total_turns += turns_used
        if found_ids & relevant:
            solved += 1

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
    p.add_argument("--save", type=Path, default=EVAL_DIR / "baselines" / "qa_latest.json", help="Save results to")
    p.add_argument("--embed-url", default=None, help="Embedding API URL (enables vector cascade)")
    p.add_argument("--embed-model", default="qwen3-embedding:4b")
    p.add_argument("--reranker-url", default=None, help="TEI reranker URL (enables cross-encoder)")
    p.add_argument("--flashrank", action="store_true", help="Use FlashRank CPU reranker (no GPU needed)")
    p.add_argument("--agent", action="store_true", help="Run multi-turn agent benchmark (requires OpenAI key)")
    p.add_argument("--openai-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env)")
    p.add_argument("--agent-model", default="gpt-4o-mini", help="Agent LLM model")
    p.add_argument("--agent-max-turns", type=int, default=5, help="Max turns per agent query")
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
                r = await run_custom_dataset(
                    cfg,
                    embed_url=args.embed_url,
                    embed_model=args.embed_model,
                    reranker_url=args.reranker_url,
                    use_flashrank=args.flashrank,
                )
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

    # Agent benchmark (optional, requires OpenAI key)
    if args.agent:
        api_key = args.openai_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("  ⚠ --agent requires --openai-key or OPENAI_API_KEY env")
        else:
            agent_datasets = [d for d in CUSTOM_DATASETS if "Hard" in d.name]
            for cfg in agent_datasets:
                print(f"  {cfg.name} (agent, max {args.agent_max_turns} turns)...", end=" ", flush=True)
                try:
                    r = await run_agent_benchmark(
                        cfg, api_key,
                        model=args.agent_model,
                        max_turns=args.agent_max_turns,
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
