"""Quick validation of (D) LLM reranker + (A) HyDE on the two benches
where they're most likely to move the needle:

  - AutoRAG (114q): cross-encoder hurts (FTS 0.906 → bge-rerank 0.806).
                    LLM reranker should reason its way to the right hit.
  - KRRA Conv (30q): conversational queries; current MRR 0.166 single-shot.
                     HyDE projects the query into answer space, helping
                     paraphrase/colloquial matching.

Both use vLLM Qwen3.5-27B at localhost:8012/v1.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples" / "ablation"))

from local_bge import LocalBgeM3Embedder, LocalBgeRerankerV2  # noqa: E402

from synaptic.backends.memory import MemoryBackend  # noqa: E402
from synaptic.backends.sqlite_graph import SqliteGraphBackend  # noqa: E402
from synaptic.extensions.embedder_hyde import HyDEEmbedder  # noqa: E402
from synaptic.extensions.evidence_search import EvidenceSearch  # noqa: E402
from synaptic.extensions.llm_provider import OpenAILLMProvider  # noqa: E402
from synaptic.extensions.reranker_llm import LLMReranker  # noqa: E402
from synaptic.graph import SynapticGraph  # noqa: E402

VLLM = "http://localhost:8012/v1"
MODEL = "Qwen3.5-27b"


def _rr(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


async def _build_memory_graph(corpus, embedder):
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend, embedder=embedder)
    inputs = [f"{title or doc_id}\n{(text or '')[:1500]}" for doc_id, title, text in corpus]
    embs = [None] * len(corpus)
    if embedder is not None:
        BATCH = 64
        for i in range(0, len(inputs), BATCH):
            vecs = await embedder.embed_batch(inputs[i : i + BATCH])
            for j, v in enumerate(vecs):
                embs[i + j] = v if v else None
    for (doc_id, title, text), e in zip(corpus, embs):
        if not text and not title:
            continue
        await graph.add(
            title=title or doc_id, content=text,
            properties={"doc_id": doc_id}, embedding=e,
        )
    return backend


async def _measure(label, *, searcher, queries, id_field):
    mrr_total = 0.0
    hit = 0
    t0 = time.time()
    for q in queries:
        r = await searcher.search(q["query"], k=10, fts_seed_limit=30)
        retrieved: list[str] = []
        for ev in r.evidence:
            if id_field == "node_title":
                if ev.node.title and ev.node.title not in retrieved:
                    retrieved.append(ev.node.title)
            else:
                did = (ev.node.properties or {}).get("doc_id", "")
                if did and did not in retrieved:
                    retrieved.append(did)
        rel = set(q.get("relevant_docs", []))
        if not rel:
            continue
        rr = _rr(retrieved[:10], rel)
        mrr_total += rr
        if rr > 0:
            hit += 1
    elapsed = time.time() - t0
    n = len(queries)
    print(f"  {label:<35} MRR={mrr_total/max(n,1):.3f}  Hit={hit}/{n}  ({elapsed:.1f}s)")


async def autorag_d():
    print("\n=== AutoRAG (114q) — LLM reranker (D) ===")
    data = json.loads((REPO_ROOT / "tests" / "benchmark" / "data" / "autorag_retrieval.json").read_text())
    corpus = [
        (str(k), str(v.get("title", "")), str(v.get("text", "")))
        for k, v in data["corpus"].items()
    ]
    qrels = data["qrels"]
    queries = [
        {"qid": qid, "query": text, "relevant_docs": list(qrels.get(qid, {}).keys())}
        for qid, text in data["queries"].items()
        if qrels.get(qid)
    ]

    print(f"corpus: {len(corpus)}, queries: {len(queries)}")
    print("Loading bge-m3 + bge-reranker-v2-m3 ...")
    bge_emb = LocalBgeM3Embedder(device="cuda:0")
    bge_rer = LocalBgeRerankerV2(device="cuda:0")
    llm = OpenAILLMProvider(api_base=VLLM, model=MODEL, timeout=60)
    llm_rer = LLMReranker(llm=llm, max_documents=10)

    backend = await _build_memory_graph(corpus, bge_emb)

    # Baseline (current v0.17.1): bge embedder + bge reranker
    s_bge = EvidenceSearch(backend=backend, embedder=bge_emb, reranker=bge_rer)
    await _measure("baseline (bge embed + bge rer)", searcher=s_bge, queries=queries, id_field="doc_id")

    # FTS-only: no embedder, no reranker
    s_fts = EvidenceSearch(backend=backend)
    await _measure("FTS-only (ceiling on this corpus)", searcher=s_fts, queries=queries, id_field="doc_id")

    # D: bge embedder + LLM reranker
    s_d = EvidenceSearch(backend=backend, embedder=bge_emb, reranker=llm_rer)
    await _measure("D: bge embed + LLM rerank", searcher=s_d, queries=queries, id_field="doc_id")


async def krra_conv_a():
    print("\n=== KRRA Conv (30q) — HyDE (A) ===")
    backend = SqliteGraphBackend(str(REPO_ROOT / "eval" / "data" / "krra_graph.sqlite"))
    await backend.connect()
    queries = json.loads((REPO_ROOT / "eval" / "data" / "queries" / "krra_conversational.json").read_text())["queries"]
    id_field = json.loads((REPO_ROOT / "eval" / "data" / "queries" / "krra_conversational.json").read_text()).get("id_field", "doc_id")

    print(f"queries: {len(queries)}, id_field: {id_field}")
    bge_emb = LocalBgeM3Embedder(device="cuda:0")
    bge_rer = LocalBgeRerankerV2(device="cuda:0")
    llm = OpenAILLMProvider(api_base=VLLM, model=MODEL, timeout=60)
    hyde_emb = HyDEEmbedder(llm=llm, embedder=bge_emb)

    # Baseline (v0.17.1): bge embedder + bge reranker
    s_bge = EvidenceSearch(backend=backend, embedder=bge_emb, reranker=bge_rer)
    await _measure("baseline (bge embed + bge rer)", searcher=s_bge, queries=queries, id_field=id_field)

    # A: HyDE embedder + bge reranker
    s_a = EvidenceSearch(backend=backend, embedder=hyde_emb, reranker=bge_rer)
    await _measure("A: HyDE embed + bge rer", searcher=s_a, queries=queries, id_field=id_field)


async def main() -> None:
    await autorag_d()
    await krra_conv_a()


if __name__ == "__main__":
    asyncio.run(main())
