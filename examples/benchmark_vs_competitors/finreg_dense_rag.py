"""Dense (vector) RAG vs synaptic-memory — finreg multi-hop head-to-head.

Vector retrieval is the *standard* RAG baseline — BM25/FTS understates
it. This script measures plain dense RAG on the exact same finreg corpus
/ GT / strict scoring used by ``finreg_hipporag.py`` so all three systems
(dense RAG, HippoRAG2, synaptic) are compared on equal footing.

Pipeline: embed every article with BAAI/bge-m3 → embed each query →
cosine top-k → strict multi-hop hit (every GT article must be in top-k).
This is exactly what a textbook RAG retriever does; there is no graph,
no reranker, no cross-reference following.

Fairness: uses the *same* bge-m3 embedder injected into HippoRAG2, so
neither system is penalised for embedder quality — the only variable is
whether the retrieval mechanism can follow an exact statute citation.

Environment (same isolated venv as finreg_hipporag.py)::

    export LD_LIBRARY_PATH=/tmp/hrag/lib/python3.10/site-packages/nvidia/nvjitlink/lib
    /tmp/hrag/bin/python examples/benchmark_vs_competitors/finreg_dense_rag.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "eval" / "data" / "finreg" / "raw.jsonl"
MH = REPO / "eval" / "data" / "queries" / "finreg_multihop.json"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="BAAI/bge-m3")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--fp16", action="store_true", help="half-precision (fits a near-full GPU)")
ap.add_argument("--max-seq-len", type=int, default=0, help="cap encode length (0=model default)")
ap.add_argument("--top-k", type=int, default=10, help="retrieved units per query")
ap.add_argument("--limit", type=int, default=0, help="cap docs (0=all)")
ap.add_argument("--nq", type=int, default=0, help="cap queries (0=all)")
args = ap.parse_args()

# --- Data ---
arts = [json.loads(line) for line in RAW.open(encoding="utf-8") if line.strip()]
if args.limit:
    arts = arts[: args.limit]
docs = [a["text"] for a in arts]
doc_ids = [a["doc_id"] for a in arts]
print(f"docs: {len(docs)}", flush=True)

gt = json.load(MH.open(encoding="utf-8"))["queries"]
if args.nq:
    gt = gt[: args.nq]
queries = [q["query"] for q in gt]
print(f"multi-hop queries: {len(queries)}", flush=True)

# --- Embed ---
t0 = time.time()
model_kwargs = {"torch_dtype": "float16"} if args.fp16 else {}
model = SentenceTransformer(args.model, device=args.device, model_kwargs=model_kwargs)
if args.max_seq_len:
    model.max_seq_length = args.max_seq_len
doc_emb = model.encode(
    docs, normalize_embeddings=True, batch_size=args.batch_size, show_progress_bar=True
)
print(f"embedded {len(docs)} docs in {time.time() - t0:.0f}s", flush=True)

q_emb = model.encode(queries, normalize_embeddings=True, batch_size=args.batch_size)

# --- Retrieve: cosine top-k (normalised → dot product) ---
doc_emb = np.asarray(doc_emb, dtype=np.float32)
q_emb = np.asarray(q_emb, dtype=np.float32)
sims = q_emb @ doc_emb.T  # (nq, ndocs)

# --- Strict multi-hop scoring (every GT article must be in top-k) ---
solved = 0
for i, q in enumerate(gt):
    topk = np.argsort(-sims[i])[: args.top_k]
    got = {doc_ids[j] for j in topk}
    relevant = set(q["relevant_docs"])
    hit = relevant.issubset(got)
    solved += hit
    print(f"  [{q['qid']}] hit={hit}", flush=True)

print(
    f"\nDense RAG (bge-m3, top-{args.top_k}) finreg multi-hop: "
    f"{solved}/{len(gt)} ({solved / len(gt) * 100:.0f}%)",
    flush=True,
)
