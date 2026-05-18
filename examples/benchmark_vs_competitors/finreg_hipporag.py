"""HippoRAG2 vs synaptic-memory — finreg multi-hop head-to-head (v0.25 WS-2).

Measured result (2026-05-18, same corpus / GT / model / strict scoring):

    vanilla RAG        0/120   (0%)
    HippoRAG2         30/120  (25%)
    synaptic-memory  100/120  (83%)

HippoRAG2 (NeurIPS'24 — Personalized PageRank over an LLM-extracted
entity graph) extracts *fuzzy entity triples* via OpenIE; it cannot
capture an exact statute cross-reference ("제30조") as a clean edge, so
on multi-hop "follow the citation" queries it lands at 25% — above
single-shot RAG but far below synaptic's REFERENCES-edge 83%.

Environment
-----------
HippoRAG pulls a torch build whose bundled CUDA libs can mismatch the
host; run it in a *separate* venv, and add the nvjitlink lib to the
loader path::

    uv venv /tmp/hrag --python 3.10
    uv pip install --python /tmp/hrag/bin/python hipporag sentence-transformers
    export LD_LIBRARY_PATH=/tmp/hrag/lib/python3.10/site-packages/nvidia/nvjitlink/lib
    /tmp/hrag/bin/python examples/benchmark_vs_competitors/finreg_hipporag.py

Fairness: HippoRAG2 ships only English embedders (contriever /
NV-Embed-v2 / GritLM). To avoid penalising its graph algorithm for a
missing feature, this script injects a Korean embedder (bge-m3) — the
same calibre synaptic's full pipeline uses — via a small adapter class.
The OpenIE LLM is the same local vLLM Qwen3.6-27B the synaptic agent
benchmark used.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "eval" / "data" / "finreg" / "raw.jsonl"
MH = REPO / "eval" / "data" / "queries" / "finreg_multihop.json"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="cap docs (0=all)")
ap.add_argument("--nq", type=int, default=0, help="cap queries (0=all)")
ap.add_argument("--save-dir", default="/tmp/hrag_finreg_store")  # noqa: S108
ap.add_argument("--device", default="cpu", help="embedder device (cpu avoids GPU OOM)")
ap.add_argument("--llm-base-url", default="http://localhost:8012/v1")
ap.add_argument("--llm-model", default="Qwen3.6-27B")
args = ap.parse_args()

os.environ.setdefault("OPENAI_API_KEY", "dummy")

# --- Fair-comparison Korean embedder for HippoRAG ---
from hipporag.embedding_model.base import BaseEmbeddingModel, EmbeddingConfig


class BGEEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, global_config=None, embedding_model_name=None):
        super().__init__(global_config=global_config)
        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
        from sentence_transformers import SentenceTransformer

        self._st = SentenceTransformer(self.embedding_model_name, device=args.device)
        self.embedding_dim = self._st.get_sentence_embedding_dimension()
        self.embedding_config = EmbeddingConfig.from_dict(
            {"embedding_model_name": self.embedding_model_name, "norm": True,
             "encode_params": {"batch_size": 32, "instruction": "", "max_length": 8192}}
        )

    def batch_encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        emb = self._st.encode(
            texts, normalize_embeddings=True, batch_size=32,
            show_progress_bar=len(texts) > 64,
        )
        return np.asarray(emb, dtype=np.float32)


import hipporag  # noqa: F401  (load submodules)
from hipporag import embedding_model as _em

_orig_get = _em._get_embedding_model_class


def _patched_get(embedding_model_name="nvidia/NV-Embed-v2"):
    if "bge" in (embedding_model_name or "").lower():
        return BGEEmbeddingModel
    return _orig_get(embedding_model_name)


# HippoRAG.py copied the function reference at import time; patch every
# module namespace that holds a copy.
_patched_any = False
for _mod in list(sys.modules.values()):
    if _mod is not None and getattr(_mod, "_get_embedding_model_class", None) is not None:
        _mod._get_embedding_model_class = _patched_get
        _patched_any = True
assert _patched_any, "monkeypatch found no _get_embedding_model_class to replace"

# --- Data ---
arts = [json.loads(line) for line in RAW.open(encoding="utf-8") if line.strip()]
if args.limit:
    arts = arts[: args.limit]
docs = [a["text"] for a in arts]
text2did = {a["text"]: a["doc_id"] for a in arts}
print(f"docs: {len(docs)} (unique texts: {len(text2did)})", flush=True)

gt = json.load(MH.open(encoding="utf-8"))["queries"]
if args.nq:
    gt = gt[: args.nq]
queries = [q["query"] for q in gt]
print(f"multi-hop queries: {len(queries)}", flush=True)

# --- HippoRAG ---
from hipporag import HippoRAG

t0 = time.time()
hr = HippoRAG(
    save_dir=args.save_dir,
    llm_model_name=args.llm_model,
    llm_base_url=args.llm_base_url,
    embedding_model_name="BAAI/bge-m3",
)
print("HippoRAG constructed", flush=True)

hr.index(docs)
print(f"indexed {len(docs)} docs in {time.time() - t0:.0f}s", flush=True)

t1 = time.time()
sols = hr.retrieve(queries, num_to_retrieve=10)
print(f"retrieved {len(queries)} queries in {time.time() - t1:.0f}s", flush=True)

# --- Strict multi-hop scoring (every GT article must be in top-10) ---
solved = 0
for q, sol in zip(gt, sols):
    relevant = set(q["relevant_docs"])
    got = {text2did[d] for d in (sol.docs or []) if d in text2did}
    hit = relevant.issubset(got)
    solved += hit
    print(f"  [{q['qid']}] hit={hit} ({len(got)} mapped)", flush=True)

print(f"\nHippoRAG2 finreg multi-hop: {solved}/{len(gt)} ({solved / len(gt) * 100:.0f}%)", flush=True)
