"""v0.28 Q3 validation — sweep ``rerank_std_deadzone`` on the public benches.

Measures whether the per-query cross-encoder std deadzone recovers the
AutoRAG cross-encoder regression WITHOUT regressing the corpora where the
reranker genuinely helps (PublicHealthQA, HotPotQA, Allganize).

Config: vLLM Qwen3-Embedding-4B embedder (:8013) + bge-reranker-v2-m3
(torch, cuda:1). This is NOT the CLAUDE.md baseline embedder (bge-m3), so
absolute MRR differs — the experiment is the RELATIVE comparison of
deadzone>0 against this config's own deadzone=0 baseline.

Run:  uv run python examples/ablation/sweep_rerank_deadzone.py

RESULT (2026-06-04, Qwen3-Embedding-4B + bge-reranker-v2-m3):

    Dataset             dz=0.0   dz=1.0   dz=2.0   dz=3.0
    HotPotQA-24         0.9722   0.9722   0.9722   1.0000   reranker-helps
    Allganize RAG-ko    0.9790   0.9790   0.9790   0.9733   reranker-helps
    Allganize RAG-Eval  0.9544   0.9544   0.9558   0.9535   reranker-helps
    PublicHealthQA      0.8093   0.8093   0.8093   0.7891   reranker-helps (+0.20)
    AutoRAG             0.7983   0.7983   0.8107   0.8415   reranker-HARMFUL

  - The dz=0 AutoRAG regression reproduces (0.798 vs FTS-only ~0.895).
  - The deadzone monotonically RECOVERS AutoRAG (+0.043 by dz=3) — the
    mechanism works as designed.
  - dz=2.0 is a strict Pareto win for THIS config: AutoRAG +0.012, every
    reranker-helps corpus unchanged.
  - But full recovery needs dz>=3, where PHQA (-0.020) and Allganize-ko
    (-0.006) start to lose: AutoRAG's per-query std OVERLAPS PHQA's with
    this embedder. The code comment's "AutoRAG std~=0.53" is bge-m3-
    specific; the std SCALE shifts with the embedder, so no single fixed
    threshold is universally correct.
  - Conclusion: KEEP the deadzone opt-in (default 0.0). It is a useful
    per-deployment knob (dz~2 safe-positive here), not a fixed default.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "eval"))
sys.path.insert(0, str(_HERE.parent))

from local_bge import LocalBgeRerankerV2
from run_all import PUBLIC_DATASETS, run_public_dataset

from synaptic.extensions.embedder import OpenAIEmbeddingProvider

WANTED = {
    "AutoRAG",  # reranker HARMFUL — must recover toward FTS-only
    "PublicHealthQA",  # reranker helps +0.20 — must NOT regress
    "HotPotQA-24",  # reranker helps +0.10 — must NOT regress
    "Allganize RAG-ko",  # reranker helps +0.04 — must NOT regress
    "Allganize RAG-Eval",  # reranker helps +0.04 — must NOT regress
}
DEADZONES = [float(x) for x in os.environ.get("SWEEP_DZ", "0.0,1.0,2.0,3.0").split(",")]


async def main() -> None:
    datasets = [d for d in PUBLIC_DATASETS if d.name in WANTED]
    print(f"datasets: {[d.name for d in datasets]}")
    print(f"deadzones: {DEADZONES}")

    embedder = OpenAIEmbeddingProvider(
        api_base="http://localhost:8013/v1", model="Qwen3-Embedding-4B"
    )
    print("loading bge-reranker-v2-m3 on cuda:1 ...", flush=True)
    reranker = LocalBgeRerankerV2(device="cuda:1")

    # results[name][dz] = mrr
    results: dict[str, dict[float, float]] = {d.name: {} for d in datasets}
    for dz in DEADZONES:
        os.environ["SYNAPTIC_RERANK_STD_DEADZONE"] = str(dz)
        for cfg in datasets:
            r = await run_public_dataset(cfg, embedder=embedder, reranker=reranker)
            results[cfg.name][dz] = r.mrr
            print(f"  dz={dz:<4} {cfg.name:<20} MRR={r.mrr:.4f} ({r.elapsed:.1f}s)", flush=True)

    # Trade-off table
    print("\n" + "=" * 72)
    hdr = f"{'Dataset':<20}" + "".join(f"dz={dz:<6}" for dz in DEADZONES) + "Δ(best-dz0)"
    print(hdr)
    print("-" * 72)
    for name, row in results.items():
        base = row[DEADZONES[0]]
        best = max(row.values())
        cells = "".join(f"{row[dz]:<9.4f}" for dz in DEADZONES)
        print(f"{name:<20}{cells}{best - base:+.4f}")


if __name__ == "__main__":
    asyncio.run(main())
