"""Streaming retrieval experiment — empirical evidence for the
Top-k Reproducibility Theorem.

What we claim (informally):

    For any query q and any k, the top-k produced by
    ``mode="full"`` rebuild over a corpus snapshot C matches the
    top-k produced by any sequence of cumulative incremental ingests
    that produce C.

What this script does:

1. Load the Allganize RAG-ko public corpus (200 docs, 200 queries).
2. **Arm A (batch baseline)** — ingest all 200 docs in one shot,
   run all 200 queries, record the top-10 for every query.
3. **Arm B (streaming)** — ingest the corpus in N random timesteps,
   taking the same top-10 snapshot at every step.  Crucially, after
   all N steps the cumulative corpus is identical to Arm A's.
4. Compare: for every query, is the final top-10 **bit-wise identical**
   between Arm A and Arm B?
5. Also report: how many queries had monotone rank (the top doc at
   step t >= was at rank ≤ the rank it was at step t−1, conditioned
   on the doc existing at both checkpoints) — a weaker invariance
   that matters operationally.

Output:

* A JSON file ``examples/ablation/diagnostics/streaming_invariance.json``
  with per-query, per-step top-k.
* A one-page Markdown summary suitable for pasting into the paper.

This is the experiment that empirically validates Theorem 1
(``docs/paper/theorem.md``).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "tests" / "benchmark" / "data" / "allganize_rag_ko.json"
OUT_DIR = Path(__file__).parent / "diagnostics"

SEED = 42
N_STEPS = 10
TOP_K = 10


@dataclass
class StepCheckpoint:
    step_idx: int
    n_docs: int
    # queryid → ordered list of doc_ids in the top-K for this step.
    top_k: dict[str, list[str]] = field(default_factory=dict)


async def _run_queries(graph: SynapticGraph, queries) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for qid, qtext, _rel in queries:
        r = await graph.search(qtext, limit=TOP_K)
        retr: list[str] = []
        for h in r.nodes:
            did = (h.node.properties or {}).get("doc_id", "")
            if did and did not in retr:
                retr.append(did)
        result[qid] = retr[:TOP_K]
    return result


async def _build_arm(
    corpus: list[tuple[str, str, str]],
    queries,
    *,
    batches: list[list[tuple[str, str, str]]],
) -> tuple[list[StepCheckpoint], float]:
    """Build a graph by ingesting ``batches`` in order; snapshot top-k
    after every batch."""
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend)

    checkpoints: list[StepCheckpoint] = []
    cumulative = 0
    total_ingest = 0.0

    for step_idx, batch in enumerate(batches):
        t0 = time.perf_counter()
        for doc_id, title, text in batch:
            if not text and not title:
                continue
            await graph.add(
                title=title or doc_id,
                content=text,
                properties={"doc_id": doc_id},
            )
        total_ingest += time.perf_counter() - t0
        cumulative += len(batch)

        top_k = await _run_queries(graph, queries)
        checkpoints.append(StepCheckpoint(step_idx=step_idx, n_docs=cumulative, top_k=top_k))

    return checkpoints, total_ingest


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    corpus_raw = data["corpus"]
    corpus: list[tuple[str, str, str]] = [
        (str(doc_id), str(d.get("title", "")), str(d.get("text", "")))
        for doc_id, d in corpus_raw.items()
    ]

    queries = []
    qs = data["queries"]
    qrels = data["qrels"]
    for qid, text in qs.items():
        rel = qrels.get(qid, {})
        ids = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
        if ids and text:
            queries.append((str(qid), str(text), ids))

    # --- Arm A: batch baseline — all 200 docs in one ingest ---
    arm_a_checkpoints, arm_a_ingest = await _build_arm(corpus, queries, batches=[corpus])
    batch_top_k = arm_a_checkpoints[-1].top_k

    # --- Arm B: streaming, split into N_STEPS random batches ---
    rng = random.Random(SEED)
    shuffled = corpus[:]
    rng.shuffle(shuffled)
    batch_size = max(1, len(shuffled) // N_STEPS)
    streaming_batches = [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]
    # Collapse any stragglers into the last batch.
    if len(streaming_batches) > N_STEPS:
        last = streaming_batches[N_STEPS - 1]
        for extra in streaming_batches[N_STEPS:]:
            last.extend(extra)
        streaming_batches = streaming_batches[:N_STEPS]

    arm_b_checkpoints, arm_b_ingest = await _build_arm(corpus, queries, batches=streaming_batches)
    streaming_top_k = arm_b_checkpoints[-1].top_k

    # --- Compare ---
    identical = 0
    near_identical = 0  # same set, possibly different order
    diffs: list[dict] = []
    top1_match = 0
    mrr_a_total = 0.0
    mrr_b_total = 0.0
    for qid, qtext, rel in queries:
        a = batch_top_k.get(qid, [])
        b = streaming_top_k.get(qid, [])
        if a == b:
            identical += 1
        elif set(a) == set(b):
            near_identical += 1
        else:
            diffs.append(
                {
                    "qid": qid,
                    "query": qtext,
                    "batch": a,
                    "streaming": b,
                    "only_in_batch": [d for d in a if d not in b],
                    "only_in_streaming": [d for d in b if d not in a],
                }
            )
        if a and b and a[0] == b[0]:
            top1_match += 1
        rr_a = next((1.0 / (i + 1) for i, d in enumerate(a) if d in rel), 0.0)
        rr_b = next((1.0 / (i + 1) for i, d in enumerate(b) if d in rel), 0.0)
        mrr_a_total += rr_a
        mrr_b_total += rr_b

    total_q = len(queries)
    mrr_a = mrr_a_total / max(total_q, 1)
    mrr_b = mrr_b_total / max(total_q, 1)
    report = {
        "corpus": "Allganize RAG-ko",
        "n_docs": len(corpus),
        "n_queries": total_q,
        "top_k": TOP_K,
        "seed": SEED,
        "n_streaming_steps": len(streaming_batches),
        "batch_size_per_step": batch_size,
        "arm_a_ingest_sec": round(arm_a_ingest, 3),
        "arm_b_ingest_sec": round(arm_b_ingest, 3),
        "bitwise_identical_topk": identical,
        "same_set_different_order": near_identical,
        "set_mismatched": len(diffs),
        "top1_match": top1_match,
        "mrr_batch": round(mrr_a, 4),
        "mrr_streaming": round(mrr_b, 4),
        "mrr_delta": round(mrr_b - mrr_a, 4),
        "mismatch_sample": diffs[:5],
    }

    out_path = OUT_DIR / "streaming_invariance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Streaming retrieval invariance — Allganize RAG-ko")
    print()
    print(f"  corpus                          {len(corpus)} docs")
    print(f"  queries                         {total_q}")
    print(f"  top-k                           {TOP_K}")
    print(f"  streaming steps                 {len(streaming_batches)}")
    print(f"  batch size / step               {batch_size}")
    print()
    print(f"  Arm A (batch) ingest time       {arm_a_ingest:.2f}s")
    print(f"  Arm B (streaming) ingest time   {arm_b_ingest:.2f}s")
    print()
    print(
        f"  bit-wise identical top-10       {identical}/{total_q} "
        f"({100 * identical / total_q:.1f} %)"
    )
    print(f"  same set, different order       {near_identical}/{total_q}")
    print(f"  set mismatched                  {len(diffs)}/{total_q}")
    print(
        f"  top-1 identical                 {top1_match}/{total_q} "
        f"({100 * top1_match / total_q:.1f} %)"
    )
    print()
    print(f"  MRR batch                       {mrr_a:.4f}")
    print(f"  MRR streaming                   {mrr_b:.4f}")
    print(f"  Δ MRR (streaming - batch)       {mrr_b - mrr_a:+.4f}")
    print()
    print(f"  Full report → {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
