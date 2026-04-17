"""Per-query failure diagnostic for the Allganize FTS-only baseline.

Goal: surface **why** the 20/200 queries that miss are missing — token
overlap? language mix? query length? — so improvements can be
targeted, not sprayed.

Usage::

    python examples/ablation/failure_diagnostic.py

Writes a per-query JSON to ``examples/ablation/diagnostics/`` and
prints a failure-pattern summary.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "tests" / "benchmark" / "data" / "allganize_rag_ko.json"
OUT_DIR = Path(__file__).parent / "diagnostics"


@dataclass
class QueryDiag:
    qid: str
    query: str
    relevant: list[str]
    retrieved: list[str]
    hit_rank: int  # 0 means miss (no hit in top-10)
    reciprocal_rank: float
    query_len_chars: int
    query_len_tokens: int
    hangul_ratio: float
    has_number: bool
    has_english: bool
    notes: list[str] = field(default_factory=list)


def _hangul_ratio(s: str) -> float:
    if not s:
        return 0.0
    hangul = sum(1 for c in s if 0xAC00 <= ord(c) <= 0xD7A3)
    alpha = sum(1 for c in s if c.isalpha())
    return hangul / max(alpha, 1)


def _annotate(q: QueryDiag) -> None:
    if q.query_len_tokens <= 2:
        q.notes.append("very_short_query")
    if q.query_len_tokens >= 15:
        q.notes.append("very_long_query")
    if q.has_english and q.hangul_ratio > 0.3:
        q.notes.append("mixed_lang")
    if q.has_number:
        q.notes.append("contains_number")
    # Detect overly generic questions ("무엇인가", "어떻게 하는가")
    generic = ["무엇인가", "무엇입니까", "어떻게", "왜", "어떤", "언제", "어디"]
    if any(g in q.query for g in generic):
        q.notes.append("generic_question_form")


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Load corpus
    corpus = []
    for doc_id, doc in data["corpus"].items():
        corpus.append((str(doc_id), str(doc.get("title", "")), str(doc.get("text", ""))))

    # Build graph
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend)
    for doc_id, title, text in corpus:
        if not text and not title:
            continue
        await graph.add(title=title or doc_id, content=text, properties={"doc_id": doc_id})

    # Load queries (BEIR-style)
    raw_queries = data.get("queries", [])
    qrels = data.get("relevant_docs", data.get("qrels", {}))

    queries: list[tuple[str, str, set[str]]] = []
    if isinstance(raw_queries, dict):
        for qid, text in raw_queries.items():
            rel = qrels.get(qid, {})
            ids = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
            if ids and text:
                queries.append((str(qid), str(text), ids))
    elif isinstance(raw_queries, list):
        for q in raw_queries:
            qid = str(q.get("qid") or q.get("query_id") or q.get("_id") or "")
            text = str(q.get("query") or q.get("question") or "")
            rel = q.get("relevant_docs") or []
            ids = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
            if ids and text:
                queries.append((qid, text, ids))

    # Run + diagnose
    diags: list[QueryDiag] = []
    t0 = time.time()
    for qid, qtext, relevant in queries:
        result = await graph.search(qtext, limit=20)
        retrieved: list[str] = []
        for hit in result.nodes:
            did = (hit.node.properties or {}).get("doc_id", "")
            if did and did not in retrieved:
                retrieved.append(did)
        hit_rank = 0
        for i, did in enumerate(retrieved[:10], 1):
            if did in relevant:
                hit_rank = i
                break
        rr = 1.0 / hit_rank if hit_rank > 0 else 0.0
        tokens = [t for t in re.split(r"\s+", qtext) if t]
        q_diag = QueryDiag(
            qid=qid,
            query=qtext,
            relevant=sorted(relevant),
            retrieved=retrieved[:10],
            hit_rank=hit_rank,
            reciprocal_rank=rr,
            query_len_chars=len(qtext),
            query_len_tokens=len(tokens),
            hangul_ratio=_hangul_ratio(qtext),
            has_number=bool(re.search(r"\d", qtext)),
            has_english=bool(re.search(r"[A-Za-z]", qtext)),
        )
        _annotate(q_diag)
        diags.append(q_diag)

    elapsed = time.time() - t0

    # Write full detail
    per_query = OUT_DIR / "allganize_rag_ko_per_query.json"
    with open(per_query, "w", encoding="utf-8") as f:
        json.dump([d.__dict__ for d in diags], f, ensure_ascii=False, indent=2)

    # Summary stats
    misses = [d for d in diags if d.hit_rank == 0]
    weak = [d for d in diags if 0 < d.hit_rank > 3]  # hit but at rank 4+
    top1 = [d for d in diags if d.hit_rank == 1]

    print(f"Diagnostic for Allganize RAG-ko — {len(diags)} queries, {elapsed:.1f}s")
    print()
    print(f"  hit@10 = {len(diags) - len(misses)}/{len(diags)}")
    print(f"  rank=1 = {len(top1)} ({100 * len(top1) / len(diags):.1f}%)")
    print(f"  rank 4+ (weak hit) = {len(weak)}")
    print(f"  miss (no hit in top-10) = {len(misses)}")
    print()

    print("Failure pattern counts (misses only):")
    note_counter: Counter[str] = Counter()
    for d in misses:
        for n in d.notes:
            note_counter[n] += 1
        if not d.notes:
            note_counter["unclassified"] += 1
    for note, cnt in note_counter.most_common():
        print(f"  {note:30s} {cnt:3d} / {len(misses)}")
    print()

    print("Sample misses (first 10):")
    for d in misses[:10]:
        print(f"  [{d.qid}] {d.query!r}")
        print(f"     relevant: {d.relevant[:3]}  notes: {d.notes}")
    print()

    print("Sample weak hits (rank 4+, first 10):")
    for d in weak[:10]:
        print(f"  [{d.qid}] rank={d.hit_rank}  {d.query!r}")
        print(f"     relevant: {d.relevant[:1]}  retrieved top-3: {d.retrieved[:3]}")

    print()
    print(f"Full per-query diagnostic → {per_query.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
