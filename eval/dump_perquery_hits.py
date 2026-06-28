"""Per-query deterministic single-shot hit@k dump (v0.29 E1 / Track A).

Fills the ``single_shot_hit`` axis of the routing GT
(``eval/routing_gt.py --hits-jsonl CORPUS=PATH``) with the deterministic
top-k id-hit of the cheap single-shot path — FTS-only (no embedder, no
reranker, GPU 0). Output is JSONL, one object per scored query::

    {"qid": "a001", "query": "...", "hit": false, "rank": null,
     "scores": [0.66, 0.61, ...], "has_table_row": false}

``hit`` is true iff any gold id appears in the deduped top-k retrieved
list; ``rank`` is the 1-based position of the first gold id within that
top-k window (null on miss) — exactly ``reciprocal_rank > 0`` over
``retrieved[:k]`` as ``tests/benchmark/metrics.py`` scores it, so a
file's hit-rate equals the ``hits/total`` column of ``eval/run_all.py``
when run at the same k.

``scores`` (top-k retrieval scores, descending) and ``has_table_row``
(any ``_table_name`` property in the top-k nodes) are the s2/s3 signal
inputs of ``eval/routing_signal_auc.py`` — the same file serves both
the routing-GT hit axis and the AUC harness's retrieval pass (its
loader reads scores/hit/has_table_row and ignores the rest).

Mirroring contract — the retrieval + matching logic below is a 1:1 copy
of ``eval/run_all.py`` so the GT axis stays consistent with the
published FTS-only baselines. run_all has *two* FTS-only code paths and
this module mirrors whichever applies to the input:

- **custom corpora** (pre-built sqlite + separate query file, e.g.
  assort/x2bee/krra): ``EvidenceSearch(backend, embedder=None,
  reranker=None, table_query_hints=<eval/data/profiles/{stem}.toml>)``
  with ``search(query, k=k*2, fts_seed_limit=30)``; gold matching keyed
  by the query file's ``id_field`` — ``node_title`` matches
  ``ev.node.title``, anything else (default ``doc_id``) matches
  ``ev.document_id or ev.node.properties["doc_id"]``. Queries with an
  empty ``relevant_docs`` are skipped, exactly as
  ``run_custom_dataset`` skips them (they never enter the MRR
  denominator there either).
- **public bench JSON** (corpus embedded in the same file, e.g.
  ``tests/benchmark/data/autorag_retrieval.json``): run_all's FTS-only
  branch does NOT use EvidenceSearch — ``run_public_dataset`` falls back
  to ``SynapticGraph.search(query, limit=k*2)`` when no embedder is
  wired, and matches ``hit.node.properties["doc_id"]`` only. Mirrored
  verbatim, including the corpus/query/qrels normalisation (BEIR dict
  and list formats).

qid convention: the query record's own ``qid`` (or ``id``) field when
present — these match the ``gt_datasets.xlsx`` sheet qids, which is what
lets ``routing_gt._merge_axes`` complete a 2x2 label. Without one, the
zero-based index fallback ``q{i:03d}`` applies — the same convention
``rag_vs_agent_answer.py --out-jsonl`` / the routing-GT finreg loader
use. BEIR dict-format queries use the dict key. NOTE: the finreg query
files DO carry ``qid`` fields (s001/m001...) but the finreg per-query
JSONL is keyed q000/q001... (rag_vs_agent_answer.py only consults
``id``); a finreg hits dump therefore needs ``--qid-mode index`` to
merge against the finreg JSONL axis in routing_gt.

Determinism: FTS-only retrieval over a fixed sqlite file is
deterministic — same ``--graph`` + ``--queries`` + ``--k`` produce a
byte-identical ``--out`` file. (Building a *new* graph from a public
JSON assigns fresh node ids, so determinism is per built graph file —
keep and reuse the built sqlite.)

Usage::

    # custom corpora (graph sqlite already exists)
    uv run python eval/dump_perquery_hits.py \
        --graph eval/data/assort_graph.sqlite \
        --queries eval/data/queries/assort_hard.json \
        --k 5 --out eval/data/hits/assort_hard_hits_fts_k5.jsonl

    # AutoRAG (public bench JSON; corpus is embedded in the file).
    # When --graph does not exist it is built first by ingesting the
    # embedded corpus FTS-only (CPU, ~1 min for the 720-doc AutoRAG
    # corpus) — the same transient ingest run_public_dataset performs
    # per run, persisted so reruns are deterministic:
    uv run python eval/dump_perquery_hits.py \
        --graph eval/data/autorag_graph.sqlite \
        --queries tests/benchmark/data/autorag_retrieval.json \
        --k 5 --out eval/data/hits/autorag_hits_fts_k5.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EVAL_DIR = Path(__file__).resolve().parent
PROFILE_DIR = EVAL_DIR / "data" / "profiles"


def _log(msg: str) -> None:
    print(f"[dump_perquery_hits] {msg}", file=sys.stderr)


# --- qid convention -------------------------------------------------


def resolve_qid(q: dict, index: int, *, qid_mode: str = "field") -> str:
    """The hits-file qid for one custom-format query record.

    ``field`` (default): the record's own ``qid``/``id`` when present —
    matching the gt_datasets.xlsx sheet qids so routing_gt's axis merge
    keys align — else the zero-based ``q{i:03d}`` index fallback.
    ``index``: always ``q{i:03d}`` — the rag_vs_agent_answer.py JSONL
    convention (needed to merge a finreg hits dump against the finreg
    per-query JSONL, whose qids ignore the file's ``qid`` field).
    """
    if qid_mode == "field":
        qid = q.get("qid") or q.get("id")
        if qid:
            return str(qid)
    return f"q{index:03d}"


# --- hit / rank (mirrors tests/benchmark/metrics.py reciprocal_rank) -


def hit_rank(retrieved: list[str], relevant: set[str], k: int) -> int | None:
    """1-based rank of the first gold id within ``retrieved[:k]``.

    ``rank is not None`` is exactly run_all's per-query hit criterion
    (``mrr > 0`` with ``retrieved[:k]``).
    """
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            return i + 1
    return None


# --- custom corpora (mirrors run_all.run_custom_dataset) ------------


def load_table_query_hints(graph_path: Path) -> dict[str, list[str]] | None:
    """DomainProfile table_query_hints for this corpus, if a profile exists.

    Mirrors run_all: ``assort_graph.sqlite`` → ``eval/data/profiles/
    assort.toml``; missing/broken profiles silently fall back to None.
    """
    corpus_stem = graph_path.stem.removesuffix("_graph")
    profile_path = PROFILE_DIR / f"{corpus_stem}.toml"
    if not profile_path.exists():
        return None
    try:
        from synaptic.extensions.domain_profile import DomainProfile

        profile = DomainProfile.load(profile_path)
        if profile.table_query_hints:
            return dict(profile.table_query_hints)
    except Exception:
        pass
    return None


async def _score_custom(
    graph_path: str,
    data: dict,
    *,
    k: int,
    qid_mode: str,
    table_query_hints: dict[str, list[str]] | None,
) -> tuple[list[dict], int]:
    """Per-query hits for a custom corpus (pre-built sqlite + query file).

    Returns ``(rows, skipped)`` — ``skipped`` counts the queries without
    ``relevant_docs`` (run_all drops them from the denominator; so do we:
    no gold set means no defined hit axis).
    """
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.extensions.evidence_search import EvidenceSearch

    queries = data.get("queries", [])
    id_field = data.get("id_field", "doc_id")

    backend = SqliteGraphBackend(graph_path)
    await backend.connect()
    try:
        searcher = EvidenceSearch(
            backend=backend,
            embedder=None,
            reranker=None,
            table_query_hints=table_query_hints,
        )

        rows: list[dict] = []
        skipped = 0
        for qi, q in enumerate(queries):
            qid = resolve_qid(q, qi, qid_mode=qid_mode)
            query_text = q.get("query", "")
            relevant = set(q.get("relevant_docs", []))
            if not relevant:
                skipped += 1
                continue

            result = await searcher.search(query_text, k=k * 2, fts_seed_limit=30)

            retrieved: list[str] = []
            if id_field == "node_title":
                for ev in result.evidence:
                    title = ev.node.title
                    if title and title not in retrieved:
                        retrieved.append(title)
            else:
                for ev in result.evidence:
                    doc_id = ev.document_id or (ev.node.properties or {}).get("doc_id", "")
                    if doc_id and doc_id not in retrieved:
                        retrieved.append(doc_id)

            rank = hit_rank(retrieved, relevant, k)
            top = result.evidence[:k]
            rows.append(
                {
                    "qid": qid,
                    "query": query_text,
                    "hit": rank is not None,
                    "rank": rank,
                    "scores": [round(float(ev.score), 6) for ev in top],
                    "has_table_row": any("_table_name" in (ev.node.properties or {}) for ev in top),
                }
            )
    finally:
        await backend.close()
    return rows, skipped


# --- public bench JSON (mirrors run_all.run_public_dataset) ----------


def is_public_dataset(data: dict) -> bool:
    """Public bench JSON embeds its corpus next to the queries."""
    return bool(data.get("corpus") or data.get("documents"))


def parse_public_corpus(data: dict) -> list[tuple[str, str, str]]:
    """Normalise the embedded corpus to (doc_id, title, text) triples."""
    raw_corpus = data.get("corpus", data.get("documents", []))
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
    return corpus


def parse_public_queries(data: dict) -> list[tuple[str, str, set[str]]]:
    """Normalise queries/qrels to (qid, text, relevant_ids) triples.

    Both BEIR dict format (``queries={qid: text}`` + ``qrels={qid:
    {doc_id: score}}``) and list format are supported — copied from
    ``run_public_dataset`` so the same queries are kept/dropped.
    """
    queries = data.get("queries", [])
    qrels = data.get("relevant_docs", data.get("qrels", {}))
    query_list: list[tuple[str, str, set[str]]] = []

    if isinstance(queries, dict):
        for qid, text in queries.items():
            rel = qrels.get(qid, {})
            if isinstance(rel, dict):
                relevant = set(str(x) for x in rel.keys())
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
                relevant = set(str(x) for x in rel_raw.keys())
            elif isinstance(rel_raw, list):
                relevant = set(str(x) for x in rel_raw)
            else:
                continue
            if relevant and text:
                query_list.append((qid, text, relevant))
    return query_list


async def _build_public_graph(graph_path: str, corpus: list[tuple[str, str, str]]) -> None:
    """Ingest an embedded public corpus into a new sqlite graph, FTS-only.

    The same ``graph.add(title=title or doc_id, content=text,
    properties={"doc_id": doc_id})`` ingest ``run_public_dataset``
    performs per run — persisted at ``graph_path`` so subsequent dumps
    are deterministic against one fixed graph file.
    """
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.graph import SynapticGraph

    backend = SqliteGraphBackend(graph_path)
    await backend.connect()
    try:
        graph = SynapticGraph(backend)
        for doc_id, title, text in corpus:
            if not text and not title:
                continue
            await graph.add(title=title or doc_id, content=text, properties={"doc_id": doc_id})
    finally:
        await backend.close()


async def _score_public(
    graph_path: str,
    query_list: list[tuple[str, str, set[str]]],
    *,
    k: int,
) -> list[dict]:
    """Per-query hits against an (existing) graph built from a public corpus.

    run_all's FTS-only public path is ``SynapticGraph.search`` (NOT
    EvidenceSearch — that branch only activates with an embedder or
    reranker), matching ``properties["doc_id"]``.
    """
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.graph import SynapticGraph

    backend = SqliteGraphBackend(graph_path)
    await backend.connect()
    try:
        graph = SynapticGraph(backend)
        rows: list[dict] = []
        for qid, query_text, relevant in query_list:
            result = await graph.search(query_text, limit=k * 2)
            retrieved: list[str] = []
            for hit in result.nodes:
                doc_id = (hit.node.properties or {}).get("doc_id", "")
                if doc_id and doc_id not in retrieved:
                    retrieved.append(doc_id)
            rank = hit_rank(retrieved, relevant, k)
            top = result.nodes[:k]
            rows.append(
                {
                    "qid": qid,
                    "query": query_text,
                    "hit": rank is not None,
                    "rank": rank,
                    # graph.search exposes activation, not the EvidenceSearch
                    # score — both are descending relevance, which is all the
                    # s2 shape signals (flatness / margin) consume.
                    "scores": [round(float(h.activation), 6) for h in top],
                    "has_table_row": any("_table_name" in (h.node.properties or {}) for h in top),
                }
            )
    finally:
        await backend.close()
    return rows


def _count_public_queries(data: dict) -> int:
    queries = data.get("queries", [])
    return len(queries) if isinstance(queries, dict | list) else 0


# --- entry point -----------------------------------------------------


def dump_hits(
    graph_path: Path,
    queries_path: Path,
    *,
    k: int = 5,
    out: Path | None = None,
    qid_mode: str = "field",
) -> list[dict]:
    """Dump per-query single-shot hits; optionally write them as JSONL.

    Dispatches on the queries file: an embedded ``corpus`` key means the
    public-bench mirror path (auto-building ``graph_path`` from that
    corpus when missing), otherwise the custom-corpus mirror path.
    Returns the rows in query-file order; the written file is
    byte-identical across runs for the same inputs. Synchronous wrapper —
    all file I/O happens here, the async cores only await backend calls.
    """
    graph_path = Path(graph_path)
    queries_path = Path(queries_path)
    data = json.loads(queries_path.read_text(encoding="utf-8"))

    if is_public_dataset(data):
        query_list = parse_public_queries(data)
        skipped = _count_public_queries(data) - len(query_list)
        if not graph_path.exists():
            corpus = parse_public_corpus(data)
            if not corpus:
                raise SystemExit(
                    f"graph not found and no embedded corpus to build it: {graph_path}"
                )
            _log(f"building graph from embedded corpus ({len(corpus)} docs) → {graph_path}")
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            asyncio.run(_build_public_graph(str(graph_path), corpus))
            _log(f"built in {time.time() - t0:.0f}s")
        rows = asyncio.run(_score_public(str(graph_path), query_list, k=k))
    else:
        if not graph_path.exists():
            raise SystemExit(f"graph not found: {graph_path}")
        rows, skipped = asyncio.run(
            _score_custom(
                str(graph_path),
                data,
                k=k,
                qid_mode=qid_mode,
                table_query_hints=load_table_query_hints(graph_path),
            )
        )

    hits = sum(1 for r in rows if r["hit"])
    skipped_note = f" ({skipped} queries without gold ids skipped)" if skipped else ""
    _log(
        f"{queries_path.name}: {hits}/{len(rows)} hit@{k} ({hits / len(rows):.3f}){skipped_note}"
        if rows
        else f"{queries_path.name}: no scorable queries{skipped_note}"
    )

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Dump deterministic per-query single-shot hit@k (FTS-only) as JSONL."
    )
    ap.add_argument(
        "--graph",
        required=True,
        help="sqlite graph path (built when missing and the queries file embeds a public corpus)",
    )
    ap.add_argument(
        "--queries", required=True, help="query file (custom format or public bench JSON)"
    )
    ap.add_argument("--k", type=int, default=5, help="hit@k cutoff (default 5)")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument(
        "--qid-mode",
        choices=("field", "index"),
        default="field",
        help="'field' (default): the query file's qid/id field, index fallback; "
        "'index': always q{i:03d} — the rag_vs_agent_answer.py JSONL convention "
        "(use for finreg dumps that must merge against the finreg per-query JSONL)",
    )
    args = ap.parse_args(argv)

    dump_hits(
        Path(args.graph),
        Path(args.queries),
        k=args.k,
        out=Path(args.out),
        qid_mode=args.qid_mode,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
