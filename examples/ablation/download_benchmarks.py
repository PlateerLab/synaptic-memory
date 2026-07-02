"""Download Tier-1 English multi-hop retrieval benchmarks and convert
them to the BEIR-style JSON format that ``run_ablation.py`` already
consumes.

Datasets:
  * HotPotQA full dev (distractor setting, 7,405 queries)
  * MuSiQue-Ans dev (multi-hop with decomposition, 2,417 queries)
  * 2WikiMultiHopQA dev (~12k queries)

Each emits ``tests/benchmark/data/{name}.json`` with schema::

    {
      "name": "<pretty>",
      "source": "<huggingface path>",
      "corpus":   {doc_id: {"title": "...", "text": "..."}},
      "queries":  {qid: "question text"},
      "qrels":    {qid: {doc_id: 1}}
    }

The JSON files are gitignored (``tests/benchmark/data/*.json``); this
script is how you regenerate them.

Usage::

    pip install datasets
    python examples/ablation/download_benchmarks.py
    python examples/ablation/download_benchmarks.py --only hotpotqa_full
    python examples/ablation/download_benchmarks.py --only musique,2wiki
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tests" / "benchmark" / "data"
MSMARCO_DEFAULT_CORPUS_LIMIT = 1_000_000


def _hash_doc(title: str, text: str) -> str:
    """Stable doc_id based on content — dedupes across questions."""
    return hashlib.blake2b((title + "||" + text).encode("utf-8"), digest_size=8).hexdigest()


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(
        f"  → {_display_path(path)}  "
        f"({size_mb:.1f} MB, {len(obj['corpus'])} docs, {len(obj['queries'])} queries)"
    )


def _write_manifest(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    corpus_path = path.parent / str(obj.get("corpus_path", ""))
    corpus_size_mb = corpus_path.stat().st_size / (1024 * 1024)
    manifest_size_kb = path.stat().st_size / 1024
    print(
        f"  → {_display_path(path)}  "
        f"({manifest_size_kb:.1f} KB manifest, "
        f"{corpus_size_mb:.1f} MB corpus jsonl, "
        f"{obj['corpus_size']} docs, {len(obj['queries'])} queries)"
    )


# --- HotPotQA --------------------------------------------------------


def build_hotpotqa(out_path: Path) -> None:
    """Distractor setting — each question ships with 10 paragraphs
    (2 gold + 8 distractor). We deduplicate paragraphs across
    questions by (title, text) hash."""
    from datasets import load_dataset

    print("Loading hotpot_qa (distractor, validation)...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    corpus: dict[str, dict] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for ex in ds:
        qid = str(ex["id"])
        queries[qid] = str(ex["question"])

        # Index this question's 10 paragraphs into the shared corpus.
        title_to_docid: dict[str, str] = {}
        titles = ex["context"]["title"]
        sentences_list = ex["context"]["sentences"]
        for title, sents in zip(titles, sentences_list):
            text = " ".join(sents).strip()
            if not text:
                continue
            doc_id = _hash_doc(title, text)
            if doc_id not in corpus:
                corpus[doc_id] = {"title": str(title), "text": text}
            title_to_docid[str(title)] = doc_id

        # Gold = any doc whose title is in supporting_facts.
        rel: dict[str, int] = {}
        for sf_title in ex["supporting_facts"]["title"]:
            did = title_to_docid.get(str(sf_title))
            if did:
                rel[did] = 1
        if rel:
            qrels[qid] = rel

    # Drop queries with no resolvable gold (shouldn't happen but be safe).
    queries = {q: t for q, t in queries.items() if q in qrels}

    _write(
        out_path,
        {
            "name": "HotPotQA dev (distractor)",
            "source": "huggingface: hotpot_qa/distractor/validation",
            "corpus_size": len(corpus),
            "query_size": len(queries),
            "qrels_size": len(qrels),
            "corpus": corpus,
            "queries": queries,
            "qrels": qrels,
        },
    )


# --- MuSiQue ---------------------------------------------------------


def build_musique(out_path: Path) -> None:
    """MuSiQue-Ans dev split. Each question has 20 paragraphs
    (2-4 gold + distractors from 2Wiki). Uses an ``is_supporting``
    flag to mark gold paragraphs."""
    from datasets import load_dataset

    print("Loading musique (MuSiQue-Ans, validation)...")
    # Official release is under dgslibisey/MuSiQue; the Answerable
    # subset is the one used by HippoRAG2.
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    corpus: dict[str, dict] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for ex in ds:
        qid = str(ex["id"])
        queries[qid] = str(ex["question"])
        rel: dict[str, int] = {}
        for para in ex.get("paragraphs", []):
            title = str(para.get("title") or "").strip()
            text = str(para.get("paragraph_text") or para.get("text") or "").strip()
            if not text:
                continue
            doc_id = _hash_doc(title or "untitled", text)
            if doc_id not in corpus:
                corpus[doc_id] = {"title": title, "text": text}
            if para.get("is_supporting"):
                rel[doc_id] = 1
        if rel:
            qrels[qid] = rel

    queries = {q: t for q, t in queries.items() if q in qrels}

    _write(
        out_path,
        {
            "name": "MuSiQue-Ans dev",
            "source": "huggingface: dgslibisey/MuSiQue/validation",
            "corpus_size": len(corpus),
            "query_size": len(queries),
            "qrels_size": len(qrels),
            "corpus": corpus,
            "queries": queries,
            "qrels": qrels,
        },
    )


# --- 2WikiMultiHopQA -------------------------------------------------


def build_2wiki(out_path: Path) -> None:
    """2WikiMultiHopQA dev split. Similar shape to HotPotQA:
    a question, 10 context paragraphs (2 gold + 8 distractor)
    addressed by (title, sent_id) supporting facts."""
    from datasets import load_dataset

    print("Loading 2wikimultihop (validation)...")
    ds = load_dataset("voidful/2WikiMultihopQA", split="validation")

    corpus: dict[str, dict] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for ex in ds:
        qid = str(ex["_id"])
        queries[qid] = str(ex["question"])

        title_to_docid: dict[str, str] = {}
        ctx = ex["context"]
        # Two possible shapes: dict-of-lists or list-of-lists.
        titles = ctx.get("title") if isinstance(ctx, dict) else None
        contents = ctx.get("content") if isinstance(ctx, dict) else None
        if titles is None or contents is None:
            # Fallback — list of [title, [sent1, sent2, ...]] pairs.
            titles = [c[0] for c in ctx]
            contents = [c[1] for c in ctx]

        for title, sents in zip(titles, contents):
            text = " ".join(sents).strip() if isinstance(sents, list) else str(sents)
            if not text:
                continue
            doc_id = _hash_doc(str(title), text)
            if doc_id not in corpus:
                corpus[doc_id] = {"title": str(title), "text": text}
            title_to_docid[str(title)] = doc_id

        rel: dict[str, int] = {}
        sf = ex.get("supporting_facts", {})
        sf_titles = (
            sf.get("title")
            if isinstance(sf, dict)
            else [s[0] for s in sf]
            if isinstance(sf, list)
            else []
        )
        for sf_title in sf_titles:
            did = title_to_docid.get(str(sf_title))
            if did:
                rel[did] = 1
        if rel:
            qrels[qid] = rel

    queries = {q: t for q, t in queries.items() if q in qrels}

    _write(
        out_path,
        {
            "name": "2WikiMultihopQA dev",
            "source": "huggingface: voidful/2WikiMultihopQA/validation",
            "corpus_size": len(corpus),
            "query_size": len(queries),
            "qrels_size": len(qrels),
            "corpus": corpus,
            "queries": queries,
            "qrels": qrels,
        },
    )


# --- BEIR subset (English domain diversity for v0.18 generality check) -----


def _build_beir(corpus_repo: str, split: str, label: str, out_path: Path) -> None:
    """Generic BEIR-style builder.

    BEIR ships each dataset as 3 HuggingFace splits:
      - ``corpus`` (doc_id, title, text)
      - ``queries`` (qid, text)
      - ``qrels/{split}`` (qid, doc_id, score)

    The resulting JSON matches the schema run_all.py / run_tier1 expect.
    """
    from datasets import load_dataset

    print(f"Loading BEIR {label} ({corpus_repo}, split={split})...")

    corpus_ds = load_dataset(corpus_repo, "corpus", split="corpus")
    queries_ds = load_dataset(corpus_repo, "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{label}-qrels", split=split)

    corpus: dict[str, dict] = {}
    for row in corpus_ds:
        did = str(row["_id"])
        corpus[did] = {
            "title": str(row.get("title") or ""),
            "text": str(row.get("text") or ""),
        }

    queries: dict[str, str] = {}
    for row in queries_ds:
        qid = str(row["_id"])
        text = str(row.get("text") or "").strip()
        if text:
            queries[qid] = text

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row.get("score") or 0)
        if score <= 0:
            continue
        qrels.setdefault(qid, {})[did] = score

    queries = {q: t for q, t in queries.items() if q in qrels}

    _write(
        out_path,
        {
            "name": f"BEIR {label} {split}",
            "source": f"huggingface: {corpus_repo}",
            "corpus_size": len(corpus),
            "query_size": len(queries),
            "qrels_size": len(qrels),
            "corpus": corpus,
            "queries": queries,
            "qrels": qrels,
        },
    )


def build_trec_covid(out_path: Path) -> None:
    """BEIR TREC-COVID — biomedical domain, 50 queries, ~171k docs.
    Tests retrieval generality on English biomedical text — paraphrase-
    heavy, technical vocabulary, very different from Wikipedia."""
    _build_beir("BeIR/trec-covid", "test", "trec-covid", out_path)


def build_fiqa(out_path: Path) -> None:
    """BEIR FiQA — financial QA, 648 queries, ~57k docs.
    Tests retrieval on a third English domain (finance) — short
    factoid queries, varied document length."""
    _build_beir("BeIR/fiqa", "test", "fiqa", out_path)


def build_scifact(out_path: Path) -> None:
    """BEIR SciFact — scientific claim verification, 300 queries,
    ~5k docs. Small + tightly-bounded relevance."""
    _build_beir("BeIR/scifact", "test", "scifact", out_path)


def _build_beir_jsonl_shard(
    corpus_repo: str,
    split: str,
    label: str,
    out_path: Path,
    *,
    corpus_limit: int,
    numeric_docid_index: bool = False,
) -> None:
    """Build a large BEIR shard as metadata JSON + corpus JSONL.

    The small BEIR datasets fit comfortably in one JSON object. MS MARCO
    does not: the source corpus has millions of passages. This writer
    keeps all positive qrel docs for the selected split, then fills the
    shard with corpus-order distractors up to ``corpus_limit``.
    """
    from datasets import load_dataset

    if corpus_limit <= 0:
        raise ValueError("--large-corpus-limit must be positive")

    print(
        f"Loading BEIR {label} ({corpus_repo}, split={split}, "
        f"jsonl shard limit={corpus_limit:,})..."
    )

    queries_ds = load_dataset(corpus_repo, "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{label}-qrels", split=split)

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row.get("score") or 0)
        if score <= 0:
            continue
        qrels.setdefault(qid, {})[did] = score

    queries: dict[str, str] = {}
    for row in queries_ds:
        qid = str(row["_id"])
        text = str(row.get("text") or "").strip()
        if text and qid in qrels:
            queries[qid] = text

    qrels = {qid: rel for qid, rel in qrels.items() if qid in queries}
    gold_doc_ids = {did for rel in qrels.values() for did in rel}
    filler_budget = max(corpus_limit - len(gold_doc_ids), 0)

    def row_payload(row: dict, doc_id: str) -> dict[str, str]:
        return {
            "_id": doc_id,
            "title": str(row.get("title") or ""),
            "text": str(row.get("text") or ""),
        }

    corpus_path = out_path.with_suffix(".corpus.jsonl")
    written_gold: set[str] = set()
    written_docs = 0
    filler_docs = 0

    corpus_ds = load_dataset(
        corpus_repo,
        "corpus",
        split="corpus",
        streaming=not numeric_docid_index,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        if numeric_docid_index:
            for did in sorted(gold_doc_ids, key=lambda value: int(value)):
                row = corpus_ds[int(did)]
                if str(row.get("_id")) != did:
                    continue
                f.write(json.dumps(row_payload(row, did), ensure_ascii=False) + "\n")
                written_gold.add(did)
                written_docs += 1

        for row in corpus_ds:
            did = str(row["_id"])
            if did in gold_doc_ids:
                if did in written_gold:
                    continue
                written_gold.add(did)
            elif filler_docs < filler_budget:
                filler_docs += 1
            else:
                continue

            f.write(json.dumps(row_payload(row, did), ensure_ascii=False) + "\n")
            written_docs += 1

            if filler_docs >= filler_budget and len(written_gold) >= len(gold_doc_ids):
                break

    missing_gold = sorted(gold_doc_ids - written_gold)
    _write_manifest(
        out_path,
        {
            "name": f"BEIR {label} {split} large shard",
            "schema": "beir_jsonl_v1",
            "source": f"huggingface: {corpus_repo}",
            "source_corpus": "MS MARCO passage ranking (~8.8M passages)",
            "corpus_path": corpus_path.name,
            "corpus_limit": corpus_limit,
            "corpus_size": written_docs,
            "query_size": len(queries),
            "qrels_size": len(qrels),
            "qrels_rows": sum(len(rel) for rel in qrels.values()),
            "preserved_gold_docs": len(written_gold),
            "missing_gold_docs": missing_gold[:100],
            "queries": queries,
            "qrels": qrels,
        },
    )


def build_msmarco_passage(out_path: Path, *, corpus_limit: int) -> None:
    """BEIR MS MARCO passage dev — web-scale passage retrieval.

    The full source corpus is ~8.8M passages. The default local shard is
    1M passages, preserving validation positives before filling with
    distractors. Increase ``--large-corpus-limit`` for heavier runs.
    """
    _build_beir_jsonl_shard(
        "BeIR/msmarco",
        "validation",
        "msmarco",
        out_path,
        corpus_limit=corpus_limit,
        numeric_docid_index=True,
    )


BUILDERS = {
    "hotpotqa_full": (build_hotpotqa, "hotpotqa_full.json"),
    "musique": (build_musique, "musique_dev.json"),
    "2wiki": (build_2wiki, "2wiki_dev.json"),
    "trec_covid": (build_trec_covid, "trec_covid.json"),
    "fiqa": (build_fiqa, "fiqa.json"),
    "scifact": (build_scifact, "scifact.json"),
}
LARGE_BUILDERS = {
    "msmarco_passage": (build_msmarco_passage, "msmarco_passage.json"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        default=",".join(BUILDERS),
        help="comma-separated dataset names (default: all)",
    )
    p.add_argument(
        "--large-corpus-limit",
        type=int,
        default=MSMARCO_DEFAULT_CORPUS_LIMIT,
        help=(
            "Corpus rows to keep for large JSONL-sharded datasets such as "
            f"msmarco_passage (default: {MSMARCO_DEFAULT_CORPUS_LIMIT:,})."
        ),
    )
    args = p.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()]
    available = {**BUILDERS, **LARGE_BUILDERS}
    unknown = [n for n in names if n not in available]
    if unknown:
        print(f"Unknown datasets: {unknown}; available: {list(available)}")
        sys.exit(1)

    for name in names:
        if name in LARGE_BUILDERS:
            builder, filename = LARGE_BUILDERS[name]
            out_path = OUT_DIR / filename
            print(f"\n=== {name} ===")
            builder(out_path, corpus_limit=args.large_corpus_limit)
            continue
        builder, filename = BUILDERS[name]
        out_path = OUT_DIR / filename
        print(f"\n=== {name} ===")
        builder(out_path)

    print("\nDone. JSON files are gitignored; re-run this script on any clean clone.")


if __name__ == "__main__":
    main()
