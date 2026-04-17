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


def _hash_doc(title: str, text: str) -> str:
    """Stable doc_id based on content — dedupes across questions."""
    return hashlib.blake2b(
        (title + "||" + text).encode("utf-8"), digest_size=8
    ).hexdigest()


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(
        f"  → {path.relative_to(REPO_ROOT)}  "
        f"({size_mb:.1f} MB, {len(obj['corpus'])} docs, {len(obj['queries'])} queries)"
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


BUILDERS = {
    "hotpotqa_full": (build_hotpotqa, "hotpotqa_full.json"),
    "musique": (build_musique, "musique_dev.json"),
    "2wiki": (build_2wiki, "2wiki_dev.json"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        default=",".join(BUILDERS),
        help="comma-separated dataset names (default: all)",
    )
    args = p.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = [n for n in names if n not in BUILDERS]
    if unknown:
        print(f"Unknown datasets: {unknown}; available: {list(BUILDERS)}")
        sys.exit(1)

    for name in names:
        builder, filename = BUILDERS[name]
        out_path = OUT_DIR / filename
        print(f"\n=== {name} ===")
        builder(out_path)

    print("\nDone. JSON files are gitignored; re-run this script on any clean clone.")


if __name__ == "__main__":
    main()
