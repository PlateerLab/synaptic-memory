"""외부 벤치마크 데이터셋 다운로드 — HuggingFace에서 5종 IR/QA 데이터셋 수집.

실행: uv run python tests/benchmark/download_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _load_beir_dataset(hf_path: str, name: str, out_file: str) -> None:
    """BeIR 형식 데이터셋 (corpus/queries/qrels configs) 로드."""
    from datasets import get_dataset_split_names, load_dataset

    print(f"Downloading {name}...")

    # corpus
    corpus_split = get_dataset_split_names(hf_path, "corpus")[0]
    corpus_ds = load_dataset(hf_path, "corpus", split=corpus_split)
    corpus = {}
    for row in corpus_ds:
        corpus[str(row["_id"])] = {"title": row.get("title", ""), "text": row.get("text", "")}

    # queries
    queries_split = get_dataset_split_names(hf_path, "queries")[0]
    queries_ds = load_dataset(hf_path, "queries", split=queries_split)
    queries = {}
    for row in queries_ds:
        queries[str(row["_id"])] = row.get("text", "")

    # qrels
    qrels_split = get_dataset_split_names(hf_path, "qrels")[0]
    qrels_ds = load_dataset(hf_path, "qrels", split=qrels_split)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row.get("query-id", ""))
        cid = str(row.get("corpus-id", ""))
        score = row.get("score", 1)
        if qid and cid:
            qrels.setdefault(qid, {})[cid] = score

    out = {
        "name": name,
        "source": hf_path,
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "qrels_size": sum(len(v) for v in qrels.values()),
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }
    path = DATA_DIR / out_file
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  Saved: {path} (corpus={len(corpus)}, queries={len(queries)}, qrels={sum(len(v) for v in qrels.values())})")


def download_ko_strategyqa() -> None:
    _load_beir_dataset("mteb/Ko-StrategyQA", "Ko-StrategyQA", "ko_strategyqa.json")


def download_autorag_retrieval() -> None:
    _load_beir_dataset("mteb/AutoRAGRetrieval", "AutoRAGRetrieval", "autorag_retrieval.json")


def download_miracl_ko() -> None:
    """MIRACL Korean — dev split, passages만 추출."""
    from datasets import load_dataset

    print("Downloading MIRACL (ko) — dev split...")
    # miracl/miracl은 legacy script 방식이므로 miracl-ko-queries-22-12 사용
    try:
        dev_ds = load_dataset("miracl/miracl-ko-queries-22-12", split="dev")
    except Exception:
        print("  SKIP: MIRACL dataset not available in current format")
        return

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    corpus: dict[str, dict[str, str]] = {}

    for row in dev_ds:
        qid = str(row["query_id"])
        queries[qid] = row["query"]
        qrels[qid] = {}

        for pp in row.get("positive_passages", []):
            docid = str(pp["docid"])
            if docid not in corpus:
                corpus[docid] = {"title": pp.get("title", ""), "text": pp.get("text", "")}
            qrels[qid][docid] = 1

        for np_ in row.get("negative_passages", []):
            docid = str(np_["docid"])
            if docid not in corpus:
                corpus[docid] = {"title": np_.get("title", ""), "text": np_.get("text", "")}

    out = {
        "name": "MIRACL-ko (dev)",
        "source": "miracl/miracl (ko, dev)",
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "qrels_size": sum(len(v) for v in qrels.values()),
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }
    path = DATA_DIR / "miracl_ko.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  Saved: {path} (corpus={len(corpus)}, queries={len(queries)}, qrels={sum(len(v) for v in qrels.values())})")


def download_mrtydi_ko() -> None:
    """Mr. TyDi Korean — test split."""
    from datasets import load_dataset

    print("Downloading Mr. TyDi (ko) — test split...")
    try:
        ds = load_dataset("castorini/mr-tydi", "korean", split="test")
    except Exception:
        print("  SKIP: Mr. TyDi dataset not available in current format")
        return

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    corpus: dict[str, dict[str, str]] = {}

    for row in ds:
        qid = str(row["query_id"])
        queries[qid] = row["query"]
        qrels[qid] = {}

        for pp in row.get("positive_passages", []):
            docid = str(pp["docid"])
            corpus[docid] = {"title": pp.get("title", ""), "text": pp.get("text", "")}
            qrels[qid][docid] = 1

        for np_ in row.get("negative_passages", []):
            docid = str(np_["docid"])
            if docid not in corpus:
                corpus[docid] = {"title": np_.get("title", ""), "text": np_.get("text", "")}

    out = {
        "name": "Mr.TyDi-ko (test)",
        "source": "castorini/mr-tydi (korean, test)",
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "qrels_size": sum(len(v) for v in qrels.values()),
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }
    path = DATA_DIR / "mrtydi_ko.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  Saved: {path} (corpus={len(corpus)}, queries={len(queries)}, qrels={sum(len(v) for v in qrels.values())})")


def download_klue_mrc() -> None:
    """KLUE-MRC — QA→IR 변환, dev split."""
    from datasets import load_dataset

    print("Downloading KLUE-MRC — dev split...")
    ds = load_dataset("klue", "mrc", split="validation")

    corpus: dict[str, dict[str, str]] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for i, row in enumerate(ds):
        qid = f"klue_{i}"
        cid = f"klue_doc_{i}"
        queries[qid] = row["question"]
        corpus[cid] = {"title": row.get("title", ""), "text": row.get("context", "")}
        qrels[qid] = {cid: 1}

    out = {
        "name": "KLUE-MRC (dev)",
        "source": "klue/klue (mrc, validation)",
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "qrels_size": sum(len(v) for v in qrels.values()),
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }
    path = DATA_DIR / "klue_mrc.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  Saved: {path} (corpus={len(corpus)}, queries={len(queries)}, qrels={sum(len(v) for v in qrels.values())})")


def download_allganize_rag_ko_eval() -> None:
    """Allganize RAG Evaluation — 엔터프라이즈 5개 도메인 (금융/공공/의료/법률/커머스)."""
    from datasets import load_dataset

    print("Downloading Allganize RAG-Evaluation-Dataset-KO...")
    ds = load_dataset("allganize/RAG-Evaluation-Dataset-KO", split="test")

    corpus: dict[str, dict[str, str]] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for i, row in enumerate(ds):
        qid = f"allganize_{i}"
        # target_answer를 document로, question을 query로
        cid = f"allganize_doc_{i}"
        queries[qid] = row["question"]
        corpus[cid] = {
            "title": f"[{row.get('domain', '')}] {row.get('target_file_name', '')}",
            "text": row.get("target_answer", ""),
        }
        qrels[qid] = {cid: 1}

    out = {
        "name": "Allganize RAG-Eval-KO",
        "source": "allganize/RAG-Evaluation-Dataset-KO",
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "qrels_size": sum(len(v) for v in qrels.values()),
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }
    path = DATA_DIR / "allganize_rag_eval.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  Saved: {path} (corpus={len(corpus)}, queries={len(queries)})")


def download_allganize_rag_ko() -> None:
    """Allganize rag-ko — Golden + Negative context 포함."""
    from datasets import load_dataset

    print("Downloading Allganize rag-ko...")
    ds = load_dataset("allganize/rag-ko", split="test")

    corpus: dict[str, dict[str, str]] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    for i, row in enumerate(ds):
        qid = f"ragko_{i}"
        queries[qid] = row["human"]

        # system에 context가 들어있음 — golden context를 corpus로
        cid = f"ragko_doc_{i}"
        corpus[cid] = {
            "title": row.get("answer_context_title", ""),
            "text": row.get("answer_context_summary", "") or row.get("system", ""),
        }
        qrels[qid] = {cid: 1}

    out = {
        "name": "Allganize rag-ko",
        "source": "allganize/rag-ko (test)",
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "qrels_size": sum(len(v) for v in qrels.values()),
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }
    path = DATA_DIR / "allganize_rag_ko.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  Saved: {path} (corpus={len(corpus)}, queries={len(queries)})")


def download_hotpotqa() -> None:
    """HotPotQA — multi-hop QA 데이터셋 (영어).

    Cognee 벤치마크 비교용. distractor 설정 사용.
    각 question의 context에 10개 문서(relevant + distractor)가 포함됨.
    supporting_facts로 relevant 문서 식별.

    두 가지 서브셋 생성:
      - hotpotqa_24.json: Cognee와 동일 규모 (24 queries)
      - hotpotqa.json: 200 queries (더 큰 평가 셋)
    """
    from datasets import load_dataset

    print("Downloading HotPotQA (distractor, validation)...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    import random

    # 각 row를 파싱하여 per-question 데이터 구축
    all_questions: list[dict] = []
    for row in ds:
        qid = f"hotpot_{row['id']}"
        titles = row["context"]["title"]
        sentences_list = row["context"]["sentences"]
        sf_titles = set(row["supporting_facts"]["title"])

        docs: dict[str, dict[str, str]] = {}
        rels: dict[str, int] = {}
        for title, sents in zip(titles, sentences_list):
            doc_text = " ".join(sents)
            cid = f"hotpot_doc_{title}"
            docs[cid] = {"title": title, "text": doc_text}
            if title in sf_titles:
                rels[cid] = 1

        all_questions.append({
            "qid": qid,
            "question": row["question"],
            "type": row["type"],
            "docs": docs,
            "qrels": rels,
        })

    def _build_subset(selected: list[dict]) -> tuple[dict, dict, dict]:
        """선택된 question들의 context 문서만 모아서 corpus/queries/qrels 생성."""
        corpus: dict[str, dict[str, str]] = {}
        queries: dict[str, str] = {}
        qrels: dict[str, dict[str, int]] = {}
        for q in selected:
            queries[q["qid"]] = q["question"]
            qrels[q["qid"]] = q["qrels"]
            corpus.update(q["docs"])
        return corpus, queries, qrels

    # ── 200 queries 셋 ──
    random.seed(42)
    sampled_200 = random.sample(all_questions, min(200, len(all_questions)))
    corpus_200, queries_200, qrels_200 = _build_subset(sampled_200)

    out_200 = {
        "name": "HotPotQA (distractor)",
        "source": "hotpot_qa (distractor, validation, 200 sample)",
        "corpus_size": len(corpus_200),
        "query_size": len(queries_200),
        "qrels_size": sum(len(v) for v in qrels_200.values()),
        "corpus": corpus_200,
        "queries": queries_200,
        "qrels": qrels_200,
    }
    path_200 = DATA_DIR / "hotpotqa.json"
    with open(path_200, "w") as f:
        json.dump(out_200, f, ensure_ascii=False)
    print(f"  Saved: {path_200} (corpus={len(corpus_200)}, queries={len(queries_200)}, "
          f"qrels={sum(len(v) for v in qrels_200.values())})")

    # ── 24 queries 셋 (Cognee 비교용) ──
    # multi-hop 특성 다양성을 위해 bridge 16 + comparison 8
    bridge_qs = [q for q in all_questions if q["type"] == "bridge"]
    comparison_qs = [q for q in all_questions if q["type"] == "comparison"]
    random.seed(42)
    sampled_24 = (
        random.sample(bridge_qs, min(16, len(bridge_qs)))
        + random.sample(comparison_qs, min(8, len(comparison_qs)))
    )[:24]

    corpus_24, queries_24, qrels_24 = _build_subset(sampled_24)
    out_24 = {
        "name": "HotPotQA-24 (Cognee comparison)",
        "source": "hotpot_qa (distractor, validation, 24 sample — Cognee benchmark subset)",
        "corpus_size": len(corpus_24),
        "query_size": len(queries_24),
        "qrels_size": sum(len(v) for v in qrels_24.values()),
        "corpus": corpus_24,
        "queries": queries_24,
        "qrels": qrels_24,
        "metadata": {
            "cognee_human_correctness": 0.925,
            "note": "Cognee uses 24-question subset with LLM-based answer evaluation",
        },
    }
    path_24 = DATA_DIR / "hotpotqa_24.json"
    with open(path_24, "w") as f:
        json.dump(out_24, f, ensure_ascii=False)
    print(f"  Saved: {path_24} (corpus={len(corpus_24)}, queries={len(queries_24)}, "
          f"qrels={sum(len(v) for v in qrels_24.values())})")


def _load_multilingual_beir_dataset(
    hf_path: str,
    name: str,
    out_file: str,
    lang_prefix: str,
    *,
    max_corpus: int = 0,
) -> None:
    """다국어 BeIR 형식 데이터셋 로드 (ko-corpus, ko-queries, ko-qrels 등)."""
    from datasets import get_dataset_split_names, load_dataset

    print(f"Downloading {name}...")

    corpus_config = f"{lang_prefix}-corpus"
    queries_config = f"{lang_prefix}-queries"
    qrels_config = f"{lang_prefix}-qrels"

    # corpus
    try:
        corpus_split = get_dataset_split_names(hf_path, corpus_config)[0]
    except Exception:
        print(f"  SKIP: {name} — config '{corpus_config}' not available")
        return
    corpus_ds = load_dataset(hf_path, corpus_config, split=corpus_split)
    corpus = {}
    id_key = "_id" if "_id" in corpus_ds.column_names else "id"
    for row in corpus_ds:
        corpus[str(row[id_key])] = {"title": row.get("title", ""), "text": row.get("text", "")}

    # queries
    queries_split = get_dataset_split_names(hf_path, queries_config)[0]
    queries_ds = load_dataset(hf_path, queries_config, split=queries_split)
    queries = {}
    q_id_key = "_id" if "_id" in queries_ds.column_names else "id"
    for row in queries_ds:
        queries[str(row[q_id_key])] = row.get("text", "")

    # qrels
    qrels_split = get_dataset_split_names(hf_path, qrels_config)[0]
    qrels_ds = load_dataset(hf_path, qrels_config, split=qrels_split)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row.get("query-id", ""))
        cid = str(row.get("corpus-id", ""))
        score = row.get("score", 1)
        if qid and cid:
            qrels.setdefault(qid, {})[cid] = score

    # 대규모 corpus 샘플링: qrels 관련 문서 + 랜덤 negative
    if max_corpus > 0 and len(corpus) > max_corpus:
        import random
        random.seed(42)
        relevant_ids = set()
        for rels in qrels.values():
            relevant_ids.update(rels.keys())

        sampled_corpus = {cid: corpus[cid] for cid in relevant_ids if cid in corpus}

        remaining = [cid for cid in corpus if cid not in relevant_ids]
        n_neg = max_corpus - len(sampled_corpus)
        if n_neg > 0 and remaining:
            neg_sample = random.sample(remaining, min(n_neg, len(remaining)))
            for cid in neg_sample:
                sampled_corpus[cid] = corpus[cid]

        print(f"  Sampled corpus: {len(corpus)} → {len(sampled_corpus)} "
              f"(relevant={len(relevant_ids & set(corpus.keys()))}, negative={len(sampled_corpus) - len(relevant_ids & set(sampled_corpus.keys()))})")
        corpus = sampled_corpus

    out = {
        "name": name,
        "source": hf_path,
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "qrels_size": sum(len(v) for v in qrels.values()),
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }
    path = DATA_DIR / out_file
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  Saved: {path} (corpus={len(corpus)}, queries={len(queries)}, qrels={sum(len(v) for v in qrels.values())})")


# ── BeIR 영문 데이터셋 ──


def _load_mteb_beir_dataset(hf_path: str, name: str, out_file: str, *, qrels_split: str = "test") -> None:
    """MTEB BeIR 형식 데이터셋 — corpus/queries config + default config(=qrels)."""
    from datasets import load_dataset

    print(f"Downloading {name}...")

    # corpus
    corpus_ds = load_dataset(hf_path, "corpus", split="corpus")
    corpus = {}
    for row in corpus_ds:
        corpus[str(row["_id"])] = {"title": row.get("title", ""), "text": row.get("text", "")}

    # queries
    queries_ds = load_dataset(hf_path, "queries", split="queries")
    queries = {}
    for row in queries_ds:
        queries[str(row["_id"])] = row.get("text", "")

    # qrels (default config, test split)
    qrels_ds = load_dataset(hf_path, "default", split=qrels_split)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row.get("query-id", ""))
        cid = str(row.get("corpus-id", ""))
        score = row.get("score", 1)
        if qid and cid:
            qrels.setdefault(qid, {})[cid] = int(score)

    out = {
        "name": name,
        "source": hf_path,
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "qrels_size": sum(len(v) for v in qrels.values()),
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }
    path = DATA_DIR / out_file
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  Saved: {path} (corpus={len(corpus)}, queries={len(queries)}, qrels={sum(len(v) for v in qrels.values())})")


def download_nfcorpus() -> None:
    """NFCorpus — 의료/영양 도메인 (MTEB BeIR)."""
    _load_mteb_beir_dataset("mteb/NFCorpus", "NFCorpus", "nfcorpus.json")


def download_scifact() -> None:
    """SciFact — 과학적 주장 검증 (MTEB BeIR)."""
    _load_mteb_beir_dataset("mteb/SciFact", "SciFact", "scifact.json")


def download_fiqa() -> None:
    """FiQA — 금융 QA (MTEB BeIR, 57K corpus)."""
    _load_mteb_beir_dataset("mteb/FiQA", "FiQA", "fiqa.json")


# ── MTEB 한국어 데이터셋 ──


def download_miracl_retrieval_ko() -> None:
    """MIRACLRetrieval Korean — MTEB 핵심 한국어 검색 벤치마크 (1.49M corpus → 샘플링)."""
    _load_multilingual_beir_dataset(
        "mteb/MIRACLRetrieval",
        "MIRACLRetrieval-ko",
        "miracl_retrieval_ko.json",
        "ko",
        max_corpus=10000,
    )


def download_multilongdoc_ko() -> None:
    """MultiLongDocRetrieval Korean — 장문서 검색 벤치마크."""
    _load_multilingual_beir_dataset(
        "mteb/MultiLongDocRetrieval",
        "MultiLongDocRetrieval-ko",
        "multilongdoc_ko.json",
        "ko",
    )


def download_xpqa_ko() -> None:
    """XPQARetrieval Korean — 다도메인 한국어 검색."""
    _load_multilingual_beir_dataset(
        "mteb/XPQARetrieval",
        "XPQARetrieval-ko",
        "xpqa_ko.json",
        "kor-kor",
    )


def download_publichealthqa_ko() -> None:
    """PublicHealthQA Korean — 의료/공중보건 도메인 (BeIR 형식, korean- prefix)."""
    from datasets import load_dataset

    print("Downloading PublicHealthQA (korean)...")

    corpus_ds = load_dataset("mteb/PublicHealthQA", "korean-corpus", split="test")
    queries_ds = load_dataset("mteb/PublicHealthQA", "korean-queries", split="test")
    qrels_ds = load_dataset("mteb/PublicHealthQA", "korean-qrels", split="test")

    corpus = {}
    for row in corpus_ds:
        corpus[str(row["_id"])] = {"title": row.get("title", ""), "text": row.get("text", "")}

    queries = {}
    for row in queries_ds:
        queries[str(row["_id"])] = row.get("text", "")

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row.get("query-id", ""))
        cid = str(row.get("corpus-id", ""))
        score = row.get("score", 1)
        if qid and cid:
            qrels.setdefault(qid, {})[cid] = score

    out = {
        "name": "PublicHealthQA-ko",
        "source": "mteb/PublicHealthQA (korean)",
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "qrels_size": sum(len(v) for v in qrels.values()),
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }
    path = DATA_DIR / "publichealthqa_ko.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  Saved: {path} (corpus={len(corpus)}, queries={len(queries)})")


def main() -> None:
    print("=" * 60)
    print("Downloading benchmark datasets from HuggingFace")
    print("=" * 60)

    # 기존 데이터셋
    download_ko_strategyqa()
    download_autorag_retrieval()
    download_miracl_ko()
    download_mrtydi_ko()
    download_klue_mrc()
    download_allganize_rag_ko_eval()
    download_allganize_rag_ko()
    download_publichealthqa_ko()
    download_hotpotqa()

    # 신규: BeIR 영문 3종
    download_nfcorpus()
    download_scifact()
    download_fiqa()

    # 신규: MTEB 한국어 3종
    download_miracl_retrieval_ko()
    download_multilongdoc_ko()
    download_xpqa_ko()

    print("\n" + "=" * 60)
    print("All datasets downloaded!")
    print("=" * 60)

    for f in sorted(DATA_DIR.glob("*.json")):
        if f.name in ("enterprise_scenario.json", "enterprise_scenario_v2.json",
                       "wikipedia_ko_tech.json", "github_commits.json", "github_issues.json"):
            continue
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
