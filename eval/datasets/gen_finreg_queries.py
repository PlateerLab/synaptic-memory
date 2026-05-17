"""Generate GT benchmark queries for the financial-statute corpus.

Reads ``eval/data/finreg/raw.jsonl`` and writes verifiable query files —
each record carries not just the question and ``relevant_docs`` but the
**full evidence text** of every GT article plus a **gold answer**, so a
human (or a stricter scorer) can audit whether the GT is actually correct.

Record shape:
    {
      "qid", "type", "query",
      "relevant_docs": [doc_id, ...],          # for the harness id-match
      "evidence": [{role, doc_id, law, article_no, title, text}, ...],
      "answer": "gold answer derived from the evidence",
      "cross_reference": "제N조"               # multi-hop only
    }

Two query types:

* **single-hop** — question answerable from one article. GT = [doc_id].
* **multi-hop** — article A cites article B in the same law. The question
  is about A's scenario but its complete answer also needs B, and the
  query surface exposes only A. Every candidate is verified against the
  real FTS index and kept only if a bare top-k search **fails to retrieve
  B** — mechanically guaranteeing the multi-hop set is unsolvable by
  single-shot RAG. GT = [A.doc_id, B.doc_id].

Usage:
    uv run python eval/datasets/gen_finreg_queries.py \
        --llm-base-url http://localhost:8012/v1 --model Qwen3.6-27B \
        --single 120 --multi 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
RAW = REPO_ROOT / "eval" / "data" / "finreg" / "raw.jsonl"
GRAPH = REPO_ROOT / "eval" / "data" / "finreg_graph.sqlite"
QDIR = REPO_ROOT / "eval" / "data" / "queries"

_REF_RE = re.compile(r"제(\d+)조(?:의(\d+))?")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SINGLE_PROMPT = """다음은 금융 법령의 한 조문이다.

[{law} {article_no}({title})]
{text}

이 조문을 근거로 다음을 만들어라:
1. question — 실무자가 물어볼 법한 자연스러운 질문 한 문장. 조문 번호("제N조")나
   법령명을 그대로 노출하지 말고 구어체 패러프레이즈로.
2. answer — 그 질문의 정확한 정답. 위 조문 내용만으로 도출하고, 근거를 구체적으로.

JSON 한 줄로만 출력 (다른 텍스트 금지):
{{"question": "...", "answer": "..."}}"""

_MULTI_PROMPT = """다음은 같은 금융 법령의 두 조문이다. 조문 A는 조문 B를 인용한다.

[조문 A — {a_no}({a_title})]
{a_text}

[조문 B (A가 인용하는 조문) — {b_no}({b_title})]
{b_text}

다음을 만들어라:
1. question — 조문 A의 상황에 대한 실무 질문 한 문장. 엄격한 규칙:
   - 완전히 답하려면 A의 내용 + A가 인용하는 B의 내용이 둘 다 필요해야 한다.
   - 질문 표면에는 A의 용어·상황만 담는다.
   - B의 조문 번호("{b_no}"), B의 제목, B에만 등장하는 고유 용어는 질문에 절대 넣지 마라.
   - 질문만 보고서는 B를 직접 키워드 검색할 단서가 없어야 한다.
2. answer — 그 질문의 정확한 정답. A와 B를 합쳐서 도출하고, 두 조문의 근거를 모두 명시.

JSON 한 줄로만 출력 (다른 텍스트 금지):
{{"question": "...", "answer": "..."}}"""


async def _gen_qa(client, model: str, prompt: str) -> tuple[str, str]:
    """Ask the LLM for a {question, answer} pair. Returns ('','') on failure."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"  ! gen error: {exc}")
        return "", ""
    m = _JSON_RE.search(raw)
    if not m:
        return "", ""
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "", ""
    q = str(obj.get("question", "")).strip()
    a = str(obj.get("answer", "")).strip()
    return (q, a) if q and a else ("", "")


def _evidence(art: dict, role: str | None = None) -> dict:
    """Full GT evidence record for an article."""
    ev = {
        "doc_id": art["doc_id"],
        "law": art["law"],
        "article_no": art["article_no"],
        "title": art["title"],
        "text": art["text"],
    }
    if role:
        ev = {"role": role, **ev}
    return ev


def _load_articles() -> list[dict]:
    with RAW.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _find_multihop_pairs(arts: list[dict]) -> list[tuple[dict, dict, str]]:
    """(A, B, ref) where A's text cites article B (== ref) within the same law."""
    by_law: dict[str, dict[str, dict]] = {}
    for a in arts:
        by_law.setdefault(a["law"], {})[a["article_no"]] = a
    pairs: list[tuple[dict, dict, str]] = []
    for a in arts:
        law_idx = by_law[a["law"]]
        for m in _REF_RE.finditer(a["text"]):
            ref = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
            if ref != a["article_no"] and ref in law_idx:
                pairs.append((a, law_idx[ref], ref))
                break  # one cross-ref per source article
    return pairs


async def _gen_single(client, model, substantive, n) -> list[dict]:
    sample = random.sample(substantive, min(n, len(substantive)))
    out: list[dict] = []
    for i, a in enumerate(sample, 1):
        q, ans = await _gen_qa(
            client,
            model,
            _SINGLE_PROMPT.format(
                law=a["law"], article_no=a["article_no"], title=a["title"],
                text=a["text"][:1400],
            ),
        )
        if q:
            out.append(
                {
                    "qid": f"s{len(out) + 1:03d}",
                    "type": "single_hop",
                    "query": q,
                    "relevant_docs": [a["doc_id"]],
                    "evidence": [_evidence(a)],
                    "answer": ans,
                }
            )
        if i % 30 == 0:
            print(f"  single-hop {i}/{len(sample)} (kept {len(out)})")
    return out


async def _gen_multi(client, model, pairs, n, *, top_k: int) -> list[dict]:
    """Generate multi-hop queries, keeping only FTS-verified RAG-hard ones."""
    from synaptic.backends.sqlite_graph import SqliteGraphBackend

    backend = SqliteGraphBackend(str(GRAPH))
    await backend.connect()

    out: list[dict] = []
    attempts = leaked = no_entry = 0
    for a, b, ref in pairs:
        if len(out) >= n:
            break
        attempts += 1
        q, ans = await _gen_qa(
            client,
            model,
            _MULTI_PROMPT.format(
                a_no=a["article_no"], a_title=a["title"], a_text=a["text"][:1100],
                b_no=b["article_no"], b_title=b["title"], b_text=b["text"][:1100],
            ),
        )
        if not q:
            continue
        # FTS verification — query must retrieve A (entry point) but NOT B.
        nodes = await backend.search_fts(q, limit=top_k)
        found = {(nd.properties or {}).get("doc_id", "") for nd in nodes}
        if b["doc_id"] in found:
            leaked += 1
            continue  # RAG could reach B directly — not multi-hop
        if a["doc_id"] not in found:
            no_entry += 1
            continue  # A unreachable — agent has no entry hop either
        out.append(
            {
                "qid": f"m{len(out) + 1:03d}",
                "type": "multi_hop",
                "query": q,
                "relevant_docs": [a["doc_id"], b["doc_id"]],
                "evidence": [_evidence(a, "entry"), _evidence(b, "referenced")],
                "answer": ans,
                "cross_reference": ref,
            }
        )
        if attempts % 30 == 0:
            print(
                f"  multi-hop attempt {attempts}: kept {len(out)} "
                f"(leaked B {leaked}, no entry A {no_entry})"
            )
    await backend.close()
    print(
        f"  multi-hop done: {len(out)} kept / {attempts} attempts "
        f"(leaked B {leaked}, no entry A {no_entry})"
    )
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description="Generate finreg GT queries")
    ap.add_argument("--llm-base-url", default=None)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--single", type=int, default=120, help="# single-hop queries")
    ap.add_argument("--multi", type=int, default=120, help="# multi-hop queries (kept)")
    ap.add_argument("--top-k", type=int, default=10, help="FTS top-k for the RAG-hard filter")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not RAW.exists():
        print(f"ERROR: {RAW} not found — run build_finreg.py first.")
        return

    random.seed(args.seed)
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    client = AsyncOpenAI(base_url=args.llm_base_url) if args.llm_base_url else AsyncOpenAI()

    arts = _load_articles()
    substantive = [a for a in arts if len(a["text"]) >= 180]
    print(f"{len(arts)} articles ({len(substantive)} substantive)")

    single = await _gen_single(client, args.model, substantive, args.single)

    pairs = _find_multihop_pairs(arts)
    random.shuffle(pairs)
    print(f"multi-hop candidate pairs: {len(pairs)}")
    multi = await _gen_multi(client, args.model, pairs, args.multi, top_k=args.top_k)

    QDIR.mkdir(parents=True, exist_ok=True)
    for fname, qs, desc in [
        ("finreg.json", single, "financial statutes — single-hop"),
        (
            "finreg_multihop.json",
            multi,
            "financial statutes — multi-hop cross-reference (FTS-verified RAG-hard)",
        ),
    ]:
        out = QDIR / fname
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {"dataset": "finreg", "description": desc, "id_field": "doc_id", "queries": qs},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"wrote {len(qs)} queries -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
