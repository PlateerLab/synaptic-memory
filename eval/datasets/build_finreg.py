"""Fetch Korean financial-sector statutes from law.go.kr (headless browser).

No API key / OC needed — scrapes the public law viewer (lsInfoP.do). Each
조(article) becomes one document record; the resulting corpus is heavily
cross-referential ("제15조제2항에 따라", "별표 3") — exactly the multi-hop
territory where a tool-using agent should beat single-shot RAG.

Output: eval/data/finreg/raw.jsonl  (one JSON object per article)
    {doc_id, law, kind, article_no, title, text}

Usage:
    LD_LIBRARY_PATH=/tmp/plwlibs/.../x86_64-linux-gnu \
      uv run python eval/datasets/build_finreg.py --limit 3
    ... --with-decree        # also fetch 시행령 / 시행규칙
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parents[1] / "data" / "finreg" / "raw.jsonl"

# Major financial-sector statutes (금융위원회 소관 중심).
FINANCIAL_LAWS = [
    "은행법",
    "자본시장과 금융투자업에 관한 법률",
    "보험업법",
    "여신전문금융업법",
    "상호저축은행법",
    "금융지주회사법",
    "전자금융거래법",
    "금융소비자 보호에 관한 법률",
    "신용정보의 이용 및 보호에 관한 법률",
    "예금자보호법",
    "금융실명거래 및 비밀보장에 관한 법률",
    "특정 금융거래정보의 보고 및 이용 등에 관한 법률",
    "금융위원회의 설치 등에 관한 법률",
    "금융산업의 구조개선에 관한 법률",
    "금융회사의 지배구조에 관한 법률",
    "자산유동화에 관한 법률",
    "외국환거래법",
    "대부업 등의 등록 및 금융이용자 보호에 관한 법률",
    "신용협동조합법",
    "새마을금고법",
    "한국산업은행법",
    "중소기업은행법",
    "한국주택금융공사법",
    "서민의 금융생활 지원에 관한 법률",
    "온라인투자연계금융업 및 이용자 보호에 관한 법률",
]

# 제1조(목적) / 제21조의2(정보보호최고책임자) ...
_ART_RE = re.compile(r"^\s*(제\d+조(?:의\d+)?)\s*\(([^)]*)\)\s*(.*)", re.DOTALL)


def _doc_id(law: str, article_no: str) -> str:
    return hashlib.blake2b(f"{law}|{article_no}".encode(), digest_size=8).hexdigest()


async def _resolve_seq(page, name: str) -> str | None:
    """law.go.kr/법령/{name} redirects to lsInfoP.do?lsiSeq=NNN — grab NNN."""
    try:
        await page.goto(
            f"https://www.law.go.kr/법령/{name}",
            wait_until="networkidle",
            timeout=40000,
        )
        await page.wait_for_timeout(800)
    except Exception as exc:
        print(f"    ! goto failed: {exc}")
        return None
    for fr in page.frames:
        m = re.search(r"lsiSeq=(\d+)", fr.url)
        if m:
            return m.group(1)
    return None


async def _fetch_articles(page, seq: str, law: str, kind: str) -> list[dict]:
    """Open the law body and split every .lawcon block into an article record."""
    await page.goto(
        f"https://www.law.go.kr/LSW//lsInfoP.do?lsiSeq={seq}&urlMode=lsInfoP",
        wait_until="networkidle",
        timeout=40000,
    )
    await page.wait_for_timeout(1000)
    out: list[dict] = []
    for el in await page.query_selector_all(".lawcon"):
        text = (await el.inner_text()).strip()
        m = _ART_RE.match(text)
        if not m:
            continue
        article_no, title, _body = m.groups()
        out.append(
            {
                "doc_id": _doc_id(law, article_no),
                "law": law,
                "kind": kind,
                "article_no": article_no,
                "title": title.strip(),
                "text": re.sub(r"[ \t]+", " ", text).strip(),
            }
        )
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch financial statutes from law.go.kr")
    ap.add_argument("--limit", type=int, default=0, help="Cap number of base laws (0=all)")
    ap.add_argument(
        "--with-decree",
        action="store_true",
        help="Also fetch 시행령 / 시행규칙 for each law",
    )
    args = ap.parse_args()

    laws = FINANCIAL_LAWS[: args.limit] if args.limit else FINANCIAL_LAWS
    targets: list[tuple[str, str]] = []  # (name, kind)
    for law in laws:
        targets.append((law, "법률"))
        if args.with_decree:
            targets.append((f"{law} 시행령", "시행령"))
            targets.append((f"{law} 시행규칙", "시행규칙"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    seen_ids: set[str] = set()
    t0 = time.time()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        for i, (name, kind) in enumerate(targets, 1):
            seq = await _resolve_seq(page, name)
            if not seq:
                print(f"  [{i}/{len(targets)}] {name}: no lsiSeq — skip")
                continue
            try:
                arts = await _fetch_articles(page, seq, name, kind)
            except Exception as exc:
                print(f"  [{i}/{len(targets)}] {name}: fetch error {exc}")
                continue
            fresh = [a for a in arts if a["doc_id"] not in seen_ids]
            for a in fresh:
                seen_ids.add(a["doc_id"])
            records.extend(fresh)
            print(f"  [{i}/{len(targets)}] {name} (seq={seq}): {len(fresh)} articles")
            await page.wait_for_timeout(600)
        await browser.close()

    with open(OUT, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    laws_ok = len({r["law"] for r in records})
    print(
        f"\n{len(records)} articles from {laws_ok} documents "
        f"-> {OUT}  ({time.time() - t0:.1f}s)"
    )


if __name__ == "__main__":
    asyncio.run(main())
