"""Parse KRRA (마사회) documents using xgen-doc2chunk.

Walks the raw document directory, extracts text + chunks from each file,
and writes standardized JSONL to eval/data/parsed/krra/.

Usage:
    uv run python eval/scripts/parse_krra.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "마사회"
OUT_DIR = REPO_ROOT / "eval" / "data" / "parsed" / "krra"

SUPPORTED_EXTS = {
    ".pdf", ".txt", ".md", ".docx", ".doc", ".rtf",
    ".hwp", ".hwpx",
    ".xlsx", ".xls", ".csv", ".tsv",
    ".pptx", ".odp",
    ".png", ".jpg", ".jpeg",
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


@dataclass(slots=True)
class ParsedChunk:
    chunk_id: str
    doc_id: str
    text: str
    index: int
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass(slots=True)
class ParsedDocument:
    doc_id: str
    source_path: str
    title: str
    doc_type: str
    category: str
    year: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0


def _doc_id(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:16]


def _extract_year(filename: str) -> int | None:
    m = re.match(r"(\d{4})년도", filename)
    return int(m.group(1)) if m else None


def _extract_category(rel_path: Path) -> str:
    parts = rel_path.parts
    return parts[0] if parts else "unknown"


def _extract_title(filename: str) -> str:
    # Remove year prefix and extension
    name = re.sub(r"^\d{4}년도_", "", filename)
    name = Path(name).stem
    # Remove common prefixes
    name = re.sub(r"^\(본문\)\s*", "", name)
    name = re.sub(r"^\(붙임[#\d]*\)\s*", "", name)
    name = re.sub(r"^붙임\d*\s*", "", name)
    name = re.sub(r"^\[붙임\d*\]\s*", "", name)
    name = re.sub(r"^\(별첨\d*\)\s*", "", name)
    return name.strip() or filename


def parse_all() -> None:
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs_path = OUT_DIR / "documents.jsonl"
    chunks_path = OUT_DIR / "chunks.jsonl"
    errors_path = OUT_DIR / "errors.jsonl"

    from xgen_doc2chunk import DocumentProcessor

    processor = DocumentProcessor()

    files = [
        f for f in sorted(RAW_DIR.rglob("*"))
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_EXTS
        and not f.name.startswith(".")
        and "/PDF/" not in str(f)  # 한글(HWP)과 중복 — PDF 폴더 제외
        and "/PDF" not in str(f.parent.name)
    ]

    print(f"Found {len(files)} parseable files in {RAW_DIR}")

    total_docs = 0
    total_chunks = 0
    total_errors = 0
    start = time.time()

    with (
        open(docs_path, "w", encoding="utf-8") as docs_f,
        open(chunks_path, "w", encoding="utf-8") as chunks_f,
        open(errors_path, "w", encoding="utf-8") as errors_f,
    ):
        for i, fpath in enumerate(files):
            rel = fpath.relative_to(RAW_DIR)
            doc_id = _doc_id(str(rel))
            category = _extract_category(rel)
            year = _extract_year(fpath.name)
            title = _extract_title(fpath.name)
            doc_type = fpath.suffix.lower().lstrip(".")

            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - start
                print(
                    f"  [{i+1}/{len(files)}] {elapsed:.0f}s "
                    f"docs={total_docs} chunks={total_chunks} errors={total_errors}"
                )

            try:
                result = processor.extract_chunks(
                    str(fpath),
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    include_position_metadata=True,
                )

                chunks_with_meta = list(result.chunks_with_metadata)
                if not chunks_with_meta:
                    # Fallback: try plain text extraction
                    text = processor.extract_text(str(fpath))
                    if text and text.strip():
                        chunks_with_meta = [{"text": text, "page_number": None}]

                if not chunks_with_meta:
                    errors_f.write(json.dumps({
                        "doc_id": doc_id,
                        "path": str(rel),
                        "error": "empty extraction",
                    }, ensure_ascii=False) + "\n")
                    total_errors += 1
                    continue

                doc = ParsedDocument(
                    doc_id=doc_id,
                    source_path=str(rel),
                    title=title,
                    doc_type=doc_type,
                    category=category,
                    year=year,
                    metadata={"original_filename": fpath.name},
                    chunk_count=len(chunks_with_meta),
                )
                docs_f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
                total_docs += 1

                for idx, chunk_data in enumerate(chunks_with_meta):
                    text = chunk_data.get("text", "") if isinstance(chunk_data, dict) else str(chunk_data)
                    if not text.strip():
                        continue
                    chunk = ParsedChunk(
                        chunk_id=f"{doc_id}_c{idx:04d}",
                        doc_id=doc_id,
                        text=text.strip(),
                        index=idx,
                        page_number=chunk_data.get("page_number") if isinstance(chunk_data, dict) else None,
                        line_start=chunk_data.get("line_start") if isinstance(chunk_data, dict) else None,
                        line_end=chunk_data.get("line_end") if isinstance(chunk_data, dict) else None,
                    )
                    chunks_f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                    total_chunks += 1

            except Exception as exc:
                errors_f.write(json.dumps({
                    "doc_id": doc_id,
                    "path": str(rel),
                    "error": str(exc)[:500],
                }, ensure_ascii=False) + "\n")
                total_errors += 1

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"KRRA Parse Complete — {elapsed:.1f}s")
    print(f"  Documents: {total_docs}")
    print(f"  Chunks:    {total_chunks}")
    print(f"  Errors:    {total_errors}")
    print(f"  Output:    {OUT_DIR.relative_to(REPO_ROOT)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parse_all()
