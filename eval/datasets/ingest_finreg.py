"""Ingest fetched financial statutes into a graph for the eval harness.

Reads ``eval/data/finreg/raw.jsonl`` (one 조/article per line, produced by
``build_finreg.py``), converts each article into a document record, and
streams them through the generic ``DocumentIngester`` — the same path
KRRA / assort use. No finreg-specific logic lives here; the only domain
config is ``eval/data/profiles/finreg.toml``.

Each article becomes its own retrievable document (``doc_id`` preserved
into ``properties.doc_id`` on both the document and its chunk), so the
GT query files can reference articles by ``doc_id`` directly.

Usage:
    uv run python eval/datasets/ingest_finreg.py --clean
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.extensions.document_ingester import DocumentIngester, JsonlDocumentSource
from synaptic.extensions.domain_profile import DomainProfile

RAW = REPO_ROOT / "eval" / "data" / "finreg" / "raw.jsonl"
DOCS = REPO_ROOT / "eval" / "data" / "finreg" / "finreg_docs.jsonl"
PROFILE = REPO_ROOT / "eval" / "data" / "profiles" / "finreg.toml"
GRAPH = REPO_ROOT / "eval" / "data" / "finreg_graph.sqlite"


def _convert(raw_path: Path, docs_path: Path) -> int:
    """raw article records -> DocumentIngester document records."""
    n = 0
    with raw_path.open(encoding="utf-8") as fin, docs_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            a = json.loads(line)
            doc = {
                "doc_id": a["doc_id"],
                "title": f"{a['law']} {a['article_no']}({a['title']})",
                "content": a["text"],
                "category": a["law"],
                "properties": {
                    "law": a["law"],
                    "kind": a["kind"],
                    "article_no": a["article_no"],
                },
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    return n


def _clean(path: Path) -> None:
    if path.exists():
        path.unlink()
    for sib in path.parent.glob(f"{path.name}.*"):
        try:
            sib.unlink() if sib.is_file() else shutil.rmtree(sib)
        except OSError:
            pass


async def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest financial statutes")
    ap.add_argument("--clean", action="store_true", help="Rebuild graph from scratch")
    args = ap.parse_args()

    if not RAW.exists():
        print(f"ERROR: {RAW} not found — run build_finreg.py first.")
        return 1

    n_docs = _convert(RAW, DOCS)
    print(f"Converted {n_docs} articles -> {DOCS}")

    if args.clean:
        _clean(GRAPH)
        print(f"Cleaned graph at {GRAPH}")

    profile = DomainProfile.load(PROFILE)
    print(f"Profile: {profile.name} (locale={profile.locale})")

    from synaptic.backends.sqlite_graph import SqliteGraphBackend

    backend = SqliteGraphBackend(str(GRAPH))
    await backend.connect()

    ingester = DocumentIngester(profile=profile, backend=backend, merge_strategy="replace")
    t0 = time.time()
    stats = await ingester.ingest(JsonlDocumentSource(DOCS))
    elapsed = time.time() - t0

    print(f"\n{'=' * 56}")
    print(f"Ingest complete — {elapsed:.1f}s")
    print(f"  Documents:  {stats.documents_ingested}")
    print(f"  Chunks:     {stats.chunks_created}")
    print(f"  Categories: {stats.categories_created}")
    print(f"  Edges:      {stats.edges_created}")
    print(f"  Graph:      {GRAPH}")
    print(f"{'=' * 56}")

    await backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
