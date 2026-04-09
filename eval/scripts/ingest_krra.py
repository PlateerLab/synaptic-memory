"""Ingest parsed KRRA chunks into a SynapticGraph (Kuzu backend).

Reads eval/data/parsed/krra/{documents,chunks}.jsonl and builds a graph
with Document nodes, Chunk nodes, Category nodes, and edges.

Usage:
    uv run python eval/scripts/ingest_krra.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.backends.kuzu import KuzuBackend  # noqa: E402
from synaptic.graph import SynapticGraph  # noqa: E402
from synaptic.models import EdgeKind, NodeKind  # noqa: E402

PARSED_DIR = REPO_ROOT / "eval" / "data" / "parsed" / "krra"
GRAPH_DIR = REPO_ROOT / "eval" / "data" / "krra_graph.kuzu"


async def main() -> int:
    docs_path = PARSED_DIR / "documents.jsonl"
    chunks_path = PARSED_DIR / "chunks.jsonl"

    if not docs_path.exists():
        print(f"ERROR: {docs_path} not found. Run parse_krra.py first.")
        return 1

    # Load parsed data
    docs: list[dict] = []
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    chunks: list[dict] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded {len(docs)} documents, {len(chunks)} chunks")

    # Build chunk lookup: doc_id → [chunk_dicts]
    doc_chunks: dict[str, list[dict]] = {}
    for c in chunks:
        doc_chunks.setdefault(c["doc_id"], []).append(c)

    # Create graph
    import shutil

    if GRAPH_DIR.exists():
        shutil.rmtree(GRAPH_DIR)

    backend = KuzuBackend(str(GRAPH_DIR))
    await backend.connect()
    graph = SynapticGraph(backend)  # no ontology constraints → 범용 데이터 허용

    start = time.time()

    # Phase 1: Category nodes
    categories = sorted({d["category"] for d in docs})
    cat_ids: dict[str, str] = {}
    for cat_name in categories:
        node = await graph.add(
            title=cat_name,
            content=f"마사회 문서 카테고리: {cat_name}",
            kind=NodeKind.CONCEPT,
            tags=["category", "krra"],
        )
        cat_ids[cat_name] = node.id
    print(f"  Created {len(categories)} category nodes")

    # Phase 2: Document + Chunk nodes
    total_doc_nodes = 0
    total_chunk_nodes = 0

    for i, doc in enumerate(docs):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(docs)}] {elapsed:.0f}s — docs={total_doc_nodes} chunks={total_chunk_nodes}")

        # Document node
        doc_node = await graph.add(
            title=doc["title"],
            content="",  # content is in chunks
            kind=NodeKind.ENTITY,
            tags=["document", "krra", doc.get("doc_type", "")],
            source=doc["source_path"],
            properties={
                "doc_id": doc["doc_id"],
                "doc_type": doc.get("doc_type", ""),
                "year": str(doc["year"]) if doc.get("year") else "",
                "category": doc.get("category", ""),
                "original_filename": doc.get("metadata", {}).get("original_filename", ""),
                "chunk_count": str(doc.get("chunk_count", 0)),
            },
        )
        total_doc_nodes += 1

        # Link to category
        cat_id = cat_ids.get(doc.get("category", ""))
        if cat_id:
            await graph.link(doc_node.id, cat_id, kind=EdgeKind.PART_OF)

        # Chunk nodes
        doc_chunk_list = doc_chunks.get(doc["doc_id"], [])
        prev_chunk_id: str | None = None

        for chunk_data in sorted(doc_chunk_list, key=lambda x: x["index"]):
            chunk_node = await graph.add(
                title=f"{doc['title']} #{chunk_data['index']}",
                content=chunk_data["text"],
                kind=NodeKind.CHUNK,
                tags=["chunk", "krra"],
                source=doc["source_path"],
                properties={
                    "doc_id": doc["doc_id"],
                    "chunk_index": str(chunk_data["index"]),
                    "page_number": str(chunk_data.get("page_number") or ""),
                },
            )
            total_chunk_nodes += 1

            # Document → Chunk
            await graph.link(doc_node.id, chunk_node.id, kind=EdgeKind.CONTAINS)

            # Sequential chunk linking
            if prev_chunk_id:
                await graph.link(prev_chunk_id, chunk_node.id, kind=EdgeKind.NEXT_CHUNK)
            prev_chunk_id = chunk_node.id

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"KRRA Ingest Complete — {elapsed:.1f}s")
    print(f"  Categories:  {len(categories)}")
    print(f"  Documents:   {total_doc_nodes}")
    print(f"  Chunks:      {total_chunk_nodes}")
    print(f"  Graph path:  {GRAPH_DIR.relative_to(REPO_ROOT)}")
    print(f"{'='*60}")

    # Quick test: search
    print("\n[Quick search test]")
    for q in ["경마 운영계획", "인권경영", "정보기술 시스템"]:
        result = await graph.search(q, limit=3)
        hits = len(result.nodes)
        top = result.nodes[0].node.title if result.nodes else "-"
        print(f"  '{q}' → {hits} hits, top: {top}")

    await backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
