"""Ingest parsed KRRA chunks into a graph via the generic pipeline.

Phase E of the refactor — this script contains NO KRRA-specific logic.
Everything domain-specific lives in ``eval/data/profiles/krra.toml``.
The script just:

1. Loads the KRRA ``DomainProfile`` from TOML
2. Opens the requested backend (SQLite graph by default, Kuzu optional)
3. Streams documents + chunks through the generic ``DocumentIngester``

The same script template will work for any other corpus by pointing
``--profile`` at a different TOML file and ``--docs`` / ``--chunks`` at
different JSONL files.

Usage::

    # Default: SQLite graph backend (zero-install, file-based)
    uv run python eval/scripts/ingest_krra.py

    # Explicit backend choice
    uv run python eval/scripts/ingest_krra.py --backend sqlite
    uv run python eval/scripts/ingest_krra.py --backend kuzu

    # Override paths (for reuse with other corpora)
    uv run python eval/scripts/ingest_krra.py \\
        --profile eval/data/profiles/some_other.toml \\
        --docs eval/data/parsed/other/documents.jsonl \\
        --chunks eval/data/parsed/other/chunks.jsonl \\
        --graph eval/data/other_graph.sqlite

Prerequisites:
    - ``eval/data/parsed/krra/{documents,chunks}.jsonl`` exists (from parse_krra.py)
    - ``eval/data/profiles/krra.toml`` exists
    - For sqlite backend: ``pip install synaptic-memory[sqlite]``
    - For kuzu backend:   ``pip install synaptic-memory[kuzu]``
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.extensions.document_ingester import (
    DocumentIngester,
    JsonlDocumentSource,
)
from synaptic.extensions.domain_profile import DomainProfile

DEFAULT_PROFILE = REPO_ROOT / "eval" / "data" / "profiles" / "krra.toml"
DEFAULT_DOCS = REPO_ROOT / "eval" / "data" / "parsed" / "krra" / "documents.jsonl"
DEFAULT_CHUNKS = REPO_ROOT / "eval" / "data" / "parsed" / "krra" / "chunks.jsonl"
DEFAULT_SQLITE_GRAPH = REPO_ROOT / "eval" / "data" / "krra_graph.sqlite"
DEFAULT_KUZU_GRAPH = REPO_ROOT / "eval" / "data" / "krra_graph.kuzu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend",
        choices=["sqlite", "kuzu"],
        default="sqlite",
        help="Graph backend to write to (default: sqlite)",
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Graph file/dir path. Defaults depend on --backend.",
    )
    parser.add_argument(
        "--merge",
        choices=["skip", "replace"],
        default="replace",
        help="Merge strategy when doc_id already exists (default: replace)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the graph file/dir before ingesting (full rebuild)",
    )
    parser.add_argument(
        "--embed-url",
        default=None,
        help="Embedding API base URL (e.g. http://14.6.220.78:11434/v1)",
    )
    parser.add_argument(
        "--embed-model",
        default="qwen3-embedding:4b",
        help="Embedding model name (default: qwen3-embedding:4b)",
    )
    parser.add_argument(
        "--entity-link",
        action="store_true",
        help="Run EntityLinker after ingestion (creates MENTIONS edges)",
    )
    return parser.parse_args()


def _clean_graph_path(path: Path) -> None:
    """Remove a graph file or directory and any sibling WAL / lock files."""
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for sibling in path.parent.glob(f"{path.name}.*"):
        try:
            if sibling.is_dir():
                shutil.rmtree(sibling)
            else:
                sibling.unlink()
        except OSError:
            pass


async def _open_backend(backend_name: str, graph_path: Path):
    """Instantiate and connect the chosen backend."""
    if backend_name == "sqlite":
        from synaptic.backends.sqlite_graph import SqliteGraphBackend

        backend = SqliteGraphBackend(str(graph_path))
        await backend.connect()
        return backend

    if backend_name == "kuzu":
        from synaptic.backends.kuzu import KuzuBackend

        backend = KuzuBackend(str(graph_path))
        await backend.connect()
        return backend

    msg = f"Unknown backend: {backend_name}"
    raise ValueError(msg)


def _rel(path: Path) -> str:
    """Display path relative to repo root if inside it, absolute otherwise."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


async def main() -> int:
    args = _parse_args()

    if not args.docs.exists():
        print(f"ERROR: {args.docs} not found. Run parse_krra.py first.")
        return 1
    if not args.profile.exists():
        print(f"ERROR: profile {args.profile} not found.")
        return 1

    graph_path = args.graph or (
        DEFAULT_SQLITE_GRAPH if args.backend == "sqlite" else DEFAULT_KUZU_GRAPH
    )

    if args.clean:
        print(f"Cleaning existing graph at {_rel(graph_path)}...")
        _clean_graph_path(graph_path)

    print(f"Loading profile: {_rel(args.profile)}")
    profile = DomainProfile.load(args.profile)
    print(f"  name={profile.name}  locale={profile.locale}")
    print(f"  ontology_hints: {len(profile.ontology_hints)} categories")

    print(f"Opening backend: {args.backend}  at {_rel(graph_path)}")
    backend = await _open_backend(args.backend, graph_path)

    print(f"Source docs:   {_rel(args.docs)}")
    print(f"Source chunks: {_rel(args.chunks)}")
    source = JsonlDocumentSource(args.docs, args.chunks)

    ingester = DocumentIngester(
        profile=profile,
        backend=backend,
        merge_strategy=args.merge,
    )

    start = time.time()
    stats = await ingester.ingest(source)
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"Ingest complete — {elapsed:.1f}s")
    print(f"  Documents ingested:  {stats.documents_ingested}")
    print(f"  Documents skipped:   {stats.documents_skipped}")
    print(f"  Chunks created:      {stats.chunks_created}")
    print(f"  Categories created:  {stats.categories_created}")
    print(f"  Edges created:       {stats.edges_created}")
    print(f"  Graph path:          {_rel(graph_path)}")
    print(f"{'=' * 60}")

    # Embedding pass — add embeddings to all chunk + document nodes.
    # Runs after ingestion because DocumentIngester saves nodes without
    # embeddings (the embedder is a search-time concern, not a storage
    # concern in the generic ingester). This pass is optional and skipped
    # when --embed-url is not provided.
    if args.embed_url:
        from synaptic.extensions.embedder import OpenAIEmbeddingProvider
        from synaptic.models import NodeKind

        embedder = OpenAIEmbeddingProvider(
            api_base=args.embed_url,
            model=args.embed_model,
        )
        print(f"\n[Embedding pass] {args.embed_url} model={args.embed_model}")

        nodes = await backend.list_nodes(kind=None, limit=100_000)
        embed_targets = [
            n
            for n in nodes
            if n.kind in (NodeKind.CHUNK, NodeKind.ENTITY, NodeKind.CONCEPT)
            or "document" in (n.tags or [])
        ]
        print(f"  Embedding {len(embed_targets)} nodes...")

        batch_size = 32
        embedded_count = 0
        embed_start = time.time()
        for i in range(0, len(embed_targets), batch_size):
            batch = embed_targets[i : i + batch_size]
            texts = [f"{n.title}\n{n.content[:300]}" if n.content else n.title for n in batch]
            try:
                vectors = await embedder.embed_batch(texts)
            except Exception as exc:
                print(f"  ⚠ batch {i // batch_size} failed: {exc}")
                continue

            for node, vec in zip(batch, vectors):
                if vec:
                    node.embedding = vec
                    await backend.save_node(node)
                    embedded_count += 1

            if (i // batch_size) % 20 == 0 and i > 0:
                print(f"  ... {embedded_count}/{len(embed_targets)}")

        embed_elapsed = time.time() - embed_start
        print(f"  Embedded {embedded_count}/{len(embed_targets)} nodes in {embed_elapsed:.1f}s")

    # Entity linking pass — creates MENTIONS edges from chunks to hub
    # entity nodes via DF-filtered phrase extraction. Gives the graph
    # expander an entity-mention path that's otherwise absent.
    if args.entity_link:
        from synaptic.extensions.entity_linker import EntityLinker
        from synaptic.extensions.phrase_extractor import create_phrase_extractor

        extractor = create_phrase_extractor(profile)
        linker = EntityLinker(extractor=extractor, profile=profile)
        print("\n[Entity linking pass]")
        link_stats = await linker.link(backend, source_kind=NodeKind.CHUNK)
        print(f"  Sources scanned:     {link_stats.source_nodes_scanned}")
        print(f"  Raw phrases:         {link_stats.raw_phrase_candidates}")
        print(f"  Kept (DF filtered):  {link_stats.kept_phrases}")
        print(f"  Hub nodes created:   {link_stats.phrase_nodes_created}")
        print(f"  MENTIONS edges:      {link_stats.mentions_edges_created}")
        print(f"  Elapsed:             {link_stats.elapsed_seconds:.1f}s")
        if link_stats.top_phrases_by_df:
            print(f"  Top phrases:         {link_stats.top_phrases_by_df[:10]}")

    # Quick sanity probe
    if stats.documents_ingested > 0:
        from synaptic.graph import SynapticGraph

        graph = SynapticGraph(backend)
        print("\n[Quick search probe]")
        for q in ["경마 운영계획", "인권경영", "정보기술 시스템"]:
            result = await graph.search(q, limit=3)
            hits = len(result.nodes)
            top = result.nodes[0].node.title[:60] if result.nodes else "-"
            print(f"  '{q}' → {hits} hits, top: {top}")

    await backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
