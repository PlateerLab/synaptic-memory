"""Generic document → graph ingester (Phase C).

Replaces domain-specific ingest scripts with a domain-agnostic pipeline
driven by a ``DomainProfile``. Per-corpus configuration (stopwords,
metadata patterns, category → NodeKind mapping) lives in a TOML profile
next to the corpus data, not in code.

Given:

- A :class:`CorpusSource` that yields :class:`DocumentRecord` objects
- A ``StorageBackend`` to write to
- A ``DomainProfile`` for ontology / NodeKind mapping

Builds a graph with:

- **Category nodes** (``NodeKind.CONCEPT``) — one per distinct category
- **Document nodes** — ``kind`` taken from ``profile.ontology_hints`` for
  the document's category (defaults to ``NodeKind.ENTITY``)
- **Chunk nodes** (``NodeKind.CHUNK``) — one per ``ChunkRecord``
- **Edges**:
  - ``PART_OF``: doc → category
  - ``CONTAINS``: doc → chunk
  - ``NEXT_CHUNK``: chunk → next chunk (sequential)

All ids are deterministic so re-ingesting the same corpus is idempotent
under the ``skip`` merge strategy.

Example::

    from synaptic.extensions.domain_profile import DomainProfile
    from synaptic.extensions.document_ingester import (
        DocumentIngester,
        JsonlDocumentSource,
    )

    profile = DomainProfile.load("profiles/my_corpus.toml")
    source = JsonlDocumentSource(
        "data/documents.jsonl",
        "data/chunks.jsonl",
    )
    ingester = DocumentIngester(profile=profile, backend=backend)
    stats = await ingester.ingest(source)
    print(stats.documents_ingested, stats.chunks_created)
"""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
)

if TYPE_CHECKING:
    from synaptic.extensions.domain_profile import DomainProfile
    from synaptic.protocols import StorageBackend


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if s else s


# --- Record types ---


@dataclass(slots=True)
class ChunkRecord:
    """A single chunk of a document."""

    chunk_id: str
    doc_id: str
    text: str
    index: int = 0
    page_number: int | None = None
    properties: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentRecord:
    """A single document ready for ingestion.

    ``content`` is optional — if ``chunks`` is populated, the document
    node itself stores the title (for FTS matching on queries that
    reference the document by name) while the full body text lives in
    the chunks.
    """

    doc_id: str
    title: str
    content: str = ""
    source: str = ""
    category: str = ""
    year: int | None = None
    properties: dict[str, str] = field(default_factory=dict)
    chunks: list[ChunkRecord] = field(default_factory=list)


# --- Sources ---


class CorpusSource(Protocol):
    """Iterator-based source of documents for ingestion.

    Implementations yield ``DocumentRecord`` instances one at a time.
    The ingester does not assume any ordering — categories are deduped
    on the fly and chunks within a document are sorted by ``index``.
    """

    def documents(self) -> Iterator[DocumentRecord]: ...


class JsonlDocumentSource:
    """Corpus source that reads from JSONL files.

    Supports two file layouts:

    1. **Docs-only** — a single JSONL file where each line is a full
       document (including inline ``chunks`` array).
    2. **Docs + separate chunks** — one JSONL with documents and a
       second JSONL with chunks referencing ``doc_id``. This matches
       the output of a two-stage parser that emits docs and chunks
       into different files.

    Expected doc JSON keys (all optional except ``doc_id``):
        ``doc_id``, ``title``, ``content``, ``source_path``,
        ``category``, ``year``, ``properties``, ``chunks``.

    Expected chunk JSON keys:
        ``chunk_id``, ``doc_id``, ``text``, ``index``, ``page_number``.

    Unknown keys are ignored. Files are NOT loaded fully into memory —
    documents stream one line at a time. Chunks, if supplied as a
    separate file, are loaded once into an in-memory dict keyed by
    ``doc_id`` because they must be joined back to their parent.
    """

    __slots__ = ("_chunks_path", "_docs_path")

    def __init__(
        self,
        docs_path: Path | str,
        chunks_path: Path | str | None = None,
    ) -> None:
        self._docs_path = Path(docs_path)
        self._chunks_path = Path(chunks_path) if chunks_path else None

    def documents(self) -> Iterator[DocumentRecord]:
        chunks_by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
        if self._chunks_path is not None and self._chunks_path.exists():
            with self._chunks_path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    c = json.loads(line)
                    doc_id = str(c.get("doc_id", ""))
                    if not doc_id:
                        continue
                    chunks_by_doc[doc_id].append(
                        ChunkRecord(
                            chunk_id=str(c.get("chunk_id", "")),
                            doc_id=doc_id,
                            text=str(c.get("text", "")),
                            index=int(c.get("index", 0) or 0),
                            page_number=c.get("page_number"),
                        )
                    )

        with self._docs_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                doc_id = str(d.get("doc_id", ""))
                if not doc_id:
                    continue

                # Inline chunks override the separate file
                inline_chunks_raw = d.get("chunks")
                if isinstance(inline_chunks_raw, list):
                    inline_chunks = [
                        ChunkRecord(
                            chunk_id=str(c.get("chunk_id", "")),
                            doc_id=doc_id,
                            text=str(c.get("text", "")),
                            index=int(c.get("index", 0) or 0),
                            page_number=c.get("page_number"),
                        )
                        for c in inline_chunks_raw
                        if isinstance(c, dict)
                    ]
                else:
                    inline_chunks = chunks_by_doc.get(doc_id, [])

                props_raw = d.get("properties", {})
                props = (
                    {k: str(v) for k, v in props_raw.items()}
                    if isinstance(props_raw, dict)
                    else {}
                )

                year_raw = d.get("year")
                year = int(year_raw) if isinstance(year_raw, (int, float)) else None

                yield DocumentRecord(
                    doc_id=doc_id,
                    title=str(d.get("title", "")),
                    content=str(d.get("content", "")),
                    source=str(d.get("source_path", d.get("source", ""))),
                    category=str(d.get("category", "")),
                    year=year,
                    properties=props,
                    chunks=inline_chunks,
                )


class InMemoryDocumentSource:
    """Convenience source that wraps a pre-built list of records.

    Useful for tests and for callers that build documents programmatically.
    """

    __slots__ = ("_docs",)

    def __init__(self, docs: list[DocumentRecord]) -> None:
        self._docs = list(docs)

    def documents(self) -> Iterator[DocumentRecord]:
        yield from self._docs


# --- Stats ---


@dataclass(slots=True)
class IngestStats:
    """Summary of one ``DocumentIngester.ingest`` run."""

    documents_ingested: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    categories_created: int = 0
    edges_created: int = 0
    elapsed_seconds: float = 0.0


# --- Ingester ---


MergeStrategy = Literal["skip", "replace"]


class DocumentIngester:
    """Generic corpus → graph ingester driven by a ``DomainProfile``.

    Args:
        profile: Domain profile. Only ``profile.ontology_hints`` is
            consulted — stopwords, metadata patterns, etc. belong to
            the extractor / entity linker, not the ingester.
        backend: Graph backend to write to.
        merge_strategy: What to do when a document with the same
            ``doc_id`` already exists:

            - ``"skip"`` (default) — leave existing doc alone, increment
              ``documents_skipped``. Idempotent for repeated runs.
            - ``"replace"`` — delete existing doc + connected chunks +
              outgoing/incoming edges, then re-ingest. Use when you
              re-parsed a corpus and want fresh content.

    Notes:
        - Ids are deterministic so ``skip`` is genuinely idempotent on
          repeated runs against the same data.
        - The ingester does **not** run phrase extraction or entity
          linking. Use :class:`EntityLinker` as a separate post-processing
          pass — that separation keeps extraction reruns cheap.
    """

    __slots__ = ("_backend", "_merge_strategy", "_profile")

    def __init__(
        self,
        *,
        profile: DomainProfile,
        backend: StorageBackend,
        merge_strategy: MergeStrategy = "skip",
    ) -> None:
        if merge_strategy not in ("skip", "replace"):
            msg = f"Unknown merge_strategy: {merge_strategy}"
            raise ValueError(msg)
        self._profile = profile
        self._backend = backend
        self._merge_strategy = merge_strategy

    async def ingest(self, source: CorpusSource) -> IngestStats:
        """Walk the corpus and materialize category / document / chunk nodes."""
        stats = IngestStats()
        t0 = time.time()

        category_ids: dict[str, str] = {}

        for doc in source.documents():
            doc_node_id = _doc_node_id(doc.doc_id)
            doc_category = _nfc(doc.category)
            doc_title = _nfc(doc.title)

            existing = await self._backend.get_node(doc_node_id)
            if existing is not None:
                if self._merge_strategy == "skip":
                    stats.documents_skipped += 1
                    continue
                # replace
                await self._delete_document_cascade(doc_node_id)

            # Ensure category node
            category_id = await self._ensure_category(doc_category, category_ids, stats)

            doc_kind = self._profile.ontology_hints.get(
                doc_category, NodeKind.ENTITY
            )

            doc_props = dict(doc.properties)
            doc_props["doc_id"] = doc.doc_id
            if doc.year is not None:
                doc_props["year"] = str(doc.year)
                # ``year`` is also the temporal anchor — we surface it
                # under ``valid_from`` so the agent tools (and future
                # temporal filters) have a single, kind-agnostic field.
                doc_props.setdefault("valid_from", str(doc.year))
            if doc_category:
                doc_props["category"] = doc_category

            # Authority tag from the profile. Only set when the profile
            # has an explicit mapping — we never guess a default
            # authority, because wrong authority values propagate into
            # conflict resolution and silently change agent behaviour.
            authority = self._profile.authority_by_kind.get(doc_kind)
            if authority is not None:
                doc_props["authority"] = str(authority)

            # Document content. Historical default was title-only, which
            # meant Document nodes didn't participate meaningfully in FTS
            # — the agent had to traverse CONTAINS to reach searchable
            # text. When ``profile.enrich_document_content`` is True
            # (default) we glue the title together with the opening
            # chunks so each Document node carries its own semantic
            # signal. The limit caps the worst case at ~600 chars so
            # Document rows stay compact even on very long corpora.
            if doc.content:
                doc_content = doc.content
            elif self._profile.enrich_document_content and doc.chunks:
                doc_content = _build_document_preview(
                    title=doc_title,
                    chunks=sorted(doc.chunks, key=lambda c: c.index),
                    limit=self._profile.document_preview_chars,
                )
            else:
                doc_content = doc_title

            await self._backend.save_node(
                Node(
                    id=doc_node_id,
                    kind=doc_kind,
                    title=doc_title,
                    content=doc_content,
                    tags=["document"],
                    source=doc.source,
                    properties=doc_props,
                    level=ConsolidationLevel.L0_RAW,
                )
            )
            stats.documents_ingested += 1

            if category_id is not None:
                await self._backend.save_edge(
                    Edge(
                        id=_edge_id("part_of", doc_node_id, category_id),
                        source_id=doc_node_id,
                        target_id=category_id,
                        kind=EdgeKind.PART_OF,
                        weight=1.0,
                    )
                )
                stats.edges_created += 1

            # Chunks
            prev_chunk_node_id: str | None = None
            for chunk in sorted(doc.chunks, key=lambda c: c.index):
                chunk_node_id = _chunk_node_id(chunk.chunk_id)

                chunk_props = dict(chunk.properties)
                chunk_props["doc_id"] = doc.doc_id
                chunk_props["chunk_index"] = str(chunk.index)
                if chunk.page_number is not None:
                    chunk_props["page_number"] = str(chunk.page_number)

                await self._backend.save_node(
                    Node(
                        id=chunk_node_id,
                        kind=NodeKind.CHUNK,
                        title=f"{doc_title} #{chunk.index}",
                        content=chunk.text,
                        tags=["chunk"],
                        source=doc.source,
                        properties=chunk_props,
                        level=ConsolidationLevel.L0_RAW,
                    )
                )
                stats.chunks_created += 1

                await self._backend.save_edge(
                    Edge(
                        id=_edge_id("contains", doc_node_id, chunk_node_id),
                        source_id=doc_node_id,
                        target_id=chunk_node_id,
                        kind=EdgeKind.CONTAINS,
                        weight=1.0,
                    )
                )
                stats.edges_created += 1

                if prev_chunk_node_id is not None:
                    await self._backend.save_edge(
                        Edge(
                            id=_edge_id(
                                "next", prev_chunk_node_id, chunk_node_id
                            ),
                            source_id=prev_chunk_node_id,
                            target_id=chunk_node_id,
                            kind=EdgeKind.NEXT_CHUNK,
                            weight=0.9,
                        )
                    )
                    stats.edges_created += 1

                prev_chunk_node_id = chunk_node_id

        stats.elapsed_seconds = time.time() - t0
        return stats

    async def _ensure_category(
        self,
        category_name: str,
        cache: dict[str, str],
        stats: IngestStats,
    ) -> str | None:
        if not category_name:
            return None
        cached = cache.get(category_name)
        if cached is not None:
            return cached

        cat_id = _category_node_id(category_name)
        existing = await self._backend.get_node(cat_id)
        if existing is None:
            await self._backend.save_node(
                Node(
                    id=cat_id,
                    kind=NodeKind.CONCEPT,
                    title=category_name,
                    content=category_name,
                    tags=["category"],
                    level=ConsolidationLevel.L0_RAW,
                )
            )
            stats.categories_created += 1
        cache[category_name] = cat_id
        return cat_id

    async def _delete_document_cascade(self, doc_node_id: str) -> None:
        """Delete a doc + its outgoing CONTAINS chunks + all touching edges.

        Called in ``replace`` mode. Walks outgoing CONTAINS edges to find
        chunks, deletes them, then deletes the doc itself. Any stray
        incoming edges to the doc are cleaned up by ``delete_node`` on
        backends that implement cascade; otherwise they'll dangle but
        not corrupt queries since the target id is gone.
        """
        outgoing = await self._backend.get_edges(doc_node_id, direction="outgoing")
        for edge in outgoing:
            if edge.kind == EdgeKind.CONTAINS:
                await self._backend.delete_node(edge.target_id)
        await self._backend.delete_node(doc_node_id)


# --- Content enrichment ---


def _build_document_preview(
    *,
    title: str,
    chunks: list[ChunkRecord],
    limit: int,
) -> str:
    """Build a searchable preview for a Document node.

    Joins the title with the opening chunks' text, stopping as soon as
    we hit ``limit`` characters. The title is always included so a
    title-only FTS match still works; the chunk text is only there to
    raise recall on queries that would otherwise miss the document.

    The preview is NOT a summary — it's a prefix. For proper
    summarisation use ``ConsolidationCascade`` on the chunks once the
    graph is built. The goal here is purely retrieval: give FTS
    something of substance to match against.
    """
    title = (title or "").strip()
    parts: list[str] = []
    remaining = limit

    if title:
        parts.append(title)
        remaining -= len(title) + 1  # +1 for the newline separator
        if remaining <= 0:
            return title

    for chunk in chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue
        if len(text) > remaining:
            parts.append(text[: max(0, remaining)])
            remaining = 0
            break
        parts.append(text)
        remaining -= len(text) + 1
        if remaining <= 0:
            break

    return "\n".join(parts).strip()


# --- Deterministic id helpers ---


def _doc_node_id(doc_id: str) -> str:
    return f"doc_{_hash16(doc_id)}"


def _chunk_node_id(chunk_id: str) -> str:
    return f"chunk_{_hash16(chunk_id)}"


def _category_node_id(category: str) -> str:
    return f"cat_{_hash16(category)}"


def _edge_id(prefix: str, source_id: str, target_id: str) -> str:
    return f"{prefix}_{_hash16(f'{source_id}->{target_id}')}"


def _hash16(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
