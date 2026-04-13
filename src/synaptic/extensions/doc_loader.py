"""Document loader — **optional** wrapper around xgen-doc2chunk.

This module is purely a convenience layer. It is *not required* for
synaptic-memory to ingest documents — you can always:

1. Parse / chunk documents with your own tool (LangChain splitters,
   Unstructured, custom OCR, etc.) and pass the chunks straight to
   :meth:`SynapticGraph.from_chunks`.
2. Or pre-write chunks into a JSONL file and use
   :meth:`SynapticGraph.from_data`.

This loader wraps `xgen-doc2chunk
<https://pypi.org/project/xgen-doc2chunk/>`_, which handles PDF,
DOCX, PPTX, XLSX, HWP, TXT, MD with built-in chunking and table
preservation. If you install it, ``SynapticGraph.from_data()``
automatically routes office files through this loader.

Usage::

    # Option A — let the library handle the file directly
    from synaptic.extensions.doc_loader import load_document
    docs = load_document("manual.pdf")
    # → list[dict] shaped for JsonlDocumentSource

    # Option B — bring your own chunker
    chunks = my_parser.split("manual.pdf")
    graph = await SynapticGraph.from_chunks(chunks)

Optional dependency::

    pip install xgen-doc2chunk

If not installed, ``load_document`` raises :class:`ImportError` with a
clear pointer to the from_chunks alternative.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("doc-loader")

# Extensions that xgen-doc2chunk supports out of the box. Kept here so
# graph.py / from_data() can decide whether to route through this loader
# without having to import the (optional) third-party package.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".hwp",
    ".hwpx",
    ".md",
    ".txt",
    ".rtf",
)


def is_supported_extension(path: str | Path) -> bool:
    """Return True if ``path`` has an extension this loader can handle."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def _get_processor():
    try:
        from xgen_doc2chunk import DocumentProcessor
    except ImportError as exc:
        msg = (
            "Direct document loading requires 'xgen-doc2chunk' "
            "(optional). Install with:\n"
            "    pip install xgen-doc2chunk\n"
            "Or pass already-chunked documents to "
            "SynapticGraph.from_chunks() instead."
        )
        raise ImportError(msg) from exc
    return DocumentProcessor()


def load_document(
    path: str | Path,
    *,
    category: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    preserve_tables: bool = True,
    ocr: bool = False,
) -> list[dict[str, Any]]:
    """Load a single document file into a list of chunk records.

    The file is parsed by xgen-doc2chunk and split into overlapping
    chunks with table-preserving logic. Each chunk becomes its own
    record so the downstream ``DocumentIngester`` can treat them
    uniformly with native JSONL chunks.

    Args:
        path: Path to the document file (any supported extension).
        category: Optional category label assigned to every chunk.
            Defaults to the file's parent folder name when empty —
            useful when ingesting an organised document tree.
        chunk_size: Approximate target chunk length in characters.
            xgen-doc2chunk uses a recursive splitter, so the actual
            length varies a bit around this number.
        chunk_overlap: Overlap between consecutive chunks. Helps the
            retriever tolerate splits that fall mid-sentence.
        preserve_tables: When True, table rows are kept together in
            a single chunk instead of being split arbitrarily.
        ocr: When True, scanned PDFs / images get OCR'd. Slow and
            requires extra dependencies; off by default.

    Returns:
        list of dicts shaped like::

            {
              "doc_id": "<filename>_c<chunk_index>",
              "title": "<filename> (chunk N)",
              "content": "<chunk text>",
              "category": "<category>",
              "source": "<absolute path>",
              "chunk_index": <int>,
              "page": <int|None>,        # only when metadata available
            }

    Raises:
        ImportError: When ``xgen-doc2chunk`` is not installed.
        FileNotFoundError: When ``path`` does not exist.
        ValueError: When the file extension is not supported.
    """
    p = Path(path)
    if not p.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    if not is_supported_extension(p):
        msg = (
            f"Unsupported file extension: {p.suffix}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
        raise ValueError(msg)

    processor = _get_processor()

    # Request metadata so we can attach page numbers when available.
    result = processor.extract_chunks(
        str(p),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_tables=preserve_tables,
        ocr_processing=ocr,
        include_position_metadata=True,
    )

    cat = category or (p.parent.name if p.parent.name else "documents")
    stem = p.stem

    docs: list[dict[str, Any]] = []
    if result.has_metadata:
        for idx, chunk_data in enumerate(result.chunks_with_metadata):
            text = (chunk_data.get("text") or "").strip()
            if not text:
                continue
            page = chunk_data.get("page_number")
            docs.append(
                {
                    "doc_id": f"{stem}_c{idx:04d}",
                    "title": f"{stem} (chunk {idx + 1})",
                    "content": text,
                    "category": cat,
                    "source": str(p),
                    "chunk_index": idx,
                    "page": page,
                }
            )
    else:
        for idx, text in enumerate(result.chunks):
            text_str = (text or "").strip()
            if not text_str:
                continue
            docs.append(
                {
                    "doc_id": f"{stem}_c{idx:04d}",
                    "title": f"{stem} (chunk {idx + 1})",
                    "content": text_str,
                    "category": cat,
                    "source": str(p),
                    "chunk_index": idx,
                }
            )

    return docs


def load_document_dir(
    directory: str | Path,
    *,
    category: str = "",
    recursive: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Load every supported file under ``directory``.

    Each file's chunks are flattened into a single result list. Useful
    for one-shot ingestion of a manual / regulation / spec corpus that
    may contain a mix of PDFs, DOCXs, and HWPs.

    Args:
        directory: Folder to scan.
        category: Category label applied to every chunk; falls back to
            each file's parent folder name when empty.
        recursive: Walk subdirectories. When False, only direct
            children are loaded.
        **kwargs: Forwarded to :func:`load_document` (chunk_size,
            chunk_overlap, preserve_tables, ocr).
    """
    d = Path(directory)
    if not d.is_dir():
        msg = f"Not a directory: {directory}"
        raise NotADirectoryError(msg)

    glob = "**/*" if recursive else "*"
    out: list[dict[str, Any]] = []
    for f in sorted(d.glob(glob)):
        if not f.is_file() or not is_supported_extension(f):
            continue
        try:
            out.extend(load_document(f, category=category, **kwargs))
        except Exception as exc:
            logger.warning("skipped %s: %s", f, exc)
    return out
