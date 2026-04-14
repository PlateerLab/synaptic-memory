"""SynapticGraph — main entry point (facade)."""

from __future__ import annotations

import json
import unicodedata
from difflib import SequenceMatcher
from time import time
from typing import TYPE_CHECKING


def _nfc(s: str) -> str:
    """NFC-normalize a string. macOS HFS+ stores Korean as NFD; without this,
    substring/FTS matches silently fail when NFD content is queried with NFC."""
    return unicodedata.normalize("NFC", s) if s else s


def _parse_sqlite_url(conn: str) -> str:
    """Extract the filesystem path from a SQLite URL.

    SQLAlchemy-style:

    - ``sqlite:///relative/path.db`` → ``relative/path.db``
    - ``sqlite:////abs/path.db``     → ``/abs/path.db``
    - ``sqlite:path.db``             → ``path.db``

    The legacy ``rsplit("///")`` parser this replaces failed on
    absolute paths because four consecutive slashes are ambiguous
    under rsplit — we always want to strip exactly the three-slash
    prefix.
    """
    if conn.startswith("sqlite:///"):
        return conn[len("sqlite:///") :]
    if conn.startswith("sqlite://"):
        return conn[len("sqlite://") :]
    if conn.startswith("sqlite:"):
        return conn[len("sqlite:") :]
    return conn


from synaptic.agent_search import AgentSearch, SearchIntent, suggest_intent
from synaptic.cache import NodeCache
from synaptic.consolidation import ConsolidationCascade
from synaptic.evidence import EvidenceAssembler
from synaptic.exporter import JSONExporter, MarkdownExporter
from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
from synaptic.extensions.embedder import EmbeddingProvider
from synaptic.extensions.phrase_extractor import PhraseExtractor
from synaptic.hebbian import HebbianEngine
from synaptic.models import (
    ActivatedNode,
    BackfillResult,
    ConsolidationLevel,
    DigestResult,
    Edge,
    EdgeKind,
    EvidenceChain,
    MaintenanceResult,
    Node,
    NodeKind,
    SearchResult,
)
from synaptic.ontology import OntologyRegistry, build_agent_ontology
from synaptic.protocols import (
    Digester,
    KindClassifier,
    QueryRewriter,
    RelationDetector,
    StorageBackend,
    TagExtractor,
)
from synaptic.search import HybridSearch
from synaptic.store import Store

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider


class SynapticGraph:
    """Facade over the synaptic memory system.

    Quick Start::

        # 1. In-memory (zero-dep, testing/prototyping)
        graph = SynapticGraph.memory()

        # 2. SQLite (lightweight production)
        graph = SynapticGraph.sqlite("knowledge.db")

        # 3. Full preset with custom backend
        graph = SynapticGraph(backend, classifier=..., embedder=...)
    """

    __slots__ = (
        "_agent_search",
        "_backend",
        "_cache",
        "_chunk_entity_index",
        "_classifier",
        "_consolidation",
        "_corpus_size",
        "_embedder",
        "_hebbian",
        "_json_exporter",
        "_md_exporter",
        "_ontology",
        "_phrase_extractor",
        "_query_decomposer",
        "_relation_detector",
        "_reranker",
        "_search",
        "_store",
    )

    def __init__(
        self,
        backend: StorageBackend,
        *,
        query_rewriter: QueryRewriter | None = None,
        tag_extractor: TagExtractor | None = None,
        ontology: OntologyRegistry | None = None,
        embedder: EmbeddingProvider | None = None,
        classifier: KindClassifier | None = None,
        relation_detector: RelationDetector | None = None,
        phrase_extractor: PhraseExtractor | None = None,
        chunk_entity_index: ChunkEntityIndex | None = None,
        query_decomposer: object | None = None,
        reranker: object | None = None,
        cache_size: int = 256,
        vector_min_cosine: float | None = None,
        vector_relative_drop: float | None = None,
    ) -> None:
        self._backend = backend
        self._store = Store(backend, tag_extractor=tag_extractor)
        self._search = HybridSearch(
            query_rewriter=query_rewriter,
            chunk_entity_index=chunk_entity_index,
            vector_min_cosine=vector_min_cosine,
            vector_relative_drop=vector_relative_drop,
        )
        self._hebbian = HebbianEngine()
        self._consolidation = ConsolidationCascade()
        self._md_exporter = MarkdownExporter()
        self._json_exporter = JSONExporter()
        self._cache = NodeCache(maxsize=cache_size)
        self._ontology = ontology
        self._embedder = embedder
        self._classifier = classifier
        self._relation_detector = relation_detector
        self._phrase_extractor = phrase_extractor
        self._chunk_entity_index = chunk_entity_index
        self._query_decomposer = query_decomposer
        self._reranker = reranker
        self._agent_search = AgentSearch(hybrid=self._search)
        self._corpus_size = 0

    # --- Factory methods ---

    @classmethod
    def memory(cls, *, cache_size: int = 256) -> SynapticGraph:
        """In-memory backend — zero dependencies, for testing/prototyping.

        Example::

            graph = SynapticGraph.memory()
            await graph.add("Hello", "World")
        """
        from synaptic.backends.memory import MemoryBackend
        from synaptic.extensions.classifier_rules import RuleBasedClassifier

        return cls(
            MemoryBackend(),
            classifier=RuleBasedClassifier(),
            cache_size=cache_size,
        )

    @classmethod
    def sqlite(
        cls,
        db_path: str = "synaptic.db",
        *,
        cache_size: int = 256,
    ) -> SynapticGraph:
        """SQLite backend — lightweight production, FTS5 search support.

        Example::

            graph = SynapticGraph.sqlite("knowledge.db")
            await graph.backend.connect()
            await graph.add("Hello", "World")
        """
        from synaptic.backends.sqlite import SQLiteBackend
        from synaptic.extensions.classifier_rules import RuleBasedClassifier
        from synaptic.extensions.relation_detector import RuleBasedRelationDetector

        return cls(
            SQLiteBackend(db_path),
            classifier=RuleBasedClassifier(),
            relation_detector=RuleBasedRelationDetector(),
            ontology=build_agent_ontology(),
            cache_size=cache_size,
        )

    @classmethod
    def kuzu(
        cls,
        db_path: str = "synaptic.kuzu",
        *,
        cache_size: int = 256,
    ) -> SynapticGraph:
        """Kuzu embedded graph backend — native Cypher, property graph, MIT-licensed.

        Kuzu runs in-process (no server, no Docker) and supports
        openCypher, FTS, vector search, and graph algorithms via
        bundled extensions.

        Example::

            graph = SynapticGraph.kuzu("knowledge.kuzu")
            await graph.backend.connect()
            await graph.add("Hello", "World")
        """
        from synaptic.backends.kuzu import KuzuBackend
        from synaptic.extensions.classifier_rules import RuleBasedClassifier
        from synaptic.extensions.relation_detector import RuleBasedRelationDetector

        return cls(
            KuzuBackend(db_path),
            classifier=RuleBasedClassifier(),
            relation_detector=RuleBasedRelationDetector(),
            ontology=build_agent_ontology(),
            cache_size=cache_size,
        )

    @classmethod
    def full(
        cls,
        backend: StorageBackend,
        *,
        llm: LLMProvider | None = None,
        embed_api_base: str = "",
        embed_model: str = "default",
        embed_api_key: str = "",
        cache_size: int = 512,
    ) -> SynapticGraph:
        """Full-featured setup — LLM classification, embedding, relation detection, ontology.

        Example::

            from synaptic.backends.sqlite import SQLiteBackend
            from synaptic.extensions.llm_provider import OllamaLLMProvider

            graph = SynapticGraph.full(
                SQLiteBackend("knowledge.db"),
                llm=OllamaLLMProvider(model="gemma3:4b"),
                embed_api_base="http://localhost:8080/v1",
            )
        """
        from synaptic.extensions.classifier_rules import RuleBasedClassifier
        from synaptic.extensions.relation_detector import RuleBasedRelationDetector

        classifier: KindClassifier
        relation_detector: RelationDetector
        embedder: EmbeddingProvider | None = None

        if llm is not None:
            from synaptic.extensions.classifier_hybrid import HybridClassifier
            from synaptic.extensions.classifier_llm import LLMClassifier
            from synaptic.extensions.relation_detector_llm import (
                LLMRelationDetector,
            )

            classifier = HybridClassifier(
                llm=LLMClassifier(llm, fallback=RuleBasedClassifier()),
                rule=RuleBasedClassifier(),
            )
            relation_detector = LLMRelationDetector(llm, fallback=RuleBasedRelationDetector())
        else:
            classifier = RuleBasedClassifier()
            relation_detector = RuleBasedRelationDetector()

        if embed_api_base:
            from synaptic.extensions.embedder import OpenAIEmbeddingProvider

            embedder = OpenAIEmbeddingProvider(
                api_base=embed_api_base,
                model=embed_model,
                api_key=embed_api_key,
            )

        return cls(
            backend,
            classifier=classifier,
            relation_detector=relation_detector,
            embedder=embedder,
            ontology=build_agent_ontology(),
            phrase_extractor=PhraseExtractor(),
            cache_size=cache_size,
        )

    # --- Easy API ---

    @classmethod
    async def from_data(
        cls,
        data_path: str,
        *,
        db: str = "synaptic.db",
        embed_url: str | None = None,
        embed_model: str = "qwen3-embedding:4b",
    ) -> SynapticGraph:
        """ONE-LINE graph construction from any data source.

        Auto-detects file format, generates a DomainProfile, ingests,
        and optionally embeds. Returns a ready-to-search graph.

        Supports:
        - Directory of files → scans for CSV, JSONL, and (optionally)
          office documents
        - Single CSV → TableIngester
        - Single JSONL → DocumentIngester
        - Single office file (PDF / DOCX / PPTX / XLSX / HWP / TXT / MD)
          → DocumentIngester via xgen-doc2chunk (**optional dependency**;
          install with ``pip install xgen-doc2chunk`` or pre-chunk
          yourself and call :meth:`from_chunks` instead)
        - Glob pattern (``*.csv``) → batch ingest

        Example::

            graph = await SynapticGraph.from_data("./my_docs/")
            result = await graph.search("my question")

            # With embedding
            graph = await SynapticGraph.from_data(
                "./data.csv",
                embed_url="http://localhost:11434/v1",
            )

            # Bring your own chunker (no xgen-doc2chunk needed)
            chunks = my_parser.split("manual.pdf")
            graph = await SynapticGraph.from_chunks(chunks)
        """
        from pathlib import Path

        from synaptic.backends.sqlite_graph import SqliteGraphBackend
        from synaptic.extensions.document_ingester import (
            DocumentIngester,
            JsonlDocumentSource,
        )
        from synaptic.extensions.profile_generator import ProfileGenerator
        from synaptic.extensions.table_ingester import TableIngester

        path = Path(data_path)
        backend = SqliteGraphBackend(db)
        await backend.connect()

        # Detect data type and ingest. Document loader handles a wide
        # range of office formats (PDF, DOCX, PPTX, XLSX, HWP, MD, …)
        # — see synaptic.extensions.doc_loader.SUPPORTED_EXTENSIONS.
        from synaptic.extensions.doc_loader import (
            SUPPORTED_EXTENSIONS as _DOC_EXTS,
        )

        _accepted = {".csv", ".jsonl", ".json", *_DOC_EXTS}
        files: list[Path] = []
        if path.is_dir():
            files = sorted(
                p
                for p in path.rglob("*")
                if p.suffix.lower() in _accepted and not p.name.startswith(".")
            )
        elif path.is_file():
            files = [path]
        else:
            # Try as glob
            import glob as _glob

            files = [Path(p) for p in sorted(_glob.glob(data_path))]

        if not files:
            msg = f"No data files found at {data_path}"
            raise FileNotFoundError(msg)

        # Auto-generate profile from samples
        samples: list[str] = []
        categories: list[str] = []
        for f in files[:5]:
            if f.suffix == ".csv":
                import csv

                with f.open(encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for i, row in enumerate(reader):
                        if i >= 20:
                            break
                        samples.append(" ".join(str(v) for v in row.values()))
            elif f.suffix == ".jsonl":
                import json

                with f.open(encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        if i >= 20:
                            break
                        d = json.loads(line)
                        content = d.get("content", d.get("text", d.get("title", "")))
                        if content:
                            samples.append(str(content)[:500])
                        cat = d.get("category", "")
                        if cat:
                            categories.append(str(cat))
            elif f.suffix.lower() in _DOC_EXTS:
                try:
                    from synaptic.extensions.doc_loader import load_document

                    doc_chunks = load_document(f)
                    for d in doc_chunks[:20]:
                        samples.append(str(d.get("content", ""))[:500])
                    if doc_chunks and doc_chunks[0].get("category"):
                        categories.append(str(doc_chunks[0]["category"]))
                except ImportError:
                    pass  # xgen-doc2chunk is optional

        gen = ProfileGenerator()
        profile = await gen.generate(
            name=path.stem,
            samples=samples,
            categories=categories if categories else None,
        )

        # Ingest each file
        for f in files:
            if f.suffix == ".csv":
                import csv

                with f.open(encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    rows = list(reader)
                if rows:
                    columns = [{"name": k, "type": "str"} for k in rows[0]]
                    table_name = f.stem
                    graph_instance = cls(backend)
                    ingester = TableIngester()
                    await ingester.ingest(
                        graph_instance,
                        table_name,
                        columns,
                        rows,
                        primary_key=next(iter(rows[0].keys())),
                    )
            elif f.suffix == ".jsonl":
                # Check if it's a docs+chunks pair
                chunks_path = f.parent / f.name.replace("documents", "chunks")
                source = JsonlDocumentSource(
                    str(f),
                    str(chunks_path) if chunks_path.exists() and chunks_path != f else None,
                )
                doc_ingester = DocumentIngester(profile=profile, backend=backend)
                await doc_ingester.ingest(source)
            elif f.suffix.lower() in _DOC_EXTS:
                # PDF/DOCX/PPTX/XLSX/HWP/… → chunk records (xgen-doc2chunk
                # already handles chunking + table preservation) → temp
                # JSONL → DocumentIngester. Using JSONL as a transit
                # format keeps the document pipeline (NFC, profile hints,
                # embeddings, FTS) uniform regardless of input file type.
                from synaptic.extensions.doc_loader import load_document

                doc_chunks = load_document(f)
                if not doc_chunks:
                    continue

                import json as _json
                import tempfile

                tmp = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".jsonl",
                    delete=False,
                    encoding="utf-8",
                )
                try:
                    for doc in doc_chunks:
                        tmp.write(_json.dumps(doc, ensure_ascii=False) + "\n")
                    tmp.close()
                    source = JsonlDocumentSource(tmp.name, None)
                    doc_ingester = DocumentIngester(profile=profile, backend=backend)
                    await doc_ingester.ingest(source)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

        # Optional: embed all nodes
        if embed_url:
            from synaptic.extensions.embedder import OpenAIEmbeddingProvider

            embedder = OpenAIEmbeddingProvider(api_base=embed_url, model=embed_model)
            nodes = await backend.list_nodes(kind=None, limit=100_000)
            batch_size = 32
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]
                texts = [f"{n.title}\n{(n.content or '')[:300]}" for n in batch]
                try:
                    vecs = await embedder.embed_batch(texts)
                    for n, v in zip(batch, vecs):
                        if v:
                            n.embedding = v
                            await backend.save_node(n)
                except Exception:
                    pass

        graph_obj = cls(backend)
        return graph_obj

    @classmethod
    async def from_chunks(
        cls,
        chunks: list[dict],
        *,
        db: str = "synaptic.db",
        profile: object | None = None,
        embed_url: str | None = None,
        embed_model: str = "qwen3-embedding:4b",
    ) -> SynapticGraph:
        """Ingest pre-parsed / pre-chunked documents directly.

        Use this when you already have chunks from your own document
        parser (LangChain splitters, Unstructured, custom OCR, etc.)
        and don't want to depend on the optional xgen-doc2chunk loader.

        Each chunk dict should provide at minimum a ``content`` field.
        Recognised keys:

        ====================  =======================================
        ``content`` (req)     The chunk text — what gets indexed
        ``title``             Display title; auto-derived from first
                              line if missing
        ``doc_id``            Stable identifier; auto-generated if
                              missing
        ``category``          Category label for ontology routing
        ``source``            Original file path / URL (kept as a
                              property)
        ``chunk_index``       Position within the source document
        ``page``              Page number for paginated sources
        ====================  =======================================

        Args:
            chunks: List of chunk dicts (see field reference above).
            db: SQLite path for the new graph.
            profile: Optional DomainProfile. When omitted, a profile
                is auto-generated from the first 20 chunks.
            embed_url: OpenAI-compatible endpoint to embed nodes after
                ingest. Skipped when None.
            embed_model: Embedder model name.

        Example::

            # From your own parser (e.g. LangChain RecursiveCharacterTextSplitter)
            chunks = [
                {"content": "...", "title": "Page 1", "category": "manual"},
                {"content": "...", "title": "Page 2", "category": "manual"},
            ]
            graph = await SynapticGraph.from_chunks(chunks)
            result = await graph.search("my question")
        """
        if not chunks:
            msg = "from_chunks() requires at least one chunk"
            raise ValueError(msg)

        # Lazy imports — keep top-level synaptic import light.
        from pathlib import Path as _Path

        from synaptic.backends.sqlite_graph import SqliteGraphBackend
        from synaptic.extensions.document_ingester import (
            DocumentIngester,
            JsonlDocumentSource,
        )
        from synaptic.extensions.profile_generator import ProfileGenerator

        backend = SqliteGraphBackend(db)
        await backend.connect()

        # Auto-generate a profile from the first 20 chunks if the
        # caller didn't supply one. Same path as from_data().
        if profile is None:
            samples = [str(c.get("content", ""))[:500] for c in chunks[:20]]
            categories = [str(c.get("category", "")) for c in chunks if c.get("category")]
            gen = ProfileGenerator()
            profile = await gen.generate(
                name="from_chunks",
                samples=samples,
                categories=list(dict.fromkeys(categories)) if categories else None,
            )

        # Materialise chunks into a temp JSONL so they flow through
        # the same DocumentIngester path that JSONL files use — keeps
        # NFC, FTS indexing, edge construction, and embedder hooks
        # consistent regardless of input shape.
        import json as _json
        import tempfile
        import uuid as _uuid

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        try:
            for i, c in enumerate(chunks):
                content = str(c.get("content") or "").strip()
                if not content:
                    continue
                doc_id = c.get("doc_id") or f"chunk_{_uuid.uuid4().hex[:12]}"
                title = c.get("title") or content.split("\n", 1)[0][:80] or doc_id
                record = {
                    "doc_id": doc_id,
                    "title": title,
                    "content": content,
                    "category": c.get("category", ""),
                    "source": c.get("source", ""),
                    "chunk_index": c.get("chunk_index", i),
                }
                if c.get("page") is not None:
                    record["page"] = c["page"]
                tmp.write(_json.dumps(record, ensure_ascii=False) + "\n")
            tmp.close()

            source = JsonlDocumentSource(tmp.name, None)
            doc_ingester = DocumentIngester(profile=profile, backend=backend)
            await doc_ingester.ingest(source)
        finally:
            _Path(tmp.name).unlink(missing_ok=True)

        # Optional embedding pass — same logic as from_data().
        if embed_url:
            from synaptic.extensions.embedder import OpenAIEmbeddingProvider

            embedder = OpenAIEmbeddingProvider(api_base=embed_url, model=embed_model)
            nodes = await backend.list_nodes(kind=None, limit=100_000)
            for i in range(0, len(nodes), 32):
                batch = nodes[i : i + 32]
                texts = [f"{n.title}\n{(n.content or '')[:300]}" for n in batch]
                try:
                    vecs = await embedder.embed_batch(texts)
                    for n, v in zip(batch, vecs):
                        if v:
                            n.embedding = v
                            await backend.save_node(n)
                except Exception:
                    pass

        return cls(backend)

    @classmethod
    async def from_database(
        cls,
        connection_string: str,
        *,
        db: str = "synaptic.db",
        tables: list[str] | None = None,
        row_limit: int = 500_000,
        mode: str = "full",
    ) -> SynapticGraph:
        """ONE-LINE graph construction from a relational database.

        Auto-discovers schema, FK relationships, and data types.
        No manual configuration needed.

        Supports:
        - ``sqlite:///path/to/db.sqlite``
        - ``postgresql://user:pass@host:port/dbname``
        - ``mysql://user:pass@host:port/dbname``
        - ``oracle://user:pass@host:port/service_name``
        - ``mssql://connection_string``

        Modes:

        - ``"full"`` (default): current behavior — random UUIDs, no
          sync state recorded. Use for one-shot exports.
        - ``"cdc"``: deterministic node IDs keyed on
          ``(connection_string, table, pk)`` + sync state recorded.
          Subsequent :meth:`sync_from_database` calls do incremental
          deltas only. Phase 2 supports SQLite for the incremental
          path; other dialects still do a full deterministic reload
          until Phase 6.
        - ``"auto"``: if a prior CDC state exists in the graph file
          for this source URL, behave like ``"cdc"`` (incremental);
          otherwise fall back to ``"full"``.

        Example::

            graph = await SynapticGraph.from_database("sqlite:///shop.db")
            graph = await SynapticGraph.from_database("postgresql://user:pass@localhost/mydb")
            graph = await SynapticGraph.from_database("mysql://root:pass@localhost/shop")

            # Incremental sync mode
            graph = await SynapticGraph.from_database("sqlite:///shop.db", mode="cdc")
            result = await graph.sync_from_database("sqlite:///shop.db")
        """
        from synaptic.backends.sqlite_graph import SqliteGraphBackend
        from synaptic.extensions.db_ingester import DbIngester

        backend = SqliteGraphBackend(db)
        await backend.connect()
        graph = cls(backend)
        ingester = DbIngester()

        if mode not in ("full", "cdc", "auto"):
            msg = f"Unknown mode={mode!r}; expected 'full', 'cdc', or 'auto'"
            raise ValueError(msg)

        # 'auto' collapses to 'cdc' when prior sync state exists.
        effective_mode = mode
        if mode == "auto":
            await backend.ensure_cdc_tables()
            store = backend.cdc_state_store()
            # If ANY table already has prior state for this source URL,
            # we treat the call as incremental.
            async with backend._db().execute(
                "SELECT 1 FROM syn_cdc_state WHERE source_url = ? LIMIT 1",
                (connection_string,),
            ) as cur:
                existing = await cur.fetchone()
            effective_mode = "cdc" if existing else "full"
            del store  # silence unused

        source_url_arg = connection_string if effective_mode == "cdc" else ""

        # Route incremental SQLite through the CDC sync orchestrator —
        # first call seeds state, subsequent calls are deltas.
        if effective_mode == "cdc":
            if connection_string.startswith("sqlite"):
                db_path = _parse_sqlite_url(connection_string)
                await ingester.sync_from_sqlite(
                    db_path,
                    graph,
                    source_url=connection_string,
                    tables=tables,
                    row_limit=row_limit,
                )
                return graph
            if connection_string.startswith("postgresql"):
                await ingester.sync_from_postgres(
                    connection_string,
                    graph,
                    source_url=connection_string,
                    tables=tables,
                    row_limit=row_limit,
                )
                return graph
            if connection_string.startswith("mysql") or connection_string.startswith("mariadb"):
                await ingester.sync_from_mysql(
                    connection_string,
                    graph,
                    source_url=connection_string,
                    tables=tables,
                    row_limit=row_limit,
                )
                return graph
            # Other dialects fall through to the legacy ingest_from_*
            # path with deterministic IDs (no incremental sync yet).

        if connection_string.startswith("sqlite"):
            # sqlite:///path or sqlite:path
            db_path = _parse_sqlite_url(connection_string)
            stats = await ingester.ingest_from_sqlite(
                db_path,
                graph,
                tables=tables,
                row_limit=row_limit,
                source_url=source_url_arg,
            )
        elif connection_string.startswith("postgresql"):
            stats = await ingester.ingest_from_postgres(
                connection_string,
                graph,
                tables=tables,
                row_limit=row_limit,
            )
        elif connection_string.startswith("mysql") or connection_string.startswith("mariadb"):
            stats = await ingester.ingest_from_mysql(
                connection_string,
                graph,
                tables=tables,
                row_limit=row_limit,
            )
        elif connection_string.startswith("oracle"):
            stats = await ingester.ingest_from_oracle(
                connection_string,
                graph,
                tables=tables,
                row_limit=row_limit,
            )
        elif connection_string.startswith("mssql"):
            stats = await ingester.ingest_from_mssql(
                connection_string,
                graph,
                tables=tables,
                row_limit=row_limit,
            )
        else:
            msg = f"Unsupported database: {connection_string.split(':', maxsplit=1)[0]}. Use sqlite://, postgresql://, mysql://, oracle://, mssql://"
            raise ValueError(msg)

        import logging

        logging.getLogger("db-ingester").info(
            "from_database: %d tables, %d rows, %d nodes, %.1fs",
            stats.tables_ingested,
            stats.total_rows,
            stats.total_nodes,
            stats.elapsed_seconds,
        )
        return graph

    async def sync_from_database(
        self,
        connection_string: str,
        *,
        tables: list[str] | None = None,
        row_limit: int = 500_000,
    ):
        """Incrementally sync this graph with a live database.

        Detects tables with ``updated_at``-style columns, reads only
        rows whose change column is at or above the last watermark,
        and upserts them via deterministic node IDs. Tables without
        a change column are skipped with an error entry in the
        returned :class:`SyncResult` (Phase 5 adds a hash fallback).

        Must be called on a graph created with
        ``from_database(..., mode="cdc")`` or ``mode="auto"`` — the
        sync state tables rely on the sync run having seeded them
        during the initial load.

        Currently only ``sqlite://`` URLs are supported for the
        incremental path; other dialects land in Phase 6.
        """
        from synaptic.extensions.db_ingester import DbIngester

        ingester = DbIngester()
        if connection_string.startswith("sqlite"):
            db_path = _parse_sqlite_url(connection_string)
            return await ingester.sync_from_sqlite(
                db_path,
                self,
                source_url=connection_string,
                tables=tables,
                row_limit=row_limit,
            )
        if connection_string.startswith("postgresql"):
            return await ingester.sync_from_postgres(
                connection_string,
                self,
                source_url=connection_string,
                tables=tables,
                row_limit=row_limit,
            )
        if connection_string.startswith("mysql") or connection_string.startswith("mariadb"):
            return await ingester.sync_from_mysql(
                connection_string,
                self,
                source_url=connection_string,
                tables=tables,
                row_limit=row_limit,
            )

        msg = (
            f"sync_from_database does not yet support "
            f"{connection_string.split(':', maxsplit=1)[0]}:// — "
            "currently sqlite, postgresql, and mysql are wired."
        )
        raise NotImplementedError(msg)

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    async def _get_corpus_size(self) -> int:
        """Get corpus size for adaptive search weighting (cached)."""
        if self._corpus_size > 0:
            return self._corpus_size
        # First call: compute from backend
        if hasattr(self._backend, "_nodes"):
            self._corpus_size = len(self._backend._nodes)  # type: ignore[attr-defined]
        else:
            nodes = await self._backend.list_nodes(limit=100000)
            self._corpus_size = len(nodes)
        return self._corpus_size

    @property
    def cache(self) -> NodeCache:
        return self._cache

    @property
    def ontology(self) -> OntologyRegistry | None:
        return self._ontology

    @property
    def chunk_entity_index(self) -> ChunkEntityIndex | None:
        return self._chunk_entity_index

    @property
    def explorer(self) -> object:
        """Graph data exploration API for visualization frontends."""
        from synaptic.explorer import GraphExplorer

        return GraphExplorer(self._backend, self._chunk_entity_index)

    async def add(
        self,
        title: str,
        content: str,
        *,
        kind: str | NodeKind | None = None,
        tags: list[str] | None = None,
        source: str = "",
        embedding: list[float] | None = None,
        properties: dict[str, str] | None = None,
        node_id: str | None = None,
    ) -> Node:
        # NFC-normalize all user-provided text. Korean on macOS HFS+ arrives
        # as NFD, which breaks substring / FTS matching against NFC queries.
        title = _nfc(title)
        content = _nfc(content)
        source = _nfc(source)
        if tags:
            tags = [_nfc(t) for t in tags]
        if properties:
            properties = {k: _nfc(v) if isinstance(v, str) else v for k, v in properties.items()}

        # Auto-classify kind if not specified
        if kind is None:
            if self._classifier is not None:
                # LLM classifier: generate rich metadata via classify_async
                if hasattr(self._classifier, "classify_async"):
                    result = await self._classifier.classify_async(title, content)
                    kind = result.kind
                    if tags is None:
                        tags = result.tags
                    if properties is None:
                        properties = {}
                    if result.search_keywords:
                        properties["_search_keywords"] = ",".join(result.search_keywords)
                    if result.search_scenarios:
                        properties["_search_scenarios"] = "|".join(result.search_scenarios)
                    if result.summary:
                        properties["_summary"] = result.summary
                else:
                    kind = self._classifier.classify(title, content)
            else:
                kind = NodeKind.CONCEPT

        # Validate against ontology if available
        if self._ontology and properties:
            errors = self._ontology.validate_node(str(kind), properties)
            if errors:
                msg = f"Ontology validation failed: {'; '.join(errors)}"
                raise ValueError(msg)

        # Auto-embed if embedder is available and no embedding provided
        if embedding is None and self._embedder is not None:
            # Include LLM classifier-generated metadata in the embedding text
            embed_text = f"{title} {content}".strip()
            if properties:
                search_kw = properties.get("_search_keywords", "")
                summary = properties.get("_summary", "")
                if search_kw or summary:
                    embed_text = f"{title} {summary} {search_kw} {content}".strip()
            if embed_text:
                embedding = await self._embedder.embed(embed_text)

        node = await self._store.add_node(
            title,
            content,
            kind=kind,
            tags=tags,
            source=source,
            embedding=embedding,
            properties=properties,
            node_id=node_id,
        )
        self._cache.put(node)
        self._corpus_size += 1

        # Auto-detect relations with existing nodes
        if self._relation_detector is not None:
            self._relation_detector.index.add(node)
            relations = await self._relation_detector.detect(node, self._backend)
            for target_id, edge_kind, weight in relations:
                await self._store.add_edge(
                    node.id,
                    target_id,
                    kind=edge_kind,
                    weight=weight,
                )

        # Phrase extraction and linking (HippoRAG2 dual-node KG)
        if self._phrase_extractor is not None:
            await self._phrase_extractor.extract_and_link(
                self,
                node.id,
                title,
                content,
            )

        return node

    async def add_document(
        self,
        title: str,
        content: str,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        kind: str | NodeKind | None = None,
        tags: list[str] | None = None,
        source: str = "",
        properties: dict[str, str] | None = None,
    ) -> list[Node]:
        """긴 문서를 자동 청킹하여 여러 노드로 추가.

        chunk_size 이하 문서는 단일 노드로 추가 (add()와 동일).
        긴 문서는 문장 경계에서 분할하고 CHUNK 노드 + NEXT_CHUNK 순서 엣지로 연결.
        ChunkEntityIndex가 있으면 phrase_extractor가 만든 엔티티를 양방향 인덱스에 등록.

        Returns:
            생성된 노드 리스트 (첫 번째가 대표 노드).
        """
        use_chunk_kind = self._chunk_entity_index is not None

        # 짧은 문서는 그냥 add()
        if len(content) <= chunk_size:
            node = await self.add(
                title=title,
                content=content,
                kind=NodeKind.CHUNK if use_chunk_kind else kind,
                tags=tags,
                source=source,
                properties=properties,
            )
            # Register in chunk-entity index if available
            if use_chunk_kind and self._chunk_entity_index is not None:
                await self._register_chunk_entities(node)
            return [node]

        # 문장 경계에서 청킹
        chunks = self._split_into_chunks(content, chunk_size, chunk_overlap)
        nodes: list[Node] = []
        for i, chunk in enumerate(chunks):
            chunk_title = f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
            chunk_tags = list(tags) if tags else []
            chunk_tags.append(f"chunk:{i}")
            if len(chunks) > 1:
                chunk_tags.append(f"chunks:{len(chunks)}")

            chunk_props = dict(properties) if properties else {}
            chunk_props["chunk_index"] = str(i)
            chunk_props["total_chunks"] = str(len(chunks))
            chunk_props["parent_doc"] = title

            node = await self.add(
                title=chunk_title,
                content=chunk,
                kind=NodeKind.CHUNK if use_chunk_kind else kind,
                tags=chunk_tags,
                source=source,
                properties=chunk_props,
            )
            nodes.append(node)

        if len(nodes) > 1:
            # 청크 간 PART_OF 관계 (첫 번째가 대표 노드)
            for i in range(1, len(nodes)):
                await self.link(
                    nodes[i].id,
                    nodes[0].id,
                    kind=EdgeKind.PART_OF,
                    weight=0.9,
                )
            # 순차 청크 간 NEXT_CHUNK 엣지
            for i in range(len(nodes) - 1):
                await self.link(
                    nodes[i].id,
                    nodes[i + 1].id,
                    kind=EdgeKind.NEXT_CHUNK,
                    weight=0.7,
                )

        # Register all chunks in chunk-entity index
        if use_chunk_kind and self._chunk_entity_index is not None:
            for node in nodes:
                await self._register_chunk_entities(node)

        return nodes

    async def _register_chunk_entities(self, chunk_node: Node) -> None:
        """Register chunk-entity links in the bidirectional index.

        Scans outgoing CONTAINS/MENTIONS edges from the chunk node
        (created by phrase_extractor or entity_extractor) and registers them.
        """
        if self._chunk_entity_index is None:
            return
        edges = await self._backend.get_edges(chunk_node.id, direction="outgoing")
        for edge in edges:
            if edge.kind in (EdgeKind.CONTAINS, EdgeKind.MENTIONS):
                self._chunk_entity_index.register(chunk_node.id, edge.target_id)

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
        """문장 경계에서 텍스트 분할."""
        import re as _re

        sentences = _re.split(r"(?<=[.!?。\n])\s+", text)

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > chunk_size and current:
                chunks.append(" ".join(current))
                # overlap: 마지막 문장들 유지
                overlap_sents: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) > overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                current = overlap_sents
                current_len = overlap_len

            current.append(sent)
            current_len += len(sent)

        if current:
            chunks.append(" ".join(current))

        return chunks if chunks else [text]

    async def add_table(
        self,
        table_name: str,
        columns: list[dict[str, str]],
        rows: list[dict[str, object]],
        *,
        foreign_keys: dict[str, tuple[str, str]] | None = None,
        primary_key: str = "id",
        tags: list[str] | None = None,
        source: str = "",
    ) -> list[Node]:
        """테이블 데이터를 지식 그래프에 추가.

        각 행을 ENTITY 노드로 생성하고, FK를 엣지로 연결.
        테이블 스키마는 OntologyRegistry에 자동 등록.

        Args:
            table_name: 테이블 이름.
            columns: 컬럼 정의 [{"name": "col", "type": "str"}, ...].
            rows: 행 데이터 [{"col": value, ...}, ...].
            foreign_keys: FK 매핑 {"col": ("target_table", "target_col")}.
            primary_key: PK 컬럼 이름.
            tags: 추가 태그.
            source: 소스 식별자.

        Returns:
            생성된 ENTITY 노드 리스트.
        """
        from synaptic.extensions.table_ingester import TableIngester

        ingester = TableIngester()
        return await ingester.ingest(
            self,
            table_name,
            columns,
            rows,
            foreign_keys=foreign_keys,
            primary_key=primary_key,
            tags=tags,
            source=source,
        )

    async def link(
        self,
        source_id: str,
        target_id: str,
        *,
        kind: EdgeKind = EdgeKind.RELATED,
        weight: float = 1.0,
    ) -> Edge:
        # Validate against ontology relation constraints if available
        if self._ontology:
            src_node = await self._backend.get_node(source_id)
            tgt_node = await self._backend.get_node(target_id)
            if src_node is not None and tgt_node is not None:
                errors = self._ontology.validate_edge(
                    str(kind),
                    str(src_node.kind),
                    str(tgt_node.kind),
                )
                if errors:
                    msg = f"Ontology validation failed: {'; '.join(errors)}"
                    raise ValueError(msg)
        return await self._store.add_edge(source_id, target_id, kind=kind, weight=weight)

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        embedding: list[float] | None = None,
    ) -> SearchResult:
        # NFC-normalize query to match NFC-normalized stored content.
        query = _nfc(query)
        # Auto-embed query for vector search
        if embedding is None and self._embedder is not None:
            embedding = await self._embedder.embed(query)
        # Corpus size for adaptive vector weighting
        corpus_size = await self._get_corpus_size()

        # Query decomposition: split complex queries into sub-queries
        if self._query_decomposer is not None and hasattr(self._query_decomposer, "decompose"):
            import asyncio

            sub_queries = await self._query_decomposer.decompose(query)
            if len(sub_queries) > 1:
                # Search each sub-query in parallel
                tasks = [
                    self._search.search(
                        self._backend,
                        sq,
                        limit=limit,
                        embedding=embedding,
                        corpus_size=corpus_size,
                    )
                    for sq in sub_queries
                ]
                sub_results = await asyncio.gather(*tasks)

                # Merge results with RRF fusion
                from synaptic.search import _rrf_fusion

                rankings: list[dict[str, float]] = []
                node_map: dict[str, ActivatedNode] = {}
                for sr in sub_results:
                    ranking = {}
                    for activated in sr.nodes:
                        ranking[activated.node.id] = activated.resonance
                        # Keep the best ActivatedNode per node_id
                        existing = node_map.get(activated.node.id)
                        if existing is None or activated.resonance > existing.resonance:
                            node_map[activated.node.id] = activated
                    rankings.append(ranking)

                rrf_scores = _rrf_fusion(*rankings)
                max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

                merged: list[ActivatedNode] = []
                for nid, rrf_s in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
                    :limit
                ]:
                    an = node_map[nid]
                    merged.append(
                        ActivatedNode(
                            node=an.node,
                            activation=an.activation,
                            resonance=rrf_s / max_rrf if max_rrf > 0 else 0.0,
                            path=an.path,
                        )
                    )

                total = sum(sr.total_candidates for sr in sub_results)
                elapsed = sum(sr.search_time_ms for sr in sub_results)
                stages = ["decompose"]
                for sr in sub_results:
                    for s in sr.stages_used:
                        if s not in stages:
                            stages.append(s)

                result = SearchResult(
                    query=query,
                    nodes=merged,
                    total_candidates=total,
                    search_time_ms=elapsed,
                    stages_used=stages,
                )
                return await self._apply_reranker(query, result, limit)

        result = await self._search.search(
            self._backend,
            query,
            limit=limit,
            embedding=embedding,
            corpus_size=corpus_size,
        )
        return await self._apply_reranker(query, result, limit)

    async def _apply_reranker(self, query: str, result: SearchResult, limit: int) -> SearchResult:
        """Apply reranker to search results if configured."""
        if self._reranker is None or not hasattr(self._reranker, "rerank"):
            return result
        reranked = await self._reranker.rerank(query, result.nodes, top_k=limit)
        return SearchResult(
            query=result.query,
            nodes=reranked,
            total_candidates=result.total_candidates,
            search_time_ms=result.search_time_ms,
            stages_used=result.stages_used + ["rerank"],
        )

    async def agent_search(
        self,
        query: str,
        *,
        intent: str = "auto",
        context_tags: list[str] | None = None,
        limit: int = 10,
        embedding: list[float] | None = None,
        depth: int = 2,
    ) -> SearchResult:
        """Agent-optimized search with intent and context awareness.

        Set intent="auto" (default) to infer intent from query keywords.
        """
        # Auto-embed query for vector search
        if embedding is None and self._embedder is not None:
            embedding = await self._embedder.embed(query)
        if intent == "auto":
            search_intent = suggest_intent(query)
        else:
            search_intent = SearchIntent(intent)
        corpus_size = await self._get_corpus_size()
        return await self._agent_search.search(
            self._backend,
            query,
            intent=search_intent,
            context_tags=context_tags,
            limit=limit,
            embedding=embedding,
            depth=depth,
            corpus_size=corpus_size,
        )

    async def list(
        self,
        *,
        kind: str | NodeKind | None = None,
        level: ConsolidationLevel | None = None,
        limit: int = 100,
    ) -> list[Node]:
        """List all nodes with optional kind/level filtering."""
        return await self._backend.list_nodes(kind=kind, level=level, limit=limit)

    async def get(self, node_id: str) -> Node | None:
        cached = self._cache.get(node_id)
        if cached is not None:
            # Still track access in backend for consolidation
            cached.access_count += 1
            cached.updated_at = time()
            await self._backend.update_node(cached)
            return cached
        node = await self._store.get_node(node_id)
        if node is not None:
            self._cache.put(node)
        return node

    async def update(
        self,
        node_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        kind: str | NodeKind | None = None,
        tags: list[str] | None = None,
        properties: dict[str, str] | None = None,
        embedding: list[float] | None = None,
    ) -> Node | None:
        """Update a node's fields by ID. Returns updated node, or None if not found."""
        node = await self._backend.get_node(node_id)
        if node is None:
            return None
        if title is not None:
            node.title = title
        if content is not None:
            node.content = content
        if kind is not None:
            node.kind = kind
        if tags is not None:
            node.tags = tags
        if properties is not None:
            node.properties = properties
        if embedding is not None:
            node.embedding = embedding
        node.updated_at = time()
        await self._backend.update_node(node)
        self._cache.invalidate(node_id)
        self._cache.put(node)
        return node

    async def remove(self, node_id: str) -> bool:
        node = await self._backend.get_node(node_id)
        if node is None:
            return False
        # Remove from relation detector index
        if self._relation_detector is not None:
            self._relation_detector.index.remove(node_id)
        await self._store.delete_node(node_id)
        self._cache.invalidate(node_id)
        self._corpus_size = max(0, self._corpus_size - 1)
        return True

    async def reinforce(self, node_ids: list[str], *, success: bool = True) -> None:
        await self._hebbian.reinforce(self._backend, node_ids, success=success)
        # Invalidate cached nodes (counts changed)
        for nid in node_ids:
            self._cache.invalidate(nid)

    async def consolidate(
        self,
        digester: Digester | None = None,
        *,
        context: dict[str, object] | None = None,
    ) -> DigestResult:
        return await self._consolidation.consolidate(self._backend, digester, context=context)

    async def prune(self) -> int:
        return await self._backend.prune_edges(weight_below=0.1)

    async def decay(self) -> int:
        self._cache.clear()  # Vitality changed globally
        return await self._backend.decay_vitality(factor=0.95)

    async def maintain(
        self,
        digester: Digester | None = None,
        *,
        context: dict[str, object] | None = None,
    ) -> MaintenanceResult:
        """Run consolidate + decay + prune in one call with a unified result."""
        consolidated = await self._consolidation.consolidate(
            self._backend,
            digester,
            context=context,
        )
        decayed = await self.decay()
        pruned = await self.prune()
        return MaintenanceResult(consolidated=consolidated, decayed=decayed, pruned=pruned)

    async def backfill(
        self,
        *,
        embeddings: bool = True,
        phrases: bool = True,
        batch_size: int = 64,
        max_nodes: int | None = None,
    ) -> BackfillResult:
        """Repair existing nodes that are missing embeddings or phrase hubs.

        This is the recovery path for the silent-failure modes
        documented in v0.14.x:

        - **Empty embeddings.** A graph ingested without an embedder
          stores ``Node.embedding = []``. Wiring an embedder later
          does not retroactively embed those nodes — the HNSW
          index stays empty and vector search degrades to "FTS only"
          on the affected slice.

        - **Missing phrase hubs.** A graph ingested without a
          ``phrase_extractor`` (the default for the MCP server
          before v0.14.3) has no cross-document bridges, because
          no chunks ever got linked to shared ENTITY phrase hubs
          via CONTAINS edges. PPR / GraphExpander then can't walk
          across files.

        Both gaps used to require a full re-ingest from source.
        ``backfill()`` walks the existing graph in place and
        repairs each node where the relevant signal is missing,
        without touching nodes that are already healthy. It is
        idempotent — running twice on the same graph produces
        zero work on the second pass.

        Args:
            embeddings: If True (default) and an embedder is wired,
                fill in empty embeddings batch-by-batch. No-op when
                the graph has no embedder.
            phrases: If True (default) and a phrase extractor is
                wired, scan text-bearing nodes that have no
                outgoing CONTAINS edge and run the extractor on
                them so phrase hubs get created. No-op when the
                graph has no phrase extractor.
            batch_size: Embedding batch size handed to
                ``embedder.embed_batch``. Phrase extraction is
                already per-node so this only affects embeddings.
            max_nodes: Optional cap on the total nodes scanned —
                useful for incremental progress on huge graphs.
                When ``None`` (default), every node is inspected.

        Returns:
            :class:`BackfillResult` with per-axis counts and any
            per-node errors that were collected (best-effort —
            one bad row never aborts the rest).
        """
        from time import time as _time

        t0 = _time()
        result = BackfillResult()

        # Skip both passes early if neither would do anything —
        # avoids touching the backend at all.
        do_embeddings = embeddings and self._embedder is not None
        do_phrases = phrases and self._phrase_extractor is not None
        if not (do_embeddings or do_phrases):
            return result

        all_nodes = await self._backend.list_nodes(
            limit=max_nodes if max_nodes is not None else 1_000_000
        )

        # ─── Pass 1 — embedding backfill ──────────────────────
        # Two reasons to keep this in a separate pass from phrases:
        #   1. Embedder API is batched — collecting a contiguous
        #      list of "to embed" nodes is much faster than
        #      one-call-per-node.
        #   2. The phrase pass below will re-fetch the freshly
        #      embedded nodes anyway (their tags may matter).
        if do_embeddings:
            assert self._embedder is not None
            pending: list[tuple[Node, str]] = []
            for node in all_nodes:
                result.scanned += 1
                if node.embedding:
                    continue  # already embedded
                text = f"{node.title} {node.content}".strip()
                if not text:
                    result.skipped_no_text += 1
                    continue
                pending.append((node, text))

                if len(pending) >= batch_size:
                    await self._flush_embedding_batch(pending, result)
                    pending = []
            if pending:
                await self._flush_embedding_batch(pending, result)
        else:
            # Still need to count the scan even when not doing
            # embeddings, so the caller's "scanned" reflects total
            # graph size on a phrase-only run.
            result.scanned += len(all_nodes)

        # ─── Pass 2 — phrase hub backfill ─────────────────────
        if do_phrases:
            assert self._phrase_extractor is not None
            # A node "needs" phrase backfill when it has text and
            # no outgoing CONTAINS edge yet. Phrase hubs themselves
            # (tagged ``_phrase``) are skipped because they ARE the
            # bridge, not a candidate.
            for node in all_nodes:
                if node.tags and "_phrase" in node.tags:
                    continue
                text = f"{node.title} {node.content}".strip()
                if not text:
                    continue
                outgoing = await self._backend.get_edges(node.id, direction="outgoing")
                if any(e.kind == EdgeKind.CONTAINS for e in outgoing):
                    continue  # already linked to phrase hubs
                try:
                    new_ids = await self._phrase_extractor.extract_and_link(
                        self,
                        node.id,
                        node.title,
                        node.content,
                    )
                except Exception as exc:
                    result.errors.append(f"phrases:{node.id}: {exc}")
                    continue
                if new_ids:
                    result.phrases_linked += len(new_ids)
                    if self._chunk_entity_index is not None:
                        # Mirror the registration path that add()
                        # would normally do at ingest time.
                        await self._register_chunk_entities(node)

        result.elapsed_ms = (_time() - t0) * 1000.0
        return result

    async def _flush_embedding_batch(
        self,
        pending: list[tuple[Node, str]],
        result: BackfillResult,
    ) -> None:
        """Embed a pending batch and persist the new embeddings.

        Extracted from :meth:`backfill` to keep the main loop
        readable. ``pending`` is consumed (never returned) so the
        caller can simply reset its list and continue.
        """
        if self._embedder is None or not pending:
            return
        try:
            embeddings = await self._embedder.embed_batch([text for _, text in pending])
        except Exception as exc:
            result.errors.append(f"embed_batch: {exc}")
            return
        for (node, _), vec in zip(pending, embeddings):
            if not vec:
                continue
            node.embedding = vec
            try:
                await self._backend.update_node(node)
                result.embeddings_filled += 1
            except Exception as exc:
                result.errors.append(f"update_node:{node.id}: {exc}")

    async def export_markdown(self, *, node_ids: list[str] | None = None) -> str:
        return await self._md_exporter.export(self._backend, node_ids=node_ids)

    async def export_json(self, *, node_ids: list[str] | None = None) -> str:
        return await self._json_exporter.export(self._backend, node_ids=node_ids)

    async def merge(
        self,
        source_id: str,
        target_id: str,
    ) -> Node | None:
        """Merge source node into target. Combines content, stats, edges.

        Source node is deleted after merge.
        Returns the updated target node, or None if either node is missing.
        """
        source = await self._backend.get_node(source_id)
        target = await self._backend.get_node(target_id)
        if source is None or target is None:
            return None

        # Merge content
        if source.content and source.content not in target.content:
            target.content = f"{target.content}\n\n{source.content}".strip()

        # Merge tags (deduplicate)
        merged_tags = list(dict.fromkeys([*target.tags, *source.tags]))
        target.tags = merged_tags

        # Merge stats
        target.access_count += source.access_count
        target.success_count += source.success_count
        target.failure_count += source.failure_count
        target.vitality = max(target.vitality, source.vitality)
        target.updated_at = time()

        # Re-point source's edges to target
        source_edges = await self._backend.get_edges(source_id)
        for edge in source_edges:
            new_src = target_id if edge.source_id == source_id else edge.source_id
            new_tgt = target_id if edge.target_id == source_id else edge.target_id
            if new_src != new_tgt:  # Avoid self-loops
                new_edge = Edge(
                    source_id=new_src,
                    target_id=new_tgt,
                    kind=edge.kind,
                    weight=edge.weight,
                )
                try:
                    await self._backend.save_edge(new_edge)
                except Exception:
                    pass  # Duplicate edge — skip

        await self._backend.update_node(target)
        await self._backend.delete_node(source_id)
        self._cache.invalidate(source_id)
        self._cache.invalidate(target_id)
        return target

    async def find_duplicates(
        self,
        *,
        threshold: float = 0.85,
        limit: int = 50,
    ) -> list[tuple[Node, Node, float]]:
        """Find potential duplicate node pairs based on title similarity.

        Returns list of (node_a, node_b, similarity_score) tuples.
        """
        nodes = await self._backend.list_nodes(limit=limit * 10)
        duplicates: list[tuple[Node, Node, float]] = []

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if nodes[i].kind != nodes[j].kind:
                    continue
                sim = SequenceMatcher(None, nodes[i].title.lower(), nodes[j].title.lower()).ratio()
                if sim >= threshold:
                    duplicates.append((nodes[i], nodes[j], sim))

        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates[:limit]

    async def stats(self) -> dict[str, int | float]:
        all_nodes = await self._backend.list_nodes(limit=10000)
        by_kind: dict[str, int] = {}
        by_level: dict[str, int] = {}
        for node in all_nodes:
            by_kind[str(node.kind)] = by_kind.get(str(node.kind), 0) + 1
            by_level[str(node.level)] = by_level.get(str(node.level), 0) + 1

        result: dict[str, int | float] = {"total_nodes": len(all_nodes)}
        for k, v in sorted(by_kind.items()):
            result[f"kind_{k}"] = v
        for k, v in sorted(by_level.items()):
            result[f"level_{k}"] = v

        cache_stats = self._cache.stats()
        result["cache_hit_rate"] = cache_stats["hit_rate"]
        result["cache_size"] = cache_stats["size"]
        return result

    async def build_evidence(
        self,
        query: str,
        *,
        search_result: SearchResult | None = None,
        limit: int = 10,
        max_steps: int = 8,
        max_tokens: int = 2048,
        max_sentences_per_node: int = 5,
        relevance_threshold: float = 0.2,
        embedding: list[float] | None = None,
    ) -> EvidenceChain:
        """Convert search results into an evidence chain optimized for small LLMs."""
        if search_result is None:
            if embedding is None and self._embedder is not None:
                embedding = await self._embedder.embed(query)
            search_result = await self.search(query, limit=limit, embedding=embedding)

        assembler = EvidenceAssembler(
            max_sentences_per_node=max_sentences_per_node,
            relevance_threshold=relevance_threshold,
            max_tokens=max_tokens,
        )
        return await assembler.assemble(
            self._backend,
            query,
            search_result,
            max_steps=max_steps,
        )

    # --- Conversation helpers ---

    async def add_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        *,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> tuple[Node, Node, Node]:
        """Add a conversation turn (user + assistant) linked to a session.

        Creates a SESSION node on first call for a given session_id.
        Returns (session_node, user_node, assistant_node).
        """
        from synaptic.models import _new_id

        if session_id is None:
            session_id = f"session_{_new_id()}"

        # Get or create session node
        session_node = await self._backend.get_node(session_id)
        if session_node is None:
            session_node = await self._store.add_node(
                f"Session {session_id[:8]}",
                "",
                kind=NodeKind.SESSION,
                tags=["_session"],
                source=session_id,
            )
            # Override the auto-generated ID with session_id
            await self._backend.delete_node(session_node.id)
            session_node.id = session_id
            await self._backend.save_node(session_node)

        turn_tags = [*tags] if tags else []

        # Create user message node
        user_node = await self._store.add_node(
            "user",
            user_msg,
            kind=NodeKind.OBSERVATION,
            tags=[*turn_tags, "_turn_user"],
        )

        # Create assistant message node
        assistant_node = await self._store.add_node(
            "assistant",
            assistant_msg,
            kind=NodeKind.OBSERVATION,
            tags=[*turn_tags, "_turn_assistant"],
        )

        # Link: user → assistant (FOLLOWED_BY)
        await self._store.add_edge(
            user_node.id,
            assistant_node.id,
            kind=EdgeKind.FOLLOWED_BY,
        )

        # Link: session → user (CONTAINS)
        await self._store.add_edge(
            session_id,
            user_node.id,
            kind=EdgeKind.CONTAINS,
        )

        # Link last turn to this one (FOLLOWED_BY)
        session_edges = await self._backend.get_edges(session_id, direction="outgoing")
        contained = [
            e for e in session_edges if e.kind == EdgeKind.CONTAINS and e.target_id != user_node.id
        ]
        if contained:
            # Find the most recent contained user node
            last_user_id = contained[-1].target_id
            # Get the assistant node linked from last user
            last_edges = await self._backend.get_edges(last_user_id, direction="outgoing")
            last_assistant = [e for e in last_edges if e.kind == EdgeKind.FOLLOWED_BY]
            if last_assistant:
                await self._store.add_edge(
                    last_assistant[-1].target_id,
                    user_node.id,
                    kind=EdgeKind.FOLLOWED_BY,
                )

        return session_node, user_node, assistant_node

    # --- Ontology persistence ---

    async def save_ontology(self) -> None:
        """Persist the OntologyRegistry to the graph as a TYPE_DEF node."""
        if self._ontology is None:
            return
        data = self._ontology.to_dict()
        # Use a fixed ID so we can find/update it
        node = Node(
            id="_ontology_schema_",
            kind=NodeKind.TYPE_DEF,
            title="Ontology Schema",
            content=json.dumps(data),
            tags=["_ontology", "_system"],
            level=ConsolidationLevel.L3_PERMANENT,
        )
        await self._backend.save_node(node)

    async def load_ontology(self) -> OntologyRegistry | None:
        """Load OntologyRegistry from the graph. Returns None if not found."""
        node = await self._backend.get_node("_ontology_schema_")
        if node is None:
            return None
        try:
            data = json.loads(node.content)
            registry = OntologyRegistry.from_dict(data)
            self._ontology = registry
            return registry
        except (json.JSONDecodeError, KeyError):
            return None
