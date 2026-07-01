"""SynapticGraph — main entry point (facade)."""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from time import time
from typing import TYPE_CHECKING, cast

logger = logging.getLogger(__name__)

_DEFAULT_FTS_SEED_MIN = 20
_DEFAULT_FTS_SEED_MULTIPLIER = 1


def _default_fts_seed_limit(limit: int) -> int:
    return max(_DEFAULT_FTS_SEED_MIN, max(0, limit) * _DEFAULT_FTS_SEED_MULTIPLIER)


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


def _stable_auto_chunk_doc_id(
    chunk: dict,
    *,
    ordinal: int,
    title: str,
    content: str,
) -> str:
    """Deterministic fallback doc_id for ``from_chunks`` records.

    Prefer source/title when present so adjacent chunks from the same
    parsed document coalesce. With no stable document hint, include the
    input ordinal to preserve the historical "one anonymous chunk = one
    anonymous document" shape while remaining replay-safe.
    """
    source = str(chunk.get("source") or "")
    category = str(chunk.get("category") or "")
    if source.strip():
        payload: dict[str, object] = {"source": source, "category": category}
    elif str(chunk.get("title") or "").strip():
        payload = {"title": title, "category": category}
    else:
        payload = {"ordinal": ordinal, "content": content}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"auto_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def _node_content_hash(node: Node) -> str:
    payload = {
        "id": node.id,
        "kind": str(node.kind),
        "title": node.title,
        "content": node.content,
        "tags": list(node.tags or []),
        "properties": dict(node.properties or {}),
        "source": node.source,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _edge_content_hash(edge: Edge) -> str:
    payload = {
        "id": edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "kind": str(edge.kind),
        "weight": edge.weight,
        "properties": dict(edge.properties or {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


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
    FeedbackSignal,
    MaintenanceResult,
    MemoryEvent,
    MemoryEventKind,
    MemoryHealthReport,
    MemoryScope,
    MemoryScore,
    MemorySignal,
    MemorySignalKind,
    Node,
    NodeKind,
    RetrievalEvent,
    SearchResult,
    memory_scope_key,
)
from synaptic.ontology import OntologyRegistry, build_agent_ontology
from synaptic.protocols import (
    Digester,
    EntityExtractor,
    KindClassifier,
    QueryDecomposer,
    QueryRewriter,
    RelationDetector,
    StorageBackend,
    TagExtractor,
)
from synaptic.search import HybridSearch
from synaptic.store import Store

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider


def _feedback_score_delta(
    signal: FeedbackSignal,
    success: bool | None,
    confidence: float,
) -> float:
    if success is True:
        return 0.20 * confidence
    if success is False:
        return -0.25 * confidence
    if signal == FeedbackSignal.SELECTED:
        return 0.02 * confidence
    if signal == FeedbackSignal.IGNORED:
        return -0.01 * confidence
    return 0.0


def _prop_bool(props: dict[str, str], key: str) -> bool:
    return str(props.get(key, "")).lower() in {"1", "true", "yes", "y"}


def _prop_float(props: dict[str, str], key: str, default: float) -> float:
    try:
        return float(props.get(key, default))
    except (TypeError, ValueError):
        return default


def _prop_int(props: dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(props.get(key, default)))
    except (TypeError, ValueError):
        return default


def _prop_csv_ids(props: dict[str, str], key: str) -> list[str]:
    raw = str(props.get(key, "") or "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _retrieval_event_properties(result: SearchResult) -> dict[str, str]:
    properties = {
        "query": result.query,
        "returned_count": str(len(result.nodes)),
        "total_candidates": str(int(result.total_candidates)),
        "search_time_ms": f"{float(result.search_time_ms):.6f}",
        "stages_used": ",".join(result.stages_used),
    }
    for key, value in sorted((result.diagnostics or {}).items()):
        if not key.startswith("memory_"):
            continue
        if isinstance(value, int | float):
            properties[key] = f"{float(value):.6f}"
        else:
            properties[key] = str(value)
    return properties


def _node_source_label(node: Node) -> str:
    props = node.properties or {}
    return (
        node.source
        or props.get("source", "")
        or props.get("source_id", "")
        or props.get("doc_id", "")
        or props.get("source_event_id", "")
        or props.get("source_chunk_id", "")
        or node.id
    )


def _openie_output_tokens_for_profile(profile: str, requested: int) -> int:
    requested = max(1, int(requested))
    if profile == "deepseek_v4_flash" and requested == 1024:
        return 4096
    return requested


def _infer_openie_model_profile(model: str) -> str:
    model_l = model.lower()
    if "deepseek-v4-flash" in model_l or "deepseek_v4_flash" in model_l:
        return "deepseek_v4_flash"
    if "qwen3.6" in model_l or "qwen36" in model_l:
        return "qwen36_local"
    return ""


def _semantic_extract_profile_key(event: MemoryEvent) -> str:
    props = event.properties or {}
    source = event.source or "unknown"
    extractor = props.get("extractor", "") or event.source_id or "unknown"
    model = props.get("model", "") or "unknown"
    prompt_version = props.get("prompt_version", "") or "unknown"
    return f"source={source};extractor={extractor};model={model};prompt_version={prompt_version}"


_MEMORY_SIGNAL_MIN_PENALTY_CONFIDENCE = 0.7
_MEMORY_SIGNAL_MAX_RANKING_PENALTY = 0.05
_MEMORY_STRONG_NEGATIVE_SCORE_SIGNAL_THRESHOLD = -0.5
_MEMORY_DRIFT_MIN_FAILURES = 3
_MEMORY_DRIFT_MIN_FAILURE_RATE = 0.25
_MEMORY_PROPERTY_CONFLICT_IGNORED_KEYS = frozenset(
    {
        "chunk_id",
        "chunk_index",
        "confidence",
        "doc_id",
        "edge_ids",
        "is_openie",
        "node_ids",
        "page",
        "page_number",
        "scope_key",
        "score",
        "signal_kind",
        "source",
        "source_chunk_id",
        "source_event_id",
        "source_id",
    }
)


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
        "_connected",
        "_consolidation",
        "_corpus_size",
        "_embedder",
        "_evidence_search_cache",
        "_hebbian",
        "_json_exporter",
        "_md_exporter",
        "_ontology",
        "_phrase_extractor",
        "_query_decomposer",
        "_relation_detector",
        "_reranker",
        "_reranker_weights",
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
        phrase_extractor: EntityExtractor | None = None,
        chunk_entity_index: ChunkEntityIndex | None = None,
        query_decomposer: QueryDecomposer | None = None,
        reranker: object | None = None,
        reranker_weights: object | None = None,
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
        # Optional RerankerWeights for the hybrid reranker. None keeps the
        # built-in defaults; set it (e.g. via the ``reranker_weights`` property
        # or a factory arg) to enable the usage/time memory axis.
        self._reranker_weights = reranker_weights
        self._evidence_search_cache: dict[tuple[int, ...], object] = {}
        self._agent_search = AgentSearch(hybrid=self._search)
        self._corpus_size = 0
        # Tracks whether this graph has connected its backend, so
        # connect() is idempotent and the one-line constructors (which
        # connect eagerly) don't double-connect via ``async with``.
        self._connected = False

    # --- Lifecycle ---

    async def connect(self) -> None:
        """Connect the underlying storage backend. Idempotent.

        Gives every graph a uniform lifecycle regardless of how it was
        built: the ``memory()`` / ``sqlite()`` / ``full()`` factories
        return an unconnected graph — call this (or use ``async with``)
        instead of reaching into ``graph.backend``.
        """
        if self._connected:
            return
        connect = getattr(self._backend, "connect", None)
        if connect is not None:
            await connect()
        self._connected = True

    async def close(self) -> None:
        """Close the underlying storage backend and release its resources.

        Safe to call more than once. Pairs with the one-line
        constructors (which connect the backend for you) so a caller
        never has to reach into ``graph.backend`` to clean up.
        """
        close = getattr(self._backend, "close", None)
        if close is not None:
            await close()
        self._connected = False

    async def __aenter__(self) -> SynapticGraph:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

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
            await graph.connect()
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
            await graph.connect()
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
        openie_enabled: bool = False,
        openie_seed: int | None = 42,
        openie_cache_path: str = "",
        openie_alias_map: dict[str, str] | None = None,
        openie_relation_whitelist: tuple[str, ...] = (),
        openie_model_profile: str = "",
        openie_max_output_tokens: int = 1024,
        openie_max_triples_per_chunk: int = 24,
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
        phrase_extractor: EntityExtractor = PhraseExtractor()

        if llm is not None:
            from synaptic.extensions.classifier_hybrid import HybridClassifier
            from synaptic.extensions.classifier_llm import LLMClassifier
            from synaptic.extensions.relation_detector_llm import (
                LLMRelationDetector,
            )

            classifier = HybridClassifier(
                RuleBasedClassifier(),
                LLMClassifier(llm, fallback=RuleBasedClassifier()),
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

        if openie_enabled and llm is not None:
            from pathlib import Path

            from synaptic.extensions.entity_extractor_openie import (
                ChainedEntityExtractor,
                LLMOpenIEExtractor,
            )

            model_profile = openie_model_profile or _infer_openie_model_profile(
                str(getattr(llm, "_model", "") or getattr(llm, "model", "") or "")
            )
            phrase_extractor = ChainedEntityExtractor(
                phrase_extractor,
                LLMOpenIEExtractor(
                    llm,
                    seed=openie_seed,
                    alias_map=openie_alias_map,
                    relation_whitelist=openie_relation_whitelist,
                    max_output_tokens=_openie_output_tokens_for_profile(
                        model_profile,
                        openie_max_output_tokens,
                    ),
                    max_triples_per_chunk=openie_max_triples_per_chunk,
                    cache_path=Path(openie_cache_path) if openie_cache_path else None,
                ),
            )

        return cls(
            backend,
            classifier=classifier,
            relation_detector=relation_detector,
            embedder=embedder,
            ontology=build_agent_ontology(),
            phrase_extractor=phrase_extractor,
            cache_size=cache_size,
        )

    # --- Easy API ---

    @staticmethod
    async def _open_backend(backend: StorageBackend | None, db: str) -> StorageBackend:
        """Resolve the storage backend for a one-line constructor.

        When ``backend`` is given it is used as-is — the caller owns its
        connection lifecycle (same contract as the ``SynapticGraph``
        constructor). When omitted, a SQLite graph backend is created at
        ``db`` and connected. This is what lets the one-line API target
        Postgres / Kuzu / Composite instead of being locked to SQLite.
        """
        if backend is not None:
            return backend
        from synaptic.backends.sqlite_graph import SqliteGraphBackend

        b = SqliteGraphBackend(db)
        await b.connect()
        return b

    @staticmethod
    async def _embed_all_nodes(backend: StorageBackend, embedder: object) -> None:
        """Embed every node in the graph in batches.

        Shared by ``from_data`` / ``from_chunks`` / ``from_database`` so
        the embedding pass exists in exactly one place.
        """
        nodes = await backend.list_nodes(kind=None, limit=100_000)
        for i in range(0, len(nodes), 32):
            batch = nodes[i : i + 32]
            texts = [f"{n.title}\n{(n.content or '')[:300]}" for n in batch]
            try:
                vecs = await embedder.embed_batch(texts)  # type: ignore[attr-defined]
                changed = []
                for n, v in zip(batch, vecs):
                    if v:
                        n.embedding = v
                        changed.append(n)
                if not changed:
                    continue
                save_batch = getattr(backend, "save_nodes_batch", None)
                if callable(save_batch):
                    await save_batch(changed)
                else:
                    for node in changed:
                        await backend.save_node(node)
            except Exception:
                logger.warning("embedding pass failed for a batch", exc_info=True)

    @classmethod
    async def _finalize(
        cls,
        backend: StorageBackend,
        *,
        embed_url: str | None,
        embed_model: str,
        rerank_url: str | None = None,
        rerank_backend: str = "vllm",
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        calibrate: bool = False,
        connect: bool = False,
    ) -> SynapticGraph:
        """Run the optional embedding pass and assemble the graph.

        Builds the embedder / reranker from URLs and — crucially —
        wires both into the returned graph so query-time vector search
        and reranking work without the caller re-supplying them.

        ``calibrate`` is opt-in (default False). Auto-calibration was
        prototyped in v0.17 to disable the cross-encoder on FTS-near-
        optimal corpora, but v0.26 measurement showed pseudo-self-
        retrieval signal cannot distinguish "FTS strong + reranker
        unhelpful" (AutoRAG) from "FTS strong + reranker still helps
        via paraphrase boost" (Allganize, HotPotQA, PublicHealthQA).
        Auto-applying it regressed 4/5 quick benches. Call
        ``await graph.calibrate()`` manually if you have measured your
        corpus and want to override the default blend.
        """
        embedder: object | None = None
        if embed_url:
            from synaptic.extensions.embedder import OpenAIEmbeddingProvider

            embedder = OpenAIEmbeddingProvider(api_base=embed_url, model=embed_model)
            await cls._embed_all_nodes(backend, embedder)

        reranker: object | None = None
        if rerank_url:
            from synaptic.extensions.reranker_cross import reranker_from_url

            reranker = reranker_from_url(rerank_url, backend=rerank_backend, model=rerank_model)

        graph = cls(backend, embedder=embedder, reranker=reranker)
        # The one-line path connected the backend in _open_backend.
        graph._connected = True
        if calibrate:
            try:
                await graph.calibrate()
            except Exception as exc:  # pragma: no cover - non-fatal
                logger.warning("auto-calibration failed: %s", exc)
        if connect:
            try:
                await graph.connect_components()
            except Exception as exc:  # pragma: no cover - non-fatal
                logger.warning("connect_components failed: %s", exc)
        return graph

    async def calibrate(self, *, sample_size: int = 20) -> object | None:
        """Sample FTS-only MRR on the current corpus and persist a
        per-corpus rerank_blend to backend metadata.

        Runs once at the end of bulk ingest (the ``from_*`` helpers call
        this automatically). Callers using the lower-level constructor +
        ``graph.add()`` ingest path should invoke it explicitly after
        their last write. Cheap (N FTS calls, no LLM, no embedder).

        Returns the :class:`CalibrationResult` for inspection / logging,
        or None when the corpus has no content-bearing nodes.
        """
        from synaptic.extensions.calibration import (
            calibrate_corpus,
            write_calibration,
        )

        result = await calibrate_corpus(self._backend, sample_size=sample_size)
        await write_calibration(self._backend, result)
        logger.info(
            "calibration written: sample_mrr=%.3f → rerank_blend=%s",
            result.sample_mrr,
            result.rerank_blend,
        )
        return result

    async def navigability(self) -> object:
        """Diagnose how navigable the graph is — components + isolated %.

        A read-only structure-health check: how fragmented is the corpus? A
        well-structured graph is one (or few) component(s) with ~0 % isolated;
        real corpora often aren't (e.g. KRRA 28.9 % isolated). When this reports
        fragmentation, :meth:`connect_components` fixes it. No writes.

        Returns a :class:`BridgeStats` with ``components_before`` /
        ``isolated_before`` populated (``*_after`` mirror them — nothing changed).
        """
        from synaptic.extensions.connectivity import bridge_components

        stats = await bridge_components(self._backend, dry_run=True)
        logger.info("navigability: %s", stats.summary())
        return stats

    async def connect_components(
        self,
        *,
        k: int = 10,
        min_similarity: float = 0.0,
        max_bridges: int | None = None,
    ) -> object:
        """Make a fragmented corpus navigable — add a minimal semantic backbone.

        Real corpora leave nodes with no containment / FK / co-occurrence edge
        as unreachable islands (an agent can't traverse an edge that doesn't
        exist). This bridges every embedded node into one component using a
        Max-Spanning-Forest over HNSW-nearest cross-component pairs — fewest,
        highest-quality edges, no LLM, no per-domain logic. Only islands are
        queried, so the dense core is left untouched. Opt-in (not auto-run):
        call after ingest. Idempotent.

        Returns a :class:`BridgeStats` (components / isolated before-and-after).
        """
        from synaptic.extensions.connectivity import bridge_components

        stats = await bridge_components(
            self._backend, k=k, min_similarity=min_similarity, max_bridges=max_bridges
        )
        logger.info("connect_components: %s", stats.summary())
        return stats

    @classmethod
    async def from_data(
        cls,
        data_path: str,
        *,
        db: str = "synaptic.db",
        backend: StorageBackend | None = None,
        profile: object | None = None,
        openie_extractor: EntityExtractor | None = None,
        openie_enabled: bool = False,
        embed_url: str | None = None,
        embed_model: str = "qwen3-embedding:4b",
        rerank_url: str | None = None,
        rerank_backend: str = "vllm",
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        connect: bool = False,
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

            # Ontology relations — pass a profile that declares the
            # document identifier so cross-references become graph edges
            graph = await SynapticGraph.from_data(
                "./statutes/", profile="profiles/finreg.toml"
            )

        Args:
            backend: Optional pre-built :class:`StorageBackend` — pass a
                connected Postgres / Kuzu / Composite backend to target
                it instead of the default SQLite. When given, the caller
                owns its connection lifecycle. When omitted, a SQLite
                graph backend is created at ``db``.
            profile: Optional :class:`DomainProfile` (or a TOML path).
                When omitted, a profile is auto-generated from samples —
                that builds the Category→Document→Chunk hierarchy but
                **not** typed relation edges. To get REFERENCES edges
                (multi-hop ontology) the profile must declare
                ``reference_key_property``; the auto-generated one never
                does. See ``docs/PLAN-v0.24-relation-enrichment.md``.
            openie_extractor: Optional opt-in OpenIE extractor. The
                extractor runs only when the profile has
                ``openie_enabled=True`` or this factory's
                ``openie_enabled`` flag is set.
            embed_url: OpenAI-compatible ``/v1`` base URL for the
                embedder. When set, nodes are embedded at ingest time
                and the embedder is wired into the returned graph for
                query-time vector search.
            rerank_url: Base URL of a cross-encoder reranker server.
                When set, a reranker is wired into the returned graph
                (EvidenceSearch step 4b). ``rerank_backend`` selects
                the wire format — ``"vllm"`` (default; ``vllm serve
                <model> --task score``), ``"ollama"``, or ``"tei"``.
            connect: Opt-in (default False). After ingest, run
                :meth:`connect_components` to bridge fragmented islands into one
                navigable graph (real corpora leave 29 %+ of nodes unreachable).
                Off by default because its effect on retrieval is not yet
                measured; call :meth:`navigability` to see your fragmentation
                first. LLM-free, deterministic.

        Example with a vLLM-served reranker::

            graph = await SynapticGraph.from_data(
                "./docs/",
                embed_url="http://localhost:8000/v1",
                rerank_url="http://localhost:8001",
                rerank_model="BAAI/bge-reranker-v2-m3",
            )
        """
        from pathlib import Path

        from synaptic.extensions.document_ingester import (
            DocumentIngester,
            JsonlDocumentSource,
        )
        from synaptic.extensions.profile_generator import ProfileGenerator
        from synaptic.extensions.table_ingester import TableIngester

        path = Path(data_path)
        backend = await cls._open_backend(backend, db)

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

        # A caller-supplied profile is what enables ontology relation
        # enrichment (REFERENCES edges): the auto-generated profile never
        # declares ``reference_key_property``, so without an explicit
        # profile the one-line API only builds the Category→Document→
        # Chunk hierarchy. Accepts a DomainProfile instance or a TOML path.
        if isinstance(profile, str):
            from synaptic.extensions.domain_profile import DomainProfile

            profile = DomainProfile.load(profile)

        if profile is None:
            # Auto-generate a profile from samples of the input files.
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
        if openie_enabled and profile is not None:
            profile.openie_enabled = True  # type: ignore[attr-defined]

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
                doc_ingester = DocumentIngester(
                    profile=profile,
                    backend=backend,
                    openie_extractor=openie_extractor,
                )
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
                    doc_ingester = DocumentIngester(
                        profile=profile,
                        backend=backend,
                        openie_extractor=openie_extractor,
                    )
                    await doc_ingester.ingest(source)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

        return await cls._finalize(
            backend,
            embed_url=embed_url,
            embed_model=embed_model,
            rerank_url=rerank_url,
            rerank_backend=rerank_backend,
            rerank_model=rerank_model,
            connect=connect,
        )

    @classmethod
    async def from_chunks(
        cls,
        chunks: list[dict],
        *,
        db: str = "synaptic.db",
        backend: StorageBackend | None = None,
        profile: object | None = None,
        openie_extractor: EntityExtractor | None = None,
        openie_enabled: bool = False,
        embed_url: str | None = None,
        embed_model: str = "qwen3-embedding:4b",
        rerank_url: str | None = None,
        rerank_backend: str = "vllm",
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
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
            backend: Optional pre-built :class:`StorageBackend` (see
                :meth:`from_data`). Defaults to SQLite at ``db``.
            openie_extractor: Optional opt-in OpenIE extractor. Runs
                only when ``openie_enabled=True`` or the supplied profile
                has ``openie_enabled=True``.
            embed_url: OpenAI-compatible endpoint to embed nodes after
                ingest. When set, the embedder is also wired into the
                returned graph for query-time vector search.
            embed_model: Embedder model name.
            rerank_url: Optional cross-encoder reranker server URL — see
                :meth:`from_data` for ``rerank_backend`` / ``rerank_model``.

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

        from synaptic.extensions.document_ingester import (
            DocumentIngester,
            JsonlDocumentSource,
        )
        from synaptic.extensions.profile_generator import ProfileGenerator

        backend = await cls._open_backend(backend, db)

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
        if openie_enabled and profile is not None:
            profile.openie_enabled = True  # type: ignore[attr-defined]

        # Materialise chunks into a temp JSONL so they flow through
        # the same DocumentIngester path that JSONL files use — keeps
        # NFC, FTS indexing, edge construction, and embedder hooks
        # consistent regardless of input shape.
        import json as _json
        import tempfile

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        try:
            records_by_doc: dict[str, dict[str, object]] = {}
            for i, c in enumerate(chunks):
                content = str(c.get("content") or "").strip()
                if not content:
                    continue
                title_hint = str(c.get("title") or content.split("\n", 1)[0][:80])
                doc_id = str(
                    c.get("doc_id")
                    or _stable_auto_chunk_doc_id(
                        c,
                        ordinal=i,
                        title=title_hint,
                        content=content,
                    )
                )
                title = str(c.get("title") or title_hint or doc_id)
                record = records_by_doc.get(doc_id)
                if record is None:
                    record = {
                        "doc_id": doc_id,
                        "title": title,
                        "content": "",
                        "category": c.get("category", ""),
                        "source": c.get("source", ""),
                        "chunks": [],
                    }
                    records_by_doc[doc_id] = record
                record_chunks_obj = record["chunks"]
                if not isinstance(record_chunks_obj, list):
                    continue
                record_chunks = record_chunks_obj
                chunk_index = int(c.get("chunk_index", len(record_chunks)) or 0)
                chunk_record = {
                    "chunk_id": str(c.get("chunk_id") or f"{doc_id}:{chunk_index:05d}"),
                    "doc_id": doc_id,
                    "text": content,
                    "index": chunk_index,
                }
                if c.get("page") is not None:
                    chunk_record["page_number"] = c["page"]
                record_chunks.append(chunk_record)
            for record in records_by_doc.values():
                tmp.write(_json.dumps(record, ensure_ascii=False) + "\n")
            tmp.close()

            source = JsonlDocumentSource(tmp.name, None)
            doc_ingester = DocumentIngester(
                profile=profile,
                backend=backend,
                openie_extractor=openie_extractor,
            )
            await doc_ingester.ingest(source)
        finally:
            _Path(tmp.name).unlink(missing_ok=True)

        return await cls._finalize(
            backend,
            embed_url=embed_url,
            embed_model=embed_model,
            rerank_url=rerank_url,
            rerank_backend=rerank_backend,
            rerank_model=rerank_model,
        )

    @classmethod
    async def from_database(
        cls,
        connection_string: str,
        *,
        db: str = "synaptic.db",
        backend: StorageBackend | None = None,
        tables: list[str] | None = None,
        row_limit: int = 500_000,
        mode: str = "full",
        embed_url: str | None = None,
        embed_model: str = "qwen3-embedding:4b",
        rerank_url: str | None = None,
        rerank_backend: str = "vllm",
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
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

            # With vector search + reranker (parity with from_data)
            graph = await SynapticGraph.from_database(
                "sqlite:///shop.db",
                embed_url="http://localhost:8000/v1",
                rerank_url="http://localhost:8001",
            )

        Args:
            backend: Optional pre-built :class:`StorageBackend` for the
                graph store — see :meth:`from_data`. Defaults to SQLite.
            embed_url / rerank_url: Optional embedder / reranker server
                URLs — same semantics as :meth:`from_data`.
        """
        from synaptic.extensions.db_ingester import DbIngester

        backend = await cls._open_backend(backend, db)
        graph = cls(backend)
        ingester = DbIngester()
        cdc_done = False

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
                cdc_done = True
            elif connection_string.startswith("postgresql"):
                await ingester.sync_from_postgres(
                    connection_string,
                    graph,
                    source_url=connection_string,
                    tables=tables,
                    row_limit=row_limit,
                )
                cdc_done = True
            elif connection_string.startswith("mysql") or connection_string.startswith("mariadb"):
                await ingester.sync_from_mysql(
                    connection_string,
                    graph,
                    source_url=connection_string,
                    tables=tables,
                    row_limit=row_limit,
                )
                cdc_done = True
            # Other dialects fall through to the legacy ingest_from_*
            # path with deterministic IDs (no incremental sync yet).

        if cdc_done:
            pass
        elif connection_string.startswith("sqlite"):
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

        if not cdc_done:
            logging.getLogger("db-ingester").info(
                "from_database: %d tables, %d rows, %d nodes, %.1fs",
                stats.tables_ingested,
                stats.total_rows,
                stats.total_nodes,
                stats.elapsed_seconds,
            )

        return await cls._finalize(
            backend,
            embed_url=embed_url,
            embed_model=embed_model,
            rerank_url=rerank_url,
            rerank_backend=rerank_backend,
            rerank_model=rerank_model,
        )

    # --- Synchronous constructors ---
    #
    # Thin blocking wrappers for callers not running an event loop
    # (plain scripts, notebooks doing a one-time build). They only
    # cover graph *construction* — the returned graph's methods
    # (search, add, …) remain async.

    @staticmethod
    def _run_sync(coro: object) -> SynapticGraph:
        """Run a constructor coroutine to completion on a fresh loop.

        Raises a clear error when called from inside a running event
        loop, where ``asyncio.run`` would deadlock — use the async
        constructor directly there.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)  # type: ignore[arg-type]
        coro.close()  # type: ignore[attr-defined]
        msg = (
            "*_sync() cannot run inside an active event loop "
            "(e.g. Jupyter / async code) — await the async constructor instead."
        )
        raise RuntimeError(msg)

    @classmethod
    def from_data_sync(cls, *args: object, **kwargs: object) -> SynapticGraph:
        """Blocking wrapper for :meth:`from_data` — same arguments."""
        return cls._run_sync(cls.from_data(*args, **kwargs))  # type: ignore[arg-type]

    @classmethod
    def from_chunks_sync(cls, *args: object, **kwargs: object) -> SynapticGraph:
        """Blocking wrapper for :meth:`from_chunks` — same arguments."""
        return cls._run_sync(cls.from_chunks(*args, **kwargs))  # type: ignore[arg-type]

    @classmethod
    def from_database_sync(cls, *args: object, **kwargs: object) -> SynapticGraph:
        """Blocking wrapper for :meth:`from_database` — same arguments."""
        return cls._run_sync(cls.from_database(*args, **kwargs))  # type: ignore[arg-type]

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

    @property
    def reranker_weights(self) -> object | None:
        """Weights for the hybrid reranker (a :class:`RerankerWeights`).

        ``None`` (default) keeps the built-in weights. Set this to enable
        the usage/time **memory axis** — retrieval that evolves as nodes are
        reinforced/decayed, which a static index cannot do::

            from synaptic.extensions.hybrid_reranker import RerankerWeights
            g.reranker_weights = RerankerWeights(
                lexical=0.35, semantic=0.20, graph=0.10,
                structural=0.10, memory=0.25,
            )

        Read fresh on every :meth:`search`, so it can be changed at runtime.
        """
        return self._reranker_weights

    @reranker_weights.setter
    def reranker_weights(self, weights: object | None) -> None:
        self._reranker_weights = weights
        self._evidence_search_cache.clear()

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
            embed_text = self._compose_embed_text(title, content, properties)
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

        await self._save_memory_event(
            MemoryEvent(
                kind=MemoryEventKind.INGEST,
                source=node.source or "graph",
                source_id=node.id,
                content_hash=_node_content_hash(node),
                node_ids=[node.id],
                edge_ids=await self._touching_edge_ids(node.id),
                properties={"operation": "SynapticGraph.add", "kind": str(node.kind)},
            )
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
        edge = await self._store.add_edge(source_id, target_id, kind=kind, weight=weight)
        await self._save_memory_event(
            MemoryEvent(
                kind=MemoryEventKind.INGEST,
                source="graph",
                source_id=edge.id,
                content_hash=_edge_content_hash(edge),
                node_ids=[source_id, target_id],
                edge_ids=[edge.id],
                properties={
                    "operation": "SynapticGraph.link",
                    "edge_kind": str(edge.kind),
                    "weight": str(edge.weight),
                },
            )
        )
        return edge

    async def unlink(
        self,
        source_id: str,
        target_id: str,
        *,
        kind: EdgeKind | str | None = None,
    ) -> int:
        """Delete edges from ``source_id`` to ``target_id``.

        If ``kind`` is given, only edges of that kind are removed; otherwise
        every outgoing edge from source to target is removed. Returns the
        number of edges deleted.
        """
        kind_str = str(kind) if kind is not None else None
        edges = await self._backend.get_edges(source_id, direction="outgoing")
        removed_edges: list[Edge] = []
        removed = 0
        for edge in edges:
            if edge.target_id != target_id:
                continue
            if kind_str is not None and str(edge.kind) != kind_str:
                continue
            removed_edges.append(edge)
            await self._backend.delete_edge(edge.id)
            removed += 1
        if removed_edges:
            await self._save_memory_event(
                MemoryEvent(
                    kind=MemoryEventKind.DELETE,
                    source="graph",
                    source_id=",".join(edge.id for edge in removed_edges),
                    content_hash=hashlib.sha256(
                        "|".join(_edge_content_hash(edge) for edge in removed_edges).encode()
                    ).hexdigest(),
                    node_ids=list(dict.fromkeys([source_id, target_id])),
                    edge_ids=[edge.id for edge in removed_edges],
                    properties={
                        "operation": "SynapticGraph.unlink",
                        "edge_count": str(len(removed_edges)),
                        "edge_kinds": ",".join(sorted({str(edge.kind) for edge in removed_edges})),
                    },
                )
            )
        return removed

    async def update_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        kind: EdgeKind | str | None = None,
        new_weight: float | None = None,
        new_kind: EdgeKind | str | None = None,
    ) -> int:
        """Update edges matching (source_id, target_id[, kind]).

        ``kind`` filters which edges to update (None = all between the pair).
        ``new_weight`` / ``new_kind`` are the values to apply. When ``new_kind``
        is set and an ontology is bound, the new kind is re-validated against
        the source/target node kinds.

        Returns the number of edges updated.
        """
        if new_weight is None and new_kind is None:
            return 0
        filter_kind = str(kind) if kind is not None else None
        resolved_new_kind: EdgeKind | None = None
        if new_kind is not None:
            resolved_new_kind = (
                new_kind if isinstance(new_kind, EdgeKind) else EdgeKind(str(new_kind))
            )
            if self._ontology:
                src_node = await self._backend.get_node(source_id)
                tgt_node = await self._backend.get_node(target_id)
                if src_node is not None and tgt_node is not None:
                    errors = self._ontology.validate_edge(
                        str(resolved_new_kind),
                        str(src_node.kind),
                        str(tgt_node.kind),
                    )
                    if errors:
                        msg = f"Ontology validation failed: {'; '.join(errors)}"
                        raise ValueError(msg)
        edges = await self._backend.get_edges(source_id, direction="outgoing")
        updated = 0
        updated_edges: list[Edge] = []
        previous_hashes: dict[str, str] = {}
        for edge in edges:
            if edge.target_id != target_id:
                continue
            if filter_kind is not None and str(edge.kind) != filter_kind:
                continue
            previous_hashes[edge.id] = _edge_content_hash(edge)
            if resolved_new_kind is not None:
                edge.kind = resolved_new_kind
            if new_weight is not None:
                edge.weight = new_weight
            await self._backend.update_edge(edge)
            updated_edges.append(edge)
            updated += 1
        if updated_edges:
            await self._save_memory_event(
                MemoryEvent(
                    kind=MemoryEventKind.UPDATE,
                    source="graph",
                    source_id=",".join(edge.id for edge in updated_edges),
                    content_hash=hashlib.sha256(
                        "|".join(_edge_content_hash(edge) for edge in updated_edges).encode()
                    ).hexdigest(),
                    node_ids=list(dict.fromkeys([source_id, target_id])),
                    edge_ids=[edge.id for edge in updated_edges],
                    properties={
                        "operation": "SynapticGraph.update_edge",
                        "edge_count": str(len(updated_edges)),
                        "previous_edge_hashes": json.dumps(
                            previous_hashes,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "new_kind": str(new_kind) if new_kind is not None else "",
                        "new_weight": str(new_weight) if new_weight is not None else "",
                    },
                )
            )
        return updated

    async def merge_nodes(
        self,
        keep_id: str,
        drop_id: str,
        *,
        merge_tags: bool = True,
        merge_properties: bool = True,
        append_content: bool = False,
    ) -> Node | None:
        """Merge ``drop_id`` into ``keep_id``.

        All edges incident to ``drop_id`` are re-pointed to ``keep_id``.
        Duplicate edges (same other-endpoint + kind) are deduplicated by
        keeping the higher weight. Self-loops created by the re-point are
        dropped. Then ``drop_id`` is deleted.

        - ``merge_tags``: union the two tag lists onto ``keep``.
        - ``merge_properties``: keys from ``drop`` fill in keys that
          ``keep`` doesn't already have (keep wins on conflict).
        - ``append_content``: append drop.content to keep.content with a
          separator. Default off to avoid silent content bloat.

        Returns the updated keep node, or None if either ID is missing.
        """
        if keep_id == drop_id:
            msg = "merge_nodes: keep_id and drop_id must differ"
            raise ValueError(msg)

        keep = await self._backend.get_node(keep_id)
        drop = await self._backend.get_node(drop_id)
        if keep is None or drop is None:
            return None

        # Index keep's existing edges so we can dedupe.
        keep_out = await self._backend.get_edges(keep_id, direction="outgoing")
        keep_in = await self._backend.get_edges(keep_id, direction="incoming")
        out_by_pair: dict[tuple[str, str], Edge] = {(e.target_id, str(e.kind)): e for e in keep_out}
        in_by_pair: dict[tuple[str, str], Edge] = {(e.source_id, str(e.kind)): e for e in keep_in}

        # Re-point drop's outgoing edges.
        for edge in await self._backend.get_edges(drop_id, direction="outgoing"):
            new_target = edge.target_id
            if new_target == keep_id:
                # would become a self-loop on keep — discard
                await self._backend.delete_edge(edge.id)
                continue
            key = (new_target, str(edge.kind))
            existing = out_by_pair.get(key)
            if existing is not None:
                # dedupe: keep the higher-weight edge, drop the rest
                if edge.weight > existing.weight:
                    existing.weight = edge.weight
                    await self._backend.update_edge(existing)
                await self._backend.delete_edge(edge.id)
            else:
                edge.source_id = keep_id
                await self._backend.update_edge(edge)
                out_by_pair[key] = edge

        # Re-point drop's incoming edges.
        for edge in await self._backend.get_edges(drop_id, direction="incoming"):
            new_source = edge.source_id
            if new_source == keep_id:
                await self._backend.delete_edge(edge.id)
                continue
            key = (new_source, str(edge.kind))
            existing = in_by_pair.get(key)
            if existing is not None:
                if edge.weight > existing.weight:
                    existing.weight = edge.weight
                    await self._backend.update_edge(existing)
                await self._backend.delete_edge(edge.id)
            else:
                edge.target_id = keep_id
                await self._backend.update_edge(edge)
                in_by_pair[key] = edge

        # Combine metadata onto keep.
        if merge_tags and drop.tags:
            seen = set(keep.tags)
            for t in drop.tags:
                if t not in seen:
                    keep.tags.append(t)
                    seen.add(t)
        if merge_properties and drop.properties:
            for k, v in drop.properties.items():
                keep.properties.setdefault(k, v)
        if append_content and drop.content:
            sep = "\n\n" if keep.content else ""
            keep.content = f"{keep.content}{sep}{drop.content}"
        keep.updated_at = time()
        await self._backend.update_node(keep)

        # Delete drop. Use the higher-level remove() so caches/relation
        # indexes are cleaned up alongside the cascade.
        await self.remove(drop_id)
        self._cache.invalidate(keep_id)
        self._cache.put(keep)
        return keep

    async def chat(
        self,
        query: str,
        *,
        llm_client: object,
        model: str = "gpt-4o-mini",
        max_turns: int = 5,
        system_prompt: str | None = None,
        extra_context: str | None = None,
        prime_with_snapshot: bool = True,
        sufficiency_gate: bool = True,
        gate_bridge: bool = False,
        efficiency_hint: bool = True,
        record_trace: bool = False,
    ):
        """Multi-turn agent loop — Synaptic's measured-strongest mode.

        Drop-in upgrade for ``graph.search()`` when Hard / Conv questions
        outpace single-shot retrieval. Measured outcome on the 6 custom
        Hard / Conv benches: mean 81 % solved with Qwen3.5-27B vLLM,
        versus 0.30 mean MRR for the equivalent single-shot path.

        The name distinguishes this from :meth:`agent_search` (legacy
        intent-routing single-shot) — ``chat`` is a true multi-turn
        tool-using LLM dialogue.

        Args:
            query: User question.
            llm_client: OpenAI-compatible async client (e.g.
                ``openai.AsyncOpenAI``). Must implement
                ``chat.completions.create(model, messages, tools, max_tokens)``.
                vLLM / Ollama / Anthropic shims all work.
            model: Model name forwarded to the LLM client.
            max_turns: Max LLM dialogue turns (each may emit multiple
                tool calls). Loop ends early on a non-tool message.
            system_prompt: Override the default agent system prompt
                (``synaptic.agent_loop.AGENT_SYSTEM``). The graph
                context is always appended.
            extra_context: Additional per-corpus instructions appended
                to the system prompt.
            sufficiency_gate: Default True (measured +3.2pp, 0 regressions).
                Before accepting the agent's first final answer, asks the LLM
                once whether it's actually supported by the gathered evidence;
                on a clear gap it injects the gap and keeps retrieving
                (fail-open, bias-to-sufficient, bounded retries, <1.1x latency).
                Set False (or env ``SYNAPTIC_SUFFICIENCY_GATE=0``) to disable.
            gate_bridge: Opt-in (default False, env
                ``SYNAPTIC_GATE_BRIDGE=1``). When the sufficiency gate fires on
                a multi-hop gap, ask the judge to name the concrete follow-up
                search query (with the bridge entity from the evidence spelled
                out) and inject it as an explicit chained search instead of the
                generic "use the search tools" nudge. Pending agent A/B.
            efficiency_hint: Default True (env ``SYNAPTIC_AGENT_EFFICIENCY=0``
                to disable). Appends an efficiency directive to the system
                prompt — trust search snippets, batch document reads, stop when
                answered. Measured: tool_calls −15..−26%, per-query latency
                −9..−20%, solve non-negative on single-hop AND multi-hop benches.
            prime_with_snapshot: If True (default), inject a markdown
                snapshot of the graph (categories, top phrase hubs,
                tables) into the system prompt to skip the agent's
                cold-start exploration turns. Set to False on very
                large graphs (>100k nodes) where the snapshot overhead
                approaches the saved-turn benefit, or when ``extra_context``
                already provides equivalent priming.

        Returns:
            :class:`synaptic.agent_loop.AgentSearchResult` with
            ``final_answer``, ``found_ids``, ``nodes``, ``turns_used``,
            ``tool_calls_made``, ``elapsed_ms``.

        Example::

            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url="http://localhost:8012/v1", api_key="ollama")
            r = await graph.chat(
                "어떤 상품이 가장 인기인가?",
                llm_client=client, model="Qwen3.5-27b",
            )
            print(r.final_answer)
        """
        from synaptic.agent_loop import run_agent_loop

        # Auto-priming via graph snapshot — measured value: cold-start
        # agent typically wastes turn 0 on "what categories exist /
        # what tables can I filter". Snapshot preempts that probe by
        # injecting the answer up front. Cheap (sub-second on 100k
        # nodes), additive to the existing build_graph_context already
        # done inside run_agent_loop.
        priming_context = extra_context or ""
        if prime_with_snapshot:
            try:
                from synaptic.snapshot import generate_snapshot

                snapshot_md = await generate_snapshot(self._backend, include_sample_queries=True)
                priming_block = (
                    "[Graph snapshot — already provided so you can skip "
                    "the usual probe turns]\n\n" + snapshot_md
                )
                priming_context = (
                    priming_block + "\n\n" + priming_context if priming_context else priming_block
                )
            except Exception:
                # Snapshot is best-effort priming; never block the chat.
                pass

        return await run_agent_loop(
            client=llm_client,
            backend=self._backend,
            query=query,
            model=model,
            max_turns=max_turns,
            embedder=self._embedder,
            system_prompt=system_prompt,
            extra_context=priming_context or None,
            sufficiency_gate=sufficiency_gate,
            gate_bridge=gate_bridge,
            efficiency_hint=efficiency_hint,
            record_trace=record_trace,
        )

    async def ask(
        self,
        question: str,
        *,
        llm_client: object,
        model: str = "gpt-4o-mini",
        mode: str = "auto",
        k: int = 10,
        max_turns: int = 5,
    ):
        """Single entry point with honest routing — cheap when cheap is enough.

        Routes between the two answer paths this library ships:

        - **single_shot** — ``search(k)`` + ONE synthesis call (the
          naive-RAG arm of the rag_vs_agent harness, same prompt).
        - **agent** — the multi-turn :meth:`chat` loop (measured value:
          structured / multi-hop questions where single-shot is ~0).

        Decision layers (``mode="auto"``):

        1. **tier-0** — :func:`synaptic.router.decide_route`,
           deterministic, zero LLM calls. Conservative default — tier-0
           signal set pending E2 validation (PLAN-v0.29 §E2): only
           structured-operation lexis over a corpus with typed table
           nodes goes straight to the agent.
        2. **tier-1** — the cheap answer is judged by the sufficiency
           gate (``agent_loop._judge_sufficiency``, temperature 0); on a
           clear gap the query escalates to the agent loop. An empty
           cheap synthesis also escalates. Fail-open: a broken judge
           keeps the cheap answer.

        Args:
            question: User question.
            llm_client: OpenAI-compatible async client — same contract
                as :meth:`chat` (required; there is no graph-level
                default client).
            model: Model name forwarded to the LLM client.
            mode: ``"auto"`` (route), ``"search"`` (force the cheap
                path, never escalate), ``"agent"`` (force the agent
                loop).
            k: Evidence count for the cheap path's retrieval.
            max_turns: Turn budget when the agent loop runs.

        Returns:
            :class:`synaptic.router.AskResult` — answer, the route that
            produced it, the route reasons, whether tier-1 escalated,
            total prompt/completion tokens across every LLM call made
            (synthesis + judge + agent loop), and the evidence nodes.
        """
        from synaptic.agent_loop import _add_usage, _judge_sufficiency
        from synaptic.router import (
            RAG_SYNTHESIS_SYSTEM,
            AskResult,
            RouteDecision,
            corpus_has_table_nodes,
            decide_route,
        )

        if mode not in ("auto", "search", "agent"):
            raise ValueError(f"mode must be 'auto', 'search' or 'agent', got {mode!r}")

        usage = {"prompt": 0, "completion": 0}

        # --- tier-0: deterministic route decision (no LLM) ----------
        if mode == "agent":
            decision = RouteDecision(
                route="agent",
                reasons=["mode='agent' forced by caller"],
                signals={"mode_forced": True},
            )
        elif mode == "search":
            decision = RouteDecision(
                route="single_shot",
                reasons=["mode='search' forced by caller — no escalation"],
                signals={"mode_forced": True},
            )
        else:
            decision = decide_route(
                question,
                has_table_nodes=await corpus_has_table_nodes(self._backend),
            )

        async def _run_agent():
            r = await self.chat(question, llm_client=llm_client, model=model, max_turns=max_turns)
            usage["prompt"] += r.prompt_tokens
            usage["completion"] += r.completion_tokens
            return r

        if decision.route == "agent":
            agent_res = await _run_agent()
            return AskResult(
                answer=agent_res.final_answer,
                route="agent",
                route_reasons=list(decision.reasons),
                escalated=False,
                prompt_tokens=usage["prompt"],
                completion_tokens=usage["completion"],
                evidence=list(agent_res.nodes),
            )

        # --- cheap path: single-shot retrieval + one synthesis call --
        sr = await self.search(question, limit=k)
        snippets = [(an.node.content or an.node.title or "")[:700] for an in sr.nodes[:k]]
        context = "\n---\n".join(snippets)
        resp = await llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RAG_SYNTHESIS_SYSTEM},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
                },
            ],
            max_tokens=512,
        )
        _add_usage(usage, resp)
        answer = resp.choices[0].message.content or ""
        evidence: list = [an.node for an in sr.nodes]
        reasons = list(decision.reasons)
        route = "single_shot"
        escalated = False

        # --- tier-1: sufficiency gate over the cheap answer ---------
        if mode == "auto":
            escalate_reason = ""
            if not answer.strip():
                escalate_reason = "cheap synthesis returned an empty answer"
            elif snippets:
                # The judge reads role="tool" messages; present the cheap
                # path's evidence snippets in that shape so both paths are
                # judged against identical evidence framing.
                pseudo_messages = [{"role": "tool", "content": s} for s in snippets]
                verdict = await _judge_sufficiency(
                    llm_client,
                    model,
                    question,
                    pseudo_messages,
                    answer,
                    usage=usage,
                    temperature=0.0,
                )
                if verdict is not None and not verdict["sufficient"]:
                    gap = verdict["gap"] or "unspecified gap"
                    escalate_reason = (
                        f"tier-1 sufficiency judge: cheap answer insufficient (missing: {gap})"
                    )
            if escalate_reason:
                reasons.append(escalate_reason + " — escalated to the agent loop")
                agent_res = await _run_agent()
                escalated = True
                if agent_res.final_answer.strip():
                    answer = agent_res.final_answer
                    evidence = list(agent_res.nodes)
                    route = "agent"
                else:
                    # Non-empty guarantee: an escalation that comes back blank
                    # must not erase a usable cheap answer. route stays
                    # "single_shot" — that's the path that produced `answer`.
                    reasons.append(
                        "agent returned an empty answer — kept the single-shot synthesis"
                    )

        return AskResult(
            answer=answer,
            route=route,
            route_reasons=reasons,
            escalated=escalated,
            prompt_tokens=usage["prompt"],
            completion_tokens=usage["completion"],
            evidence=evidence,
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        embedding: list[float] | None = None,
        rerank: bool | None = None,
        fts_seed_limit: int | None = None,
        per_document_cap: int | None = None,
        record: bool = False,
        scope: MemoryScope | None = None,
    ) -> SearchResult:
        """Hybrid search across the graph.

        Args:
            query: User query string. NFC-normalised before lookup.
            limit: Maximum number of nodes to return.
            embedding: Pre-computed query embedding. When omitted and
                an embedder is wired into the graph, the query is
                embedded automatically.
            rerank: Per-call cross-encoder reranker override. ``None``
                uses the graph's configured reranker; ``False`` skips
                reranking for this query even when one is wired; ``True``
                is the same as ``None``.
            fts_seed_limit: Per-call FTS seed-pool size. ``None`` uses
                the ``max(20, limit)`` default heuristic.
            per_document_cap: Per-call cap on evidence items from any
                single document — lower = more source diversity.
                ``None`` uses the pipeline default (2).
            record: When true, persist a retrieval ledger event and
                attach its id to ``SearchResult.event_id``. Default is
                false so normal search remains side-effect free.
            scope: Optional workspace/user/session scope used for
                side-effectful retrieval logging and bounded ranking
                boosts/penalties from prior feedback and memory signals.

        Returns:
            ``SearchResult`` — the evidence pipeline (BM25 + HNSW + PPR +
            cross-encoder + MMR) runs through an internal adapter that
            mirrors the SearchResult shape so all existing callers in this
            repo continue to work unchanged. ``embedding`` is accepted for
            backwards compatibility; the evidence pipeline re-embeds the
            query internally via the configured embedder.
        """
        # NFC-normalize query to match NFC-normalized stored content.
        query = _nfc(query)
        return await self._search_via_evidence(
            query,
            limit=limit,
            embedding=embedding,
            rerank=rerank,
            fts_seed_limit=fts_seed_limit,
            per_document_cap=per_document_cap,
            record=record,
            scope=scope,
        )

    async def _search_via_evidence(
        self,
        query: str,
        *,
        limit: int,
        embedding: list[float] | None = None,
        rerank: bool | None = None,
        fts_seed_limit: int | None = None,
        per_document_cap: int | None = None,
        record: bool = False,
        scope: MemoryScope | None = None,
    ) -> SearchResult:
        """Run the evidence pipeline and adapt its result to the
        ``SearchResult`` shape used across the codebase.

        ``EvidenceSearch`` is imported lazily so the pipeline only loads
        on first search. The adapter mirrors the legacy ``SearchResult``
        fields so all existing callers keep working.
        """
        # rerank=False disables the cross-encoder for this query only.
        active_reranker = None if rerank is False else self._reranker
        searcher = self._get_evidence_search(active_reranker)
        search_kwargs: dict[str, object] = {
            "k": limit,
            # Keep the lexical seed pool bounded before graph expansion.
            "fts_seed_limit": (
                fts_seed_limit if fts_seed_limit is not None else _default_fts_seed_limit(limit)
            ),
        }
        if per_document_cap is not None:
            search_kwargs["per_document_cap"] = per_document_cap
        if embedding is not None:
            search_kwargs["query_embedding"] = embedding
        ev_result = await searcher.search(query, **search_kwargs)

        # Adapter: Evidence (node + score + reason) → ActivatedNode
        # (node + activation + resonance). The legacy callers rely on
        # `resonance` for ordering and the `nodes` list for iteration;
        # both are populated faithfully.
        nodes: list[ActivatedNode] = []
        for ev in ev_result.evidence:
            nodes.append(
                ActivatedNode(
                    node=ev.node,
                    activation=ev.score,  # cosine-rerank score in [0,1]
                    resonance=ev.score,
                    path=[],
                )
            )

        # `stages_used` is informational. The evidence pipeline always
        # runs the same steps so we list them statically; UIs that
        # branch on stage names ("synonym" / "rewriter") will see an
        # empty signal, which is correct — those legacy stages do not
        # exist on the modern path.
        stages = ["evidence", "fts"]
        if self._embedder is not None:
            stages.append("vector")

        result = SearchResult(
            query=ev_result.query,
            nodes=nodes,
            total_candidates=len(ev_result.scored),
            search_time_ms=ev_result.elapsed_ms,
            timings_ms=dict(ev_result.timings_ms),
            diagnostics=dict(getattr(ev_result, "diagnostics", {}) or {}),
            stages_used=stages,
        )
        if scope is not None:
            await self._apply_scope_boost(result, scope)
            await self._apply_memory_signal_penalties(result, scope=scope)
        if record:
            event = await self._record_retrieval_event(result, scope=scope)
            if event is not None:
                result.event_id = event.id
        return result

    async def _record_retrieval_event(
        self,
        result: SearchResult,
        *,
        scope: MemoryScope | None,
    ) -> RetrievalEvent | None:
        save = getattr(self._backend, "save_retrieval_event", None)
        if not callable(save):
            return None
        properties = _retrieval_event_properties(result)
        event = RetrievalEvent(
            query=result.query,
            scope=scope or MemoryScope(),
            returned_node_ids=[item.node.id for item in result.nodes],
            signal=FeedbackSignal.SELECTED,
            success=None,
            confidence=1.0,
            properties=properties,
        )
        await save(event)
        await self._save_memory_event(
            MemoryEvent(
                kind=MemoryEventKind.RETRIEVAL,
                scope=event.scope,
                source="search",
                source_id=event.id,
                node_ids=list(event.returned_node_ids),
                confidence=event.confidence,
                properties=properties,
                created_at=event.created_at,
            )
        )
        return event

    def _get_evidence_search(self, active_reranker: object | None) -> object:
        """Return a reusable EvidenceSearch for the graph's current runtime wiring."""
        from synaptic.extensions.evidence_search import EvidenceSearch

        key = (
            id(self._backend),
            id(self._embedder),
            id(self._phrase_extractor),
            id(active_reranker),
            id(self._reranker_weights),
            id(self._query_decomposer),
        )
        searcher = self._evidence_search_cache.get(key)
        if searcher is None:
            if len(self._evidence_search_cache) > 8:
                self._evidence_search_cache.clear()
            searcher = EvidenceSearch(
                backend=self._backend,
                embedder=self._embedder,
                phrase_extractor=self._phrase_extractor,
                reranker=active_reranker,
                reranker_weights=self._reranker_weights,
                decomposer=self._query_decomposer,
            )
            self._evidence_search_cache[key] = searcher
        return searcher

    async def record_feedback(
        self,
        event_id: str = "",
        *,
        success: bool | None = None,
        node_ids: list[str] | None = None,
        signal: FeedbackSignal | str = FeedbackSignal.EXPLICIT_POSITIVE,
        confidence: float = 1.0,
        scope: MemoryScope | None = None,
    ) -> RetrievalEvent:
        """Record observed retrieval outcome and update scoped reinforcement.

        The method intentionally treats implicit signals as weak evidence.
        Scoped negative feedback stays scope-local by default so one user or
        workspace cannot globally suppress a memory. Positive task/test
        outcomes can promote globally; callers can also explicitly promote a
        scope when they know the feedback should apply across scopes.
        """
        signal = FeedbackSignal(signal)
        confidence = max(0.0, min(1.0, float(confidence)))
        parent = await self._get_retrieval_event(event_id) if event_id else None
        effective_scope = scope or (parent.scope if parent is not None else MemoryScope())
        inferred_success = self._infer_feedback_success(signal, success)
        target_node_ids = list(node_ids or [])
        if not target_node_ids and parent is not None:
            target_node_ids = list(parent.selected_node_ids or parent.returned_node_ids)

        event = RetrievalEvent(
            query=parent.query if parent is not None else "",
            scope=effective_scope,
            returned_node_ids=list(parent.returned_node_ids) if parent is not None else [],
            selected_node_ids=target_node_ids,
            success=inferred_success,
            signal=signal,
            confidence=confidence,
            properties={"parent_event_id": event_id} if event_id else {},
        )
        save = getattr(self._backend, "save_retrieval_event", None)
        if callable(save):
            await save(event)
        await self._save_memory_event(
            MemoryEvent(
                kind=MemoryEventKind.FEEDBACK,
                scope=effective_scope,
                source="retrieval_feedback",
                source_id=event.id,
                node_ids=target_node_ids,
                confidence=confidence,
                properties={
                    "parent_event_id": event_id,
                    "signal": str(signal),
                    "success": "" if inferred_success is None else str(inferred_success).lower(),
                },
                created_at=event.created_at,
            )
        )

        await self._update_scope_scores(
            effective_scope,
            target_node_ids,
            signal=signal,
            success=inferred_success,
            confidence=confidence,
        )
        promote_globally = self._should_promote_feedback_globally(effective_scope, signal)
        has_distinct_global_scope = (
            promote_globally and memory_scope_key(effective_scope) != "global"
        )
        if has_distinct_global_scope:
            await self._update_scope_scores(
                MemoryScope(promote_to_global=True),
                target_node_ids,
                signal=signal,
                success=inferred_success,
                confidence=confidence,
                global_scope=True,
            )
        if promote_globally and inferred_success is not None and target_node_ids:
            await self._hebbian.reinforce(
                self._backend,
                target_node_ids,
                success=inferred_success,
            )
            for node_id in target_node_ids:
                self._cache.invalidate(node_id)
        if inferred_success is not None and target_node_ids:
            await self._update_edge_scope_scores(
                effective_scope,
                target_node_ids,
                signal=signal,
                success=inferred_success,
                confidence=confidence,
            )
            if has_distinct_global_scope:
                await self._update_edge_scope_scores(
                    MemoryScope(promote_to_global=True),
                    target_node_ids,
                    signal=signal,
                    success=inferred_success,
                    confidence=confidence,
                    global_scope=True,
                )
        return event

    async def _get_retrieval_event(self, event_id: str) -> RetrievalEvent | None:
        getter = getattr(self._backend, "get_retrieval_event", None)
        if callable(getter):
            return await getter(event_id)
        lister = getattr(self._backend, "list_retrieval_events", None)
        if not callable(lister):
            return None
        for event in await lister(limit=1000):
            if event.id == event_id:
                return event
        return None

    async def _save_memory_event(self, event: MemoryEvent) -> None:
        save = getattr(self._backend, "save_memory_event", None)
        if callable(save):
            await save(event)

    async def _touching_edge_ids(self, node_id: str) -> list[str]:
        try:
            edges = await self._backend.get_edges(node_id, direction="both")
        except Exception:
            logger.debug("failed to collect edge ids for memory event", exc_info=True)
            return []
        return sorted({edge.id for edge in edges})

    @staticmethod
    def _infer_feedback_success(
        signal: FeedbackSignal,
        success: bool | None,
    ) -> bool | None:
        if success is not None:
            return success
        if signal in (
            FeedbackSignal.EXPLICIT_POSITIVE,
            FeedbackSignal.TASK_SUCCESS,
            FeedbackSignal.TEST_PASS,
        ):
            return True
        if signal in (
            FeedbackSignal.EXPLICIT_NEGATIVE,
            FeedbackSignal.TASK_FAILURE,
            FeedbackSignal.TEST_FAIL,
        ):
            return False
        return None

    @staticmethod
    def _should_promote_feedback_globally(scope: MemoryScope, signal: FeedbackSignal) -> bool:
        if scope.promote_to_global or memory_scope_key(scope) == "global":
            return True
        return signal in (
            FeedbackSignal.TASK_SUCCESS,
            FeedbackSignal.TEST_PASS,
        )

    async def _update_scope_scores(
        self,
        scope: MemoryScope,
        node_ids: list[str],
        *,
        signal: FeedbackSignal,
        success: bool | None,
        confidence: float,
        global_scope: bool = False,
    ) -> None:
        if not node_ids:
            return
        get_score = getattr(self._backend, "get_memory_score", None)
        save_score = getattr(self._backend, "save_memory_score", None)
        if not callable(get_score) or not callable(save_score):
            return
        scope_key = memory_scope_key(scope, global_scope=global_scope)
        score_delta = _feedback_score_delta(signal, success, confidence)
        for node_id in dict.fromkeys(node_ids):
            score = await get_score(scope_key, node_id=node_id)
            if score is None:
                score = MemoryScore(scope_key=scope_key, node_id=node_id)
            score.access_count += 1
            if success is True:
                score.success_count += 1
            elif success is False:
                score.failure_count += 1
            score.score = max(-1.0, min(1.0, score.score + score_delta))
            score.updated_at = time()
            await save_score(score)

    async def _update_edge_scope_scores(
        self,
        scope: MemoryScope,
        node_ids: list[str],
        *,
        signal: FeedbackSignal,
        success: bool,
        confidence: float,
        global_scope: bool = False,
    ) -> None:
        unique_node_ids = list(dict.fromkeys(node_ids))
        if len(unique_node_ids) < 2:
            return
        get_score = getattr(self._backend, "get_memory_score", None)
        save_score = getattr(self._backend, "save_memory_score", None)
        if not callable(get_score) or not callable(save_score):
            return
        edge_ids = await self._coactivated_edge_ids(unique_node_ids)
        if not edge_ids:
            return
        scope_key = memory_scope_key(scope, global_scope=global_scope)
        score_delta = _feedback_score_delta(signal, success, confidence)
        for edge_id in edge_ids:
            score = await get_score(scope_key, edge_id=edge_id)
            if score is None:
                score = MemoryScore(scope_key=scope_key, edge_id=edge_id)
            score.access_count += 1
            if success is True:
                score.success_count += 1
            else:
                score.failure_count += 1
            score.score = max(-1.0, min(1.0, score.score + score_delta))
            score.updated_at = time()
            await save_score(score)

    async def _coactivated_edge_ids(self, node_ids: list[str]) -> list[str]:
        wanted = set(node_ids)
        seen: set[str] = set()
        for node_id in node_ids:
            for edge in await self._backend.get_edges(node_id, direction="both"):
                if edge.id in seen:
                    continue
                if edge.source_id in wanted and edge.target_id in wanted:
                    seen.add(edge.id)
        return sorted(seen)

    async def _apply_scope_boost(self, result: SearchResult, scope: MemoryScope) -> None:
        if not result.nodes:
            return
        lister = getattr(self._backend, "list_memory_scores", None)
        if not callable(lister):
            return
        node_ids = [item.node.id for item in result.nodes]
        node_id_set = set(node_ids)
        scope_key = memory_scope_key(scope)
        local_scores = await lister(scope_key=scope_key, node_ids=node_ids, limit=len(node_ids))
        global_scores: list[MemoryScore] = []
        if scope_key != "global":
            global_scores = await lister(
                scope_key="global",
                node_ids=node_ids,
                limit=len(node_ids),
            )
        local_by_node = {score.node_id: score.score for score in local_scores}
        global_by_node = {score.node_id: score.score for score in global_scores}
        edge_targets = await self._candidate_edge_targets(candidate_node_ids=node_id_set)
        edge_ids = list(edge_targets)
        local_edge_scores: list[MemoryScore] = []
        global_edge_scores: list[MemoryScore] = []
        if edge_ids:
            local_edge_scores = await lister(
                scope_key=scope_key,
                edge_ids=edge_ids,
                limit=len(edge_ids),
            )
            if scope_key != "global":
                global_edge_scores = await lister(
                    scope_key="global",
                    edge_ids=edge_ids,
                    limit=len(edge_ids),
                )
        edge_by_node: dict[str, float] = {}
        for score in local_edge_scores:
            for node_id in edge_targets.get(score.edge_id, set()):
                edge_by_node[node_id] = edge_by_node.get(node_id, 0.0) + score.score
        for score in global_edge_scores:
            for node_id in edge_targets.get(score.edge_id, set()):
                edge_by_node[node_id] = edge_by_node.get(node_id, 0.0) + 0.5 * score.score
        if not local_by_node and not global_by_node and not edge_by_node:
            return
        base_resonances = [item.resonance for item in result.nodes]
        boosted_nodes = 0
        demoted_nodes = 0
        adjusted_nodes = 0
        clamped_nodes = 0
        max_abs_boost = 0.0
        max_positive_boost = 0.0
        max_demotion = 0.0
        for idx, item in enumerate(result.nodes):
            raw = (
                local_by_node.get(item.node.id, 0.0)
                + (0.5 * global_by_node.get(item.node.id, 0.0))
                + edge_by_node.get(item.node.id, 0.0)
            )
            raw = max(-1.0, min(1.0, raw))
            boost = max(-0.10, min(0.10, raw * 0.10))
            if boost != 0.0:
                adjusted_nodes += 1
                max_abs_boost = max(max_abs_boost, abs(boost))
                if boost > 0.0:
                    boosted_nodes += 1
                    max_positive_boost = max(max_positive_boost, boost)
                else:
                    demoted_nodes += 1
                    max_demotion = max(max_demotion, abs(boost))
            item.activation = max(0.0, item.activation * (1.0 + boost))
            item.resonance = max(0.0, item.resonance * (1.0 + boost))
            if boost > 0.0 and idx > 0 and item.resonance > base_resonances[idx - 1]:
                item.resonance = base_resonances[idx - 1]
                item.activation = min(item.activation, base_resonances[idx - 1])
                clamped_nodes += 1
        result.nodes.sort(key=lambda item: item.resonance, reverse=True)
        result.diagnostics["memory_scope_boosted_nodes"] = float(boosted_nodes)
        result.diagnostics["memory_scope_demoted_nodes"] = float(demoted_nodes)
        result.diagnostics["memory_scope_adjusted_nodes"] = float(adjusted_nodes)
        result.diagnostics["memory_scope_node_score_hits"] = float(
            len(set(local_by_node) | set(global_by_node))
        )
        result.diagnostics["memory_scope_edge_score_hits"] = float(len(edge_by_node))
        result.diagnostics["memory_scope_max_abs_boost"] = max_abs_boost
        result.diagnostics["memory_scope_max_positive_boost"] = max_positive_boost
        result.diagnostics["memory_scope_max_demotion"] = max_demotion
        result.diagnostics["memory_scope_order_clamps"] = float(clamped_nodes)

    async def _apply_memory_signal_penalties(
        self,
        result: SearchResult,
        *,
        scope: MemoryScope | None,
    ) -> None:
        if not result.nodes:
            return
        node_ids = {item.node.id for item in result.nodes}
        if not node_ids:
            return
        current_scope_key = memory_scope_key(scope or MemoryScope())
        signal_nodes = await self._backend.list_nodes(kind=NodeKind.OBSERVATION, limit=100_000)
        relevant_signals: list[tuple[str, dict[str, str], float]] = []
        signal_edge_ids: set[str] = set()
        penalties: dict[str, float] = {}
        penalty_signal_ids: set[str] = set()
        penalty_edge_ids: set[str] = set()
        for signal_node in signal_nodes:
            tags = set(signal_node.tags or [])
            if "_memory_signal" not in tags or "_memory_suspect" not in tags:
                continue
            props = signal_node.properties or {}
            signal_scope = props.get("scope_key", "")
            if signal_scope not in {"", "global", current_scope_key}:
                continue
            confidence = _prop_float(props, "confidence", 0.0)
            if confidence < _MEMORY_SIGNAL_MIN_PENALTY_CONFIDENCE:
                continue
            penalty = min(
                _MEMORY_SIGNAL_MAX_RANKING_PENALTY,
                max(0.0, confidence * _MEMORY_SIGNAL_MAX_RANKING_PENALTY),
            )
            relevant_signals.append((signal_node.id, props, penalty))
            signal_edge_ids.update(_prop_csv_ids(props, "edge_ids"))
        if not relevant_signals:
            return
        edge_targets = await self._candidate_edge_targets(
            candidate_node_ids=node_ids,
            edge_ids=signal_edge_ids,
        )
        for signal_id, props, penalty in relevant_signals:
            target_ids = set(_prop_csv_ids(props, "node_ids"))
            edge_ids = _prop_csv_ids(props, "edge_ids")
            for edge_id in edge_ids:
                target_ids.update(edge_targets.get(edge_id, set()))
            for node_id in target_ids:
                if node_id in node_ids:
                    penalties[node_id] = max(penalties.get(node_id, 0.0), penalty)
                    penalty_signal_ids.add(signal_id)
                    penalty_edge_ids.update(edge_ids)
        if not penalties:
            return
        for item in result.nodes:
            penalty = penalties.get(item.node.id, 0.0)
            if penalty <= 0.0:
                continue
            factor = 1.0 - penalty
            item.activation = max(0.0, item.activation * factor)
            item.resonance = max(0.0, item.resonance * factor)
        result.nodes.sort(key=lambda item: item.resonance, reverse=True)
        result.diagnostics["memory_signal_penalized_nodes"] = float(len(penalties))
        result.diagnostics["memory_signal_max_penalty"] = max(penalties.values())
        result.diagnostics["memory_signal_penalized_node_ids"] = ",".join(sorted(penalties))
        result.diagnostics["memory_signal_source_ids"] = ",".join(sorted(penalty_signal_ids))
        result.diagnostics["memory_signal_edge_ids"] = ",".join(sorted(penalty_edge_ids))

    async def _candidate_edge_targets(
        self,
        *,
        candidate_node_ids: set[str],
        edge_ids: set[str] | None = None,
    ) -> dict[str, set[str]]:
        targets: dict[str, set[str]] = {edge_id: set() for edge_id in (edge_ids or set())}
        if not candidate_node_ids:
            return targets
        batch_get_edges = getattr(self._backend, "get_edges_batch", None)
        edges_to_scan: list[Edge] = []
        if callable(batch_get_edges):
            edge_groups = await batch_get_edges(list(candidate_node_ids), direction="both")
            for edges_for_node in edge_groups.values():
                edges_to_scan.extend(edges_for_node)
        else:
            for node_id in candidate_node_ids:
                edges_to_scan.extend(await self._backend.get_edges(node_id, direction="both"))
        for edge in edges_to_scan:
            if edge_ids is not None and edge.id not in edge_ids:
                continue
            target_ids: set[str] = set()
            if edge.source_id in candidate_node_ids:
                target_ids.add(edge.source_id)
            if edge.target_id in candidate_node_ids:
                target_ids.add(edge.target_id)
            if target_ids:
                targets.setdefault(edge.id, set()).update(target_ids)
        return targets

    async def scan_memory_signals(
        self,
        *,
        scope: MemoryScope | None = None,
        since: float | None = None,
        persist: bool = True,
    ) -> list[MemorySignal]:
        """Scan for suspicious or noteworthy memory signals.

        Signals are persisted as observation nodes by default, never used for
        automatic deletion. Repeated scans upsert the same deterministic signal
        nodes. Set ``persist=False`` for read-only diagnostics/eval reports.
        """
        effective_scope = scope or MemoryScope()
        nodes = await self._backend.list_nodes(limit=100_000)
        nodes_by_id = {node.id: node for node in nodes}
        edges = await self._all_edges(nodes)
        edges_by_id = {edge.id: edge for edge in edges}
        signals: list[MemorySignal] = []
        stale_before = time() - 365 * 24 * 3600
        repeated_failure_targets: set[str] = set()
        entity_property_groups: dict[tuple[str, str], dict[str, list[Node]]] = {}

        for node in nodes:
            if (
                node.failure_count >= 3
                and node.failure_count > node.success_count
                and "_memory_signal" not in (node.tags or [])
            ):
                signals.append(
                    self._make_signal(
                        MemorySignalKind.REPEATED_FAILURE,
                        effective_scope,
                        node_ids=[node.id],
                        confidence=0.8,
                        reason="node has repeated failure feedback",
                    )
                )
                repeated_failure_targets.add(node.id)
            if (
                node.updated_at < stale_before
                and node.access_count == 0
                and "_memory_signal" not in (node.tags or [])
            ):
                signals.append(
                    self._make_signal(
                        MemorySignalKind.STALE_MEMORY,
                        effective_scope,
                        node_ids=[node.id],
                        confidence=0.6,
                        reason="node has not been accessed for more than one year",
                    )
                )
            if since is not None and node.created_at >= since and node.kind == NodeKind.ENTITY:
                signals.append(
                    self._make_signal(
                        MemorySignalKind.NEW_ENTITY,
                        effective_scope,
                        node_ids=[node.id],
                        confidence=0.7,
                        reason="new entity appeared in recent memory events",
                    )
                )
            if node.kind == NodeKind.ENTITY and "_memory_signal" not in (node.tags or []):
                entity_key = node.title.strip().casefold()
                if entity_key:
                    for prop_key, prop_value in (node.properties or {}).items():
                        prop_key = str(prop_key)
                        if (
                            not prop_key
                            or prop_key.startswith("_")
                            or prop_key in _MEMORY_PROPERTY_CONFLICT_IGNORED_KEYS
                        ):
                            continue
                        prop_value = str(prop_value).strip()
                        if not prop_value:
                            continue
                        bucket = entity_property_groups.setdefault((entity_key, prop_key), {})
                        bucket.setdefault(prop_value, []).append(node)

        for (entity_key, prop_key), by_value in entity_property_groups.items():
            if len(by_value) < 2:
                continue
            source_labels = {
                _node_source_label(node)
                for nodes_for_value in by_value.values()
                for node in nodes_for_value
            }
            if len(source_labels) < 2:
                continue
            conflicting_nodes = [
                node for nodes_for_value in by_value.values() for node in nodes_for_value
            ]
            node_ids = [node.id for node in conflicting_nodes]
            values = sorted(by_value)
            signals.append(
                self._make_signal(
                    MemorySignalKind.POSSIBLE_CONFLICT,
                    effective_scope,
                    node_ids=node_ids,
                    confidence=0.7,
                    reason=(
                        "entity property has conflicting source values "
                        f"entity={entity_key} property={prop_key}"
                    ),
                    properties={
                        "conflict_type": "entity_property_value",
                        "entity_key": entity_key,
                        "property_key": prop_key,
                        "property_values": "|".join(values[:8]),
                        "source_labels": "|".join(sorted(source_labels)[:8]),
                    },
                )
            )

        for edge in edges:
            if edge.kind == EdgeKind.CONTRADICTS:
                signals.append(
                    self._make_signal(
                        MemorySignalKind.POSSIBLE_CONFLICT,
                        effective_scope,
                        edge_ids=[edge.id],
                        node_ids=[edge.source_id, edge.target_id],
                        confidence=0.8,
                        reason="contradiction edge is present",
                    )
                )
            if edge.kind == EdgeKind.SUPERSEDES:
                signals.append(
                    self._make_signal(
                        MemorySignalKind.POSSIBLE_SUPERSESSION,
                        effective_scope,
                        edge_ids=[edge.id],
                        node_ids=[edge.source_id, edge.target_id],
                        confidence=0.7,
                        reason="supersession edge is present",
                    )
                )
                superseding = nodes_by_id.get(edge.source_id)
                superseded = nodes_by_id.get(edge.target_id)
                if (
                    superseded is not None
                    and "_memory_signal" not in (superseded.tags or [])
                    and (
                        edge.created_at >= superseded.updated_at
                        or (
                            superseding is not None
                            and superseding.updated_at >= superseded.updated_at
                        )
                    )
                ):
                    signals.append(
                        self._make_signal(
                            MemorySignalKind.STALE_MEMORY,
                            effective_scope,
                            edge_ids=[edge.id],
                            node_ids=[edge.target_id],
                            confidence=0.75,
                            reason="memory is superseded by newer evidence",
                            properties={
                                "stale_reason": "superseded",
                                "superseding_node_id": edge.source_id,
                                "superseded_node_id": edge.target_id,
                                "supersession_edge_id": edge.id,
                            },
                        )
                    )
            if (
                _prop_bool(edge.properties, "is_openie")
                and _prop_float(edge.properties, "confidence", 1.0) < 0.6
            ):
                signals.append(
                    self._make_signal(
                        MemorySignalKind.LOW_CONFIDENCE_RELATION,
                        effective_scope,
                        edge_ids=[edge.id],
                        node_ids=[edge.source_id, edge.target_id],
                        confidence=0.7,
                        reason="low-confidence OpenIE relation is present",
                    )
                )
            if _prop_int(edge.properties, "support_count", 0) >= 2 or edge.weight > 1.5:
                signals.append(
                    self._make_signal(
                        MemorySignalKind.RELATION_REINFORCED,
                        effective_scope,
                        edge_ids=[edge.id],
                        node_ids=[edge.source_id, edge.target_id],
                        confidence=0.6,
                        reason="relation has repeated support or high weight",
                    )
                )
            if since is not None and edge.created_at >= since:
                signals.append(
                    self._make_signal(
                        MemorySignalKind.NEW_RELATION,
                        effective_scope,
                        edge_ids=[edge.id],
                        node_ids=[edge.source_id, edge.target_id],
                        confidence=0.6,
                        reason="new relation appeared in recent memory events",
                    )
                )

        scope_key = memory_scope_key(effective_scope)
        score_scopes: list[tuple[str, MemoryScope]] = [(scope_key, effective_scope)]
        if scope_key != "global":
            score_scopes.append(("global", MemoryScope()))
        for score_scope_key, signal_scope in score_scopes:
            for score in await self._list_memory_scores(scope_key=score_scope_key, limit=100_000):
                if since is not None and score.updated_at < since:
                    continue
                target = score.node_id or score.edge_id
                if not target or target in repeated_failure_targets:
                    continue
                repeated_failure = (
                    score.failure_count >= 3 and score.failure_count > score.success_count
                )
                strong_negative = score.score <= _MEMORY_STRONG_NEGATIVE_SCORE_SIGNAL_THRESHOLD
                if not repeated_failure and not strong_negative:
                    continue
                edge_ids = [score.edge_id] if score.edge_id else []
                if score.node_id:
                    node_ids = [score.node_id]
                else:
                    edge = edges_by_id.get(score.edge_id)
                    node_ids = [edge.source_id, edge.target_id] if edge is not None else []
                signal_type = (
                    "repeated_negative_feedback"
                    if repeated_failure
                    else "strong_negative_scope_score"
                )
                confidence = (
                    0.75 if repeated_failure else min(0.9, max(0.7, 0.55 + abs(score.score) * 0.35))
                )
                reason = (
                    "scope score has repeated negative retrieval feedback"
                    if repeated_failure
                    else "scope score is strongly negative from retrieval feedback"
                )
                signals.append(
                    self._make_signal(
                        MemorySignalKind.REPEATED_FAILURE,
                        signal_scope,
                        node_ids=node_ids,
                        edge_ids=edge_ids,
                        confidence=confidence,
                        reason=reason,
                        properties={
                            "score_signal_type": signal_type,
                            "score_scope_key": score.scope_key,
                            "score_failure_count": str(score.failure_count),
                            "score_success_count": str(score.success_count),
                            "score_access_count": str(score.access_count),
                            "score": f"{score.score:.6f}",
                        },
                    )
                )
                repeated_failure_targets.add(target)

        signals.extend(await self._semantic_extract_drift_signals(effective_scope, since=since))
        if persist:
            for signal in signals:
                await self._persist_memory_signal(signal)
        return signals

    async def memory_health(
        self,
        *,
        scope: MemoryScope | None = None,
        since: float | None = None,
        persist_signals: bool = True,
    ) -> MemoryHealthReport:
        """Return a compact operational report for the memory layer.

        By default this also persists deterministic signal nodes via
        :meth:`scan_memory_signals`. Set ``persist_signals=False`` for read-only
        health snapshots.
        """
        effective_scope = scope or MemoryScope()
        signals = await self.scan_memory_signals(
            scope=effective_scope,
            since=since,
            persist=persist_signals,
        )
        nodes = await self._backend.list_nodes(limit=100_000)
        edges = await self._all_edges(nodes)
        memory_events = await self._list_memory_events(scope=scope, since=since, limit=100_000)
        retrieval_events = await self._list_retrieval_events(
            scope=scope, since=since, limit=100_000
        )
        score_scope_key = memory_scope_key(effective_scope)
        all_scores = await self._list_memory_scores(
            scope_key=None,
            limit=100_000,
        )
        node_scores = await self._list_memory_scores(
            scope_key=score_scope_key,
            edge_ids=[""],
            limit=100_000,
        )
        edge_scores = await self._list_memory_scores(
            scope_key=score_scope_key,
            node_ids=[""],
            limit=100_000,
        )
        top_node_scores = [score for score in node_scores if score.node_id and score.score > 0][:10]
        top_edge_scores = [score for score in edge_scores if score.edge_id and score.score > 0][:10]
        top_demoted_node_scores = sorted(
            (score for score in node_scores if score.node_id and score.score < 0),
            key=lambda score: score.score,
        )[:10]
        top_demoted_edge_scores = sorted(
            (score for score in edge_scores if score.edge_id and score.score < 0),
            key=lambda score: score.score,
        )[:10]

        memory_event_kind_counts = Counter(str(event.kind) for event in memory_events)
        memory_event_scope_counts = Counter(
            memory_scope_key(event.scope) for event in memory_events
        )
        retrieval_event_scope_counts = Counter(
            memory_scope_key(event.scope) for event in retrieval_events
        )
        memory_score_scope_counts = Counter(score.scope_key for score in all_scores)
        memory_score_node_count = sum(1 for score in all_scores if score.node_id)
        memory_score_edge_count = sum(1 for score in all_scores if score.edge_id)
        memory_score_positive_count = sum(1 for score in all_scores if score.score > 0)
        memory_score_negative_count = sum(1 for score in all_scores if score.score < 0)
        memory_score_neutral_count = sum(1 for score in all_scores if score.score == 0)
        semantic_events = [
            event for event in memory_events if str(event.kind) == MemoryEventKind.SEMANTIC_EXTRACT
        ]
        failures = sum(
            _prop_int(event.properties, "extraction_failures", 0) for event in semantic_events
        )
        semantic_attempt_count = sum(
            max(
                _prop_int(event.properties, "chunks_selected", 0),
                _prop_int(event.properties, "extraction_failures", 0),
            )
            for event in semantic_events
        )
        semantic_failure_counts: Counter[str] = Counter()
        semantic_attempt_counts: Counter[str] = Counter()
        for event in semantic_events:
            profile_key = _semantic_extract_profile_key(event)
            event_failures = _prop_int(event.properties, "extraction_failures", 0)
            event_selected = _prop_int(event.properties, "chunks_selected", 0)
            event_attempts = max(event_selected, event_failures)
            if event_attempts > 0:
                semantic_attempt_counts[profile_key] += event_attempts
            if event_failures > 0:
                semantic_failure_counts[profile_key] += event_failures
        top_semantic_failure_keys = [
            profile_key for profile_key, _ in semantic_failure_counts.most_common(10)
        ]
        signal_kinds = [MemorySignalKind(str(signal.kind)) for signal in signals]
        signal_kind_counts = Counter(str(kind) for kind in signal_kinds)
        suspect_kinds = {
            MemorySignalKind.POSSIBLE_CONFLICT,
            MemorySignalKind.POSSIBLE_SUPERSESSION,
            MemorySignalKind.STALE_MEMORY,
            MemorySignalKind.LOW_CONFIDENCE_RELATION,
            MemorySignalKind.REPEATED_FAILURE,
            MemorySignalKind.DRIFT_SPIKE,
        }
        suspect_node_counts: Counter[str] = Counter()
        suspect_edge_counts: Counter[str] = Counter()
        for signal in signals:
            if MemorySignalKind(str(signal.kind)) not in suspect_kinds:
                continue
            suspect_node_counts.update(signal.node_ids)
            suspect_edge_counts.update(signal.edge_ids)
        openie_nodes = sum(
            1
            for node in nodes
            if "_openie" in (node.tags or []) or "_openie_entity" in (node.tags or [])
        )
        openie_edges = sum(
            1
            for edge in edges
            if edge.id.startswith("openie_") or _prop_bool(edge.properties, "is_openie")
        )
        boosted_retrieval_count = 0
        demoted_retrieval_count = 0
        adjusted_retrieval_count = 0
        penalized_retrieval_count = 0
        boosted_node_count = 0
        demoted_node_count = 0
        adjusted_node_count = 0
        penalized_node_count = 0
        feedback_event_count = 0
        feedback_success_count = 0
        feedback_failure_count = 0
        feedback_neutral_count = 0
        feedback_signal_counts: Counter[str] = Counter()
        feedback_node_counts: Counter[str] = Counter()
        feedback_success_node_counts: Counter[str] = Counter()
        feedback_failure_node_counts: Counter[str] = Counter()
        feedback_neutral_node_counts: Counter[str] = Counter()
        max_scope_boost = 0.0
        max_scope_demotion = 0.0
        max_scope_adjustment = 0.0
        max_signal_penalty = 0.0
        penalty_signal_counts: Counter[str] = Counter()
        penalized_node_counts: Counter[str] = Counter()
        penalty_edge_counts: Counter[str] = Counter()
        for event in retrieval_events:
            props = event.properties or {}
            signal = str(event.signal)
            is_feedback_event = (
                bool(event.selected_node_ids)
                or event.success is not None
                or signal != str(FeedbackSignal.SELECTED)
                or bool(props.get("parent_event_id"))
            )
            if is_feedback_event:
                feedback_event_count += 1
                feedback_signal_counts[signal] += 1
                feedback_target_ids = list(event.selected_node_ids or [])
                feedback_node_counts.update(feedback_target_ids)
                if event.success is True:
                    feedback_success_count += 1
                    feedback_success_node_counts.update(feedback_target_ids)
                elif event.success is False:
                    feedback_failure_count += 1
                    feedback_failure_node_counts.update(feedback_target_ids)
                else:
                    feedback_neutral_count += 1
                    feedback_neutral_node_counts.update(feedback_target_ids)
            boosted_nodes = _prop_int(props, "memory_scope_boosted_nodes", 0)
            demoted_nodes = _prop_int(props, "memory_scope_demoted_nodes", 0)
            adjusted_nodes = _prop_int(
                props,
                "memory_scope_adjusted_nodes",
                boosted_nodes + demoted_nodes,
            )
            penalized_nodes = _prop_int(props, "memory_signal_penalized_nodes", 0)
            if boosted_nodes > 0:
                boosted_retrieval_count += 1
                boosted_node_count += boosted_nodes
                max_scope_boost = max(
                    max_scope_boost,
                    _prop_float(
                        props,
                        "memory_scope_max_positive_boost",
                        _prop_float(props, "memory_scope_max_abs_boost", 0.0),
                    ),
                )
            if demoted_nodes > 0:
                demoted_retrieval_count += 1
                demoted_node_count += demoted_nodes
                max_scope_demotion = max(
                    max_scope_demotion,
                    _prop_float(
                        props,
                        "memory_scope_max_demotion",
                        _prop_float(props, "memory_scope_max_abs_boost", 0.0),
                    ),
                )
            if adjusted_nodes > 0:
                adjusted_retrieval_count += 1
                adjusted_node_count += adjusted_nodes
                max_scope_adjustment = max(
                    max_scope_adjustment,
                    _prop_float(props, "memory_scope_max_abs_boost", 0.0),
                )
            if penalized_nodes > 0:
                penalized_retrieval_count += 1
                penalized_node_count += penalized_nodes
                penalty_signal_counts.update(_prop_csv_ids(props, "memory_signal_source_ids"))
                penalized_node_counts.update(
                    _prop_csv_ids(props, "memory_signal_penalized_node_ids")
                )
                penalty_edge_counts.update(_prop_csv_ids(props, "memory_signal_edge_ids"))
                max_signal_penalty = max(
                    max_signal_penalty,
                    _prop_float(props, "memory_signal_max_penalty", 0.0),
                )

        return MemoryHealthReport(
            scope_key=memory_scope_key(effective_scope),
            total_nodes=len(nodes),
            total_edges=len(edges),
            memory_events=len(memory_events),
            memory_event_kind_counts=dict(memory_event_kind_counts),
            memory_event_scope_counts=dict(memory_event_scope_counts),
            retrieval_events=len(retrieval_events),
            retrieval_event_scope_counts=dict(retrieval_event_scope_counts),
            memory_score_scope_counts=dict(memory_score_scope_counts),
            memory_score_count=len(all_scores),
            memory_score_node_count=memory_score_node_count,
            memory_score_edge_count=memory_score_edge_count,
            memory_score_positive_count=memory_score_positive_count,
            memory_score_negative_count=memory_score_negative_count,
            memory_score_neutral_count=memory_score_neutral_count,
            signal_count=len(signals),
            new_entity_count=signal_kinds.count(MemorySignalKind.NEW_ENTITY),
            new_relation_count=signal_kinds.count(MemorySignalKind.NEW_RELATION),
            relation_reinforced_count=signal_kinds.count(MemorySignalKind.RELATION_REINFORCED),
            suspect_count=sum(1 for kind in signal_kinds if kind in suspect_kinds),
            conflict_signal_count=signal_kinds.count(MemorySignalKind.POSSIBLE_CONFLICT),
            possible_supersession_count=signal_kinds.count(MemorySignalKind.POSSIBLE_SUPERSESSION),
            stale_signal_count=signal_kinds.count(MemorySignalKind.STALE_MEMORY),
            repeated_failure_count=signal_kinds.count(MemorySignalKind.REPEATED_FAILURE),
            low_confidence_relation_count=signal_kinds.count(
                MemorySignalKind.LOW_CONFIDENCE_RELATION
            ),
            drift_spike_count=signal_kinds.count(MemorySignalKind.DRIFT_SPIKE),
            signal_kind_counts=dict(signal_kind_counts),
            feedback_event_count=feedback_event_count,
            feedback_success_count=feedback_success_count,
            feedback_failure_count=feedback_failure_count,
            feedback_neutral_count=feedback_neutral_count,
            feedback_signal_counts=dict(feedback_signal_counts),
            top_feedback_node_ids=[node_id for node_id, _ in feedback_node_counts.most_common(10)],
            top_feedback_success_node_ids=[
                node_id for node_id, _ in feedback_success_node_counts.most_common(10)
            ],
            top_feedback_failure_node_ids=[
                node_id for node_id, _ in feedback_failure_node_counts.most_common(10)
            ],
            top_feedback_neutral_node_ids=[
                node_id for node_id, _ in feedback_neutral_node_counts.most_common(10)
            ],
            top_feedback_node_counts={
                node_id: count for node_id, count in feedback_node_counts.most_common(10)
            },
            top_feedback_success_node_counts={
                node_id: count for node_id, count in feedback_success_node_counts.most_common(10)
            },
            top_feedback_failure_node_counts={
                node_id: count for node_id, count in feedback_failure_node_counts.most_common(10)
            },
            top_feedback_neutral_node_counts={
                node_id: count for node_id, count in feedback_neutral_node_counts.most_common(10)
            },
            openie_artifact_count=openie_nodes + openie_edges,
            semantic_extract_failure_count=failures,
            semantic_extract_attempt_count=semantic_attempt_count,
            openie_failure_rate=(failures / semantic_attempt_count)
            if semantic_attempt_count
            else 0.0,
            semantic_extract_failure_counts={
                profile_key: semantic_failure_counts[profile_key]
                for profile_key in top_semantic_failure_keys
            },
            semantic_extract_attempt_counts={
                profile_key: semantic_attempt_counts[profile_key]
                for profile_key in top_semantic_failure_keys
            },
            semantic_extract_failure_rates={
                profile_key: (
                    semantic_failure_counts[profile_key] / semantic_attempt_counts[profile_key]
                    if semantic_attempt_counts[profile_key]
                    else 0.0
                )
                for profile_key in top_semantic_failure_keys
            },
            memory_boosted_retrieval_count=boosted_retrieval_count,
            memory_demoted_retrieval_count=demoted_retrieval_count,
            memory_adjusted_retrieval_count=adjusted_retrieval_count,
            memory_penalized_retrieval_count=penalized_retrieval_count,
            memory_boosted_node_count=boosted_node_count,
            memory_demoted_node_count=demoted_node_count,
            memory_adjusted_node_count=adjusted_node_count,
            memory_penalized_node_count=penalized_node_count,
            max_memory_scope_boost=max_scope_boost,
            max_memory_scope_demotion=max_scope_demotion,
            max_memory_scope_adjustment=max_scope_adjustment,
            max_memory_signal_penalty=max_signal_penalty,
            top_reinforced_node_ids=[score.node_id for score in top_node_scores if score.node_id],
            top_reinforced_edge_ids=[score.edge_id for score in top_edge_scores if score.edge_id],
            top_reinforced_node_scores={
                score.node_id: score.score for score in top_node_scores if score.node_id
            },
            top_reinforced_edge_scores={
                score.edge_id: score.score for score in top_edge_scores if score.edge_id
            },
            top_demoted_node_ids=[
                score.node_id for score in top_demoted_node_scores if score.node_id
            ],
            top_demoted_edge_ids=[
                score.edge_id for score in top_demoted_edge_scores if score.edge_id
            ],
            top_demoted_node_scores={
                score.node_id: score.score for score in top_demoted_node_scores if score.node_id
            },
            top_demoted_edge_scores={
                score.edge_id: score.score for score in top_demoted_edge_scores if score.edge_id
            },
            top_suspect_node_ids=[node_id for node_id, _ in suspect_node_counts.most_common(10)],
            top_suspect_edge_ids=[edge_id for edge_id, _ in suspect_edge_counts.most_common(10)],
            top_suspect_node_counts={
                node_id: count for node_id, count in suspect_node_counts.most_common(10)
            },
            top_suspect_edge_counts={
                edge_id: count for edge_id, count in suspect_edge_counts.most_common(10)
            },
            top_penalty_signal_ids=[
                signal_id for signal_id, _ in penalty_signal_counts.most_common(10)
            ],
            top_penalized_node_ids=[
                node_id for node_id, _ in penalized_node_counts.most_common(10)
            ],
            top_penalty_edge_ids=[edge_id for edge_id, _ in penalty_edge_counts.most_common(10)],
            top_penalty_signal_counts={
                signal_id: count for signal_id, count in penalty_signal_counts.most_common(10)
            },
            top_penalized_node_counts={
                node_id: count for node_id, count in penalized_node_counts.most_common(10)
            },
            top_penalty_edge_counts={
                edge_id: count for edge_id, count in penalty_edge_counts.most_common(10)
            },
        )

    async def _semantic_extract_drift_signals(
        self,
        scope: MemoryScope,
        *,
        since: float | None,
    ) -> list[MemorySignal]:
        events = await self._list_memory_events(scope=scope, since=since, limit=100_000)
        buckets: dict[tuple[str, str, str, str], dict[str, object]] = {}
        for event in events:
            if str(event.kind) != MemoryEventKind.SEMANTIC_EXTRACT:
                continue
            failures = _prop_int(event.properties, "extraction_failures", 0)
            if failures <= 0:
                continue
            selected = max(0, _prop_int(event.properties, "chunks_selected", 0))
            source = event.source or "unknown"
            extractor = event.properties.get("extractor", "") or event.source_id or "unknown"
            model = event.properties.get("model", "")
            prompt_version = event.properties.get("prompt_version", "")
            key = (source, extractor, model, prompt_version)
            bucket = buckets.setdefault(
                key,
                {
                    "failures": 0,
                    "selected": 0,
                    "events": 0,
                    "node_ids": [],
                    "edge_ids": [],
                },
            )
            bucket["failures"] = int(bucket["failures"]) + failures
            bucket["selected"] = int(bucket["selected"]) + selected
            bucket["events"] = int(bucket["events"]) + 1
            cast_node_ids = cast(list[str], bucket["node_ids"])
            cast_edge_ids = cast(list[str], bucket["edge_ids"])
            cast_node_ids.extend(event.node_ids)
            cast_edge_ids.extend(event.edge_ids)

        signals: list[MemorySignal] = []
        for (source, extractor, model, prompt_version), bucket in buckets.items():
            failures = int(bucket["failures"])
            selected = int(bucket["selected"])
            attempts = max(selected, failures)
            failure_rate = failures / attempts if attempts else 0.0
            if (
                failures < _MEMORY_DRIFT_MIN_FAILURES
                or attempts < _MEMORY_DRIFT_MIN_FAILURES
                or failure_rate < _MEMORY_DRIFT_MIN_FAILURE_RATE
            ):
                continue
            confidence = min(0.95, max(0.7, 0.55 + failure_rate * 0.4))
            node_ids = cast(list[str], bucket["node_ids"])
            edge_ids = cast(list[str], bucket["edge_ids"])
            signals.append(
                self._make_signal(
                    MemorySignalKind.DRIFT_SPIKE,
                    scope,
                    node_ids=[str(node_id) for node_id in node_ids],
                    edge_ids=[str(edge_id) for edge_id in edge_ids],
                    confidence=confidence,
                    reason=(
                        "semantic extraction failure spike "
                        f"source={source} extractor={extractor} "
                        f"model={model or 'unknown'} "
                        f"prompt_version={prompt_version or 'unknown'} "
                        f"failures={failures}/{attempts}"
                    ),
                    properties={
                        "source": source,
                        "extractor": extractor,
                        "model": model,
                        "prompt_version": prompt_version,
                        "failure_count": str(failures),
                        "attempt_count": str(attempts),
                        "failure_rate": f"{failure_rate:.6f}",
                        "event_count": str(bucket["events"]),
                    },
                )
            )
        return signals

    async def _all_edges(self, nodes: list[Node]) -> list[Edge]:
        seen: set[str] = set()
        edges: list[Edge] = []
        for node in nodes:
            for edge in await self._backend.get_edges(node.id):
                if edge.id in seen:
                    continue
                seen.add(edge.id)
                edges.append(edge)
        return edges

    def _make_signal(
        self,
        kind: MemorySignalKind,
        scope: MemoryScope,
        *,
        node_ids: list[str] | None = None,
        edge_ids: list[str] | None = None,
        confidence: float,
        reason: str,
        properties: dict[str, str] | None = None,
    ) -> MemorySignal:
        node_ids = list(dict.fromkeys(node_ids or []))
        edge_ids = list(dict.fromkeys(edge_ids or []))
        properties = dict(properties or {})
        payload = json.dumps(
            {
                "kind": str(kind),
                "scope": memory_scope_key(scope),
                "node_ids": node_ids,
                "edge_ids": edge_ids,
                "reason": reason,
                "properties": properties,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        signal_id = f"memsig_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"
        return MemorySignal(
            id=signal_id,
            kind=kind,
            scope=scope,
            node_ids=node_ids,
            edge_ids=edge_ids,
            confidence=confidence,
            reason=reason,
            properties=properties,
        )

    async def _persist_memory_signal(self, signal: MemorySignal) -> None:
        existing = await self._backend.get_node(signal.id)
        suspect = MemorySignalKind(str(signal.kind)) in {
            MemorySignalKind.POSSIBLE_CONFLICT,
            MemorySignalKind.POSSIBLE_SUPERSESSION,
            MemorySignalKind.STALE_MEMORY,
            MemorySignalKind.LOW_CONFIDENCE_RELATION,
            MemorySignalKind.REPEATED_FAILURE,
            MemorySignalKind.DRIFT_SPIKE,
        }
        tags = ["_memory_signal"]
        if suspect:
            tags.append("_memory_suspect")
        properties = {
            "signal_kind": str(signal.kind),
            "scope_key": memory_scope_key(signal.scope),
            "node_ids": ",".join(signal.node_ids),
            "edge_ids": ",".join(signal.edge_ids),
            "confidence": str(signal.confidence),
        }
        properties.update(signal.properties)
        node = Node(
            id=signal.id,
            kind=NodeKind.OBSERVATION,
            title=f"Memory signal: {signal.kind}",
            content=signal.reason,
            tags=tags,
            properties=properties,
            source="memory_monitor",
            created_at=signal.created_at,
            updated_at=time(),
        )
        await self._backend.save_node(node)
        if existing is None:
            await self._save_memory_event(
                MemoryEvent(
                    id=f"evt_{signal.id}",
                    kind=MemoryEventKind.SIGNAL,
                    scope=signal.scope,
                    source="memory_monitor",
                    source_id=signal.id,
                    node_ids=signal.node_ids + [signal.id],
                    edge_ids=signal.edge_ids,
                    confidence=signal.confidence,
                    properties={**properties, "reason": signal.reason},
                    created_at=signal.created_at,
                )
            )

    async def _list_memory_events(
        self,
        *,
        scope: MemoryScope | None,
        since: float | None,
        limit: int,
    ) -> list[MemoryEvent]:
        lister = getattr(self._backend, "list_memory_events", None)
        if not callable(lister):
            return []
        return await lister(scope=scope, since=since, limit=limit)

    async def _list_retrieval_events(
        self,
        *,
        scope: MemoryScope | None,
        since: float | None,
        limit: int,
    ) -> list[RetrievalEvent]:
        lister = getattr(self._backend, "list_retrieval_events", None)
        if not callable(lister):
            return []
        return await lister(scope=scope, since=since, limit=limit)

    async def _list_memory_scores(
        self,
        *,
        scope_key: str | None,
        node_ids: list[str] | None = None,
        edge_ids: list[str] | None = None,
        limit: int,
    ) -> list[MemoryScore]:
        lister = getattr(self._backend, "list_memory_scores", None)
        if not callable(lister):
            return []
        return await lister(
            scope_key=scope_key,
            node_ids=node_ids,
            edge_ids=edge_ids,
            limit=limit,
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

    @staticmethod
    def _compose_embed_text(
        title: str,
        content: str,
        properties: dict[str, str] | None,
    ) -> str:
        """Build the text passed to the embedder for a node.

        Mirrors the composition used at ingest time so an update produces
        the same vector that ``add`` would have produced for equivalent
        fields. LLM-classifier metadata (``_summary``, ``_search_keywords``)
        is folded in when present.
        """
        embed_text = f"{title} {content}".strip()
        if properties:
            search_kw = properties.get("_search_keywords", "")
            summary = properties.get("_summary", "")
            if search_kw or summary:
                embed_text = f"{title} {summary} {search_kw} {content}".strip()
        return embed_text

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
        reembed: bool = True,
    ) -> Node | None:
        """Update a node's fields by ID. Returns updated node, or None if not found.

        When ``title``, ``content``, or ``properties`` change and an embedder
        is wired into the graph, the node's vector is automatically recomputed
        so search stays consistent. Pass ``embedding`` to supply a vector
        directly (skips re-embed), or ``reembed=False`` to suppress it.
        """
        node = await self._backend.get_node(node_id)
        if node is None:
            return None
        previous_hash = _node_content_hash(node)
        text_changed = title is not None or content is not None or properties is not None
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
        elif reembed and text_changed and self._embedder is not None:
            embed_text = self._compose_embed_text(node.title, node.content, node.properties)
            if embed_text:
                node.embedding = await self._embedder.embed(embed_text)
        changed_fields = [
            name
            for name, value in (
                ("title", title),
                ("content", content),
                ("kind", kind),
                ("tags", tags),
                ("properties", properties),
                ("embedding", embedding),
            )
            if value is not None
        ]
        node.updated_at = time()
        await self._backend.update_node(node)
        self._cache.invalidate(node_id)
        self._cache.put(node)
        await self._save_memory_event(
            MemoryEvent(
                kind=MemoryEventKind.UPDATE,
                source=node.source or "graph",
                source_id=node.id,
                content_hash=_node_content_hash(node),
                node_ids=[node.id],
                edge_ids=await self._touching_edge_ids(node.id),
                properties={
                    "operation": "SynapticGraph.update",
                    "changed_fields": ",".join(changed_fields),
                    "previous_content_hash": previous_hash,
                    "kind": str(node.kind),
                },
            )
        )
        return node

    async def remove(self, node_id: str) -> bool:
        node = await self._backend.get_node(node_id)
        if node is None:
            return False
        edge_ids = await self._touching_edge_ids(node_id)
        content_hash = _node_content_hash(node)
        # Remove from relation detector index
        if self._relation_detector is not None:
            self._relation_detector.index.remove(node_id)
        await self._store.delete_node(node_id)
        self._cache.invalidate(node_id)
        self._corpus_size = max(0, self._corpus_size - 1)
        await self._save_memory_event(
            MemoryEvent(
                kind=MemoryEventKind.DELETE,
                source=node.source or "graph",
                source_id=node.id,
                content_hash=content_hash,
                node_ids=[node.id],
                edge_ids=edge_ids,
                properties={"operation": "SynapticGraph.remove", "kind": str(node.kind)},
            )
        )
        return True

    async def reinforce(self, node_ids: list[str], *, success: bool = True) -> None:
        signal = FeedbackSignal.TASK_SUCCESS if success else FeedbackSignal.TASK_FAILURE
        await self.record_feedback(node_ids=node_ids, signal=signal)

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
