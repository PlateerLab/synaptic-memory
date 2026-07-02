"""Synaptic Memory — Brain-inspired knowledge graph for LLM agents.

Quick Start
-----------

1. In-memory (zero dependencies)::

    from synaptic import SynapticGraph

    graph = SynapticGraph.memory()
    await graph.add("API Incident Response", "Recovered after server restart", kind=NodeKind.LESSON)
    result = await graph.search("incident response")

2. SQLite (lightweight production)::

    graph = SynapticGraph.sqlite("knowledge.db")
    await graph.connect()
    await graph.add("Deploy Policy", "Auto-deploy after PR merge", kind=NodeKind.RULE)

3. Full-featured (LLM classification + embedding + relation detection)::

    from synaptic.backends.sqlite import SQLiteBackend
    from synaptic.extensions.llm_provider import OllamaLLMProvider

    graph = SynapticGraph.full(
        SQLiteBackend("knowledge.db"),
        llm=OllamaLLMProvider(model="gemma3:4b"),
        embed_api_base="http://localhost:8080/v1",
    )
    await graph.connect()

Backends
--------
- ``MemoryBackend`` — testing/development (zero-dep)
- ``SQLiteBackend`` — lightweight production (``pip install synaptic-memory[sqlite]``)
- ``KuzuBackend`` — embedded property graph DB (``pip install synaptic-memory[kuzu]``)
- ``PostgreSQLBackend`` — production with pgvector (``pip install synaptic-memory[postgresql]``)
- ``CompositeBackend`` — Kuzu + Qdrant + MinIO combined (``pip install synaptic-memory[scale]``)
"""

from __future__ import annotations

from synaptic.activity import ActivityTracker
from synaptic.agent_search import AgentSearch, SearchIntent, suggest_intent
from synaptic.evidence import EvidenceAssembler
from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
from synaptic.extensions.classifier_rules import RuleBasedClassifier
from synaptic.extensions.embedder import EmbeddingProvider, MockEmbeddingProvider
from synaptic.extensions.phrase_extractor import PhraseExtractor
from synaptic.extensions.relation_detector import (
    EmbeddingRelationDetector,
    RuleBasedRelationDetector,
)
from synaptic.graph import SynapticGraph
from synaptic.indexing import (
    CandidateProvider,
    CandidateScoreSource,
    CandidateSearchRequest,
    CandidateSearchResult,
    IndexFilter,
    IndexHealthBackend,
    IndexLagReport,
    IndexRouter,
    IngestionJob,
    IngestionJobStage,
    IngestionJobStatus,
    IngestionJobStore,
    ScoredCandidate,
    unique_candidates,
)
from synaptic.models import (
    ActivatedNode,
    ConsolidationLevel,
    DigestResult,
    Edge,
    EdgeKind,
    EvidenceChain,
    EvidenceStep,
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
from synaptic.ontology import (
    OntologyRegistry,
    PropertyDef,
    RelationConstraint,
    TypeDef,
    build_agent_ontology,
)
from synaptic.ppr import personalized_pagerank
from synaptic.protocols import (
    Digester,
    EntityExtractor,
    GraphTraversal,
    KindClassifier,
    MemoryEventBackend,
    MemoryScoreBackend,
    QueryRewriter,
    RelationDetector,
    StorageBackend,
    TagExtractor,
)
from synaptic.resonance import ResonanceWeights

__version__ = "0.27.0"

__all__ = [
    "ActivatedNode",
    "ActivityTracker",
    "AgentSearch",
    "ChainedEntityExtractor",
    "CandidateProvider",
    "CandidateScoreSource",
    "CandidateSearchRequest",
    "CandidateSearchResult",
    "ChunkEntityIndex",
    "ClassificationResult",
    "ConsolidationLevel",
    "DigestResult",
    "Digester",
    "Edge",
    "EdgeKind",
    "EmbeddingProvider",
    "EmbeddingRelationDetector",
    "EntityExtractor",
    "EvidenceAssembler",
    "EvidenceChain",
    "EvidenceStep",
    "FeedbackSignal",
    "GraphTraversal",
    "HybridClassifier",
    "HybridEntityExtractor",
    "IndexFilter",
    "IndexHealthBackend",
    "IndexLagReport",
    "IndexRouter",
    "IngestionJob",
    "IngestionJobStage",
    "IngestionJobStatus",
    "IngestionJobStore",
    "KindClassifier",
    "LLMClassifier",
    "LLMOpenIEExtractor",
    "LLMRelationDetector",
    "MaintenanceResult",
    "MemoryEvent",
    "MemoryEventBackend",
    "MemoryEventKind",
    "MemoryHealthReport",
    "MemoryScope",
    "MemoryScore",
    "MemoryScoreBackend",
    "MemorySignal",
    "MemorySignalKind",
    "MockEmbeddingProvider",
    "Node",
    "NodeKind",
    "OllamaLLMProvider",
    "OntologyRegistry",
    "OpenAILLMProvider",
    "OpenIELinker",
    "OpenIESelectionPolicy",
    "PhraseExtractor",
    "PropertyDef",
    "QueryRewriter",
    "RelationConstraint",
    "RelationDetector",
    "RetrievalEvent",
    "ResonanceWeights",
    "RuleBasedClassifier",
    "RuleBasedRelationDetector",
    "SearchIntent",
    "SearchResult",
    "ScoredCandidate",
    "SpaCyEntityExtractor",
    "StorageBackend",
    "SynapticGraph",
    "TableIngester",
    "TagExtractor",
    "TypeDef",
    "build_agent_ontology",
    "personalized_pagerank",
    "purge_openie_artifacts",
    "memory_scope_key",
    "suggest_intent",
    "unique_candidates",
    # v0.12
    "DomainProfile",
    "ProfileGenerator",
    "OntologyClassifier",
    "DocumentIngester",
    "JsonlDocumentSource",
    "EntityLinker",
    "EvidenceSearch",
    "SearchSession",
    "SessionStore",
    "SqliteGraphBackend",
]


def __getattr__(name: str) -> object:
    """Lazy import for optional-dep providers (avoids crash when aiohttp not installed)."""
    if name == "OpenAIEmbeddingProvider":
        from synaptic.extensions.embedder import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider
    if name == "OllamaEmbeddingProvider":
        from synaptic.extensions.embedder import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider
    if name == "HybridClassifier":
        from synaptic.extensions.classifier_hybrid import HybridClassifier

        return HybridClassifier
    if name == "LLMClassifier":
        from synaptic.extensions.classifier_llm import LLMClassifier

        return LLMClassifier
    if name == "ClassificationResult":
        from synaptic.extensions.classifier_llm import ClassificationResult

        return ClassificationResult
    if name == "LLMRelationDetector":
        from synaptic.extensions.relation_detector_llm import LLMRelationDetector

        return LLMRelationDetector
    if name == "OllamaLLMProvider":
        from synaptic.extensions.llm_provider import OllamaLLMProvider

        return OllamaLLMProvider
    if name == "OpenAILLMProvider":
        from synaptic.extensions.llm_provider import OpenAILLMProvider

        return OpenAILLMProvider
    if name == "SpaCyEntityExtractor":
        from synaptic.extensions.entity_extractor_spacy import SpaCyEntityExtractor

        return SpaCyEntityExtractor
    if name == "HybridEntityExtractor":
        from synaptic.extensions.entity_extractor_hybrid import HybridEntityExtractor

        return HybridEntityExtractor
    if name == "TableIngester":
        from synaptic.extensions.table_ingester import TableIngester

        return TableIngester
    # v0.12: agent tool layer + domain profile + 3rd-gen pipeline
    _LAZY_V012 = {
        "DomainProfile": "synaptic.extensions.domain_profile",
        "ProfileGenerator": "synaptic.extensions.profile_generator",
        "OntologyClassifier": "synaptic.extensions.ontology_classifier",
        "DocumentIngester": "synaptic.extensions.document_ingester",
        "JsonlDocumentSource": "synaptic.extensions.document_ingester",
        "EntityLinker": "synaptic.extensions.entity_linker",
        "ChainedEntityExtractor": "synaptic.extensions.entity_extractor_openie",
        "LLMOpenIEExtractor": "synaptic.extensions.entity_extractor_openie",
        "OpenIELinker": "synaptic.extensions.entity_extractor_openie",
        "OpenIESelectionPolicy": "synaptic.extensions.entity_extractor_openie",
        "purge_openie_artifacts": "synaptic.extensions.entity_extractor_openie",
        "EvidenceSearch": "synaptic.extensions.evidence_search",
        "SearchSession": "synaptic.search_session",
        "SessionStore": "synaptic.search_session",
        "SqliteGraphBackend": "synaptic.backends.sqlite_graph",
    }
    if name in _LAZY_V012:
        import importlib

        mod = importlib.import_module(_LAZY_V012[name])
        return getattr(mod, name)
    msg = f"module 'synaptic' has no attribute {name!r}"
    raise AttributeError(msg)
