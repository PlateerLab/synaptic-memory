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
    await graph.backend.connect()
    await graph.add("Deploy Policy", "Auto-deploy after PR merge", kind=NodeKind.RULE)

3. Full-featured (LLM classification + embedding + relation detection)::

    from synaptic.backends.sqlite import SQLiteBackend
    from synaptic.extensions.llm_provider import OllamaLLMProvider

    graph = SynapticGraph.full(
        SQLiteBackend("knowledge.db"),
        llm=OllamaLLMProvider(model="gemma3:4b"),
        embed_api_base="http://localhost:8080/v1",
    )
    await graph.backend.connect()

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
from synaptic.models import (
    ActivatedNode,
    ConsolidationLevel,
    DigestResult,
    Edge,
    EdgeKind,
    EvidenceChain,
    EvidenceStep,
    MaintenanceResult,
    Node,
    NodeKind,
    SearchResult,
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
    QueryRewriter,
    RelationDetector,
    StorageBackend,
    TagExtractor,
)
from synaptic.resonance import ResonanceWeights

__version__ = "0.14.0"

__all__ = [
    "ActivatedNode",
    "ActivityTracker",
    "AgentSearch",
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
    "GraphTraversal",
    "HybridClassifier",
    "HybridEntityExtractor",
    "KindClassifier",
    "LLMClassifier",
    "LLMRelationDetector",
    "MaintenanceResult",
    "MockEmbeddingProvider",
    "Node",
    "NodeKind",
    "OllamaLLMProvider",
    "OntologyRegistry",
    "OpenAILLMProvider",
    "PhraseExtractor",
    "PropertyDef",
    "QueryRewriter",
    "RelationConstraint",
    "RelationDetector",
    "ResonanceWeights",
    "RuleBasedClassifier",
    "RuleBasedRelationDetector",
    "SearchIntent",
    "SearchResult",
    "SpaCyEntityExtractor",
    "StorageBackend",
    "SynapticGraph",
    "TableIngester",
    "TagExtractor",
    "TypeDef",
    "build_agent_ontology",
    "personalized_pagerank",
    "suggest_intent",
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
