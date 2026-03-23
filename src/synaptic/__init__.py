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
- ``PostgreSQLBackend`` — production (``pip install synaptic-memory[postgresql]``)
- ``Neo4jBackend`` — graph traversal (``pip install synaptic-memory[neo4j]``)
- ``CompositeBackend`` — Neo4j + Qdrant + MinIO combined (``pip install synaptic-memory[scale]``)
"""

from __future__ import annotations

from synaptic.activity import ActivityTracker
from synaptic.agent_search import AgentSearch, SearchIntent, suggest_intent
from synaptic.evidence import EvidenceAssembler
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
    GraphTraversal,
    KindClassifier,
    QueryRewriter,
    RelationDetector,
    StorageBackend,
    TagExtractor,
)
from synaptic.resonance import ResonanceWeights

__version__ = "0.9.0"

__all__ = [
    "ActivatedNode",
    "ActivityTracker",
    "AgentSearch",
    "ClassificationResult",
    "ConsolidationLevel",
    "DigestResult",
    "Digester",
    "Edge",
    "EdgeKind",
    "EmbeddingProvider",
    "EmbeddingRelationDetector",
    "EvidenceAssembler",
    "EvidenceChain",
    "EvidenceStep",
    "GraphTraversal",
    "HybridClassifier",
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
    "StorageBackend",
    "SynapticGraph",
    "TagExtractor",
    "TypeDef",
    "build_agent_ontology",
    "personalized_pagerank",
    "suggest_intent",
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
    msg = f"module 'synaptic' has no attribute {name!r}"
    raise AttributeError(msg)
