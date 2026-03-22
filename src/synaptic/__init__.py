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
from synaptic.ppr import personalized_pagerank
from synaptic.extensions.classifier_rules import RuleBasedClassifier
from synaptic.extensions.embedder import EmbeddingProvider, MockEmbeddingProvider
from synaptic.extensions.phrase_extractor import PhraseExtractor
from synaptic.extensions.relation_detector import (
    EmbeddingRelationDetector,
    RuleBasedRelationDetector,
)
from synaptic.graph import SynapticGraph
from synaptic.evidence import EvidenceAssembler
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

__version__ = "0.8.0"

__all__ = [
    "ActivatedNode",
    "ActivityTracker",
    "AgentSearch",
    "ConsolidationLevel",
    "DigestResult",
    "Digester",
    "Edge",
    "EdgeKind",
    "EvidenceAssembler",
    "EvidenceChain",
    "EvidenceStep",
    "EmbeddingProvider",
    "EmbeddingRelationDetector",
    "GraphTraversal",
    "KindClassifier",
    "MaintenanceResult",
    "MockEmbeddingProvider",
    "Node",
    "NodeKind",
    "OntologyRegistry",
    "PhraseExtractor",
    "personalized_pagerank",
    "PropertyDef",
    "QueryRewriter",
    "RelationDetector",
    "RelationConstraint",
    "ResonanceWeights",
    "ClassificationResult",
    "LLMClassifier",
    "LLMRelationDetector",
    "OllamaLLMProvider",
    "OpenAILLMProvider",
    "HybridClassifier",
    "RuleBasedClassifier",
    "RuleBasedRelationDetector",
    "SearchIntent",
    "SearchResult",
    "StorageBackend",
    "SynapticGraph",
    "TagExtractor",
    "TypeDef",
    "build_agent_ontology",
    "suggest_intent",
]


def __getattr__(name: str) -> object:
    """Lazy import for optional-dep providers (avoids crash when aiohttp not installed)."""
    if name == "OpenAIEmbeddingProvider":
        from synaptic.extensions.embedder import OpenAIEmbeddingProvider  # noqa: PLC0415

        return OpenAIEmbeddingProvider
    if name == "OllamaEmbeddingProvider":
        from synaptic.extensions.embedder import OllamaEmbeddingProvider  # noqa: PLC0415

        return OllamaEmbeddingProvider
    if name == "HybridClassifier":
        from synaptic.extensions.classifier_hybrid import HybridClassifier  # noqa: PLC0415

        return HybridClassifier
    if name == "LLMClassifier":
        from synaptic.extensions.classifier_llm import LLMClassifier  # noqa: PLC0415

        return LLMClassifier
    if name == "ClassificationResult":
        from synaptic.extensions.classifier_llm import ClassificationResult  # noqa: PLC0415

        return ClassificationResult
    if name == "LLMRelationDetector":
        from synaptic.extensions.relation_detector_llm import LLMRelationDetector  # noqa: PLC0415

        return LLMRelationDetector
    if name == "OllamaLLMProvider":
        from synaptic.extensions.llm_provider import OllamaLLMProvider  # noqa: PLC0415

        return OllamaLLMProvider
    if name == "OpenAILLMProvider":
        from synaptic.extensions.llm_provider import OpenAILLMProvider  # noqa: PLC0415

        return OpenAILLMProvider
    msg = f"module 'synaptic' has no attribute {name!r}"
    raise AttributeError(msg)
