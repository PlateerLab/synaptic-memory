"""Synaptic Memory — Brain-inspired knowledge graph for LLM agents."""

from __future__ import annotations

from synaptic.activity import ActivityTracker
from synaptic.agent_search import AgentSearch, SearchIntent, suggest_intent
from synaptic.ppr import personalized_pagerank
from synaptic.extensions.classifier_rules import RuleBasedClassifier
from synaptic.extensions.embedder import EmbeddingProvider, MockEmbeddingProvider
from synaptic.extensions.relation_detector import RuleBasedRelationDetector
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

__version__ = "0.6.0"

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
    "GraphTraversal",
    "KindClassifier",
    "MockEmbeddingProvider",
    "Node",
    "NodeKind",
    "OntologyRegistry",
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
