"""Synaptic Memory — Brain-inspired knowledge graph for LLM agents."""

from __future__ import annotations

from synaptic.activity import ActivityTracker
from synaptic.agent_search import AgentSearch, SearchIntent, suggest_intent
from synaptic.extensions.embedder import EmbeddingProvider, MockEmbeddingProvider
from synaptic.graph import SynapticGraph
from synaptic.models import (
    ActivatedNode,
    ConsolidationLevel,
    DigestResult,
    Edge,
    EdgeKind,
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
from synaptic.protocols import Digester, GraphTraversal, QueryRewriter, StorageBackend, TagExtractor
from synaptic.resonance import ResonanceWeights

__version__ = "0.5.0"

__all__ = [
    "ActivatedNode",
    "ActivityTracker",
    "AgentSearch",
    "ConsolidationLevel",
    "DigestResult",
    "Digester",
    "Edge",
    "EdgeKind",
    "EmbeddingProvider",
    "GraphTraversal",
    "MockEmbeddingProvider",
    "Node",
    "NodeKind",
    "OntologyRegistry",
    "PropertyDef",
    "QueryRewriter",
    "RelationConstraint",
    "ResonanceWeights",
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
    msg = f"module 'synaptic' has no attribute {name!r}"
    raise AttributeError(msg)
