"""Type stubs for synaptic — IDE autocomplete for lazy-imported classes."""

from synaptic.activity import ActivityTracker as ActivityTracker
from synaptic.agent_search import AgentSearch as AgentSearch
from synaptic.agent_search import SearchIntent as SearchIntent
from synaptic.agent_search import suggest_intent as suggest_intent
from synaptic.evidence import EvidenceAssembler as EvidenceAssembler
from synaptic.extensions.classifier_hybrid import HybridClassifier as HybridClassifier
from synaptic.extensions.classifier_llm import ClassificationResult as ClassificationResult
from synaptic.extensions.classifier_llm import LLMClassifier as LLMClassifier
from synaptic.extensions.classifier_rules import RuleBasedClassifier as RuleBasedClassifier
from synaptic.extensions.embedder import EmbeddingProvider as EmbeddingProvider
from synaptic.extensions.embedder import MockEmbeddingProvider as MockEmbeddingProvider
from synaptic.extensions.embedder import OllamaEmbeddingProvider as OllamaEmbeddingProvider
from synaptic.extensions.embedder import OpenAIEmbeddingProvider as OpenAIEmbeddingProvider
from synaptic.extensions.llm_provider import OllamaLLMProvider as OllamaLLMProvider
from synaptic.extensions.llm_provider import OpenAILLMProvider as OpenAILLMProvider
from synaptic.extensions.phrase_extractor import PhraseExtractor as PhraseExtractor
from synaptic.extensions.relation_detector import (
    EmbeddingRelationDetector as EmbeddingRelationDetector,
)
from synaptic.extensions.relation_detector import (
    RuleBasedRelationDetector as RuleBasedRelationDetector,
)
from synaptic.extensions.relation_detector_llm import LLMRelationDetector as LLMRelationDetector
from synaptic.graph import SynapticGraph as SynapticGraph
from synaptic.models import ActivatedNode as ActivatedNode
from synaptic.models import ConsolidationLevel as ConsolidationLevel
from synaptic.models import DigestResult as DigestResult
from synaptic.models import Edge as Edge
from synaptic.models import EdgeKind as EdgeKind
from synaptic.models import EvidenceChain as EvidenceChain
from synaptic.models import EvidenceStep as EvidenceStep
from synaptic.models import Node as Node
from synaptic.models import NodeKind as NodeKind
from synaptic.models import SearchResult as SearchResult
from synaptic.ontology import OntologyRegistry as OntologyRegistry
from synaptic.ontology import PropertyDef as PropertyDef
from synaptic.ontology import RelationConstraint as RelationConstraint
from synaptic.ontology import TypeDef as TypeDef
from synaptic.ontology import build_agent_ontology as build_agent_ontology
from synaptic.ppr import personalized_pagerank as personalized_pagerank
from synaptic.protocols import Digester as Digester
from synaptic.protocols import GraphTraversal as GraphTraversal
from synaptic.protocols import KindClassifier as KindClassifier
from synaptic.protocols import QueryRewriter as QueryRewriter
from synaptic.protocols import RelationDetector as RelationDetector
from synaptic.protocols import StorageBackend as StorageBackend
from synaptic.protocols import TagExtractor as TagExtractor
from synaptic.resonance import ResonanceWeights as ResonanceWeights

__version__: str
__all__: list[str]
