"""Type stubs for synaptic — IDE autocomplete for lazy-imported classes."""

from synaptic.activity import ActivityTracker as ActivityTracker
from synaptic.agent_search import AgentSearch as AgentSearch
from synaptic.agent_search import SearchIntent as SearchIntent
from synaptic.agent_search import suggest_intent as suggest_intent
from synaptic.backends.sqlite_graph import SqliteGraphBackend as SqliteGraphBackend
from synaptic.evidence import EvidenceAssembler as EvidenceAssembler
from synaptic.extensions.classifier_hybrid import HybridClassifier as HybridClassifier
from synaptic.extensions.classifier_llm import ClassificationResult as ClassificationResult
from synaptic.extensions.classifier_llm import LLMClassifier as LLMClassifier
from synaptic.extensions.classifier_rules import RuleBasedClassifier as RuleBasedClassifier
from synaptic.extensions.document_ingester import DocumentIngester as DocumentIngester
from synaptic.extensions.document_ingester import JsonlDocumentSource as JsonlDocumentSource
from synaptic.extensions.domain_profile import DomainProfile as DomainProfile
from synaptic.extensions.embedder import EmbeddingProvider as EmbeddingProvider
from synaptic.extensions.embedder import MockEmbeddingProvider as MockEmbeddingProvider
from synaptic.extensions.embedder import OllamaEmbeddingProvider as OllamaEmbeddingProvider
from synaptic.extensions.embedder import OpenAIEmbeddingProvider as OpenAIEmbeddingProvider
from synaptic.extensions.entity_extractor_hybrid import (
    HybridEntityExtractor as HybridEntityExtractor,
)
from synaptic.extensions.entity_extractor_openie import (
    ChainedEntityExtractor as ChainedEntityExtractor,
)
from synaptic.extensions.entity_extractor_openie import LLMOpenIEExtractor as LLMOpenIEExtractor
from synaptic.extensions.entity_extractor_openie import OpenIELinker as OpenIELinker
from synaptic.extensions.entity_extractor_openie import (
    OpenIESelectionPolicy as OpenIESelectionPolicy,
)
from synaptic.extensions.entity_extractor_openie import (
    purge_openie_artifacts as purge_openie_artifacts,
)
from synaptic.extensions.entity_extractor_spacy import (
    SpaCyEntityExtractor as SpaCyEntityExtractor,
)
from synaptic.extensions.entity_linker import EntityLinker as EntityLinker
from synaptic.extensions.evidence_search import EvidenceSearch as EvidenceSearch
from synaptic.extensions.llm_provider import OllamaLLMProvider as OllamaLLMProvider
from synaptic.extensions.llm_provider import OpenAILLMProvider as OpenAILLMProvider
from synaptic.extensions.ontology_classifier import OntologyClassifier as OntologyClassifier
from synaptic.extensions.phrase_extractor import PhraseExtractor as PhraseExtractor
from synaptic.extensions.profile_generator import ProfileGenerator as ProfileGenerator
from synaptic.extensions.relation_detector import (
    EmbeddingRelationDetector as EmbeddingRelationDetector,
)
from synaptic.extensions.relation_detector import (
    RuleBasedRelationDetector as RuleBasedRelationDetector,
)
from synaptic.extensions.relation_detector_llm import LLMRelationDetector as LLMRelationDetector
from synaptic.extensions.table_ingester import TableIngester as TableIngester
from synaptic.graph import SynapticGraph as SynapticGraph
from synaptic.models import ActivatedNode as ActivatedNode
from synaptic.models import ConsolidationLevel as ConsolidationLevel
from synaptic.models import DigestResult as DigestResult
from synaptic.models import Edge as Edge
from synaptic.models import EdgeKind as EdgeKind
from synaptic.models import EvidenceChain as EvidenceChain
from synaptic.models import EvidenceStep as EvidenceStep
from synaptic.models import FeedbackSignal as FeedbackSignal
from synaptic.models import MemoryEvent as MemoryEvent
from synaptic.models import MemoryEventKind as MemoryEventKind
from synaptic.models import MemoryHealthReport as MemoryHealthReport
from synaptic.models import MemoryScope as MemoryScope
from synaptic.models import MemoryScore as MemoryScore
from synaptic.models import MemorySignal as MemorySignal
from synaptic.models import MemorySignalKind as MemorySignalKind
from synaptic.models import Node as Node
from synaptic.models import NodeKind as NodeKind
from synaptic.models import RetrievalEvent as RetrievalEvent
from synaptic.models import SearchResult as SearchResult
from synaptic.models import memory_scope_key as memory_scope_key
from synaptic.ontology import OntologyRegistry as OntologyRegistry
from synaptic.ontology import PropertyDef as PropertyDef
from synaptic.ontology import RelationConstraint as RelationConstraint
from synaptic.ontology import TypeDef as TypeDef
from synaptic.ontology import build_agent_ontology as build_agent_ontology
from synaptic.ppr import personalized_pagerank as personalized_pagerank
from synaptic.protocols import Digester as Digester
from synaptic.protocols import EntityExtractor as EntityExtractor
from synaptic.protocols import GraphTraversal as GraphTraversal
from synaptic.protocols import KindClassifier as KindClassifier
from synaptic.protocols import MemoryEventBackend as MemoryEventBackend
from synaptic.protocols import MemoryScoreBackend as MemoryScoreBackend
from synaptic.protocols import QueryRewriter as QueryRewriter
from synaptic.protocols import RelationDetector as RelationDetector
from synaptic.protocols import StorageBackend as StorageBackend
from synaptic.protocols import TagExtractor as TagExtractor
from synaptic.resonance import ResonanceWeights as ResonanceWeights
from synaptic.search_session import SearchSession as SearchSession
from synaptic.search_session import SessionStore as SessionStore

__version__: str
__all__: list[str]
