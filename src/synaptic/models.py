"""Domain models for Synaptic Memory."""

from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from uuid import uuid4


def _new_id() -> str:
    return uuid4().hex[:16]


def _str_list() -> list[str]:
    return []


def _float_list() -> list[float]:
    return []


def _str_dict() -> dict[str, str]:
    return {}


class ConsolidationLevel(StrEnum):
    L0_RAW = "L0"
    L1_SPRINT = "L1"
    L2_MONTHLY = "L2"
    L3_PERMANENT = "L3"


class NodeKind(StrEnum):
    CONCEPT = "concept"
    ENTITY = "entity"
    LESSON = "lesson"
    DECISION = "decision"
    RULE = "rule"
    ARTIFACT = "artifact"
    AGENT = "agent"
    TASK = "task"
    SPRINT = "sprint"
    # v0.5: Agent activity & ontology
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    REASONING = "reasoning"
    OUTCOME = "outcome"
    SESSION = "session"
    TYPE_DEF = "type_def"
    # v1.0: RAG enhancement — chunk-entity graph
    CHUNK = "chunk"
    COMMUNITY = "community"


class EdgeKind(StrEnum):
    RELATED = "related"
    CAUSED = "caused"
    LEARNED_FROM = "learned_from"
    DEPENDS_ON = "depends_on"
    PRODUCED = "produced"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    # v0.5: Ontology & agent activity
    IS_A = "is_a"
    INVOKED = "invoked"
    RESULTED_IN = "resulted_in"
    PART_OF = "part_of"
    FOLLOWED_BY = "followed_by"
    CONTAINS = "contains"
    # v1.0: RAG enhancement — chunk-entity graph
    MENTIONS = "mentions"
    EXTRACTED_FROM = "extracted_from"
    NEXT_CHUNK = "next_chunk"


@dataclass(slots=True)
class Node:
    id: str = field(default_factory=_new_id)
    kind: str = NodeKind.CONCEPT
    title: str = ""
    content: str = ""
    tags: list[str] = field(default_factory=_str_list)
    level: ConsolidationLevel = ConsolidationLevel.L0_RAW
    embedding: list[float] = field(default_factory=_float_list)
    vitality: float = 1.0
    access_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    properties: dict[str, str] = field(default_factory=_str_dict)
    source: str = ""
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)


def _sparse_dict() -> dict[int, float]:
    return {}


@dataclass(slots=True)
class HybridEmbedding:
    """BGE-M3 style hybrid embedding: dense + sparse + ColBERT vectors."""

    dense: list[float] = field(default_factory=_float_list)
    sparse: dict[int, float] = field(default_factory=_sparse_dict)
    colbert: list[list[float]] | None = None


@dataclass(slots=True)
class Edge:
    id: str = field(default_factory=_new_id)
    source_id: str = ""
    target_id: str = ""
    kind: EdgeKind = EdgeKind.RELATED
    weight: float = 1.0
    created_at: float = field(default_factory=time)


def _activated_list() -> list["ActivatedNode"]:
    return []


def _node_list() -> list["Node"]:
    return []


def _edge_list() -> list["Edge"]:
    return []


@dataclass(slots=True)
class ActivatedNode:
    node: Node
    activation: float = 0.0
    resonance: float = 0.0
    path: list[str] = field(default_factory=_str_list)


@dataclass(slots=True)
class SearchResult:
    query: str = ""
    nodes: list[ActivatedNode] = field(default_factory=_activated_list)
    total_candidates: int = 0
    search_time_ms: float = 0.0
    stages_used: list[str] = field(default_factory=_str_list)


@dataclass(slots=True)
class DigestResult:
    nodes_created: list[Node] = field(default_factory=_node_list)
    edges_created: list[Edge] = field(default_factory=_edge_list)
    nodes_updated: list[str] = field(default_factory=_str_list)
    tokens_used: int = 0


@dataclass(slots=True)
class MaintenanceResult:
    """Unified result for maintenance operations (consolidate + decay + prune)."""

    consolidated: DigestResult | None = None
    decayed: int = 0
    pruned: int = 0

    @property
    def total_affected(self) -> int:
        count = self.decayed + self.pruned
        if self.consolidated:
            count += len(self.consolidated.nodes_created) + len(self.consolidated.nodes_updated)
        return count


def _evidence_step_list() -> list["EvidenceStep"]:
    return []


@dataclass(slots=True)
class EvidenceStep:
    """A single step in an evidence chain."""

    node: Node
    role: str = ""  # "seed", "bridge", "supporting"
    connection_to_next: str = ""  # connection description based on edge kind
    compressed_content: str = ""  # content after context compression
    facts: list[str] = field(default_factory=_str_list)


@dataclass(slots=True)
class EvidenceChain:
    """Search results assembled into an LLM-friendly context."""

    query: str = ""
    steps: list[EvidenceStep] = field(default_factory=_evidence_step_list)
    compressed_context: str = ""  # final assembled context string
    facts: list[str] = field(default_factory=_str_list)
    total_tokens_approx: int = 0  # approximate token count
    assembly_time_ms: float = 0.0


# --- Visualization / Explorer data models ---


def _dict_list() -> list[dict[str, object]]:
    return []


def _node_edge_list() -> list[tuple["Node", "Edge"]]:
    return []


@dataclass(slots=True)
class GraphData:
    """Graph visualization data — nodes + edges + communities + stats."""

    nodes: list[dict[str, object]] = field(default_factory=_dict_list)
    edges: list[dict[str, object]] = field(default_factory=_dict_list)
    communities: list[dict[str, object]] = field(default_factory=_dict_list)
    stats: dict[str, object] = field(default_factory=_str_dict)


@dataclass(slots=True)
class NodeDetail:
    """Full node detail with neighbors and context."""

    node: Node
    neighbors: list[tuple[Node, Edge]] = field(default_factory=_node_edge_list)
    chunk_count: int = 0
    community_id: str = ""


@dataclass(slots=True)
class EntityContext:
    """Entity with all source chunks and related entities."""

    entity: Node
    source_chunks: list[Node] = field(default_factory=_node_list)
    related_entities: list[tuple[Node, Edge]] = field(default_factory=_node_edge_list)
    community: dict[str, object] | None = None


@dataclass(slots=True)
class ChunkDetail:
    """Chunk with extracted entities and navigation."""

    chunk: Node
    extracted_entities: list[dict[str, object]] = field(default_factory=_dict_list)
    prev_chunk: Node | None = None
    next_chunk: Node | None = None
    parent_doc: str = ""


@dataclass(slots=True)
class EdgeDetail:
    """Edge with source/target nodes and evidence chunks."""

    edge: Edge
    source_node: Node | None = None
    target_node: Node | None = None
    evidence_chunks: list[Node] = field(default_factory=_node_list)


@dataclass(slots=True)
class TableRowDetail:
    """Table row node with column data and FK relations."""

    node: Node
    columns: dict[str, str] = field(default_factory=_str_dict)
    table_name: str = ""
    related_rows: list[tuple[Node, Edge]] = field(default_factory=_node_edge_list)
    schema: dict[str, object] = field(default_factory=_str_dict)


@dataclass(slots=True)
class CommunityDetail:
    """Community with members and key entities."""

    community: Node
    summary: str = ""
    members: list[Node] = field(default_factory=_node_list)
    key_entities: list[Node] = field(default_factory=_node_list)
    sub_communities: list[Node] = field(default_factory=_node_list)


@dataclass(slots=True)
class GraphStats:
    """Graph-level statistics."""

    total_nodes: int = 0
    total_edges: int = 0
    nodes_by_kind: dict[str, int] = field(default_factory=_str_dict)
    edges_by_kind: dict[str, int] = field(default_factory=_str_dict)
    entity_count: int = 0
    chunk_count: int = 0
    community_count: int = 0
    table_count: int = 0
    avg_entities_per_chunk: float = 0.0
    avg_edges_per_entity: float = 0.0
