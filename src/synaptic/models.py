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


def _float_dict() -> dict[str, float]:
    return {}


def _int_dict() -> dict[str, int]:
    return {}


def _diagnostic_dict() -> dict[str, float | str]:
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
    # Explicit cross-reference between documents (e.g. a statute article
    # citing another article). Traversable via the agent ``follow`` tool.
    REFERENCES = "references"


class MemoryEventKind(StrEnum):
    INGEST = "ingest"
    UPDATE = "update"
    DELETE = "delete"
    SEMANTIC_EXTRACT = "semantic_extract"
    RETRIEVAL = "retrieval"
    FEEDBACK = "feedback"
    MAINTENANCE = "maintenance"
    SIGNAL = "signal"


class FeedbackSignal(StrEnum):
    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    SELECTED = "selected"
    IGNORED = "ignored"
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    TEST_PASS = "test_pass"  # noqa: S105 - feedback signal name, not a secret
    TEST_FAIL = "test_fail"


class MemorySignalKind(StrEnum):
    NEW_ENTITY = "new_entity"
    NEW_RELATION = "new_relation"
    RELATION_REINFORCED = "relation_reinforced"
    POSSIBLE_CONFLICT = "possible_conflict"
    POSSIBLE_SUPERSESSION = "possible_supersession"
    STALE_MEMORY = "stale_memory"
    DRIFT_SPIKE = "drift_spike"
    LOW_CONFIDENCE_RELATION = "low_confidence_relation"
    REPEATED_FAILURE = "repeated_failure"


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


@dataclass(slots=True)
class MemoryScope:
    """Scope that keeps memory reinforcement from leaking across users/tasks."""

    workspace_id: str = ""
    user_id: str = ""
    session_id: str = ""
    domain: str = ""
    promote_to_global: bool = False

    @property
    def key(self) -> str:
        return memory_scope_key(self)


def memory_scope_key(scope: MemoryScope | None, *, global_scope: bool = False) -> str:
    """Return the storage key for a memory scope.

    Specificity is intentional: session feedback should not automatically
    dominate the whole workspace, and workspace feedback should not
    silently become global.
    """
    if global_scope:
        return "global"
    if scope is None:
        return "global"
    if scope.session_id:
        return f"session:{scope.session_id}"
    if scope.user_id:
        return f"user:{scope.user_id}"
    if scope.workspace_id:
        return f"workspace:{scope.workspace_id}"
    if scope.domain:
        return f"domain:{scope.domain}"
    return "global"


def _scope() -> MemoryScope:
    return MemoryScope()


@dataclass(slots=True)
class MemoryEvent:
    """Durable ledger entry for memory creation, extraction, feedback, and upkeep."""

    id: str = field(default_factory=_new_id)
    kind: str | MemoryEventKind = MemoryEventKind.INGEST
    scope: MemoryScope = field(default_factory=_scope)
    source: str = ""
    source_id: str = ""
    content_hash: str = ""
    node_ids: list[str] = field(default_factory=_str_list)
    edge_ids: list[str] = field(default_factory=_str_list)
    confidence: float = 1.0
    properties: dict[str, str] = field(default_factory=_str_dict)
    created_at: float = field(default_factory=time)


@dataclass(slots=True)
class RetrievalEvent:
    """One retrieval or feedback observation.

    ``success`` is tri-state on purpose: implicit signals such as
    ``selected`` are useful, but should not be treated as ground truth.
    """

    id: str = field(default_factory=_new_id)
    query: str = ""
    scope: MemoryScope = field(default_factory=_scope)
    returned_node_ids: list[str] = field(default_factory=_str_list)
    selected_node_ids: list[str] = field(default_factory=_str_list)
    success: bool | None = None
    signal: str | FeedbackSignal = FeedbackSignal.SELECTED
    confidence: float = 1.0
    created_at: float = field(default_factory=time)
    properties: dict[str, str] = field(default_factory=_str_dict)


@dataclass(slots=True)
class MemoryScore:
    """Scope-local reinforcement score for a node or edge."""

    scope_key: str = "global"
    node_id: str = ""
    edge_id: str = ""
    access_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    score: float = 0.0
    updated_at: float = field(default_factory=time)


@dataclass(slots=True)
class MemorySignal:
    """A non-destructive warning or lifecycle signal emitted by the monitor."""

    id: str = field(default_factory=_new_id)
    kind: str | MemorySignalKind = MemorySignalKind.NEW_ENTITY
    scope: MemoryScope = field(default_factory=_scope)
    node_ids: list[str] = field(default_factory=_str_list)
    edge_ids: list[str] = field(default_factory=_str_list)
    confidence: float = 1.0
    reason: str = ""
    properties: dict[str, str] = field(default_factory=_str_dict)
    created_at: float = field(default_factory=time)


@dataclass(slots=True)
class MemoryHealthReport:
    """Compact operational health summary for the memory layer."""

    scope_key: str = "global"
    total_nodes: int = 0
    total_edges: int = 0
    memory_events: int = 0
    retrieval_events: int = 0
    signal_count: int = 0
    new_entity_count: int = 0
    new_relation_count: int = 0
    relation_reinforced_count: int = 0
    suspect_count: int = 0
    conflict_signal_count: int = 0
    stale_signal_count: int = 0
    repeated_failure_count: int = 0
    low_confidence_relation_count: int = 0
    drift_spike_count: int = 0
    openie_artifact_count: int = 0
    openie_failure_rate: float = 0.0
    memory_boosted_retrieval_count: int = 0
    memory_demoted_retrieval_count: int = 0
    memory_adjusted_retrieval_count: int = 0
    memory_penalized_retrieval_count: int = 0
    memory_boosted_node_count: int = 0
    memory_demoted_node_count: int = 0
    memory_adjusted_node_count: int = 0
    memory_penalized_node_count: int = 0
    max_memory_scope_boost: float = 0.0
    max_memory_scope_demotion: float = 0.0
    max_memory_scope_adjustment: float = 0.0
    max_memory_signal_penalty: float = 0.0
    top_reinforced_node_ids: list[str] = field(default_factory=_str_list)
    top_reinforced_edge_ids: list[str] = field(default_factory=_str_list)
    top_demoted_node_ids: list[str] = field(default_factory=_str_list)
    top_demoted_edge_ids: list[str] = field(default_factory=_str_list)
    top_suspect_node_ids: list[str] = field(default_factory=_str_list)
    top_suspect_edge_ids: list[str] = field(default_factory=_str_list)
    top_suspect_node_counts: dict[str, int] = field(default_factory=_int_dict)
    top_suspect_edge_counts: dict[str, int] = field(default_factory=_int_dict)
    top_penalty_signal_ids: list[str] = field(default_factory=_str_list)
    top_penalized_node_ids: list[str] = field(default_factory=_str_list)
    top_penalty_edge_ids: list[str] = field(default_factory=_str_list)
    top_penalty_signal_counts: dict[str, int] = field(default_factory=_int_dict)
    top_penalized_node_counts: dict[str, int] = field(default_factory=_int_dict)
    top_penalty_edge_counts: dict[str, int] = field(default_factory=_int_dict)
    generated_at: float = field(default_factory=time)


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
    properties: dict[str, str] = field(default_factory=_str_dict)
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
    timings_ms: dict[str, float] = field(default_factory=_float_dict)
    diagnostics: dict[str, float | str] = field(default_factory=_diagnostic_dict)
    stages_used: list[str] = field(default_factory=_str_list)
    event_id: str = ""


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


@dataclass(slots=True)
class BackfillResult:
    """Counts from one ``SynapticGraph.backfill()`` run.

    ``backfill`` is the recovery path for the silent-failure modes
    documented in v0.14.x: graphs ingested without an embedder or
    without a phrase extractor end up missing data that downstream
    search depends on, and there used to be no way to recover
    short of re-ingesting the source.

    Attributes:
        scanned: Total nodes inspected (regardless of whether any
            work was needed).
        embeddings_filled: Nodes that gained an embedding because
            ``embeddings=True`` was requested *and* their previous
            embedding was empty.
        phrases_linked: Phrase-hub CONTAINS edges newly created
            by re-running the extractor on text-bearing nodes.
            Only counts edges to *new* hubs, not duplicates.
        skipped_no_text: Nodes skipped because they had no title
            and no content to embed or phrase-extract.
        elapsed_ms: Wall-clock time of the backfill call.
        errors: Per-node error messages — empty when every node
            processed cleanly. Backfill is best-effort: a single
            failing row never aborts the rest of the batch.
    """

    scanned: int = 0
    embeddings_filled: int = 0
    phrases_linked: int = 0
    skipped_no_text: int = 0
    elapsed_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


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
