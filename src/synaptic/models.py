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


@dataclass(slots=True)
class Node:
    id: str = field(default_factory=_new_id)
    kind: NodeKind = NodeKind.CONCEPT
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
