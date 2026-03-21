"""Protocol definitions for pluggable backends and extensions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from synaptic.models import (
    ConsolidationLevel,
    DigestResult,
    Edge,
    Node,
    NodeKind,
)


class StorageBackend(Protocol):
    """Storage backend for nodes and edges."""

    # Lifecycle
    async def connect(self) -> None: ...
    async def close(self) -> None: ...

    # Node CRUD
    async def save_node(self, node: Node) -> None: ...
    async def get_node(self, node_id: str) -> Node | None: ...
    async def update_node(self, node: Node) -> None: ...
    async def delete_node(self, node_id: str) -> None: ...
    async def list_nodes(
        self,
        *,
        kind: NodeKind | None = None,
        level: ConsolidationLevel | None = None,
        limit: int = 100,
    ) -> list[Node]: ...

    # Edge CRUD
    async def save_edge(self, edge: Edge) -> None: ...
    async def get_edges(
        self, node_id: str, *, direction: Literal["both", "incoming", "outgoing"] = "both"
    ) -> list[Edge]: ...
    async def update_edge(self, edge: Edge) -> None: ...
    async def delete_edge(self, edge_id: str) -> None: ...

    # Search
    async def search_fts(self, query: str, *, limit: int = 20) -> list[Node]: ...
    async def search_fuzzy(
        self, query: str, *, limit: int = 20, threshold: float = 0.3
    ) -> list[Node]: ...
    async def search_vector(self, embedding: list[float], *, limit: int = 20) -> list[Node]: ...

    # Graph traversal
    async def get_neighbors(self, node_id: str, *, depth: int = 1) -> list[tuple[Node, Edge]]: ...

    # Batch
    async def save_nodes_batch(self, nodes: Sequence[Node]) -> None: ...
    async def save_edges_batch(self, edges: Sequence[Edge]) -> None: ...

    # Maintenance
    async def prune_edges(self, *, weight_below: float = 0.1) -> int: ...
    async def decay_vitality(self, *, factor: float = 0.95) -> int: ...


class GraphTraversal(Protocol):
    """Extended protocol for graph-native backends (Neo4j, etc.).

    Provides advanced traversal operations beyond StorageBackend.get_neighbors().
    """

    async def shortest_path(
        self, from_id: str, to_id: str, *, max_depth: int = 5
    ) -> list[tuple[Node, Edge]]: ...

    async def pattern_match(
        self, pattern: str, *, limit: int = 20
    ) -> list[dict[str, object]]: ...

    async def find_by_type_hierarchy(
        self, type_name: str, *, limit: int = 50
    ) -> list[Node]: ...


class Digester(Protocol):
    """Converts structured context into knowledge nodes and edges."""

    async def digest(self, context: dict[str, object]) -> DigestResult: ...


class QueryRewriter(Protocol):
    """Rewrites a search query into expanded forms (e.g. via LLM)."""

    async def rewrite(self, query: str) -> list[str]: ...


class TagExtractor(Protocol):
    """Extracts tags from text content."""

    def extract(self, text: str) -> list[str]: ...
