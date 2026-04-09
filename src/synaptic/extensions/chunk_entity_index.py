"""E2GraphRAG-style bidirectional index: entity ↔ chunk.

Maintains two in-memory dicts for O(1) lookup:
  - entity_to_chunks: {entity_node_id: set[chunk_node_id]}
  - chunk_to_entities: {chunk_node_id: set[entity_node_id]}

Persisted as MENTIONS / EXTRACTED_FROM edge relationships in the graph.
The in-memory index is rebuilt from edges on startup via rebuild_from_backend().
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from synaptic.models import EdgeKind, NodeKind

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend


class ChunkEntityIndex:
    """Bidirectional chunk ↔ entity index for grounded retrieval.

    When search finds an entity, this index instantly resolves which original
    text chunks mention it — providing passage-level grounding that pure
    triple-based KGs lack.

    Example::

        index = ChunkEntityIndex()
        index.register("chunk_001", "entity_042")
        index.chunks_for_entity("entity_042")  # {"chunk_001"}
        index.entities_for_chunk("chunk_001")   # {"entity_042"}
    """

    __slots__ = ("_chunk_to_entities", "_entity_to_chunks")

    def __init__(self) -> None:
        self._entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        self._chunk_to_entities: dict[str, set[str]] = defaultdict(set)

    # --- Registration ---

    def register(self, chunk_id: str, entity_id: str) -> None:
        """Register a chunk-entity link (called during ingestion)."""
        self._entity_to_chunks[entity_id].add(chunk_id)
        self._chunk_to_entities[chunk_id].add(entity_id)

    def unregister_chunk(self, chunk_id: str) -> None:
        """Remove a chunk and all its entity links."""
        entity_ids = self._chunk_to_entities.pop(chunk_id, set())
        for eid in entity_ids:
            s = self._entity_to_chunks.get(eid)
            if s:
                s.discard(chunk_id)
                if not s:
                    del self._entity_to_chunks[eid]

    def unregister_entity(self, entity_id: str) -> None:
        """Remove an entity and all its chunk links."""
        chunk_ids = self._entity_to_chunks.pop(entity_id, set())
        for cid in chunk_ids:
            s = self._chunk_to_entities.get(cid)
            if s:
                s.discard(entity_id)
                if not s:
                    del self._chunk_to_entities[cid]

    # --- Lookup ---

    def chunks_for_entity(self, entity_id: str) -> set[str]:
        """Get all chunks mentioning this entity."""
        return set(self._entity_to_chunks.get(entity_id, ()))

    def entities_for_chunk(self, chunk_id: str) -> set[str]:
        """Get all entities extracted from this chunk."""
        return set(self._chunk_to_entities.get(chunk_id, ()))

    def shared_entities(self, chunk_ids: list[str]) -> dict[str, int]:
        """Find entities shared across multiple chunks (bridge detection).

        Returns:
            {entity_id: number_of_chunks_mentioning_it} for entities
            appearing in 2+ of the given chunks.
        """
        counts: dict[str, int] = defaultdict(int)
        for cid in chunk_ids:
            for eid in self._chunk_to_entities.get(cid, ()):
                counts[eid] += 1
        return {eid: cnt for eid, cnt in counts.items() if cnt >= 2}

    def chunks_for_entities(self, entity_ids: list[str]) -> dict[str, float]:
        """Get chunks ranked by how many of the given entities they mention.

        Returns:
            {chunk_id: entity_overlap_count} sorted descending.
        """
        scores: dict[str, float] = defaultdict(float)
        for eid in entity_ids:
            for cid in self._entity_to_chunks.get(eid, ()):
                scores[cid] += 1.0
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    # --- Stats ---

    @property
    def entity_count(self) -> int:
        return len(self._entity_to_chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_to_entities)

    def stats(self) -> dict[str, int | float]:
        """Return index statistics."""
        entity_counts = [len(v) for v in self._entity_to_chunks.values()]
        chunk_counts = [len(v) for v in self._chunk_to_entities.values()]
        return {
            "entity_count": self.entity_count,
            "chunk_count": self.chunk_count,
            "avg_chunks_per_entity": (
                sum(entity_counts) / len(entity_counts) if entity_counts else 0.0
            ),
            "avg_entities_per_chunk": (
                sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0.0
            ),
        }

    # --- Persistence ---

    async def rebuild_from_backend(self, backend: StorageBackend) -> None:
        """Rebuild in-memory index from stored MENTIONS/EXTRACTED_FROM edges.

        Call this on startup to restore the index from persisted edges.
        """
        self._entity_to_chunks.clear()
        self._chunk_to_entities.clear()

        # Scan all chunk nodes and their edges
        chunk_nodes = await backend.list_nodes(kind=NodeKind.CHUNK, limit=100_000)
        for chunk_node in chunk_nodes:
            edges = await backend.get_edges(chunk_node.id, direction="outgoing")
            for edge in edges:
                if edge.kind == EdgeKind.MENTIONS:
                    self.register(chunk_node.id, edge.target_id)

        # Also scan EXTRACTED_FROM edges (entity → chunk direction)
        entity_nodes = await backend.list_nodes(kind=NodeKind.ENTITY, limit=100_000)
        for entity_node in entity_nodes:
            edges = await backend.get_edges(entity_node.id, direction="outgoing")
            for edge in edges:
                if edge.kind == EdgeKind.EXTRACTED_FROM:
                    self.register(edge.target_id, entity_node.id)
