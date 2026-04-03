"""Entity resolution — merge duplicate entities in the knowledge graph.

Detects duplicates via:
  1. String similarity (SequenceMatcher on titles) — zero-dep
  2. Embedding similarity (cosine) — when embeddings available

Merges duplicates using SynapticGraph.merge() which combines content,
stats, and re-points edges.

Hooks into SynapticGraph.maintain() for periodic cleanup.

Usage::

    resolver = EntityResolver(threshold=0.85)
    merged = await resolver.resolve(graph)
    # merged = [("kept_id", "removed_id"), ...]
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from synaptic.models import NodeKind

if TYPE_CHECKING:
    from synaptic.graph import SynapticGraph

logger = logging.getLogger("entity-resolver")


def _title_similarity(a: str, b: str) -> float:
    """String similarity between two titles (0-1)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class EntityResolver:
    """Resolves duplicate entities via title + embedding similarity.

    Example::

        resolver = EntityResolver(threshold=0.85)
        merged = await resolver.resolve(graph)
        print(f"Merged {len(merged)} duplicate pairs")

    Threshold controls how similar two entities must be to merge:
      - 0.9+: very strict (nearly identical titles)
      - 0.85: moderate (recommended)
      - 0.7: aggressive (may merge related-but-different entities)
    """

    __slots__ = ("_embedding_weight", "_max_comparisons", "_threshold")

    def __init__(
        self,
        *,
        threshold: float = 0.85,
        embedding_weight: float = 0.4,
        max_comparisons: int = 5000,
    ) -> None:
        self._threshold = threshold
        self._embedding_weight = embedding_weight
        self._max_comparisons = max_comparisons

    async def resolve(self, graph: SynapticGraph) -> list[tuple[str, str]]:
        """Find and merge duplicate entities.

        Returns:
            List of (kept_id, removed_id) pairs that were merged.
        """
        backend = graph.backend
        entities = await backend.list_nodes(kind=NodeKind.ENTITY, limit=100_000)

        # Filter out phrase nodes (internal)
        entities = [e for e in entities if "_phrase" not in (e.tags or [])]

        if len(entities) < 2:
            return []

        # Find duplicate pairs
        duplicates: list[tuple[str, str, float]] = []
        comparisons = 0

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if comparisons >= self._max_comparisons:
                    break
                comparisons += 1

                a, b = entities[i], entities[j]

                # Title similarity
                title_sim = _title_similarity(a.title, b.title)

                # Embedding similarity (if available)
                embed_sim = 0.0
                if a.embedding and b.embedding:
                    embed_sim = _cosine_sim(a.embedding, b.embedding)

                # Combined score
                if embed_sim > 0:
                    combined = (
                        (1 - self._embedding_weight) * title_sim
                        + self._embedding_weight * embed_sim
                    )
                else:
                    combined = title_sim

                if combined >= self._threshold:
                    duplicates.append((a.id, b.id, combined))

            if comparisons >= self._max_comparisons:
                break

        # Sort by similarity (highest first) and merge
        duplicates.sort(key=lambda x: x[2], reverse=True)

        merged: list[tuple[str, str]] = []
        removed: set[str] = set()

        for kept_id, remove_id, sim in duplicates:
            # Skip if either was already removed
            if kept_id in removed or remove_id in removed:
                continue

            # Keep the one with more access/edges (more established)
            kept = await backend.get_node(kept_id)
            to_remove = await backend.get_node(remove_id)
            if kept is None or to_remove is None:
                continue

            # Prefer the node with more activity
            if to_remove.access_count > kept.access_count:
                kept_id, remove_id = remove_id, kept_id

            result = await graph.merge(remove_id, kept_id)
            if result is not None:
                merged.append((kept_id, remove_id))
                removed.add(remove_id)

                # Update chunk-entity index if available
                chunk_idx = getattr(graph, "_chunk_entity_index", None)
                if chunk_idx is not None:
                    # Move chunk links from removed to kept
                    chunks = chunk_idx.chunks_for_entity(remove_id)
                    for cid in chunks:
                        chunk_idx.register(cid, kept_id)
                    chunk_idx.unregister_entity(remove_id)

                logger.info(
                    f"Merged entity '{to_remove.title}' into '{kept.title}' "
                    f"(similarity={sim:.2f})"
                )

        if merged:
            logger.info(f"Resolved {len(merged)} duplicate entity pairs")

        return merged

    async def find_candidates(
        self, graph: SynapticGraph, *, limit: int = 20
    ) -> list[tuple[str, str, float]]:
        """Find duplicate candidates without merging (dry run).

        Returns:
            List of (entity_a_id, entity_b_id, similarity) above threshold.
        """
        backend = graph.backend
        entities = await backend.list_nodes(kind=NodeKind.ENTITY, limit=100_000)
        entities = [e for e in entities if "_phrase" not in (e.tags or [])]

        candidates: list[tuple[str, str, float]] = []
        comparisons = 0

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if comparisons >= self._max_comparisons:
                    break
                comparisons += 1

                a, b = entities[i], entities[j]
                title_sim = _title_similarity(a.title, b.title)

                embed_sim = 0.0
                if a.embedding and b.embedding:
                    embed_sim = _cosine_sim(a.embedding, b.embedding)

                combined = (
                    (1 - self._embedding_weight) * title_sim
                    + self._embedding_weight * embed_sim
                    if embed_sim > 0
                    else title_sim
                )

                if combined >= self._threshold:
                    candidates.append((a.id, b.id, combined))

            if comparisons >= self._max_comparisons:
                break

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:limit]
