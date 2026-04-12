"""GraphExpander — shallow 1-hop expansion from query anchors.

The 3rd-generation retrieval pattern is:

    query → anchor → shallow expansion → rerank → evidence

This module owns the expansion step. It takes the ``QueryAnchors``
produced by ``QueryAnchorExtractor`` plus an initial candidate set
(typically the top FTS hits) and walks the graph **one hop** to pull
in neighbours that share a category, a document parent, or an entity
link.

Why only one hop:

- 1st-gen GraphRAG went multi-hop with LLM summarisation. Cost exploded.
- 2nd-gen (LazyGraphRAG, LightRAG) capped hops to save money.
- 3rd-gen (LinearRAG, Practical GraphRAG) showed that **shallow
  expansion already captures most of the recall**, with deeper hops
  only adding noise. "One-hop is enough" is now the prevailing wisdom.

Expansion paths the expander considers:

1. **CONTAINS**: from a Document anchor → its Chunks.
2. **PART_OF**: from a Document → its Category; from a Chunk → its Document.
3. **NEXT_CHUNK**: chunk-sequence neighbours (both directions).
4. **Category siblings**: from a Category node → all Documents that
   ``PART_OF`` that category. This is the key path for cross-document
   queries — a query matching ``"규정 및 지침"`` can surface sibling
   rule documents even if they don't lexically overlap with the query.
5. **MENTIONS** (optional): entity → sources that mention it. Only
   triggered when the corpus has ``NodeKind.ENTITY`` hubs (built by
   ``EntityLinker`` post-processing).

Budget discipline — expansion is **capped** at every step so a popular
category with 10,000 documents can't poison the candidate set. The
caller controls the budget via ``max_per_anchor`` and
``max_total_expanded``.

Example::

    from synaptic.extensions.graph_expander import GraphExpander
    from synaptic.extensions.query_anchor import QueryAnchorExtractor

    anchors = await anchor_extractor.extract("경마 운영계획 인권경영")
    expander = GraphExpander(backend=backend)

    seed_nodes = [hit.node for hit in fts_hits]
    expanded = await expander.expand(
        anchors=anchors,
        seed_nodes=seed_nodes,
        max_per_anchor=20,
        max_total_expanded=60,
    )
    # expanded is a list[ExpandedNode] — seeds first, then the new hops,
    # tagged with the expansion path so the reranker can weight them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from synaptic.models import EdgeKind, Node, NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.query_anchor import QueryAnchors
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("graph-expander")


@dataclass(slots=True)
class ExpandedNode:
    """A graph node plus the reason it was included.

    The ``reason`` tag lets the downstream reranker apply different
    weights to different expansion paths (a category-sibling chunk is
    more noisy than a same-document sibling chunk, for instance).

    Attributes:
        node: The actual ``Node`` object.
        reason: Short tag describing why the expander included this
            node. One of ``"seed"``, ``"category_sibling"``,
            ``"document_chunk"``, ``"chunk_next"``, ``"entity_mention"``.
        hops: Minimum number of edges from the nearest seed. ``0`` for
            seeds themselves, ``1`` for direct neighbours, etc.
        anchor_hit: Which anchor ID pulled this node in. Useful for
            diagnostics and for score fusion that cares about the
            strength of the anchor (category > entity > keyword).
    """

    node: Node
    reason: str
    hops: int = 0
    anchor_hit: str | None = None


@dataclass(slots=True)
class ExpansionBudget:
    """Caps on expansion fan-out.

    Every limit defaults to a conservative value tuned for KRRA-sized
    corpora (~20K nodes). Increase for larger graphs or aggressive
    recall settings; decrease when latency matters more than coverage.

    Attributes:
        max_per_anchor: Maximum neighbours one anchor may contribute.
            Prevents a single popular category from flooding the set.
        max_total_expanded: Absolute cap on the final expanded list
            (seeds + hops combined). Hard upper bound on reranker cost.
        max_hops: How many graph layers to walk. ``1`` is the default
            and matches the 3rd-gen "shallow" doctrine. Setting this
            higher turns the expander into a small PPR step.
        category_sibling_limit: Max documents pulled per category
            sibling expansion — categories can be huge, so this is
            usually tighter than ``max_per_anchor``.
    """

    max_per_anchor: int = 20
    max_total_expanded: int = 100
    max_hops: int = 1
    category_sibling_limit: int = 10


class GraphExpander:
    """Walk the graph one layer out from the query anchors and seeds.

    The expander is intentionally stateless — all state lives on
    the per-call ``_ExpansionState`` helper so the same expander can
    serve concurrent queries. The only cached data is what the
    backend caches (chunk node lookups); the expander re-issues its
    own queries each call.

    Args:
        backend: Storage backend providing ``get_neighbors``,
            ``get_edges``, and ``list_nodes``. Any backend implementing
            ``StorageBackend`` works — Memory, SQLite, Kuzu alike.
    """

    __slots__ = ("_backend",)

    def __init__(self, *, backend: StorageBackend) -> None:
        self._backend = backend

    async def expand(
        self,
        *,
        anchors: QueryAnchors,
        seed_nodes: list[Node],
        budget: ExpansionBudget | None = None,
    ) -> list[ExpandedNode]:
        """Produce an expanded candidate list from anchors and seeds.

        Returns the seed nodes followed by newly-discovered neighbours.
        Order is deterministic within each group (seeds first, then
        category siblings, then document-scoped expansion, then
        chunk-next walk) so tests can assert on it without sorting.
        """
        budget = budget or ExpansionBudget()
        state = _ExpansionState(budget)

        # Step 1 — seeds are always included first.
        for node in seed_nodes:
            state.add(ExpandedNode(node=node, reason="seed", hops=0))

        # Step 2 — walk category siblings. Categories are a cheap way
        # to surface cross-document context that lexical FTS misses.
        await self._expand_category_siblings(anchors, state)

        # Step 3 — for every seed document, pull its chunks; for every
        # seed chunk, pull its parent document (and its sibling chunks).
        # This is the "stay inside the same document" expansion.
        await self._expand_document_scope(seed_nodes, state)

        # Step 4 — chunk-next sequence walk. Cheap and often useful for
        # narrative documents where the relevant answer spans neighbours.
        await self._expand_chunk_next(seed_nodes, state)

        # Step 5 — entity mentions. Only triggers if the corpus has
        # ENTITY hub nodes (post-processed by EntityLinker).
        await self._expand_entity_mentions(seed_nodes, state)

        # Step 6 — RELATED edges (FK relationships for structured data).
        # For ENTITY nodes from TableIngester/DbIngester, RELATED edges
        # represent foreign-key relationships (e.g., product→sales,
        # product→reviews). These are valuable for cross-table discovery.
        await self._expand_related(seed_nodes, state)

        return state.results()

    # --- per-path helpers ---

    async def _expand_category_siblings(
        self,
        anchors: QueryAnchors,
        state: _ExpansionState,
    ) -> None:
        """From category anchors, surface documents in the same category.

        Uses ``get_neighbors`` with ``depth=1`` so we go category → doc
        in a single hop. The backend's neighbour call returns the edge
        too, but we only care about the node here.
        """
        for cat_id in anchors.category_node_ids:
            if state.is_full():
                return
            try:
                hops = await self._backend.get_neighbors(cat_id, depth=1)
            except Exception as exc:
                logger.debug("category expansion failed for %s: %s", cat_id, exc)
                continue

            added = 0
            for node, _edge in hops:
                if state.is_full() or added >= state.budget.category_sibling_limit:
                    break
                if node.id == cat_id:
                    continue
                state.add(
                    ExpandedNode(
                        node=node,
                        reason="category_sibling",
                        hops=1,
                        anchor_hit=cat_id,
                    )
                )
                added += 1

    async def _expand_document_scope(
        self,
        seed_nodes: list[Node],
        state: _ExpansionState,
    ) -> None:
        """Pull sibling chunks for seed chunks and child chunks for seed docs.

        The goal is to give the reranker the whole neighbourhood of a
        hit chunk — if q001 hits chunk 17 of Doc A, chunks 14-20 of
        Doc A are probably all relevant too. We fetch them via the
        shared CONTAINS / PART_OF edges.
        """
        for seed in seed_nodes:
            if state.is_full():
                return
            try:
                edges = await self._backend.get_edges(seed.id, direction="both")
            except Exception as exc:
                logger.debug("edge fetch failed for %s: %s", seed.id, exc)
                continue

            added = 0
            for edge in edges:
                if state.is_full() or added >= state.budget.max_per_anchor:
                    break
                # Only follow the structural edges — skip RELATED / MENTIONS
                if edge.kind not in (EdgeKind.CONTAINS, EdgeKind.PART_OF):
                    continue
                other_id = edge.target_id if edge.source_id == seed.id else edge.source_id
                if state.contains(other_id):
                    continue
                other = await self._backend.get_node(other_id)
                if other is None:
                    continue
                state.add(
                    ExpandedNode(
                        node=other,
                        reason="document_chunk",
                        hops=1,
                        anchor_hit=seed.id,
                    )
                )
                added += 1

    async def _expand_chunk_next(
        self,
        seed_nodes: list[Node],
        state: _ExpansionState,
    ) -> None:
        """Walk NEXT_CHUNK edges forward and backward from seed chunks."""
        chunks = [n for n in seed_nodes if n.kind == NodeKind.CHUNK]
        if not chunks:
            return

        for seed in chunks:
            if state.is_full():
                return
            try:
                edges = await self._backend.get_edges(seed.id, direction="both")
            except Exception as exc:
                logger.debug("chunk-next fetch failed for %s: %s", seed.id, exc)
                continue

            for edge in edges:
                if state.is_full():
                    break
                if edge.kind != EdgeKind.NEXT_CHUNK:
                    continue
                other_id = edge.target_id if edge.source_id == seed.id else edge.source_id
                if state.contains(other_id):
                    continue
                other = await self._backend.get_node(other_id)
                if other is None:
                    continue
                state.add(
                    ExpandedNode(
                        node=other,
                        reason="chunk_next",
                        hops=1,
                        anchor_hit=seed.id,
                    )
                )

    async def _expand_entity_mentions(
        self,
        seed_nodes: list[Node],
        state: _ExpansionState,
    ) -> None:
        """If a seed is an ENTITY hub, add its MENTIONS sources.

        No-ops on corpora without EntityLinker post-processing, which
        is the common case right now. Kept in place so Phase G ontology
        work lights up for free.
        """
        entities = [n for n in seed_nodes if n.kind == NodeKind.ENTITY]
        if not entities:
            return

        for seed in entities:
            if state.is_full():
                return
            try:
                edges = await self._backend.get_edges(seed.id, direction="incoming")
            except Exception as exc:
                logger.debug("entity expansion failed for %s: %s", seed.id, exc)
                continue

            added = 0
            for edge in edges:
                if state.is_full() or added >= state.budget.max_per_anchor:
                    break
                if edge.kind != EdgeKind.MENTIONS:
                    continue
                src_id = edge.source_id
                if state.contains(src_id):
                    continue
                src = await self._backend.get_node(src_id)
                if src is None:
                    continue
                state.add(
                    ExpandedNode(
                        node=src,
                        reason="entity_mention",
                        hops=1,
                        anchor_hit=seed.id,
                    )
                )
                added += 1

    async def _expand_related(
        self,
        seed_nodes: list[Node],
        state: _ExpansionState,
    ) -> None:
        """Walk RELATED edges from seed ENTITY nodes (structured data FK).

        For structured data ingested via TableIngester/DbIngester, RELATED
        edges connect FK-linked rows (e.g., product → sales, product →
        reviews). This step surfaces cross-table neighbours that lexical
        search alone cannot find.

        Only expands from ENTITY nodes to keep document graphs unaffected.
        Capped at ``max_per_anchor`` per seed to prevent fan-out explosion
        on heavily-linked rows.
        """
        entities = [n for n in seed_nodes if n.kind == NodeKind.ENTITY]
        if not entities:
            return

        for seed in entities:
            if state.is_full():
                return
            try:
                edges = await self._backend.get_edges(seed.id, direction="both")
            except Exception as exc:
                logger.debug("related expansion failed for %s: %s", seed.id, exc)
                continue

            added = 0
            for edge in edges:
                if state.is_full() or added >= state.budget.max_per_anchor:
                    break
                if edge.kind != EdgeKind.RELATED:
                    continue
                other_id = edge.target_id if edge.source_id == seed.id else edge.source_id
                if state.contains(other_id):
                    continue
                other = await self._backend.get_node(other_id)
                if other is None:
                    continue
                state.add(
                    ExpandedNode(
                        node=other,
                        reason="related",
                        hops=1,
                        anchor_hit=seed.id,
                    )
                )
                added += 1


@dataclass(slots=True)
class _ExpansionState:
    """Per-call bookkeeping for an expansion run.

    Keeps a dict from node id to ``ExpandedNode`` so duplicates are
    detected in O(1) and the *first* reason a node was added wins —
    seeds beat category siblings, category siblings beat document
    siblings, etc., matching the order the expander visits paths.
    """

    budget: ExpansionBudget
    _by_id: dict[str, ExpandedNode] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def add(self, expanded: ExpandedNode) -> None:
        nid = expanded.node.id
        if nid in self._by_id:
            return
        if len(self._by_id) >= self.budget.max_total_expanded:
            return
        self._by_id[nid] = expanded
        self._order.append(nid)

    def contains(self, node_id: str) -> bool:
        return node_id in self._by_id

    def is_full(self) -> bool:
        return len(self._by_id) >= self.budget.max_total_expanded

    def results(self) -> list[ExpandedNode]:
        return [self._by_id[nid] for nid in self._order]
