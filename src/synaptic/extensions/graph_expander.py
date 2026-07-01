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
6. **Semantic entity relations**: from an OpenIE entity hub to the typed
   entity it depends on, supersedes, contradicts, etc.

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
from time import time
from typing import TYPE_CHECKING

from synaptic.extensions.graph_read_cache import GraphReadCache
from synaptic.models import EdgeKind, Node, NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.query_anchor import QueryAnchors
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("graph-expander")

_OPENIE_ENTITY_RELATION_KINDS = {
    EdgeKind.RELATED,
    EdgeKind.IS_A,
    EdgeKind.PART_OF,
    EdgeKind.DEPENDS_ON,
    EdgeKind.CAUSED,
    EdgeKind.PRODUCED,
    EdgeKind.CONTRADICTS,
    EdgeKind.SUPERSEDES,
}
_OPENIE_TYPED_RELATION_KINDS = _OPENIE_ENTITY_RELATION_KINDS - {EdgeKind.RELATED}
_ENTITY_MENTION_TAGS = frozenset({"_phrase", "_openie_entity", "_llm_enriched", "_spacy"})


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
        edge_kind: Relation kind of the edge used for the expansion.
        edge_confidence: Extractor confidence carried by provenance metadata
            when available. Defaults to 1.0 for deterministic structural edges.
    """

    node: Node
    reason: str
    hops: int = 0
    anchor_hit: str | None = None
    edge_kind: str = ""
    edge_confidence: float = 1.0


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
    max_total_expanded: int = 40
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
        query_terms: frozenset[str] | None = None,
        read_cache: GraphReadCache | None = None,
        timings_ms: dict[str, float] | None = None,
    ) -> list[ExpandedNode]:
        """Produce an expanded candidate list from anchors and seeds.

        Returns the seed nodes followed by newly-discovered neighbours.
        Order is deterministic within each group (seeds first, then
        category siblings, then document-scoped expansion, then
        chunk-next walk) so tests can assert on it without sorting.

        ``query_terms`` (opt-in) makes the budget relevance-aware: when the
        1-hop neighbourhood exceeds the budget, the most query-relevant
        neighbours win the slots instead of the first-visited ones. Helps the
        agent find evidence in large neighbourhoods where relevant neighbours
        would otherwise be dropped before the reranker. None = prior behaviour.
        """
        budget = budget or ExpansionBudget()
        reads = read_cache or GraphReadCache(self._backend)
        state = _ExpansionState(budget, q_terms=query_terms or frozenset())

        # Step 1 — seeds are always included first.
        stage_t0 = time()
        for node in seed_nodes:
            state.add(ExpandedNode(node=node, reason="seed", hops=0))
        _record_timing(timings_ms, "expand_graph_seed", stage_t0)

        stage_t0 = time()
        _record_timing(timings_ms, "expand_graph_seed_prefetch", stage_t0)

        # Step 2 — REFERENCES edges (explicit document cross-references,
        # e.g. a statute article citing another article). Runs first
        # among the expansion paths: a cited document is the highest-value
        # neighbour there is, so it must get budget priority over cheaper
        # category / chunk expansion. Turns "follow the citation"
        # multi-hop retrieval into a single structural hop.
        # No-op on corpora without REFERENCES edges.
        stage_t0 = time()
        if not state.is_full():
            await self._expand_references(seed_nodes, state, reads)
        _record_timing(timings_ms, "expand_graph_references", stage_t0)

        # Step 3 — walk category siblings. Categories are a cheap way
        # to surface cross-document context that lexical FTS misses.
        stage_t0 = time()
        if not state.is_full():
            await self._expand_category_siblings(anchors, state, reads)
        _record_timing(timings_ms, "expand_graph_category", stage_t0)

        # Step 4 — for every seed document, pull its chunks; for every
        # seed chunk, pull its parent document (and its sibling chunks).
        # This is the "stay inside the same document" expansion.
        stage_t0 = time()
        if not state.is_full():
            await self._expand_document_scope(seed_nodes, state, reads)
        _record_timing(timings_ms, "expand_graph_document", stage_t0)

        # Step 5 — chunk-next sequence walk. Cheap and often useful for
        # narrative documents where the relevant answer spans neighbours.
        stage_t0 = time()
        if not state.is_full():
            await self._expand_chunk_next(seed_nodes, state, reads)
        _record_timing(timings_ms, "expand_graph_chunk_next", stage_t0)

        # Step 6 — entity mentions. Only triggers if the corpus has
        # ENTITY hub nodes (post-processed by EntityLinker).
        stage_t0 = time()
        if not state.is_full():
            await self._expand_entity_mentions(seed_nodes, state, reads)
        _record_timing(timings_ms, "expand_graph_entity", stage_t0)

        # Step 7 — RELATED edges (FK relationships for structured data) and
        # opt-in OpenIE semantic relation edges.
        # For ENTITY nodes from TableIngester/DbIngester, RELATED edges
        # represent foreign-key relationships (e.g., product→sales,
        # product→reviews). For OpenIE entity hubs, typed relation edges
        # represent memory facts (e.g., concept→depends_on→policy). These are
        # valuable for relation-only discovery that lexical RAG cannot see.
        stage_t0 = time()
        if not state.is_full():
            await self._expand_related(seed_nodes, state, reads)
        _record_timing(timings_ms, "expand_graph_related", stage_t0)

        return state.results()

    # --- per-path helpers ---

    async def _expand_category_siblings(
        self,
        anchors: QueryAnchors,
        state: _ExpansionState,
        reads: GraphReadCache,
    ) -> None:
        """From category anchors, surface documents in the same category.

        Uses filtered PART_OF edges instead of the generic neighbour walk so a
        large category does not materialize every adjacent node/edge kind before
        the expansion budget trims it.
        """
        category_ids = list(dict.fromkeys(anchors.category_node_ids))
        if not category_ids:
            return

        try:
            edge_map = await reads.get_edges_many_by_kind_light(
                category_ids,
                direction="both",
                kinds=[EdgeKind.PART_OF],
            )
        except Exception as exc:
            logger.debug("category edge batch failed: %s", exc)
            return

        groups: list[tuple[str, list[str]]] = []
        node_ids: list[str] = []
        seen: set[str] = set()
        offer_limit = state.remaining_capacity()
        for cat_id in category_ids:
            if state.is_full() or (offer_limit is not None and len(node_ids) >= offer_limit):
                break

            candidate_ids: list[str] = []
            for edge in edge_map.get(cat_id, []):
                if (
                    state.is_full()
                    or len(candidate_ids) >= state.budget.category_sibling_limit
                    or (offer_limit is not None and len(node_ids) >= offer_limit)
                ):
                    break
                if edge.kind != EdgeKind.PART_OF:
                    continue
                other_id = edge.target_id if edge.source_id == cat_id else edge.source_id
                if other_id == cat_id or state.contains(other_id) or other_id in candidate_ids:
                    continue
                candidate_ids.append(other_id)
                if other_id not in seen:
                    node_ids.append(other_id)
                    seen.add(other_id)
            if candidate_ids:
                groups.append((cat_id, candidate_ids))

        nodes = await reads.get_nodes(node_ids)
        for cat_id, candidate_ids in groups:
            added = 0
            for node_id in candidate_ids:
                if state.is_full() or added >= state.budget.category_sibling_limit:
                    break
                if state.contains(node_id):
                    continue
                node = nodes.get(node_id)
                if node is None:
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
        reads: GraphReadCache,
    ) -> None:
        """Pull sibling chunks for seed chunks and child chunks for seed docs.

        The goal is to give the reranker the whole neighbourhood of a
        hit chunk — if q001 hits chunk 17 of Doc A, chunks 14-20 of
        Doc A are probably all relevant too. We fetch them via the
        shared CONTAINS / PART_OF edges.
        """
        document_scope_seeds = [node for node in seed_nodes if _is_document_scope_seed(node)]
        if not document_scope_seeds:
            return

        groups: list[tuple[str, list[str]]] = []
        node_ids: list[str] = []
        seen: set[str] = set()
        offer_limit = state.remaining_capacity()
        try:
            edge_map = await reads.get_edges_many_by_kind_light(
                [node.id for node in document_scope_seeds],
                direction="both",
                kinds=[EdgeKind.CONTAINS, EdgeKind.PART_OF],
            )
        except Exception as exc:
            logger.debug("document-scope edge batch failed: %s", exc)
            return
        for seed in document_scope_seeds:
            if state.is_full() or (offer_limit is not None and len(node_ids) >= offer_limit):
                break
            edges = edge_map.get(seed.id, [])

            candidate_ids: list[str] = []
            for edge in edges:
                if (
                    state.is_full()
                    or len(candidate_ids) >= state.budget.max_per_anchor
                    or (offer_limit is not None and len(node_ids) >= offer_limit)
                ):
                    break
                # Only follow the structural edges — skip RELATED / MENTIONS
                if edge.kind not in (EdgeKind.CONTAINS, EdgeKind.PART_OF):
                    continue
                other_id = edge.target_id if edge.source_id == seed.id else edge.source_id
                if state.contains(other_id) or other_id in candidate_ids:
                    continue
                candidate_ids.append(other_id)
                if other_id not in seen:
                    node_ids.append(other_id)
                    seen.add(other_id)
            if candidate_ids:
                groups.append((seed.id, candidate_ids))

        nodes = await reads.get_nodes(node_ids)
        for seed_id, candidate_ids in groups:
            added = 0
            for other_id in candidate_ids:
                if state.is_full() or added >= state.budget.max_per_anchor:
                    break
                if state.contains(other_id):
                    continue
                other = nodes.get(other_id)
                if other is None:
                    continue
                state.add(
                    ExpandedNode(
                        node=other,
                        reason="document_chunk",
                        hops=1,
                        anchor_hit=seed_id,
                    )
                )
                added += 1

    async def _expand_chunk_next(
        self,
        seed_nodes: list[Node],
        state: _ExpansionState,
        reads: GraphReadCache,
    ) -> None:
        """Walk NEXT_CHUNK edges forward and backward from seed chunks."""
        chunks = [n for n in seed_nodes if n.kind == NodeKind.CHUNK]
        if not chunks:
            return

        groups: list[tuple[str, list[str]]] = []
        node_ids: list[str] = []
        seen: set[str] = set()
        offer_limit = state.remaining_capacity()
        try:
            edge_map = await reads.get_edges_many_by_kind_light(
                [node.id for node in chunks],
                direction="both",
                kinds=[EdgeKind.NEXT_CHUNK],
            )
        except Exception as exc:
            logger.debug("chunk-next edge batch failed: %s", exc)
            return
        for seed in chunks:
            if state.is_full() or (offer_limit is not None and len(node_ids) >= offer_limit):
                break
            edges = edge_map.get(seed.id, [])

            candidate_ids: list[str] = []
            for edge in edges:
                if state.is_full() or (offer_limit is not None and len(node_ids) >= offer_limit):
                    break
                if edge.kind != EdgeKind.NEXT_CHUNK:
                    continue
                other_id = edge.target_id if edge.source_id == seed.id else edge.source_id
                if state.contains(other_id) or other_id in candidate_ids:
                    continue
                candidate_ids.append(other_id)
                if other_id not in seen:
                    node_ids.append(other_id)
                    seen.add(other_id)
            if candidate_ids:
                groups.append((seed.id, candidate_ids))

        nodes = await reads.get_nodes(node_ids)
        for seed_id, candidate_ids in groups:
            for other_id in candidate_ids:
                if state.is_full():
                    break
                if state.contains(other_id):
                    continue
                other = nodes.get(other_id)
                if other is None:
                    continue
                state.add(
                    ExpandedNode(
                        node=other,
                        reason="chunk_next",
                        hops=1,
                        anchor_hit=seed_id,
                    )
                )

    async def _expand_entity_mentions(
        self,
        seed_nodes: list[Node],
        state: _ExpansionState,
        reads: GraphReadCache,
    ) -> None:
        """If a seed is an ENTITY hub, add its MENTIONS sources.

        No-ops on corpora without EntityLinker post-processing, which
        is the common case right now. Kept in place so Phase G ontology
        work lights up for free.
        """
        entities = [n for n in seed_nodes if _is_entity_mention_seed(n)]
        if not entities:
            return

        groups: list[tuple[str, list[str]]] = []
        node_ids: list[str] = []
        seen: set[str] = set()
        offer_limit = state.remaining_capacity()
        try:
            edge_map = await reads.get_edges_many_by_kind_light(
                [node.id for node in entities],
                direction="incoming",
                kinds=[EdgeKind.MENTIONS],
            )
        except Exception as exc:
            logger.debug("entity edge batch failed: %s", exc)
            return
        for seed in entities:
            if state.is_full() or (offer_limit is not None and len(node_ids) >= offer_limit):
                break
            edges = edge_map.get(seed.id, [])

            candidate_ids: list[str] = []
            for edge in edges:
                if (
                    state.is_full()
                    or len(candidate_ids) >= state.budget.max_per_anchor
                    or (offer_limit is not None and len(node_ids) >= offer_limit)
                ):
                    break
                if edge.kind != EdgeKind.MENTIONS:
                    continue
                src_id = edge.source_id
                if state.contains(src_id) or src_id in candidate_ids:
                    continue
                candidate_ids.append(src_id)
                if src_id not in seen:
                    node_ids.append(src_id)
                    seen.add(src_id)
            if candidate_ids:
                groups.append((seed.id, candidate_ids))

        nodes = await reads.get_nodes(node_ids)
        for seed_id, candidate_ids in groups:
            added = 0
            for src_id in candidate_ids:
                if state.is_full() or added >= state.budget.max_per_anchor:
                    break
                if state.contains(src_id):
                    continue
                src = nodes.get(src_id)
                if src is None:
                    continue
                state.add(
                    ExpandedNode(
                        node=src,
                        reason="entity_mention",
                        hops=1,
                        anchor_hit=seed_id,
                    )
                )
                added += 1

    async def _expand_related(
        self,
        seed_nodes: list[Node],
        state: _ExpansionState,
        reads: GraphReadCache,
    ) -> None:
        """Walk relation edges from seed ENTITY nodes.

        For structured data ingested via TableIngester/DbIngester, RELATED
        edges connect FK-linked rows (e.g., product → sales, product →
        reviews). For OpenIE, typed relation edges connect extracted entity
        hubs. This step surfaces neighbours that lexical search alone cannot
        find.

        Only expands from ENTITY nodes to keep document graphs unaffected.
        Capped at ``max_per_anchor`` per seed to prevent fan-out explosion
        on heavily-linked rows.
        """
        entities = [n for n in seed_nodes if n.kind == NodeKind.ENTITY]
        if not entities:
            return

        groups: list[tuple[str, list[tuple[str, EdgeKind, float]]]] = []
        node_ids: list[str] = []
        seen: set[str] = set()
        offer_limit = state.remaining_capacity()
        entity_ids = [node.id for node in entities]
        try:
            edge_map = await reads.get_edges_many_by_kind_selective_light(
                entity_ids,
                direction="both",
                light_kinds=[EdgeKind.RELATED],
                full_kinds=sorted(_OPENIE_TYPED_RELATION_KINDS, key=lambda kind: kind.value),
            )
        except Exception as exc:
            logger.debug("related edge batch failed for entity seeds: %s", exc)
            return
        for seed in entities:
            if state.is_full() or (offer_limit is not None and len(node_ids) >= offer_limit):
                break
            edges = edge_map.get(seed.id, [])

            candidates: list[tuple[str, EdgeKind, float]] = []
            seed_seen: set[str] = set()
            for edge in edges:
                if (
                    state.is_full()
                    or len(candidates) >= state.budget.max_per_anchor
                    or (offer_limit is not None and len(node_ids) >= offer_limit)
                ):
                    break
                is_related = edge.kind == EdgeKind.RELATED
                is_openie_relation = (
                    edge.kind in _OPENIE_ENTITY_RELATION_KINDS
                    and (edge.properties or {}).get("is_openie") == "true"
                )
                if not (is_related or is_openie_relation):
                    continue
                other_id = edge.target_id if edge.source_id == seed.id else edge.source_id
                if state.contains(other_id) or other_id in seed_seen:
                    continue
                candidates.append((other_id, edge.kind, _edge_confidence(edge)))
                seed_seen.add(other_id)
                if other_id not in seen:
                    node_ids.append(other_id)
                    seen.add(other_id)
            if candidates:
                groups.append((seed.id, candidates))

        nodes = await reads.get_nodes(node_ids)
        for seed_id, candidates in groups:
            added = 0
            for other_id, edge_kind, edge_confidence in candidates:
                if state.is_full() or added >= state.budget.max_per_anchor:
                    break
                if state.contains(other_id):
                    continue
                other = nodes.get(other_id)
                if other is None:
                    continue
                state.add(
                    ExpandedNode(
                        node=other,
                        reason="related" if edge_kind == EdgeKind.RELATED else "semantic_relation",
                        hops=1,
                        anchor_hit=seed_id,
                        edge_kind=str(edge_kind.value),
                        edge_confidence=edge_confidence,
                    )
                )
                added += 1

    async def _expand_references(
        self,
        seed_nodes: list[Node],
        state: _ExpansionState,
        reads: GraphReadCache,
    ) -> None:
        """Walk REFERENCES edges from seed nodes (explicit cross-references).

        REFERENCES edges connect a document to another document it
        explicitly cites (e.g. a statute article → the article it
        invokes). Surfacing the cited document alongside the citing one
        makes "follow the citation" multi-hop retrieval a single hop —
        single-shot search alone cannot reach the cited document when the
        query shares no vocabulary with it.

        No-op on corpora without REFERENCES edges. Capped per seed.
        """
        has_references = await _has_edges_of_kind(self._backend, EdgeKind.REFERENCES)
        if has_references is False:
            return

        groups: list[tuple[str, list[str]]] = []
        node_ids: list[str] = []
        seen: set[str] = set()
        offer_limit = state.remaining_capacity()
        try:
            edge_map = await reads.get_edges_many_by_kind_light(
                [node.id for node in seed_nodes],
                direction="both",
                kinds=[EdgeKind.REFERENCES],
            )
        except Exception as exc:
            logger.debug("reference edge batch failed: %s", exc)
            return
        for seed in seed_nodes:
            if state.is_full() or (offer_limit is not None and len(node_ids) >= offer_limit):
                break
            edges = edge_map.get(seed.id, [])

            candidate_ids: list[str] = []
            for edge in edges:
                if (
                    state.is_full()
                    or len(candidate_ids) >= state.budget.max_per_anchor
                    or (offer_limit is not None and len(node_ids) >= offer_limit)
                ):
                    break
                if edge.kind != EdgeKind.REFERENCES:
                    continue
                other_id = edge.target_id if edge.source_id == seed.id else edge.source_id
                if state.contains(other_id) or other_id in candidate_ids:
                    continue
                candidate_ids.append(other_id)
                if other_id not in seen:
                    node_ids.append(other_id)
                    seen.add(other_id)
            if candidate_ids:
                groups.append((seed.id, candidate_ids))

        nodes = await reads.get_nodes(node_ids)
        for seed_id, candidate_ids in groups:
            added = 0
            for other_id in candidate_ids:
                if state.is_full() or added >= state.budget.max_per_anchor:
                    break
                if state.contains(other_id):
                    continue
                other = nodes.get(other_id)
                if other is None:
                    continue
                state.add(
                    ExpandedNode(
                        node=other,
                        reason="references",
                        hops=1,
                        anchor_hit=seed_id,
                    )
                )
                added += 1


def _relevance(node: Node, q_terms: frozenset[str]) -> int:
    """LLM-free relevance of a node to the query: query-term hits in the title
    (weighted 2x) plus content. Used to decide which neighbour keeps a budget
    slot when the 1-hop neighbourhood is larger than the budget."""
    if not q_terms:
        return 0
    title = (node.title or "").lower()
    content = (node.content or "").lower()
    return sum((2 if t in title else 0) + (1 if t in content else 0) for t in q_terms)


def _edge_confidence(edge: object) -> float:
    props = getattr(edge, "properties", {}) or {}
    raw = props.get("confidence", getattr(edge, "weight", 1.0))
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 1.0


async def _has_edges_of_kind(backend: object, kind: EdgeKind) -> bool | None:
    has_kind = getattr(backend, "has_edges_of_kind", None)
    if not callable(has_kind):
        return None
    try:
        return bool(await has_kind(kind))
    except Exception as exc:
        logger.debug("edge-kind existence check failed for %s: %s", kind, exc)
        return None


def _is_document_scope_seed(node: Node) -> bool:
    tags = node.tags or []
    return "_openie_entity" not in tags


def _is_entity_mention_seed(node: Node) -> bool:
    if node.kind != NodeKind.ENTITY:
        return False
    tags = set(node.tags or [])
    return bool(tags & _ENTITY_MENTION_TAGS)


def _record_timing(timings_ms: dict[str, float] | None, key: str, started_at: float) -> None:
    if timings_ms is None:
        return
    timings_ms[key] = timings_ms.get(key, 0.0) + (time() - started_at) * 1000


@dataclass(slots=True)
class _ExpansionState:
    """Per-call bookkeeping for an expansion run.

    Keeps a dict from node id to ``ExpandedNode`` so duplicates are
    detected in O(1) and the *first* reason a node was added wins —
    seeds beat category siblings, category siblings beat document
    siblings, etc., matching the order the expander visits paths.

    When ``q_terms`` is non-empty the budget becomes RELEVANCE-AWARE: once
    full, a new neighbour evicts the least-query-relevant *non-seed* already
    held if the newcomer is more relevant. This stops a large structural
    neighbourhood (e.g. a 90k-node graph) from spending the whole budget on
    the first-visited paths and dropping relevant neighbours before the
    reranker ever sees them. Seeds (hops==0) are never evicted. Empty
    ``q_terms`` (default) preserves the prior first-come behaviour exactly.
    """

    budget: ExpansionBudget
    q_terms: frozenset[str] = frozenset()
    _by_id: dict[str, ExpandedNode] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _rel: dict[str, int] = field(default_factory=dict)

    def add(self, expanded: ExpandedNode) -> None:
        nid = expanded.node.id
        if nid in self._by_id:
            return
        if len(self._by_id) < self.budget.max_total_expanded:
            self._by_id[nid] = expanded
            self._order.append(nid)
            if self.q_terms:
                self._rel[nid] = _relevance(expanded.node, self.q_terms)
            return
        # Full. First-come mode (no query terms) → drop the newcomer.
        if not self.q_terms:
            return
        # Relevance-aware mode → evict the least-relevant non-seed if the
        # newcomer beats it (strictly, so ties keep the earlier-visited node).
        new_rel = _relevance(expanded.node, self.q_terms)
        victim, victim_rel = None, new_rel
        for oid in self._order:
            if self._by_id[oid].hops == 0:  # never evict a seed
                continue
            if self._rel.get(oid, 0) < victim_rel:
                victim, victim_rel = oid, self._rel.get(oid, 0)
        if victim is None:
            return
        del self._by_id[victim]
        self._order.remove(victim)
        self._rel.pop(victim, None)
        self._by_id[nid] = expanded
        self._order.append(nid)
        self._rel[nid] = new_rel

    def contains(self, node_id: str) -> bool:
        return node_id in self._by_id

    def is_full(self) -> bool:
        # In relevance mode never short-circuit the per-path walks — let every
        # path OFFER its neighbours so add() can evict-by-relevance. Per-path
        # `added` caps still bound how many each path iterates. First-come mode
        # keeps the original early-stop behaviour.
        if self.q_terms:
            return False
        return len(self._by_id) >= self.budget.max_total_expanded

    def remaining_capacity(self) -> int | None:
        """Return first-come free slots, or None when relevance mode may evict."""
        if self.q_terms:
            return None
        return max(0, self.budget.max_total_expanded - len(self._by_id))

    def results(self) -> list[ExpandedNode]:
        return [self._by_id[nid] for nid in self._order]
