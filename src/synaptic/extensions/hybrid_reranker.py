"""HybridReranker — 4-signal fusion on top of GraphExpander output.

After ``GraphExpander`` has produced a candidate pool, the job of this
module is to decide which candidates actually go into the evidence set.
The 3rd-generation retrieval playbook combines four signals:

1. **Lexical**: how well does the node's text match the query at the
   BM25 / FTS level? This is what the FTS stage already produces and
   it's the signal we trust most for surface-level matching.
2. **Semantic**: how close is the node's embedding to the query
   embedding? Only runs if both sides have vectors — the reranker
   degrades gracefully to lexical-only when no embedder is wired in.
3. **Graph importance**: where does the node sit in the expansion
   graph? Category-anchored siblings rank lower than documents pulled
   via the same-document path, which rank lower than the seeds.
4. **Structural**: does the node's category / kind match what the
   anchor extractor found in the query? Gives a small boost to
   category-aligned nodes.

Each signal is normalised to ``[0, 1]`` and the final score is a
weighted sum. Defaults are tuned for a Korean corpus like KRRA where
lexical matching is dominant; the caller can override weights to
shift the emphasis toward semantic or graph signals for different
domains.

Example::

    from synaptic.extensions.hybrid_reranker import HybridReranker
    reranker = HybridReranker()

    scored = await reranker.rerank(
        expanded=expanded_nodes,       # from GraphExpander
        fts_scores={...},              # node_id → bm25-ish score
        query_embedding=q_vec,          # optional
        anchor_categories=set(...),     # from QueryAnchors
    )
    # scored is a list[ScoredCandidate] sorted by .total descending

The reranker is pure — no IO, no backend calls — which means it's
trivial to test and cheap to re-run with different weights during
experimentation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from synaptic.extensions.node_metadata import authority_of, is_current
from synaptic.models import Node, NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.graph_expander import ExpandedNode

logger = logging.getLogger("hybrid-reranker")


# --- Expansion-reason priors ---
#
# When a node comes from the expander, the reason it was pulled in is
# itself a signal: a seed is essentially a direct FTS hit, while a
# chunk reached via NEXT_CHUNK is two steps away from the query. We
# encode that as a prior in the graph-importance axis.

_REASON_PRIOR: dict[str, float] = {
    "seed": 1.00,
    "document_chunk": 0.70,
    "chunk_next": 0.55,
    "entity_mention": 0.50,
    "related": 0.50,
    "ppr_discovery": 0.45,
    "category_sibling": 0.40,
}


@dataclass(slots=True)
class ScoredCandidate:
    """A candidate node plus the per-signal scores that produced its total.

    Keeping the components separate makes reranker tuning observable —
    you can dump the list, see exactly which signal dominated, and
    adjust weights without guessing.

    Attributes:
        node: The original ``Node``.
        total: Weighted sum of the four signals. Higher is better.
        lexical: Normalised FTS / BM25 score.
        semantic: Normalised cosine similarity with the query vector.
            ``0.0`` when no embedding was supplied.
        graph: Normalised graph-importance prior from expansion reason.
        structural: Category / kind alignment bonus.
        reason: Expansion reason (kept for diagnostics and UI).
    """

    node: Node
    total: float
    lexical: float
    semantic: float
    graph: float
    structural: float
    reason: str


@dataclass(slots=True)
class RerankerWeights:
    """Weights applied to each signal before summing.

    The defaults sum to exactly 1.0 so ``total`` stays in ``[0, 1]``
    and is directly comparable across corpora. Weights do not have to
    sum to one — the reranker doesn't normalise — but keeping them
    there makes tuning interpretable.

    Attributes:
        lexical: BM25 / FTS weight. Default ``0.45`` — dominant.
        semantic: Query-embedding weight. Default ``0.25``. When no
            embedder is wired up this contribution is always zero.
        graph: Expansion-reason prior weight. Default ``0.20``.
        structural: Category / kind alignment weight. Default ``0.10``.
    """

    lexical: float = 0.45
    semantic: float = 0.25
    graph: float = 0.20
    structural: float = 0.10


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity, safe on zero / mismatched vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalise a score dict into ``[0, 1]``.

    When every entry has the same value (common early in development
    with tiny corpora) we flatten everything to ``1.0`` — the signal
    offers no differentiating information but shouldn't drag the total
    down, either.
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


class HybridReranker:
    """Combine lexical, semantic, graph, and structural signals.

    Args:
        weights: Override the default :class:`RerankerWeights`. Pass a
            preset when you want to emphasise one signal — e.g. on a
            multilingual corpus you might raise ``semantic`` and drop
            ``lexical`` because BM25 loses discriminative power.
    """

    __slots__ = ("_weights",)

    def __init__(self, *, weights: RerankerWeights | None = None) -> None:
        self._weights = weights or RerankerWeights()

    def rerank(
        self,
        *,
        expanded: list[ExpandedNode],
        fts_scores: dict[str, float] | None = None,
        query_embedding: list[float] | None = None,
        anchor_categories: set[str] | None = None,
        anchor_kinds: set[NodeKind] | None = None,
    ) -> list[ScoredCandidate]:
        """Score and sort ``expanded``.

        Args:
            expanded: Output of :class:`GraphExpander.expand`.
            fts_scores: Optional per-node lexical score (e.g. raw BM25
                or rank-based score). Missing nodes get ``0`` before
                normalisation. If the dict is ``None`` every node
                gets a flat lexical of ``1.0``.
            query_embedding: Query vector for semantic similarity.
                When omitted the semantic component is always ``0``.
            anchor_categories: Category labels the anchor extractor
                pulled from the query. Nodes whose ``properties`` or
                ``tags`` mention any of these categories get the
                structural bonus.
            anchor_kinds: Explicit ``NodeKind`` preferences. Lets the
                caller narrow results to, say, ``{ENTITY, RULE}`` when
                the query is clearly a compliance lookup.

        Returns:
            A list of :class:`ScoredCandidate` sorted by ``total``
            descending. Length equals ``len(expanded)`` — the reranker
            does not drop candidates; that's the aggregator's job.
        """
        if not expanded:
            return []

        # --- Lexical component ---
        if fts_scores is None:
            lex_raw = {ex.node.id: 1.0 for ex in expanded}
        else:
            lex_raw = {ex.node.id: fts_scores.get(ex.node.id, 0.0) for ex in expanded}
        lex_norm = _normalise(lex_raw)

        # --- Semantic component ---
        sem_raw: dict[str, float] = {}
        if query_embedding:
            for ex in expanded:
                emb = ex.node.embedding
                sem_raw[ex.node.id] = _cosine(query_embedding, emb) if emb else 0.0
            sem_norm = _normalise(sem_raw)
        else:
            sem_norm = {ex.node.id: 0.0 for ex in expanded}

        # --- Graph component ---
        graph_raw = {
            ex.node.id: _REASON_PRIOR.get(ex.reason, 0.3) for ex in expanded
        }
        graph_norm = _normalise(graph_raw)

        # --- Structural component ---
        category_set = {c.lower() for c in (anchor_categories or set())}
        kind_set = anchor_kinds or set()

        def _structural_score(node: Node) -> float:
            score = 0.0
            if category_set:
                props = node.properties or {}
                cat_field = (props.get("category") or "").lower()
                if cat_field and any(c in cat_field for c in category_set):
                    score += 0.4
                if node.tags and any(
                    c in t.lower() for t in node.tags for c in category_set
                ):
                    score += 0.1
            if kind_set and node.kind in kind_set:
                score += 0.2
            # Authority: higher-authority nodes (RULE > DECISION > OBSERVATION)
            # get a structural boost. Normalized to 0-0.2 range (authority 0-10).
            score += authority_of(node) * 0.02
            # Temporal: current documents get a small boost over expired ones
            if is_current(node):
                score += 0.1
            return min(score, 1.0)

        # --- Combine ---
        w = self._weights
        scored: list[ScoredCandidate] = []
        for ex in expanded:
            nid = ex.node.id
            lex = lex_norm.get(nid, 0.0)
            sem = sem_norm.get(nid, 0.0)
            graph = graph_norm.get(nid, 0.0)
            struct = _structural_score(ex.node)

            total = (
                w.lexical * lex
                + w.semantic * sem
                + w.graph * graph
                + w.structural * struct
            )
            scored.append(
                ScoredCandidate(
                    node=ex.node,
                    total=total,
                    lexical=lex,
                    semantic=sem,
                    graph=graph,
                    structural=struct,
                    reason=ex.reason,
                )
            )

        # --- Document-level MaxP + coverage bonus ---
        # When multiple chunks from the same document score well, that
        # document is likely more relevant than one with a single high
        # chunk. We boost each chunk's score by the document's coverage
        # signal: max(sibling scores) + α * log(sibling count + 1).
        # This is the MaxP aggregation pattern from ColBERT / HippoRAG2.
        doc_scores: dict[str, list[float]] = {}
        doc_map: dict[str, str] = {}  # node_id → doc_id
        for s in scored:
            doc_id = (s.node.properties or {}).get("doc_id", "")
            if doc_id:
                doc_scores.setdefault(doc_id, []).append(s.total)
                doc_map[s.node.id] = doc_id

        for s in scored:
            doc_id = doc_map.get(s.node.id, "")
            if doc_id and doc_id in doc_scores:
                siblings = doc_scores[doc_id]
                if len(siblings) > 1:
                    coverage = 0.05 * math.log(len(siblings) + 1)
                    s.total = min(1.0, s.total + coverage)

        scored.sort(key=lambda s: s.total, reverse=True)
        return scored
