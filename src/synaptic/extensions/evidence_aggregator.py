"""EvidenceAggregator — final evidence selection with diversity constraints.

The reranker produces a scored list; the aggregator decides which
subset actually goes to the answer-generation step. This is where
3rd-generation retrieval earns its name: instead of returning "top-k
by score" and calling it a day, we **spread the evidence** across
documents, categories, and expansion reasons so a complex question
has multiple grounded perspectives to draw from.

Three mechanics do the work:

1. **Maximal Marginal Relevance (MMR)** — penalise candidates that
   duplicate the content of already-selected ones. The MMR formula is
   the standard ``λ · relevance − (1 − λ) · max_similarity``; we use
   a cheap Jaccard over content tokens because it's O(tokens) and
   doesn't need embeddings.

2. **Per-document cap** — no document contributes more than ``N``
   chunks. Prevents a single long document from monopolising the
   evidence set when its chunks all score high.

3. **Category coverage bonus** — if the query touched multiple
   categories (from ``QueryAnchors``) we prefer keeping at least one
   evidence per matched category. This is the mechanism that lets
   cross-category questions ("어떻게 규정과 운영계획이 충돌하나") see
   both sides of the evidence.

The aggregator is deterministic and pure — same input, same output —
which matters for regression-style eval.

Example::

    aggregator = EvidenceAggregator()
    evidence = aggregator.aggregate(
        scored=reranked,               # from HybridReranker
        k=6,
        per_document_cap=2,
        anchor_categories={"규정 및 지침", "운영계획"},
    )
    # evidence is a list[Evidence] — the final set to hand to the LLM
    # (or to return as top-k for pure retrieval use-cases).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from synaptic.models import Node

if TYPE_CHECKING:
    from synaptic.extensions.hybrid_reranker import ScoredCandidate

logger = logging.getLogger("evidence-aggregator")


# --- Tokeniser used by the Jaccard similarity check ---
#
# Cheap content fingerprint: pull Hangul / Latin runs of length ≥ 2,
# lowercase the Latin side, take the set. Good enough to detect
# near-duplicate chunks from the same document without any model call.
# Two-char minimum matches the rest of the synaptic pipeline (phrase
# extractor, query anchor tokeniser) and lets Korean bigrams through.

_TOKEN = re.compile(r"[A-Za-z가-힣]{2,}")


def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    return {
        t.lower() if t[0].isascii() else t
        for t in _TOKEN.findall(text)
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity — ``|A∩B| / |A∪B|``, zero-safe."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


@dataclass(slots=True)
class Evidence:
    """A final-selected piece of evidence plus why it was picked.

    Attributes:
        node: The underlying ``Node`` — typically a Chunk or Document.
        score: The relevance score carried over from the reranker's
            ``total``. Not re-normalised so callers can still compare
            evidence scores across queries on the same corpus.
        reason: Short tag describing why the aggregator kept this
            node. ``"top_score"`` means "it was the best remaining
            candidate"; ``"category_coverage"`` means "we kept it to
            represent category X"; ``"document_quota"`` means "it was
            the best chunk we hadn't yet taken from this document".
        document_id: Parent document id (from ``properties['doc_id']``).
            Empty when the node isn't attached to a document or the
            parent wasn't indexed. Used by downstream UI to group
            evidence by source document.
        category: Category label from ``properties['category']``,
            empty when absent.
    """

    node: Node
    score: float
    reason: str
    document_id: str = ""
    category: str = ""


class EvidenceAggregator:
    """Select the final evidence set from a reranked candidate pool.

    The aggregator is stateless — every call is independent — so one
    instance can serve concurrent queries safely.

    Args:
        mmr_lambda: MMR blending parameter. ``1.0`` disables diversity
            (pure greedy top-k); ``0.0`` only cares about diversity
            and ignores relevance. Default ``0.7`` gives relevance ~3x
            the weight of novelty, which matches published RAG tuning.
        similarity_threshold: A candidate whose Jaccard with the nearest
            already-selected evidence exceeds this is always dropped,
            regardless of the MMR penalty. Hard cutoff for near
            duplicates that the soft MMR would otherwise keep because
            they're still high-scoring.
    """

    __slots__ = ("_lambda", "_sim_threshold")

    def __init__(
        self,
        *,
        mmr_lambda: float = 0.7,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._lambda = mmr_lambda
        self._sim_threshold = similarity_threshold

    def aggregate(
        self,
        *,
        scored: list[ScoredCandidate],
        k: int = 6,
        per_document_cap: int = 2,
        anchor_categories: set[str] | None = None,
    ) -> list[Evidence]:
        """Pick the top ``k`` evidence items under diversity constraints.

        The algorithm walks the reranked list greedily. For each
        candidate we compute an adjusted score:

            adjusted = λ · relevance − (1 − λ) · max_similarity

        where ``max_similarity`` is the highest Jaccard between the
        candidate's tokens and any already-selected evidence's tokens.
        The candidate with the best adjusted score that also satisfies
        the per-document cap and the similarity threshold gets
        selected, and we repeat until we hit ``k`` or run out.

        If ``anchor_categories`` is supplied we do a **category coverage
        pass first**: for each category we keep the best-scoring
        candidate that matches it, even if it wouldn't have won the
        greedy pass on its own merit. The remaining slots are filled
        greedily.
        """
        if not scored or k <= 0:
            return []

        remaining = list(scored)
        selected: list[Evidence] = []
        selected_tokens: list[set[str]] = []
        doc_counts: dict[str, int] = {}

        # --- Pass 1: category coverage ---
        if anchor_categories:
            for cat in sorted(anchor_categories):
                if len(selected) >= k:
                    break
                pick = self._best_for_category(remaining, cat, doc_counts, per_document_cap)
                if pick is None:
                    continue
                evidence = _make_evidence(pick, reason="category_coverage")
                if self._passes_similarity(evidence, selected_tokens):
                    selected.append(evidence)
                    selected_tokens.append(_tokens(evidence.node.content))
                    if evidence.document_id:
                        doc_counts[evidence.document_id] = (
                            doc_counts.get(evidence.document_id, 0) + 1
                        )
                    remaining.remove(pick)

        # --- Pass 2: greedy MMR fill ---
        while len(selected) < k and remaining:
            best_idx = -1
            best_adj = -math.inf
            for i, cand in enumerate(remaining):
                cand_tokens = _tokens(cand.node.content)
                sim_max = max(
                    (_jaccard(cand_tokens, t) for t in selected_tokens),
                    default=0.0,
                )
                if sim_max >= self._sim_threshold:
                    continue
                adjusted = self._lambda * cand.total - (1.0 - self._lambda) * sim_max

                # Document cap check
                doc_id = (cand.node.properties or {}).get("doc_id", "")
                if doc_id and doc_counts.get(doc_id, 0) >= per_document_cap:
                    continue

                if adjusted > best_adj:
                    best_adj = adjusted
                    best_idx = i

            if best_idx < 0:
                break

            chosen = remaining.pop(best_idx)
            evidence = _make_evidence(chosen, reason="top_score")
            selected.append(evidence)
            selected_tokens.append(_tokens(evidence.node.content))
            if evidence.document_id:
                doc_counts[evidence.document_id] = (
                    doc_counts.get(evidence.document_id, 0) + 1
                )

        return selected

    # --- helpers ---

    def _best_for_category(
        self,
        remaining: list[ScoredCandidate],
        category: str,
        doc_counts: dict[str, int],
        per_document_cap: int,
    ) -> ScoredCandidate | None:
        """Return the highest-scoring candidate matching ``category``.

        "Matching" means the node's ``properties['category']`` contains
        the category label (case-insensitive substring). Respects the
        per-document cap so a single doc can't fill every category
        slot.
        """
        cat_lower = category.lower()
        best: ScoredCandidate | None = None
        for cand in remaining:
            props = cand.node.properties or {}
            node_cat = (props.get("category") or "").lower()
            if not node_cat or cat_lower not in node_cat:
                continue
            doc_id = props.get("doc_id", "")
            if doc_id and doc_counts.get(doc_id, 0) >= per_document_cap:
                continue
            if best is None or cand.total > best.total:
                best = cand
        return best

    def _passes_similarity(
        self,
        evidence: Evidence,
        existing_tokens: list[set[str]],
    ) -> bool:
        """Drop the candidate if it's too similar to anything already picked."""
        if not existing_tokens:
            return True
        cand_tokens = _tokens(evidence.node.content)
        sim_max = max((_jaccard(cand_tokens, t) for t in existing_tokens), default=0.0)
        return sim_max < self._sim_threshold


def _make_evidence(cand: ScoredCandidate, *, reason: str) -> Evidence:
    """Project a ``ScoredCandidate`` into an ``Evidence`` record."""
    props = cand.node.properties or {}
    return Evidence(
        node=cand.node,
        score=cand.total,
        reason=reason,
        document_id=props.get("doc_id", ""),
        category=props.get("category", ""),
    )


# math.inf is used in the MMR loop — importing locally so the module
# doesn't need a top-level ``import math`` for a single constant.
import math  # noqa: E402
