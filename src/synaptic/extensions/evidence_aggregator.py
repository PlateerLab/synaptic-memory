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

The aggregator is deterministic — same input, same output — and keeps only
bounded token/similarity fingerprints as an internal speed cache. That matters
for regression-style eval while avoiding repeated tokenisation and pairwise
Jaccard work across queries.

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
import math
import re
from dataclasses import dataclass
from hashlib import blake2b
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

# Max REFERENCES-companion documents pulled in alongside a single chosen
# node, to bound fan-out on heavily cross-referential corpora.
_MAX_COMPANIONS_PER_ANCHOR = 3

_TOKEN_CACHE_MAX = 4096
_SIMILARITY_CACHE_MAX = 32768

_ContentKey = tuple[str, str]


def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    return {t.lower() if t[0].isascii() else t for t in _TOKEN.findall(text)}


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

    The aggregator is deterministic. It keeps bounded per-instance token and
    pairwise-similarity caches, but every call's selection state is independent.

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

    __slots__ = (
        "_lambda",
        "_pool_limit",
        "_sim_threshold",
        "_similarity_cache",
        "_token_cache",
    )

    def __init__(
        self,
        *,
        mmr_lambda: float = 0.7,
        similarity_threshold: float = 0.85,
        candidate_pool_limit: int = 0,
    ) -> None:
        self._lambda = mmr_lambda
        self._sim_threshold = similarity_threshold
        self._pool_limit = max(0, candidate_pool_limit)
        self._token_cache: dict[_ContentKey, set[str]] = {}
        self._similarity_cache: dict[tuple[_ContentKey, _ContentKey], float] = {}

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

        Kind-aware split (v0.17.1)
        --------------------------
        Candidates whose node carries a ``_table_name`` property — i.e.
        the rows materialised by ``table_ingester`` / ``db_ingester`` —
        are treated as **structured atoms**: they bypass MMR /
        per-document cap / category coverage, which all assume a
        chunk-like hierarchy and actively hurt retrieval on structured
        corpora (assort, X2BEE tables: MRR jumps from 0.425 → 0.268
        under full pipeline otherwise). Structured path is a straight
        top-K by score. Passage-style nodes (CHUNK, CONCEPT, plain
        ENTITY without a table binding) keep the full MMR + cap +
        coverage pipeline.
        """
        if not scored or k <= 0:
            return []

        # --- Kind split: structured rows → atoms, rest → passages ---
        #
        # OpenIE entity hubs are retrieval bridges, not source evidence, when
        # they enter as direct FTS/PPR seeds. Keeping them in final evidence can
        # crowd out the source document with a model-created artifact that only
        # repeats the query phrase. Relation-expanded OpenIE targets are kept:
        # those are the graph-only facts this layer is meant to surface.
        visible_scored = [s for s in scored if not _is_openie_bridge_only_candidate(s)]
        structured = [s for s in visible_scored if (s.node.properties or {}).get("_table_name")]
        passage = [s for s in visible_scored if not (s.node.properties or {}).get("_table_name")]

        passage_evidence = self._aggregate_passages(
            passage,
            k=k,
            per_document_cap=per_document_cap,
            anchor_categories=anchor_categories,
        )

        structured_evidence: list[Evidence] = []
        for cand in structured[:k]:
            structured_evidence.append(_make_evidence(cand, reason="structured_top_score"))

        # Preserve aggregator-specific ordering when only one kind is
        # present:
        #   - passage_evidence carries MMR-adjusted order, but its
        #     ``score`` field stores the *raw* total (pre-MMR). Sorting
        #     by score here would revert the diversity work — measured
        #     as a −6.5pp regression on KRRA Hard FTS-only.
        #   - structured_evidence is already in score-descending order.
        # Only when *both* kinds are present do we re-merge by score so
        # the top-K cap respects relative magnitudes across kinds.
        if not structured_evidence:
            return passage_evidence[:k]
        if not passage_evidence:
            return structured_evidence[:k]

        merged = passage_evidence + structured_evidence
        merged.sort(key=lambda e: e.score, reverse=True)
        return merged[:k]

    def _aggregate_passages(
        self,
        scored: list[ScoredCandidate],
        *,
        k: int,
        per_document_cap: int,
        anchor_categories: set[str] | None,
    ) -> list[Evidence]:
        """Legacy passage aggregator — MMR + per-doc cap + category coverage.

        Extracted from ``aggregate`` to keep the kind-aware split readable.
        Behaviour unchanged from v0.17.0 for CHUNK / CONCEPT / etc.
        """
        if not scored or k <= 0:
            return []

        remaining = (
            _bounded_passage_pool(
                scored,
                k=k,
                per_document_cap=per_document_cap,
                anchor_categories=anchor_categories,
                limit=self._pool_limit,
            )
            if self._pool_limit
            else list(scored)
        )
        selected: list[Evidence] = []
        selected_entries: list[tuple[_ContentKey, set[str]]] = []
        doc_counts: dict[str, int] = {}
        token_cache: dict[int, tuple[_ContentKey, set[str]]] = {}
        sim_cache: dict[int, tuple[int, float]] = {}

        def cand_entry(cand: ScoredCandidate) -> tuple[_ContentKey, set[str]]:
            cache_key = id(cand)
            cached = token_cache.get(cache_key)
            if cached is None:
                key = self._content_key(cand.node)
                cached = (key, self._tokens_for_key(cand.node, key))
                token_cache[cache_key] = cached
            return cached

        def cand_tokens(cand: ScoredCandidate) -> set[str]:
            return cand_entry(cand)[1]

        def max_selected_similarity(cand: ScoredCandidate) -> float:
            if not selected_entries:
                return 0.0
            cache_key = id(cand)
            checked_count, sim_max = sim_cache.get(cache_key, (0, 0.0))
            if checked_count >= len(selected_entries):
                return sim_max
            key, tokens = cand_entry(cand)
            for selected_key, selected_tokens in selected_entries[checked_count:]:
                sim_max = max(
                    sim_max,
                    self._jaccard_for(key, tokens, selected_key, selected_tokens),
                )
            sim_cache[cache_key] = (len(selected_entries), sim_max)
            return sim_max

        # --- Pass 1: category coverage ---
        if anchor_categories:
            for cat in sorted(anchor_categories):
                if len(selected) >= k:
                    break
                pick = self._best_for_category(remaining, cat, doc_counts, per_document_cap)
                if pick is None:
                    continue
                evidence = _make_evidence(pick, reason="category_coverage")
                pick_key, pick_tokens = cand_entry(pick)
                if self._passes_entry_similarity(
                    pick_key,
                    pick_tokens,
                    selected_entries,
                    self._sim_threshold,
                ):
                    selected.append(evidence)
                    selected_entries.append((pick_key, pick_tokens))
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
                # Document cap check can reject many same-source chunks before
                # we pay the tokenisation/Jaccard cost.
                doc_id = (cand.node.properties or {}).get("doc_id", "")
                if doc_id and doc_counts.get(doc_id, 0) >= per_document_cap:
                    continue

                # A node reached via a REFERENCES edge is a deliberate
                # cross-reference, not a redundant near-duplicate — yet
                # cited documents in the same corpus share boilerplate
                # vocabulary, so MMR's Jaccard similarity wrongly flags
                # them as duplicates and skips them. Reference companions
                # bypass the diversity *skip* (they are never eliminated
                # as duplicates) but still compete on the normal MMR-
                # adjusted score, so they cannot crowd out the seeds.
                is_reference = cand.reason == "references"
                if selected_entries:
                    sim_max = max_selected_similarity(cand)
                else:
                    sim_max = 0.0
                if sim_max >= self._sim_threshold and not is_reference:
                    continue
                adjusted = self._lambda * cand.total - (1.0 - self._lambda) * sim_max

                if adjusted > best_adj:
                    best_adj = adjusted
                    best_idx = i

            if best_idx < 0:
                break

            chosen = remaining.pop(best_idx)
            evidence = _make_evidence(chosen, reason="top_score")
            selected.append(evidence)
            selected_entries.append(cand_entry(chosen))
            if evidence.document_id:
                doc_counts[evidence.document_id] = doc_counts.get(evidence.document_id, 0) + 1

            # Companion attach — a document the chosen node explicitly
            # cites (REFERENCES edge) rides in *with* it as a bundle.
            # This is the multi-hop payload: a cited provision shares no
            # query vocabulary with the query, so it can only enter as a
            # companion of the document that cites it. Bypasses MMR /
            # per-doc cap; capped per anchor to bound fan-out.
            companions = [
                c for c in remaining if c.reason == "references" and c.anchor_id == chosen.node.id
            ]
            for comp in companions[:_MAX_COMPANIONS_PER_ANCHOR]:
                remaining.remove(comp)
                comp_ev = _make_evidence(comp, reason="reference_companion")
                selected.append(comp_ev)
                selected_entries.append(cand_entry(comp))
                if comp_ev.document_id:
                    doc_counts[comp_ev.document_id] = doc_counts.get(comp_ev.document_id, 0) + 1

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

    def _content_key(self, node: Node) -> _ContentKey:
        content = node.content or ""
        digest = blake2b(content.encode("utf-8"), digest_size=12).hexdigest()
        return (node.id, f"{len(content)}:{digest}")

    def _tokens_for(self, node: Node) -> set[str]:
        return self._tokens_for_key(node, self._content_key(node))

    def _tokens_for_key(self, node: Node, key: _ContentKey) -> set[str]:
        cached = self._token_cache.get(key)
        if cached is not None:
            return cached
        tokens = _tokens(node.content or "")
        if len(self._token_cache) >= _TOKEN_CACHE_MAX:
            self._token_cache.pop(next(iter(self._token_cache)))
        self._token_cache[key] = tokens
        return tokens

    def _jaccard_for(
        self,
        left_key: _ContentKey,
        left_tokens: set[str],
        right_key: _ContentKey,
        right_tokens: set[str],
    ) -> float:
        pair = (left_key, right_key) if left_key <= right_key else (right_key, left_key)
        cached = self._similarity_cache.get(pair)
        if cached is not None:
            return cached
        similarity = _jaccard(left_tokens, right_tokens)
        if len(self._similarity_cache) >= _SIMILARITY_CACHE_MAX:
            self._similarity_cache.pop(next(iter(self._similarity_cache)))
        self._similarity_cache[pair] = similarity
        return similarity

    def _passes_entry_similarity(
        self,
        candidate_key: _ContentKey,
        candidate_tokens: set[str],
        existing_entries: list[tuple[_ContentKey, set[str]]],
        threshold: float,
    ) -> bool:
        if not existing_entries:
            return True
        sim_max = max(
            (
                self._jaccard_for(
                    candidate_key,
                    candidate_tokens,
                    existing_key,
                    existing_tokens,
                )
                for existing_key, existing_tokens in existing_entries
            ),
            default=0.0,
        )
        return sim_max < threshold

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


def _bounded_passage_pool(
    scored: list[ScoredCandidate],
    *,
    k: int,
    per_document_cap: int,
    anchor_categories: set[str] | None,
    limit: int,
) -> list[ScoredCandidate]:
    """Return a small MMR pool plus protected tail candidates.

    The reranker already sorts by relevance in production, but tests and
    external callers may pass unsorted candidates. We therefore choose the
    base pool by score while preserving original order in the returned list
    for deterministic tie handling.
    """
    if limit <= 0 or len(scored) <= limit:
        return list(scored)

    ranked = sorted(enumerate(scored), key=lambda item: item[1].total, reverse=True)
    keep: set[int] = {idx for idx, _cand in ranked[:limit]}

    # Category coverage may intentionally pick a lower-scored representative.
    # Keep the best candidate for each requested category even if it sits just
    # outside the global score head.
    if anchor_categories:
        for category in anchor_categories:
            best_idx = _best_category_index(ranked, category)
            if best_idx is not None:
                keep.add(best_idx)

    # Per-document caps can force the final selection below the score head
    # when many high-ranked chunks come from one source. Keep enough distinct
    # document representatives for the greedy pass to fill k slots.
    doc_target = k
    if per_document_cap > 1:
        doc_target = max(1, math.ceil(k / per_document_cap) + 2)
    seen_docs: set[str] = set()
    for idx, cand in ranked:
        doc_id = (cand.node.properties or {}).get("doc_id", "")
        if not doc_id or doc_id in seen_docs:
            continue
        keep.add(idx)
        seen_docs.add(doc_id)
        if len(seen_docs) >= doc_target:
            break

    # REFERENCES companions are low-scoring by design: the anchor document
    # carries the query match and the cited document rides with it. If an
    # anchor survives the pool bound, keep up to the normal companion fan-out.
    kept_anchor_ids = {
        scored[idx].node.id
        for idx in keep
        if scored[idx].reason != "references" and scored[idx].node.id
    }
    refs_by_anchor: dict[str, int] = {}
    for idx, cand in ranked:
        if cand.reason != "references" or cand.anchor_id not in kept_anchor_ids:
            continue
        count = refs_by_anchor.get(cand.anchor_id, 0)
        if count >= _MAX_COMPANIONS_PER_ANCHOR:
            continue
        keep.add(idx)
        refs_by_anchor[cand.anchor_id] = count + 1

    return [cand for idx, cand in enumerate(scored) if idx in keep]


def _best_category_index(
    ranked: list[tuple[int, ScoredCandidate]],
    category: str,
) -> int | None:
    cat_lower = category.lower()
    for idx, cand in ranked:
        props = cand.node.properties or {}
        node_cat = (props.get("category") or "").lower()
        if node_cat and cat_lower in node_cat:
            return idx
    return None


def _passes_token_similarity(
    candidate_tokens: set[str],
    existing_tokens: list[set[str]],
    threshold: float,
) -> bool:
    if not existing_tokens:
        return True
    sim_max = max((_jaccard(candidate_tokens, t) for t in existing_tokens), default=0.0)
    return sim_max < threshold


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


def _is_openie_bridge_only_candidate(cand: ScoredCandidate) -> bool:
    tags = cand.node.tags or []
    if "_openie_entity" not in tags:
        return False
    return cand.reason not in {"semantic_relation", "related"}
