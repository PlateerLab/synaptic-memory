"""EvidenceSearch — the 3rd-generation retrieval pipeline as one facade.

This module stitches together the four F4 modules
(:class:`QueryAnchorExtractor`, :class:`GraphExpander`,
:class:`HybridReranker`, :class:`EvidenceAggregator`) into a single
``search`` call so application code doesn't have to wire them up
manually every time.

Pipeline:

    query
      ↓  anchor extraction     (categories, entities, keywords)
      ↓  FTS seed retrieval    (existing backend.search_fts)
      ↓  graph expansion       (shallow 1-hop from anchors + seeds)
      ↓  hybrid reranking      (lexical + semantic + graph + structural)
      ↓  evidence aggregation  (MMR + per-doc cap + category coverage)
    final evidence set

This is deliberately a thin facade — no new search algorithm hides
inside. All of the intelligence lives in the individual modules so
they can be swapped out or tested in isolation. If you need to tune
one stage of the pipeline, edit that module; the facade never
contains stage-specific logic.

Example::

    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.extensions.evidence_search import EvidenceSearch

    backend = SqliteGraphBackend("graph.db")
    await backend.connect()

    searcher = EvidenceSearch(backend=backend)
    result = await searcher.search(
        "경마 운영계획에서 인권경영과 충돌하는 부분은?",
        k=6,
    )
    for ev in result.evidence:
        print(ev.score, ev.node.title, "→", ev.reason)

    # Per-anchor diagnostics live on ``result.anchors``, full reranker
    # breakdowns on ``result.scored``, and expansion metadata on
    # ``result.expanded`` — same data the unit tests check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING

from synaptic.extensions.evidence_aggregator import Evidence, EvidenceAggregator
from synaptic.extensions.graph_expander import (
    ExpandedNode,
    ExpansionBudget,
    GraphExpander,
)
from synaptic.extensions.hybrid_reranker import (
    HybridReranker,
    RerankerWeights,
    ScoredCandidate,
)
from synaptic.extensions.query_anchor import QueryAnchorExtractor, QueryAnchors
from synaptic.ppr import personalized_pagerank

if TYPE_CHECKING:
    from synaptic.extensions.embedder import EmbeddingProvider
    from synaptic.extensions.query_anchor import PhraseExtractorProtocol
    from synaptic.extensions.reranker_cross import RerankerProtocol
    from synaptic.protocols import QueryDecomposer, StorageBackend

# RRF (Reciprocal Rank Fusion) constant — 60 is the canonical k from the
# original Cormack et al. 2009 paper and what every baseline re-implements.
_RRF_K = 60

logger = logging.getLogger("evidence-search")


@dataclass(slots=True)
class EvidenceSearchResult:
    """The full output of one evidence search call.

    Most callers only look at ``evidence`` — that's the final set the
    pipeline selected. The other fields are kept so eval harnesses,
    UIs, and debugging sessions can inspect why a particular answer
    came out the way it did without re-running the pipeline.
    """

    query: str
    anchors: QueryAnchors
    seeds: list[str] = field(default_factory=list)
    expanded: list[ExpandedNode] = field(default_factory=list)
    scored: list[ScoredCandidate] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    elapsed_ms: float = 0.0
    sub_queries: list[str] = field(default_factory=list)


class EvidenceSearch:
    """One-call wrapper around the F4 modules.

    Args:
        backend: Storage backend to search against.
        phrase_extractor: Optional phrase extractor for anchor
            extraction. Drop-in from the ingestion pipeline.
        reranker_weights: Override :class:`RerankerWeights`. Leave as
            ``None`` for the defaults tuned for Korean corpora.
        expansion_budget: Override the expander's budget.
        mmr_lambda: Diversity/relevance balance for the aggregator.
            ``0.7`` (default) biases toward relevance.
        similarity_threshold: Hard cutoff for near-duplicate content.
    """

    __slots__ = (
        "_aggregator",
        "_anchor_extractor",
        "_backend",
        "_cross_reranker",
        "_decomposer",
        "_embedder",
        "_expander",
        "_expansion_budget",
        "_rerank_blend",
        "_reranker",
    )

    def __init__(
        self,
        *,
        backend: StorageBackend,
        embedder: EmbeddingProvider | None = None,
        reranker: RerankerProtocol | None = None,
        phrase_extractor: PhraseExtractorProtocol | None = None,
        decomposer: QueryDecomposer | None = None,
        reranker_weights: RerankerWeights | None = None,
        expansion_budget: ExpansionBudget | None = None,
        mmr_lambda: float = 0.7,
        similarity_threshold: float = 0.85,
        rerank_blend: float = 0.1,
        table_query_hints: dict[str, list[str]] | None = None,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._cross_reranker = reranker
        self._decomposer = decomposer
        self._rerank_blend = rerank_blend
        self._anchor_extractor = QueryAnchorExtractor(
            backend=backend,
            phrase_extractor=phrase_extractor,
            table_query_hints=table_query_hints,
        )
        self._expander = GraphExpander(backend=backend)
        self._reranker = HybridReranker(weights=reranker_weights)
        self._aggregator = EvidenceAggregator(
            mmr_lambda=mmr_lambda,
            similarity_threshold=similarity_threshold,
        )
        self._expansion_budget = expansion_budget or ExpansionBudget()

    async def search(
        self,
        query: str,
        *,
        k: int = 6,
        fts_seed_limit: int = 20,
        per_document_cap: int = 2,
        query_embedding: list[float] | None = None,
    ) -> EvidenceSearchResult:
        """Run the full 3rd-gen pipeline for ``query``.

        Args:
            query: The raw user query. Normalised and tokenised
                internally; callers don't need to preprocess.
            k: Final evidence set size. 4-8 is the usual range for
                downstream LLM prompting.
            fts_seed_limit: How many FTS hits to take as initial seeds
                before expansion. Bigger numbers pay off on highly
                ambiguous queries; the default 20 is a safe middle.
            per_document_cap: Max evidence items from any single
                document. Passed straight to the aggregator.
            query_embedding: Optional query vector for the semantic
                signal. When ``None`` the reranker falls back to
                lexical + graph + structural only.
        """
        t0 = time()

        # Step 0 — embed the query if an embedder is wired up.
        # The caller can also pass query_embedding directly; the
        # embedder is a convenience so callers don't have to embed
        # on their own every time.
        if query_embedding is None and self._embedder is not None:
            try:
                query_embedding = await self._embedder.embed(query)
                if not query_embedding:
                    query_embedding = None
            except Exception:
                query_embedding = None

        # Step 1 — extract anchors
        anchors = await self._anchor_extractor.extract(query)

        # Step 2a — FTS seeds (lexical).
        fts_nodes = await self._backend.search_fts(query, limit=fts_seed_limit)
        fts_scores: dict[str, float] = {}
        for rank, node in enumerate(fts_nodes):
            fts_scores[node.id] = max(0.10, 0.95 - rank * 0.03)

        # Step 2b — Vector seeds (semantic). Supplements FTS with
        # results that share meaning but not surface words. This is
        # what fixes L2 paraphrase and L7 conversational queries.
        # Only nodes NOT already found by FTS are added so lexical
        # ranking is never disrupted (cascade, not fusion).
        fts_ids = {n.id for n in fts_nodes}
        vec_seeds: list = []
        if query_embedding:
            try:
                vec_nodes = await self._backend.search_vector(query_embedding, limit=fts_seed_limit)
                for node in vec_nodes:
                    if node.id not in fts_ids:
                        vec_seeds.append(node)
                        # Vector-only seeds get a flat score below FTS floor
                        # so they never outrank a strong lexical hit, but they
                        # ARE present for the reranker's semantic signal.
                        fts_scores[node.id] = 0.08
            except Exception:
                pass

        # Step 2c — Vector PRF (Pseudo Relevance Feedback).
        # Refine the query vector using top FTS results' embeddings:
        # q' = α*q + (1-α)*mean(top_k). This pulls the query vector
        # toward the actual relevant region of the embedding space,
        # improving recall on paraphrase / conversational queries.
        prf_seeds: list = []
        if query_embedding and fts_nodes:
            top_embeddings = [
                n.embedding
                for n in fts_nodes[:3]
                if n.embedding and len(n.embedding) == len(query_embedding)
            ]
            if top_embeddings:
                alpha = 0.7
                dim = len(query_embedding)
                mean_emb = [
                    sum(e[i] for e in top_embeddings) / len(top_embeddings) for i in range(dim)
                ]
                prf_vec = [
                    alpha * query_embedding[i] + (1 - alpha) * mean_emb[i] for i in range(dim)
                ]
                try:
                    prf_nodes = await self._backend.search_vector(
                        prf_vec, limit=fts_seed_limit // 2
                    )
                    seen_ids = fts_ids | {n.id for n in vec_seeds}
                    for node in prf_nodes:
                        if node.id not in seen_ids:
                            prf_seeds.append(node)
                            fts_scores[node.id] = 0.06
                except Exception:
                    pass

        all_seeds = list(fts_nodes) + vec_seeds + prf_seeds

        # Step 2c+ — table hint seed augmentation (v0.17.1).
        # If ``DomainProfile.table_query_hints`` was wired and any hint
        # matched the query, ``anchors.preferred_tables`` carries the
        # target tables.
        #
        # We only intervene when FTS has *under-represented* the target
        # table (≤2 hits in the current seed pool). For narrow-target
        # corpora — e.g. assort q012 "LBL코리아 판매 파트너" where the
        # gold ``sales_partners:2`` never reaches FTS top-30 — we
        # augment with a targeted re-FTS (``"{table_name} {query}"``)
        # and score the hits at 0.96, just past the rank-0 FTS floor
        # of 0.95 so they can outrank cross-table noise. When the
        # hinted table already dominates FTS (X2BEE's pr_goods_base
        # appears 10+ times in top-30) we leave well enough alone —
        # further boosting flattens every row to the same top score
        # and dilutes the gold rank, which the earlier "bulk boost to
        # 0.99" implementation demonstrated with a −5%-point X2BEE
        # Hard regression.
        preferred_tables = anchors.preferred_tables
        if preferred_tables:
            existing_ids = {n.id for n in all_seeds}
            for table_name in preferred_tables:
                in_table_count = sum(
                    1 for n in fts_nodes
                    if (n.properties or {}).get("_table_name") == table_name
                )
                if in_table_count >= 3:
                    continue  # table is well-represented; FTS is doing fine

                try:
                    aug_nodes = await self._backend.search_fts(
                        f"{table_name} {query}", limit=5
                    )
                except Exception as exc:
                    logger.warning(
                        "table-hint FTS failed (%r): %s", table_name, exc
                    )
                    continue
                for node in aug_nodes:
                    if (node.properties or {}).get("_table_name") != table_name:
                        continue
                    if node.id in existing_ids:
                        fts_scores[node.id] = max(
                            fts_scores.get(node.id, 0.0), 0.96
                        )
                    else:
                        all_seeds.append(node)
                        existing_ids.add(node.id)
                        fts_scores[node.id] = 0.96

        # Step 2d — query decomposition + RRF seed fusion.
        # If a decomposer is wired and returns >1 sub-queries, run FTS for
        # each sub and fuse the ranked lists via RRF (k=60). Sub-queries
        # surface bridge documents that the original compound query
        # buries. We only re-FTS (not vec/PRF) per-sub to keep this cheap
        # — PRF's payoff scales with query specificity, which sub-queries
        # already have.
        #
        # Graph expansion and reranking (Steps 3-5) continue to operate
        # on the ORIGINAL query so relevance is scored against actual
        # user intent, not the decomposed fragments.
        sub_queries: list[str] = []
        if self._decomposer is not None:
            try:
                decomposed = await self._decomposer.decompose(query)
                if len(decomposed) > 1:
                    sub_queries = [s for s in decomposed if s and s != query]
            except Exception as exc:
                logger.warning("query decomposition failed: %s", exc)

        if sub_queries:
            ranked_lists: list[list] = [list(fts_nodes)]
            for sub in sub_queries:
                try:
                    sub_fts = await self._backend.search_fts(sub, limit=fts_seed_limit)
                    ranked_lists.append(sub_fts)
                except Exception as exc:
                    logger.warning("sub-query FTS failed (%r): %s", sub, exc)
                    continue

            rrf: dict[str, float] = {}
            for ranked in ranked_lists:
                for rank, node in enumerate(ranked):
                    rrf[node.id] = rrf.get(node.id, 0.0) + 1.0 / (_RRF_K + rank)

            existing_ids = {n.id for n in all_seeds}
            for ranked in ranked_lists[1:]:
                for node in ranked:
                    if node.id not in existing_ids:
                        all_seeds.append(node)
                        existing_ids.add(node.id)

            # Normalise RRF → [0.10, 0.95] (same band as FTS rank scores).
            # Max-combine with the original fts_scores so a node's best
            # signal wins; vec-only / PRF-only nodes keep their floor.
            if rrf:
                rrf_max = max(rrf.values())
                for node_id, score in rrf.items():
                    normalised = 0.10 + 0.85 * (score / rrf_max)
                    fts_scores[node_id] = max(
                        fts_scores.get(node_id, 0.0), normalised
                    )

        # Step 3 — shallow graph expansion
        expanded = await self._expander.expand(
            anchors=anchors,
            seed_nodes=all_seeds,
            budget=self._expansion_budget,
        )

        # Step 3b — PPR graph discovery. Uses FTS seeds as teleport
        # nodes and walks the graph via PPR to find nodes reachable
        # through structural paths (PART_OF, CONTAINS, MENTIONS) that
        # neither FTS nor vector search found. Discovered nodes are
        # added to the expanded set with a graph-based score.
        if fts_scores:
            try:
                ppr_results = await personalized_pagerank(
                    self._backend,
                    {nid: score for nid, score in fts_scores.items()},
                    damping=0.85,
                    top_k=k * 3,
                )
                from synaptic.extensions.graph_expander import ExpandedNode

                expanded_ids = {e.node.id for e in expanded}
                for node_id, ppr_score in ppr_results:
                    if node_id not in expanded_ids:
                        node = await self._backend.get_node(node_id)
                        if node:
                            expanded.append(
                                ExpandedNode(
                                    node=node,
                                    reason="ppr_discovery",
                                    hops=2,
                                    anchor_hit=None,
                                )
                            )
                            fts_scores[node_id] = ppr_score * 0.5
            except Exception:
                pass  # PPR failure is non-fatal

        # Step 4 — hybrid reranking
        anchor_category_set = set(anchors.categories)
        scored = self._reranker.rerank(
            expanded=expanded,
            fts_scores=fts_scores,
            query_embedding=query_embedding,
            anchor_categories=anchor_category_set,
        )

        # Step 4b — cross-encoder reranking (optional, highest quality).
        # Takes the top candidates from the hybrid reranker and rescores
        # each (query, content) pair jointly. This is what enables
        # paraphrase matching ("말 복지" ↔ "재활힐링승마") that neither
        # BM25 nor cosine can handle.
        #
        # Blend defaults to 0.1 (10% cross-encoder + 90% existing hybrid).
        # Earlier versions used 0.4 which maximised paraphrase wins but
        # wrecked retrieval-style corpora where FTS ranking was already
        # near-optimal (AutoRAG: 0.906 FTS → 0.642 at blend=0.4, recovered
        # to 0.766 at 0.1). 0.1 is the global optimum across 5 public
        # benches; see ``examples/ablation/sweep_rerank_blend.py``.
        #
        # v0.17.1 — structured rows (nodes with a ``_table_name``
        # property, emitted by db_ingester / table_ingester) are excluded
        # from reranking entirely. The cross-encoder's training
        # distribution is long-form paraphrase; applied to short
        # structured rows it produces near-random signal that
        # overrides FTS's near-optimal ranking on those corpora
        # (X2BEE Hard −34%, assort Conv −37% measured under blend=0.4).
        # Passage kinds (CHUNK, CONCEPT, plain ENTITY) still rerank.
        if self._cross_reranker is not None and scored:
            top_n = min(20, len(scored))
            top_candidates = scored[:top_n]
            rerank_indices = [
                i for i, s in enumerate(top_candidates)
                if not (s.node.properties or {}).get("_table_name")
            ]
            if rerank_indices:
                documents = [
                    f"{top_candidates[i].node.title}\n"
                    f"{top_candidates[i].node.content[:400]}"
                    for i in rerank_indices
                ]
                try:
                    rerank_scores = await self._cross_reranker.rerank(query, documents)
                    # Adaptive blend (v0.17.1) — scale the reranker
                    # contribution by its own discrimination strength.
                    # When the cross-encoder produces a tight cluster of
                    # logits (e.g. AutoRAG: std≈0.3) it has no signal and
                    # blending it just injects noise that displaces the
                    # FTS top-1 (measured −15 % MRR at fixed blend=0.1).
                    # When it produces a wide spread (PublicHealthQA:
                    # std≈4) the reranker is clearly picking up paraphrase
                    # similarity and the full blend pays off (+34 %).
                    # Linear interpolation: std≥3 → full blend, std=0 →
                    # zero. Threshold 3.0 was chosen from per-corpus
                    # diagnostics (see docs/PLAN-v0.18-architecture.md
                    # §Q3 + examples/ablation/diagnose_autorag.py); the
                    # worst-case std on AutoRAG was 0.53.
                    #
                    # Round 5 also tried RRF rank-fusion as a
                    # magnitude-free alternative; it was strictly worse
                    # (mean MRR 0.637 vs 0.647) — rank discretisation
                    # discards the score-magnitude information that the
                    # weighted blend uses to do small reorders. Adaptive
                    # weighted blend is the v0.17.1 default.
                    if len(rerank_scores) >= 2:
                        mean = sum(rerank_scores) / len(rerank_scores)
                        var = sum(
                            (s - mean) ** 2 for s in rerank_scores
                        ) / len(rerank_scores)
                        std = var ** 0.5
                        discriminator = min(1.0, std / 3.0)
                    else:
                        discriminator = 1.0
                    effective_blend = self._rerank_blend * discriminator
                    if effective_blend > 0:
                        for j, i in enumerate(rerank_indices):
                            if j < len(rerank_scores):
                                s = top_candidates[i]
                                blended = (
                                    effective_blend * rerank_scores[j]
                                    + (1.0 - effective_blend) * s.total
                                )
                                s.total = blended
                        scored[:top_n] = top_candidates
                        scored.sort(key=lambda s: s.total, reverse=True)
                except Exception as exc:
                    logger.warning("cross-encoder rerank failed: %s", exc)

        # Step 5 — evidence aggregation with diversity
        evidence = self._aggregator.aggregate(
            scored=scored,
            k=k,
            per_document_cap=per_document_cap,
            anchor_categories=anchor_category_set,
        )

        elapsed_ms = (time() - t0) * 1000
        logger.debug(
            "evidence-search[%r]: anchors=%d cats, %d seeds → %d expanded → %d evidence in %.1f ms",
            query,
            len(anchors.categories),
            len(fts_nodes),
            len(expanded),
            len(evidence),
            elapsed_ms,
        )

        return EvidenceSearchResult(
            query=query,
            anchors=anchors,
            seeds=[n.id for n in fts_nodes],
            expanded=expanded,
            scored=scored,
            evidence=evidence,
            elapsed_ms=elapsed_ms,
            sub_queries=sub_queries,
        )
