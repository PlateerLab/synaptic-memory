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
from synaptic.extensions.graph_read_cache import GraphReadCache
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
# Asymmetric RRF constant for the vector rank list in fusion_mode="rrf".
# Larger than _RRF_K so dense hits are weighted slightly below lexical,
# keeping synaptic's lexical-dominant identity while un-capping vector-only
# seeds from the flat 0.08 cascade floor.
_RRF_VEC_K = 90
_PPR_SEED_MIN = 32
_PPR_SEED_MULTIPLIER = 1
_PPR_RESULT_MULTIPLIER = 2
_AGGREGATE_POOL_MIN = 24
_AGGREGATE_POOL_MULTIPLIER = 1

# Anchor-coverage thresholds for the adaptive controller (SYNAPTIC_ADAPTIVE):
# a query whose anchor terms overlap the retrieved docs >= HI is lexically
# reliable (apply real-BM25 sharpening); <= LO is a paraphrase (apply vector
# fusion + semantic tilt). The middle band gets the safe default. See
# ``EvidenceSearch._anchor_coverage``.
_COVERAGE_HI = 0.8
_COVERAGE_LO = 0.3

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
    timings_ms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)
    sub_queries: list[str] = field(default_factory=list)


def _ppr_seed_limit(k: int) -> int:
    return max(_PPR_SEED_MIN, k * _PPR_SEED_MULTIPLIER)


def _bounded_ppr_seed_scores(seed_scores: dict[str, float], k: int) -> dict[str, float]:
    limit = _ppr_seed_limit(k)
    if len(seed_scores) <= limit:
        return dict(seed_scores)
    return dict(sorted(seed_scores.items(), key=lambda item: item[1], reverse=True)[:limit])


def _aggregate_candidate_pool_limit(k: int) -> int:
    return max(_AGGREGATE_POOL_MIN, max(0, k) * _AGGREGATE_POOL_MULTIPLIER)


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
        aggregate_candidate_pool_limit: Passage candidate cap for the final
            aggregation stage. ``None`` uses a dynamic cap based on ``k``;
            ``0`` disables the cap.
    """

    __slots__ = (
        "_adaptive",
        "_aggregate_candidate_pool_limit",
        "_aggregator",
        "_anchor_extractor",
        "_backend",
        "_backend_scored",
        "_calibrated_blend",
        "_calibration_loaded",
        "_cross_reranker",
        "_decomposer",
        "_embedder",
        "_expand_relevance",
        "_expander",
        "_expansion_budget",
        "_fusion_mode",
        "_graph_expansion",
        "_phrase_cache",
        "_phrase_cache_loaded",
        "_query_phrase_seed_k",
        "_query_tilt_enabled",
        "_real_scores_enabled",
        "_rerank_blend",
        "_rerank_std_deadzone",
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
        query_phrase_seed_k: int = 0,
        rerank_std_deadzone: float = 0.0,
        fusion_mode: str = "cascade",
        adaptive: bool = False,
        graph_expansion: bool = True,
        aggregate_candidate_pool_limit: int | None = None,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._cross_reranker = reranker
        self._decomposer = decomposer
        # ``_rerank_blend`` is the user-provided default. The actual
        # blend used at search time may be overridden by a per-corpus
        # calibration (see ``synaptic.extensions.calibration``) — for
        # FTS-already-strong corpora calibration sets the override to
        # 0.0, structurally avoiding the AutoRAG / X2BEE Easy
        # regression measured in v0.17.x.
        self._rerank_blend = rerank_blend
        self._calibration_loaded = False
        self._calibrated_blend: float | None = None
        # Phrase hub embedding cache for query→phrase dense seeding.
        # Lazily populated on first search() call: each phrase node's
        # title-embedding plus its L2 norm, so per-query cosine reduces
        # to dot-product math against an in-memory list. Without this,
        # every query re-scanned ``list_nodes(kind=ENTITY)`` on 10k+
        # phrases and rebuilt norms — measured >10s per query on
        # MuSiQue and dominated runtime.
        self._phrase_cache: list[tuple[object, list[float], float]] | None = None
        self._phrase_cache_loaded = False
        # v0.27 — query→phrase dense seed top-K. Set to 0 to disable
        # the bridge entirely (useful for ablation against a graph that
        # still has phrase hub nodes from entity-linker / phrase
        # extractor). Env override ``SYNAPTIC_PHRASE_SEED_K`` lets
        # ablation scripts flip behavior without threading the kwarg
        # through every facade.
        import os as _os

        env_k = _os.environ.get("SYNAPTIC_PHRASE_SEED_K")
        if env_k is not None:
            try:
                query_phrase_seed_k = int(env_k)
            except ValueError:
                pass
        self._query_phrase_seed_k = query_phrase_seed_k
        # Bound the final MMR/diversity stage by default. Graph expansion can
        # now surface 100+ candidates per query; aggregating all of them spends
        # time tokenising and pairwise-checking passages that cannot plausibly
        # enter a small evidence set. The aggregator's protected pool keeps
        # category, document-diversity, and REFERENCES companion candidates.
        env_pool = _os.environ.get("SYNAPTIC_AGGREGATE_CANDIDATE_POOL_LIMIT")
        if env_pool is not None:
            try:
                aggregate_candidate_pool_limit = int(env_pool)
            except ValueError:
                pass
        self._aggregate_candidate_pool_limit = aggregate_candidate_pool_limit
        # v0.28 — per-query reranker-score deadzone (opt-in, default 0.0
        # → never triggers, behaviour identical to the v0.17.1 std/3 ramp).
        # When the cross-encoder's top-K logits have std below this floor
        # they carry no discriminative signal, so the blend is zeroed
        # outright for that query rather than merely attenuated. The ramp
        # alone leaves a residual blend (AutoRAG worst-case std≈0.53 →
        # blend≈0.018) that still displaces the FTS top-1 and costs ~−0.10
        # MRR on retrieval-style corpora. The std SCALE is embedder-
        # dependent (measured: bge-m3 puts AutoRAG std≈0.5, while
        # Qwen3-Embedding-4B puts it ≈3 and overlaps PublicHealthQA), so
        # no single fixed floor is universal — tune per deployment via env
        # ``SYNAPTIC_RERANK_STD_DEADZONE`` and measure with
        # examples/ablation/sweep_rerank_deadzone.py. PublicHealthQA /
        # HotPotQA rely on the reranker at higher std and must not regress.
        env_dz = _os.environ.get("SYNAPTIC_RERANK_STD_DEADZONE")
        if env_dz is not None:
            try:
                rerank_std_deadzone = float(env_dz)
            except ValueError:
                pass
        self._rerank_std_deadzone = rerank_std_deadzone
        # L01 (opt-in via env ``SYNAPTIC_REAL_SCORES=1``) — feed the backend's
        # real normalized BM25 into the lexical axis instead of a synthetic
        # rank ramp. Only engages when the backend's ``search_fts`` accepts
        # ``with_scores`` (SQLite today); other backends transparently fall
        # back to rank-derived scoring, so this never breaks a backend.
        # Capability: does the backend's search_fts accept ``with_scores``?
        # (SQLite does; others fall back to rank-derived scoring.) Shared by
        # L01 and the adaptive controller.
        try:
            import inspect as _inspect

            self._backend_scored = (
                "with_scores" in _inspect.signature(backend.search_fts).parameters
            )
        except (ValueError, TypeError):
            self._backend_scored = False
        self._real_scores_enabled = (
            _os.environ.get("SYNAPTIC_REAL_SCORES") == "1" and self._backend_scored
        )
        # L02 (opt-in via ``fusion_mode="rrf"`` or env ``SYNAPTIC_FUSION_MODE``)
        # — true RRF fusion of the FTS and vector seed rank lists instead of
        # the cascade that pins vector-only seeds at a flat 0.08 below the FTS
        # floor. Default "cascade" preserves shipped behaviour exactly.
        self._fusion_mode = _os.environ.get("SYNAPTIC_FUSION_MODE", fusion_mode)
        # L05 (opt-in via env ``SYNAPTIC_QUERY_TILT=1``) — per-query
        # lexical↔semantic reranker-weight tilt driven by anchor coverage.
        # Default off; only shifts weights on low-surface-overlap (paraphrase)
        # queries, leaving lexical-confident queries on the default weights.
        self._query_tilt_enabled = _os.environ.get("SYNAPTIC_QUERY_TILT") == "1"
        # v0.28 — coverage-ADAPTIVE mode (``adaptive=True`` / env
        # ``SYNAPTIC_ADAPTIVE=1``). One signal (anchor coverage) gates the
        # three retrieval levers per query so they STACK safely: real-BM25
        # sharpening only on high-coverage queries, vector RRF fusion only on
        # low-coverage (paraphrase) queries, and the L05 semantic tilt
        # (self-gating). Measured rationale: stacking the raw flags regressed
        # FTS-optimal AutoRAG -0.021; coverage gating fixes that on the Qwen
        # embedder (net-positive, KRRA Hard +0.027, AutoRAG -0.01 borderline).
        # OPT-IN, NOT DEFAULT: it is EMBEDDER-DEPENDENT — on bge-m3 it
        # regresses AutoRAG -0.030 (the same wall as the rerank deadzone). The
        # gating LOGIC is embedder-independent (surface overlap) but the
        # levers' EFFECT is not, so the 0.8/0.3 thresholds aren't universal.
        # Recommended for Qwen-style embedders; measure on yours first.
        # Degrades to a no-op without a scored backend or an embedder.
        env_ad = _os.environ.get("SYNAPTIC_ADAPTIVE")
        self._adaptive = (env_ad == "1") if env_ad is not None else adaptive
        # Ablation knob — skip 1-hop graph expansion + PPR discovery, feeding
        # only FTS/vector seeds to the reranker. Measures how much the graph
        # structure (vs the lexical/vector seeds) actually contributes to
        # retrieval, so we know whether graph-connectivity work is worth it
        # before building it. Default True = shipped behaviour unchanged.
        env_ge = _os.environ.get("SYNAPTIC_GRAPH_EXPANSION")
        self._graph_expansion = (env_ge != "0") if env_ge is not None else graph_expansion
        # Relevance-aware expansion budget (opt-in via SYNAPTIC_EXPAND_RELEVANCE=1).
        # When the 1-hop neighbourhood exceeds the expander budget, keep the most
        # query-relevant neighbours instead of the first-visited ones — so the
        # agent finds evidence buried in large neighbourhoods. Default off until
        # measured on the agent benches.
        self._expand_relevance = _os.environ.get("SYNAPTIC_EXPAND_RELEVANCE") == "1"
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

    @staticmethod
    def _rerank_discriminator(rerank_scores: list[float], deadzone: float) -> float:
        """Map the cross-encoder's top-K logit spread to a blend
        multiplier in ``[0, 1]``.

        A wide spread (std ≥ 3) means the reranker is confidently
        separating paraphrase-relevant passages → full blend. A flat
        cluster means it has no discriminative signal → ~0, so blending
        it would only inject noise that displaces the FTS top-1.

        ``deadzone`` is an opt-in hard floor (default ``0.0`` → never
        triggers, identical to the v0.17.1 ``std/3`` ramp). When the
        observed std is below it, the multiplier is zeroed outright
        instead of merely attenuated — the ramp alone leaves a residual
        (AutoRAG's worst-case std ≈ 0.53 → ≈ 0.018) that still costs
        ~−0.10 MRR on retrieval-style corpora.
        """
        if len(rerank_scores) < 2:
            return 1.0
        mean = sum(rerank_scores) / len(rerank_scores)
        var = sum((s - mean) ** 2 for s in rerank_scores) / len(rerank_scores)
        std = var**0.5
        if std < deadzone:
            return 0.0
        return min(1.0, std / 3.0)

    @staticmethod
    def _anchor_coverage(anchors: QueryAnchors, fts_nodes: list) -> float | None:
        """Fraction of the query's anchor terms (keywords ∪ entities) that
        surface in the top FTS docs. High = lexically well-matched; low = the
        query is paraphrased relative to the corpus (surface overlap is low
        *by definition* for a paraphrase — the paraphrase-AWARE signal the
        v0.26 pseudo-self-retrieval calibration lacked). ``None`` = no signal.
        """
        terms = {t.lower() for t in [*anchors.keywords, *anchors.entities] if t}
        if not terms or not fts_nodes:
            return None
        corpus = " ".join(f"{n.title or ''} {n.content or ''}" for n in fts_nodes[:5]).lower()
        return sum(1 for t in terms if t in corpus) / len(terms)

    @staticmethod
    def _tilt_weights(anchors: QueryAnchors, fts_nodes: list) -> RerankerWeights | None:
        """Per-query lexical↔semantic reranker-weight tilt from anchor
        coverage (L05). Low coverage → tilt toward the semantic axis;
        lexical-confident queries keep the default. ``None`` = no tilt.
        """
        coverage = EvidenceSearch._anchor_coverage(anchors, fts_nodes)
        if coverage is None:
            return None
        # coverage >= HI → no tilt; <= LO → full tilt to the semantic-lead
        # preset; linear in between.
        t = max(0.0, min(1.0, (_COVERAGE_HI - coverage) / (_COVERAGE_HI - _COVERAGE_LO)))
        if t <= 0.0:
            return None
        # graph (0.20) / structural (0.10) fixed; only lexical↔semantic shift,
        # so the weights still sum to 1.0 at every t.
        return RerankerWeights(
            lexical=0.45 - t * (0.45 - 0.30),
            semantic=0.25 + t * (0.40 - 0.25),
            graph=0.20,
            structural=0.10,
        )

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
        timings_ms: dict[str, float] = {}
        diagnostics: dict[str, float] = {}

        # Step 0 — embed the query if an embedder is wired up.
        # The caller can also pass query_embedding directly; the
        # embedder is a convenience so callers don't have to embed
        # on their own every time.
        stage_t0 = time()
        if query_embedding is None and self._embedder is not None:
            try:
                query_embedding = await self._embedder.embed(query)
                if not query_embedding:
                    query_embedding = None
            except Exception:
                query_embedding = None
        timings_ms["embed"] = (time() - stage_t0) * 1000

        # Step 1 — extract anchors
        stage_t0 = time()
        anchors = await self._anchor_extractor.extract(query)
        timings_ms["anchor"] = (time() - stage_t0) * 1000

        # Step 2a — FTS seeds (lexical).
        stage_t0 = time()
        fts_scores: dict[str, float] = {}
        # Fetch real scores when L01 is on OR adaptive mode may need them.
        want_scores = self._backend_scored and (self._real_scores_enabled or self._adaptive)
        if want_scores:
            scored_fts = await self._backend.search_fts(
                query, limit=fts_seed_limit, with_scores=True
            )
            fts_nodes = [n for n, _ in scored_fts]
        else:
            scored_fts = None
            fts_nodes = await self._backend.search_fts(query, limit=fts_seed_limit)

        # Adaptive coverage signal — computed once, gates L01/L02/L05 below.
        coverage = self._anchor_coverage(anchors, fts_nodes) if self._adaptive else None

        # L01 — real normalized BM25 into the lexical axis ([0.10,0.95] band).
        # Use it when explicitly enabled, or (adaptive) only on high-coverage
        # queries where lexical is reliable; else fall back to the rank ramp
        # (sharpening an unreliable lexical signal hurts paraphrase corpora).
        use_real = scored_fts is not None and (
            self._real_scores_enabled
            or (self._adaptive and coverage is not None and coverage >= _COVERAGE_HI)
        )
        if use_real:
            for node, rel in scored_fts:
                fts_scores[node.id] = 0.10 + rel * 0.85
        else:
            for rank, node in enumerate(fts_nodes):
                fts_scores[node.id] = max(0.10, 0.95 - rank * 0.03)
        timings_ms["fts"] = (time() - stage_t0) * 1000

        # Step 2b — Vector seeds (semantic). Supplements FTS with
        # results that share meaning but not surface words. This is
        # what fixes L2 paraphrase and L7 conversational queries.
        # Only nodes NOT already found by FTS are added so lexical
        # ranking is never disrupted (cascade, not fusion).
        fts_ids = {n.id for n in fts_nodes}
        vec_seeds: list = []
        vec_nodes: list = []
        stage_t0 = time()
        if query_embedding:
            try:
                vec_nodes = await self._backend.search_vector(query_embedding, limit=fts_seed_limit)
                for node in vec_nodes:
                    if node.id not in fts_ids:
                        vec_seeds.append(node)
                        # Vector-only seeds get a flat score below FTS floor
                        # so they never outrank a strong lexical hit, but they
                        # ARE present for the reranker's semantic signal.
                        # (Overwritten below in fusion_mode="rrf".)
                        fts_scores[node.id] = 0.08
            except Exception:
                vec_nodes = []

        # L02 — true RRF fusion of the FTS and vector rank lists (opt-in).
        # The cascade above pins every vector-only seed at a flat 0.08, below
        # the FTS floor by construction, so a gold doc that shares no surface
        # tokens (the MuSiQue failure mode) can never outrank a weak lexical
        # hit. RRF-fuse the two rank lists (asymmetric k keeps lexical-lead)
        # and MAX-combine into fts_scores, so FTS hits keep their real-BM25 /
        # rank lexical score and only vector-only seeds are lifted by rank.
        use_fusion = self._fusion_mode == "rrf" or (
            self._adaptive and coverage is not None and coverage <= _COVERAGE_LO
        )
        if use_fusion and vec_nodes:
            rrf: dict[str, float] = {}
            for rank, node in enumerate(fts_nodes):
                rrf[node.id] = rrf.get(node.id, 0.0) + 1.0 / (_RRF_K + rank)
            for rank, node in enumerate(vec_nodes):
                rrf[node.id] = rrf.get(node.id, 0.0) + 1.0 / (_RRF_VEC_K + rank)
            if rrf:
                rrf_max = max(rrf.values())
                for node_id, score in rrf.items():
                    normalised = 0.10 + 0.85 * (score / rrf_max)
                    fts_scores[node_id] = max(fts_scores.get(node_id, 0.0), normalised)

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

        # Step 2c++ — query→entity-hub dense seed (v0.27/v0.30).
        # Bridge the lexical/dense gap on multi-hop and paraphrased entity
        # mentions: embed the query, match against embedded entity hub
        # nodes built by EntityLinker / PhraseExtractor / OpenIE
        # (kind=ENTITY, tag=_phrase or _openie_entity), and use the
        # chunks linked to the top-matched hubs as seeds. Hub nodes
        # themselves are NOT seeds — they
        # are bridges from "what the query is about" to "which chunks
        # mention that thing", which is the mechanism HippoRAG2's
        # query→triple linking exploits (+12.5pp R@5 on MuSiQue in their
        # Table 5 ablation).
        phrase_bridge_seeds: list = []
        if query_embedding and self._query_phrase_seed_k > 0:
            try:
                phrase_chunks = await self._seed_via_phrase_bridges(
                    query_embedding,
                    top_k_phrases=self._query_phrase_seed_k,
                    seen_ids=fts_ids | {n.id for n in vec_seeds} | {n.id for n in prf_seeds},
                )
                for node in phrase_chunks:
                    phrase_bridge_seeds.append(node)
                    # Score below FTS top but above noise floor so the
                    # reranker still sees them as candidate evidence.
                    fts_scores[node.id] = max(fts_scores.get(node.id, 0.0), 0.07)
            except Exception as exc:
                logger.debug("phrase-bridge seeding failed: %s", exc)

        all_seeds = list(fts_nodes) + vec_seeds + prf_seeds + phrase_bridge_seeds
        diagnostics["fts_seed_count"] = float(len(fts_nodes))
        diagnostics["vector_seed_count"] = float(len(vec_seeds))
        diagnostics["prf_seed_count"] = float(len(prf_seeds))
        diagnostics["phrase_bridge_seed_count"] = float(len(phrase_bridge_seeds))
        timings_ms["vector"] = (time() - stage_t0) * 1000

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
        stage_t0 = time()
        preferred_tables = anchors.preferred_tables
        if preferred_tables:
            existing_ids = {n.id for n in all_seeds}
            for table_name in preferred_tables:
                in_table_count = sum(
                    1 for n in fts_nodes if (n.properties or {}).get("_table_name") == table_name
                )
                if in_table_count >= 3:
                    continue  # table is well-represented; FTS is doing fine

                try:
                    aug_nodes = await self._backend.search_fts(f"{table_name} {query}", limit=5)
                except Exception as exc:
                    logger.warning("table-hint FTS failed (%r): %s", table_name, exc)
                    continue
                for node in aug_nodes:
                    if (node.properties or {}).get("_table_name") != table_name:
                        continue
                    if node.id in existing_ids:
                        fts_scores[node.id] = max(fts_scores.get(node.id, 0.0), 0.96)
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
                    fts_scores[node_id] = max(fts_scores.get(node_id, 0.0), normalised)
        diagnostics["seed_count"] = float(len(all_seeds))
        timings_ms["seed_extra"] = (time() - stage_t0) * 1000

        # Step 3 — shallow graph expansion
        stage_t0 = time()
        graph_reads = GraphReadCache(self._backend)
        graph_stage_t0 = time()
        if self._graph_expansion:
            q_terms = (
                frozenset(t for t in query.lower().split() if len(t) > 1)
                if self._expand_relevance
                else None
            )
            expanded = await self._expander.expand(
                anchors=anchors,
                seed_nodes=all_seeds,
                budget=self._expansion_budget,
                query_terms=q_terms,
                read_cache=graph_reads,
                timings_ms=timings_ms,
            )
        else:
            # Ablation: skip graph expansion — feed only the seeds (as 0-hop
            # ExpandedNodes) so the reranker sees the same lexical/vector
            # candidates minus everything reachable through graph structure.
            from synaptic.extensions.graph_expander import ExpandedNode

            expanded = [
                ExpandedNode(node=n, reason="seed", hops=0, anchor_hit=None) for n in all_seeds
            ]
        timings_ms["expand_graph"] = (time() - graph_stage_t0) * 1000
        diagnostics["expanded_count_before_ppr"] = float(len(expanded))

        # Step 3b — PPR graph discovery. Uses FTS seeds as teleport
        # nodes and walks the graph via PPR to find nodes reachable
        # through structural paths (PART_OF, CONTAINS, MENTIONS) that
        # neither FTS nor vector search found. Discovered nodes are
        # added to the expanded set with a graph-based score.
        ppr_stage_t0 = time()
        ppr_seed_scores = _bounded_ppr_seed_scores(fts_scores, k)
        diagnostics["ppr_seed_cap"] = float(_ppr_seed_limit(k))
        diagnostics["ppr_seed_count"] = float(len(ppr_seed_scores))
        diagnostics["ppr_result_count"] = 0.0
        diagnostics["ppr_missing_count"] = 0.0
        diagnostics["ppr_added_count"] = 0.0
        if ppr_seed_scores and self._graph_expansion:
            try:
                ppr_timings: dict[str, float] = {}
                ppr_results = await personalized_pagerank(
                    self._backend,
                    ppr_seed_scores,
                    damping=0.85,
                    # Candidate discovery only needs stable top neighbours,
                    # not high-precision stationary probabilities.
                    max_iter=20,
                    tol=1e-5,
                    # GraphExpander has already pulled 1-hop neighbours.
                    # For discovery, one PPR materialisation layer is enough
                    # to rank those direct relation targets without reading
                    # every neighbour's own edge list.
                    bfs_depth=1,
                    light_edges=True,
                    top_k=k * _PPR_RESULT_MULTIPLIER,
                    read_cache=graph_reads,
                    timings_ms=ppr_timings,
                )
                for key, value in ppr_timings.items():
                    timings_ms[f"expand_ppr_{key}"] = value
                from synaptic.extensions.graph_expander import ExpandedNode

                expanded_ids = {e.node.id for e in expanded}
                diagnostics["ppr_result_count"] = float(len(ppr_results))
                ppr_fetch_t0 = time()
                missing_node_ids = [
                    node_id for node_id, _ppr_score in ppr_results if node_id not in expanded_ids
                ]
                diagnostics["ppr_missing_count"] = float(len(missing_node_ids))
                ppr_nodes = await graph_reads.get_nodes(missing_node_ids)
                timings_ms["expand_ppr_fetch"] = (time() - ppr_fetch_t0) * 1000
                ppr_added = 0
                for node_id, ppr_score in ppr_results:
                    if node_id not in expanded_ids:
                        node = ppr_nodes.get(node_id)
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
                            ppr_added += 1
                diagnostics["ppr_added_count"] = float(ppr_added)
            except Exception:
                pass  # PPR failure is non-fatal
        timings_ms["expand_ppr"] = (time() - ppr_stage_t0) * 1000
        timings_ms["expand"] = (time() - stage_t0) * 1000
        diagnostics["expanded_count"] = float(len(expanded))

        # Step 4 — hybrid reranking
        stage_t0 = time()
        anchor_category_set = set(anchors.categories)
        tilt = (
            self._tilt_weights(anchors, fts_nodes)
            if (self._query_tilt_enabled or self._adaptive)
            else None
        )
        scored = self._reranker.rerank(
            expanded=expanded,
            fts_scores=fts_scores,
            query_embedding=query_embedding,
            anchor_categories=anchor_category_set,
            weights=tilt,
        )
        diagnostics["scored_count"] = float(len(scored))

        # Lazy-load per-corpus calibration once (auto-disable reranker
        # on FTS-already-strong corpora; see calibration.py).
        if not self._calibration_loaded:
            try:
                from synaptic.extensions.calibration import read_calibration

                cal = await read_calibration(self._backend)
                if cal is not None:
                    self._calibrated_blend = cal.rerank_blend
                    logger.info(
                        "calibration loaded: blend=%s (sample_mrr=%.3f)",
                        cal.rerank_blend,
                        cal.sample_mrr,
                    )
            except Exception:
                pass
            self._calibration_loaded = True
        active_blend = (
            self._calibrated_blend if self._calibrated_blend is not None else self._rerank_blend
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
                i
                for i, s in enumerate(top_candidates)
                if not (s.node.properties or {}).get("_table_name")
            ]
            if rerank_indices:
                documents = [
                    f"{top_candidates[i].node.title}\n{top_candidates[i].node.content[:400]}"
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
                    discriminator = self._rerank_discriminator(
                        rerank_scores, self._rerank_std_deadzone
                    )
                    effective_blend = active_blend * discriminator
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
        timings_ms["rerank"] = (time() - stage_t0) * 1000

        # Step 5 — evidence aggregation with diversity
        stage_t0 = time()
        aggregate_pool_limit = (
            _aggregate_candidate_pool_limit(k)
            if self._aggregate_candidate_pool_limit is None
            else max(0, self._aggregate_candidate_pool_limit)
        )
        evidence = self._aggregator.aggregate(
            scored=scored,
            k=k,
            per_document_cap=per_document_cap,
            anchor_categories=anchor_category_set,
            candidate_pool_limit=aggregate_pool_limit,
        )
        diagnostics["aggregate_pool_limit"] = float(aggregate_pool_limit)
        diagnostics["evidence_count"] = float(len(evidence))
        timings_ms["aggregate"] = (time() - stage_t0) * 1000

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
            timings_ms=timings_ms,
            diagnostics=diagnostics,
            sub_queries=sub_queries,
        )

    async def _seed_via_phrase_bridges(
        self,
        query_embedding: list[float],
        *,
        top_k_phrases: int,
        seen_ids: set[str],
        max_chunks_per_phrase: int = 4,
    ) -> list:
        """Dense-match the query against entity hub nodes and return
        chunks linked to the top matches.

        The hub layer is made of ``NodeKind.ENTITY`` nodes tagged either
        ``_phrase`` (built at ingest by EntityLinker / PhraseExtractor)
        or ``_openie_entity`` (built by the LLM OpenIE pass). Hubs are
        shared across documents — the same entity mentioned in two
        unrelated chunks creates a bridge between them via CONTAINS /
        MENTIONS edges. Lexical FTS misses paraphrased entity mentions
        in the query; dense matching at the hub level recovers them.

        Args:
            query_embedding: Pre-computed query vector.
            top_k_phrases: How many entity hubs to walk from.
            seen_ids: Node IDs already in the seed set — skip when
                discovering chunks to avoid double-counting.
            max_chunks_per_phrase: Cap on the chunks pulled from each
                matched hub. A very high-DF phrase ("therefore") that
                slipped past min/max-DF filtering could otherwise flood
                the seed pool.

        Returns:
            A list of chunk-level :class:`Node` objects suitable for
            inclusion in the seed pool. Entity hub nodes themselves are
            *not* returned — they are bridges, not evidence.
        """
        # 1) Pull phrase/OpenIE hubs with non-empty embeddings — cached.
        # The cache stores (nodes, matrix, l2_norms) so per-query work
        # is a single numpy matmul + argpartition. Pure-Python cosine
        # over 10k phrases × 2560-dim was ~10 s/query on MuSiQue,
        # making the bridge dominate runtime — measured before this
        # optimisation. With numpy it drops to ~1 ms.
        try:
            import numpy as _np
        except ImportError:
            _np = None  # falls back to pure-Python loop below

        if not self._phrase_cache_loaded:
            from synaptic.models import NodeKind as _NK

            phrase_nodes_raw = await self._backend.list_nodes(kind=_NK.ENTITY, limit=100_000)
            nodes: list = []
            matrix_rows: list[list[float]] = []
            for n in phrase_nodes_raw:
                if not n.embedding:
                    continue
                tags = set(n.tags or [])
                if "_phrase" not in tags and "_openie_entity" not in tags:
                    continue
                nodes.append(n)
                matrix_rows.append(n.embedding)
            if not nodes:
                self._phrase_cache = []
                self._phrase_cache_loaded = True
            else:
                if _np is not None:
                    mat = _np.asarray(matrix_rows, dtype=_np.float32)
                    norms = _np.linalg.norm(mat, axis=1)
                    safe = norms > 0
                    mat = mat[safe]
                    norms = norms[safe]
                    nodes = [n for n, ok in zip(nodes, safe.tolist(), strict=False) if ok]
                    # Store the numpy-ready form for fast per-query reuse.
                    self._phrase_cache = (nodes, mat, norms)
                else:
                    pure: list[tuple[object, list[float], float]] = []
                    for n, vec in zip(nodes, matrix_rows, strict=False):
                        norm = sum(x * x for x in vec) ** 0.5
                        if norm == 0:
                            continue
                        pure.append((n, list(vec), norm))
                    self._phrase_cache = pure
                self._phrase_cache_loaded = True

        cache = self._phrase_cache
        if not cache:
            return []

        if _np is not None and isinstance(cache, tuple):
            nodes_cached, mat, norms = cache
            if mat.shape[1] != len(query_embedding):
                return []
            q = _np.asarray(query_embedding, dtype=_np.float32)
            q_norm = float(_np.linalg.norm(q))
            if q_norm == 0:
                return []
            # cosine = (M · q) / (||M_i|| * ||q||)
            scores = (mat @ q) / (norms * q_norm)
            k = min(top_k_phrases, scores.shape[0])
            idx = _np.argpartition(-scores, k - 1)[:k]
            idx = idx[_np.argsort(-scores[idx])]
            top_phrases = [nodes_cached[int(i)] for i in idx]
        else:
            # Pure-python fallback for environments without numpy.
            if not cache:
                return []
            cached_dim = len(cache[0][1])
            if cached_dim != len(query_embedding):
                return []
            q_norm = sum(x * x for x in query_embedding) ** 0.5
            if q_norm == 0:
                return []
            scored_phrases: list[tuple[float, object]] = []
            for node, p_emb, p_norm in cache:
                dot = sum(a * b for a, b in zip(query_embedding, p_emb, strict=False))
                cos = dot / (q_norm * p_norm)
                scored_phrases.append((cos, node))
            scored_phrases.sort(key=lambda t: t[0], reverse=True)
            top_phrases = [p for _, p in scored_phrases[:top_k_phrases]]
        if not top_phrases:
            return []

        # 3) For each matched phrase, follow incoming MENTIONS /
        # CONTAINS edges back to the chunks that reference it.
        from synaptic.models import EdgeKind as _EK

        bridge_chunks: list = []
        bridge_seen: set[str] = set(seen_ids)
        for phrase in top_phrases:
            incoming = await self._backend.get_edges(phrase.id, direction="incoming")
            count_for_phrase = 0
            for edge in incoming:
                if edge.kind not in (_EK.CONTAINS, _EK.MENTIONS):
                    continue
                if edge.source_id in bridge_seen:
                    continue
                chunk = await self._backend.get_node(edge.source_id)
                if chunk is None:
                    continue
                bridge_chunks.append(chunk)
                bridge_seen.add(chunk.id)
                count_for_phrase += 1
                if count_for_phrase >= max_chunks_per_phrase:
                    break
        return bridge_chunks
