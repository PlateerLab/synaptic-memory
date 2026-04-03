"""ColBERT late-interaction reranking using per-token embeddings.

MaxSim scoring: for each query token, find the max similarity with any
document token, then sum across all query tokens.

  score = sum(max(q_i · d_j for j in doc_tokens) for i in query_tokens)

This provides token-level precision without the cost of cross-encoder
inference — 10-100x faster than LLM reranking.

Only used when HybridEmbedding.colbert vectors are available.
"""

from __future__ import annotations

import math

from synaptic.models import ActivatedNode


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


class ColBERTReranker:
    """ColBERT MaxSim reranker — token-level precision matching.

    Example::

        reranker = ColBERTReranker()
        reranked = reranker.rerank(query_colbert, candidates, top_k=10)

    Where query_colbert is list[list[float]] (per-token embeddings from BGE-M3),
    and candidates have colbert vectors stored in properties.
    """

    __slots__ = ()

    def rerank(
        self,
        query_colbert: list[list[float]],
        candidates: list[tuple[ActivatedNode, list[list[float]]]],
        *,
        top_k: int = 20,
    ) -> list[ActivatedNode]:
        """Rerank candidates using ColBERT MaxSim scoring.

        Args:
            query_colbert: Per-token embeddings for the query.
            candidates: [(ActivatedNode, doc_colbert_vectors), ...].
            top_k: Number of top candidates to return.

        Returns:
            Reranked list of ActivatedNode with updated resonance scores.
        """
        if not query_colbert or not candidates:
            return [c[0] for c in candidates[:top_k]]

        scored: list[tuple[ActivatedNode, float]] = []

        for activated, doc_colbert in candidates:
            if not doc_colbert:
                # No ColBERT vectors — keep original score
                scored.append((activated, activated.resonance))
                continue

            # MaxSim: for each query token, find best matching doc token
            maxsim_score = 0.0
            for q_vec in query_colbert:
                best_sim = max((_cosine(q_vec, d_vec) for d_vec in doc_colbert), default=0.0)
                maxsim_score += best_sim

            # Normalize by query length
            normalized = maxsim_score / len(query_colbert) if query_colbert else 0.0

            # Blend with original resonance (70% ColBERT + 30% original)
            blended = 0.7 * normalized + 0.3 * activated.resonance

            scored.append((activated, blended))

        # Sort by blended score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update resonance scores
        result: list[ActivatedNode] = []
        for activated, new_score in scored[:top_k]:
            result.append(
                ActivatedNode(
                    node=activated.node,
                    activation=activated.activation,
                    resonance=new_score,
                    path=activated.path,
                )
            )

        return result
