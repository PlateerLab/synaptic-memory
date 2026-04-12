"""OntologyClassifier — LLM-free NodeKind classification via embedding similarity.

Maps free-form category labels (folder names, tags, taxonomy entries) to
``NodeKind`` values using a bring-your-own embedder — no torch, no
transformers, no model download inside this library. The user supplies
any object implementing the ``EmbeddingProvider`` protocol: an Ollama
server, an OpenAI-compatible endpoint, a local TEI instance, or a mock
for tests.

## Why this exists

Generation-3 GraphRAG systems (LinearRAG, Practical GraphRAG, LightRAG,
MiniRAG) have shifted toward **relation-free graphs with encoder-based
extraction** — the LLM is removed from the indexing path and only kept
for query-time generation. ``DomainProfile.ontology_hints`` is one of
the last manual knobs left: for every new corpus the user has to map
category labels like ``"규정 및 지침"`` to ``NodeKind.RULE`` by hand, or
call an LLM per label.

This classifier replaces that step with a one-shot embedding cosine
match against a small table of NodeKind descriptions. For ~10 category
labels the entire pass is ~100 ms on a remote embedder and zero dollars
on an LLM.

## How it works

1. At construction time we embed a short human-written *description*
   for each NodeKind (e.g. ``NodeKind.RULE`` → "규정, 법령, 지침, policy,
   rule, regulation"). Descriptions are deliberately multilingual so the
   same classifier handles Korean and English corpora without locale
   switching.
2. ``classify(label)`` embeds the query label and returns the NodeKind
   whose description vector has the highest cosine similarity — but
   only if the score clears ``threshold``, otherwise it returns ``None``
   so the caller can decide whether to fall back to an LLM tier.
3. ``classify_many(labels)`` batches the query through ``embed_batch``
   so classifying a full category list costs one round-trip.

## Usage

    from synaptic.backends.memory import MemoryBackend  # noqa
    from synaptic.extensions.embedder import OpenAIEmbeddingProvider
    from synaptic.extensions.ontology_classifier import OntologyClassifier

    embedder = OpenAIEmbeddingProvider(
        api_base="http://localhost:11434/v1",
        model="qwen3-embedding:4b",
    )
    classifier = OntologyClassifier(embedder=embedder)
    await classifier.warm_up()

    hint = await classifier.classify("규정 및 지침")
    # hint == NodeKind.RULE

Because the embedder is injected, the same classifier instance can be
shared across ingestion, profile generation, and post-hoc re-labelling.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from synaptic.models import NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.embedder import EmbeddingProvider

logger = logging.getLogger("ontology-classifier")


# --- NodeKind descriptions ---
#
# Each entry is a short multilingual phrase that *characterises* the
# meaning of the NodeKind in 3-6 canonical terms. The goal is not to be
# exhaustive — embedding models generalise well from a handful of
# prototypes — but to cover the dominant semantic axis so Korean legal
# docs, English research notes, and Japanese incident logs all land in
# the same NodeKind bucket.
#
# Only the NodeKinds that meaningfully classify *documents* are listed.
# Runtime-emitted kinds (TOOL_CALL, SESSION, REASONING) are excluded
# because they never appear as corpus category labels.

DEFAULT_NODE_KIND_DESCRIPTIONS: dict[NodeKind, str] = {
    NodeKind.RULE: (
        "규정, 법령, 지침, 정책, 규칙, 조항, 기준, "
        "policy, rule, regulation, guideline, compliance, standard"
    ),
    NodeKind.DECISION: (
        "계획, 방침, 결정, 의사결정, 전략, 추진안, "
        "plan, decision, strategy, roadmap, proposal, resolution"
    ),
    NodeKind.OBSERVATION: (
        "조사, 평가, 분석, 보고서, 모니터링, 진단, 리뷰, "
        "observation, analysis, report, evaluation, audit, monitoring"
    ),
    NodeKind.OUTCOME: (
        "실적, 성과, 결과, 달성, 매출, 지표, outcome, result, performance, achievement, metric, kpi"
    ),
    NodeKind.CONCEPT: (
        "개념, 용어, 분류, 카테고리, 주제, 정의, "
        "concept, category, topic, definition, taxonomy, term"
    ),
    NodeKind.ARTIFACT: (
        "산출물, 시스템, 도구, 문서, 양식, 템플릿, 코드, "
        "artifact, system, tool, deliverable, template, code"
    ),
    NodeKind.ENTITY: (
        "조직, 인물, 법인, 기업, 부서, 팀, entity, organization, person, company, department, team"
    ),
    NodeKind.LESSON: (
        "교훈, 회고, 학습, 개선점, 레슨런, lesson, retrospective, learning, takeaway, postmortem"
    ),
    NodeKind.TASK: (
        "과제, 업무, 작업, 할일, 액션아이템, task, assignment, work item, action item, ticket"
    ),
}


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length vectors.

    Returns ``0.0`` when either vector is all-zero or the lengths
    differ — the classifier treats this as "no match" so zero vectors
    never accidentally win a comparison.
    """
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


class OntologyClassifier:
    """Embedding-similarity classifier for category label → NodeKind.

    Args:
        embedder: Any object implementing the ``EmbeddingProvider``
            protocol — the classifier calls ``embed`` / ``embed_batch``
            and never touches the underlying model directly. Pass an
            ``OpenAIEmbeddingProvider`` pointed at Ollama for a fully
            local setup.
        descriptions: Override the default NodeKind description table.
            Useful when the built-in prototypes don't cover the target
            domain, or when you want to narrow the classifier to a
            subset of kinds (e.g. ``{NodeKind.RULE: "...", NodeKind.DECISION: "..."}``).
        threshold: Minimum cosine score to accept a classification.
            Labels that don't clear this bar return ``None`` so the
            caller can fall back to another tier. Default ``0.35`` is
            calibrated for ``qwen3-embedding`` / ``bge-m3``; raise it
            to reduce false positives on unclear labels.
    """

    __slots__ = ("_descriptions", "_embedder", "_kind_vectors", "_threshold")

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        descriptions: dict[NodeKind, str] | None = None,
        threshold: float = 0.35,
    ) -> None:
        self._embedder = embedder
        self._descriptions = descriptions or DEFAULT_NODE_KIND_DESCRIPTIONS
        self._threshold = threshold
        self._kind_vectors: dict[NodeKind, list[float]] = {}

    async def warm_up(self) -> None:
        """Pre-compute the description vectors.

        Call this once before the first ``classify`` — it embeds every
        NodeKind description in a single batch so later classifications
        are single-query round-trips. ``classify`` will lazy-warm on
        first use, but doing it explicitly lets errors surface at
        startup instead of in the middle of a profile generation run.
        """
        if self._kind_vectors:
            return

        kinds = list(self._descriptions.keys())
        descs = [self._descriptions[k] for k in kinds]
        vectors = await self._embedder.embed_batch(descs)
        for kind, vec in zip(kinds, vectors):
            if vec:
                self._kind_vectors[kind] = vec
            else:
                logger.warning("ontology-classifier: empty vector for %s — dropped", kind)

        if not self._kind_vectors:
            msg = "ontology-classifier: all description embeddings failed"
            raise RuntimeError(msg)

        logger.info(
            "ontology-classifier: warmed up with %d NodeKind vectors",
            len(self._kind_vectors),
        )

    async def classify(self, label: str) -> NodeKind | None:
        """Return the best-matching NodeKind for ``label``.

        Returns ``None`` when the top cosine score is below
        ``threshold``. The caller is expected to treat ``None`` as "ask
        a smarter tier" rather than as an error — typical flow is
        rule-based → classifier → LLM, and each tier only handles what
        the previous one couldn't.
        """
        if not label or not label.strip():
            return None
        await self.warm_up()

        query_vec = await self._embedder.embed(label)
        if not query_vec:
            logger.warning("ontology-classifier: empty query vector for %r", label)
            return None

        scored = [(kind, _cosine(query_vec, vec)) for kind, vec in self._kind_vectors.items()]
        scored.sort(key=lambda kv: -kv[1])
        top_kind, top_score = scored[0]
        if top_score < self._threshold:
            logger.debug(
                "ontology-classifier: %r below threshold (top=%s score=%.3f)",
                label,
                top_kind,
                top_score,
            )
            return None
        return top_kind

    async def classify_many(
        self,
        labels: list[str],
    ) -> dict[str, NodeKind]:
        """Batch version — one embedding round-trip for all labels.

        Skips labels that fail to clear ``threshold`` so the result dict
        contains only confident matches. Use this when building
        ``DomainProfile.ontology_hints``: the caller merges the result
        with any LLM-supplied hints and writes the final table to TOML.
        """
        filtered = [l for l in labels if l and l.strip()]
        if not filtered:
            return {}
        await self.warm_up()

        label_vectors = await self._embedder.embed_batch(filtered)
        result: dict[str, NodeKind] = {}
        for label, vec in zip(filtered, label_vectors):
            if not vec:
                continue
            scored = [(kind, _cosine(vec, kvec)) for kind, kvec in self._kind_vectors.items()]
            scored.sort(key=lambda kv: -kv[1])
            top_kind, top_score = scored[0]
            if top_score >= self._threshold:
                result[label] = top_kind
        return result

    async def score_all(self, label: str) -> dict[NodeKind, float]:
        """Return the full cosine score table for ``label``.

        Useful for diagnostics, tuning the threshold, or building a
        soft-label UI that shows the top-3 candidates when the
        classifier is uncertain.
        """
        if not label or not label.strip():
            return {}
        await self.warm_up()
        query_vec = await self._embedder.embed(label)
        if not query_vec:
            return {}
        return {kind: _cosine(query_vec, vec) for kind, vec in self._kind_vectors.items()}
