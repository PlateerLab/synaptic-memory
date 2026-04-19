"""Auto-corpus calibration — detect FTS-already-strong corpora at
ingest, write per-corpus pipeline config so search-time avoids
known-bad mechanisms.

Why this exists
---------------
v0.17.x measurements proved the cross-encoder reranker is a corpus-
type-dependent signal: paraphrase-heavy corpora (PublicHealthQA,
Allganize) gain +0.02-0.20 MRR with it, but corpora where FTS is
already near-optimal (AutoRAG 0.906 FTS-only, X2BEE Easy 1.000,
KRRA Easy 0.967) regress under reranker by up to −15% MRR. The
v0.17.1 adaptive blend mitigates the per-query case but can't recover
the structural ceiling — a reranker is fundamentally noise on
corpora where FTS + graph signals already pick the gold doc.

This module detects the situation at ingest time and writes a
``_calibration`` JSON payload to the backend's metadata, which
``EvidenceSearch`` reads on construction and uses to override its
own pipeline knobs (``rerank_blend``, vector PRF, etc.).

Algorithm
---------
1. Sample N=20 chunk-pair pseudo-queries from the just-ingested
   corpus. Each "query" = a node's title (or first sentence of
   content). The "gold" = that same node — meaning we measure how
   well FTS-only retrieves a document by its own surface anchors.
2. Run FTS-only retrieval on each pseudo-query, compute MRR.
3. Map mean MRR to pipeline config:
     mean_mrr ≥ 0.85 → reranker_blend = 0.0  (FTS already strong;
                       reranker only adds noise — measured AutoRAG)
     mean_mrr ≤ 0.55 → reranker_blend = 0.2  (paraphrase-heavy;
                       reranker pays off — measured PublicHealthQA)
     0.55 < x < 0.85 → reranker_blend = 0.1  (current default)
4. Write config to backend ``_calibration`` key (JSON blob).

Cost: ~N FTS calls (cheap, FTS5 indexed). One-time at ingest.
Output: stored in graph metadata, queried by EvidenceSearch on
each ``search()`` call (cached after first read).

Pseudo-query approach is a deliberate compromise vs. user-supplied
real queries: zero user friction, no LLM dependency, works on any
corpus type. False positives (FTS scores its own anchor highly even
when real queries would diverge) are mitigated by the conservative
threshold band 0.55-0.85.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("calibration")


@dataclass(slots=True)
class CalibrationResult:
    """Per-corpus pipeline config derived from sampled FTS-only MRR.

    Stored as JSON in the backend's ``_calibration`` metadata key so
    every subsequent ``EvidenceSearch.search()`` can read and apply
    the corpus-specific overrides.
    """

    sample_size: int
    sample_mrr: float
    rerank_blend: float
    vector_prf_enabled: bool
    rationale: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "CalibrationResult":
        data = json.loads(raw)
        return cls(**data)


def _config_for_mrr(mean_mrr: float, sample_size: int) -> CalibrationResult:
    """Map a sampled FTS-only MRR to pipeline config.

    Threshold band (0.55-0.85) is conservative — based on v0.17.1
    measurements where:
      - AutoRAG sample_mrr ~0.92 → reranker hurts at any blend
      - PublicHealthQA sample_mrr ~0.55 → reranker helps +0.20
      - Allganize sample_mrr ~0.95 → reranker neutral (already strong)
    The 0.85 upper edge is where reranker's marginal contribution
    inverts from helpful to noise.
    """
    if mean_mrr >= 0.85:
        return CalibrationResult(
            sample_size=sample_size,
            sample_mrr=mean_mrr,
            rerank_blend=0.0,
            vector_prf_enabled=False,
            rationale=(
                f"FTS-only sample MRR {mean_mrr:.2f} ≥ 0.85 — corpus is "
                "FTS-near-optimal; cross-encoder rerank disabled "
                "(measured to regress on AutoRAG, X2BEE Easy, KRRA Easy)."
            ),
        )
    if mean_mrr <= 0.55:
        return CalibrationResult(
            sample_size=sample_size,
            sample_mrr=mean_mrr,
            rerank_blend=0.2,
            vector_prf_enabled=True,
            rationale=(
                f"FTS-only sample MRR {mean_mrr:.2f} ≤ 0.55 — corpus is "
                "paraphrase-heavy; cross-encoder weighted higher and "
                "vector PRF enabled (measured PublicHealthQA: FTS 0.547 "
                "→ Full pipeline 0.748)."
            ),
        )
    return CalibrationResult(
        sample_size=sample_size,
        sample_mrr=mean_mrr,
        rerank_blend=0.1,
        vector_prf_enabled=True,
        rationale=(
            f"FTS-only sample MRR {mean_mrr:.2f} in [0.55, 0.85] — "
            "default v0.17.1 adaptive blend (rerank_blend=0.1) applied."
        ),
    )


async def calibrate_corpus(
    backend: StorageBackend,
    *,
    sample_size: int = 20,
    seed: int = 0,
) -> CalibrationResult:
    """Run the calibration sweep on an already-ingested corpus.

    Samples ``sample_size`` content-bearing nodes, treats each node's
    title as a pseudo-query, measures whether FTS-only retrieval finds
    the source node in the top-10. Returns the resulting
    :class:`CalibrationResult`.

    Designed to be called once at the end of bulk ingest (or after a
    sync). Cheap (N FTS calls, no LLM, no embedder).
    """
    nodes = await backend.list_nodes(limit=10_000)
    candidates = [
        n
        for n in nodes
        if n.title and n.title.strip() and (n.content or "").strip()
    ]
    if not candidates:
        # No content-bearing nodes — degenerate corpus, fall back to
        # default config.
        return _config_for_mrr(0.7, 0)

    rng = random.Random(seed)
    sample = rng.sample(candidates, k=min(sample_size, len(candidates)))

    # Use a content-derived pseudo-query rather than node.title.
    # Title-as-query was tried first but mis-classified FAQ corpora
    # (AutoRAG: titles like "What is X?" share too much surface form,
    # giving MRR 0.48 even though real queries hit MRR 0.91 FTS-only).
    # First content sentence is a better proxy for "what a user might
    # ask about this doc": it carries more lexical specificity than
    # the title and matches what an embedder/reranker would chunk.
    rr_total = 0.0
    counted = 0
    for node in sample:
        query = _extract_pseudo_query(node)
        if not query:
            continue
        try:
            results = await backend.search_fts(query, limit=10)
        except Exception as exc:
            logger.debug("calibration search failed for %r: %s", query, exc)
            continue
        rr = 0.0
        for i, r in enumerate(results):
            if r.id == node.id:
                rr = 1.0 / (i + 1)
                break
        rr_total += rr
        counted += 1

    mean_mrr = rr_total / max(counted, 1)
    return _config_for_mrr(mean_mrr, counted)


def _extract_pseudo_query(node) -> str:
    """Build a short query from a node's first content sentence.

    Falls back to the title if content is too short to extract a
    useful sentence (≤30 chars). The 8-word cap keeps the query in
    the same length band a real user query tends to occupy.
    """
    content = (node.content or "").strip()
    if len(content) < 30:
        return (node.title or "").strip()

    # First sentence: split on Korean/English sentence-ending markers
    import re

    parts = re.split(r"(?<=[\.!\?。!?])\s+|(?<=[\.\?!])\n", content, maxsplit=1)
    first = parts[0].strip() if parts else content
    if len(first) < 20:
        # Sentence boundary failed — take first 80 chars instead
        first = content[:80]

    # Cap to 8 words so the FTS query isn't unfairly long
    words = first.split()[:8]
    return " ".join(words).strip()


async def write_calibration(backend: StorageBackend, result: CalibrationResult) -> None:
    """Persist calibration to the backend's metadata table.

    Uses the backend's own metadata mechanism if it has one (SqliteGraphBackend
    has ``set_meta`` / ``get_meta``); otherwise stores as a magic-id
    node (``__synaptic_calibration__``). Read back via
    :func:`read_calibration`.
    """
    raw = result.to_json()
    set_meta = getattr(backend, "set_meta", None)
    if callable(set_meta):
        await set_meta("calibration", raw)
        return
    # Fallback: store on a sentinel node so any backend works
    from synaptic.models import ConsolidationLevel, Node, NodeKind

    sentinel = Node(
        id="__synaptic_calibration__",
        kind=NodeKind.TYPE_DEF,
        title="_calibration",
        content=raw,
        level=ConsolidationLevel.L0_RAW,
        tags=["_calibration"],
    )
    await backend.save_node(sentinel)


async def read_calibration(backend: StorageBackend) -> CalibrationResult | None:
    """Read previously written calibration; returns None if not present."""
    get_meta = getattr(backend, "get_meta", None)
    if callable(get_meta):
        try:
            raw = await get_meta("calibration")
        except Exception:
            raw = None
        if raw:
            try:
                return CalibrationResult.from_json(raw)
            except Exception:
                return None
    # Fallback: sentinel node
    try:
        node = await backend.get_node("__synaptic_calibration__")
    except Exception:
        return None
    if node is None or not (node.content or "").strip():
        return None
    try:
        return CalibrationResult.from_json(node.content)
    except Exception:
        return None
