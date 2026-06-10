"""Query routing for ``graph.ask()`` — tier-0 deterministic signals.

``graph.ask()`` is the honest-routing entry point: answer with the cheap
single-shot path (one retrieval + one synthesis call) when that is
enough, and promote to the multi-turn agent loop only where the agent's
measured value lives — structured aggregation / filter / FK-join
questions over typed table nodes (single-shot MRR 0.0 → agent 91 %
solved on assort Hard).

This module is the *deterministic* half of that decision and makes zero
LLM calls: :func:`decide_route` inspects the query text plus corpus
shape and returns a :class:`RouteDecision`. Anything tier-0 does not
promote falls through to ``ask()``'s tier-1 sufficiency gate (an LLM
judge over the cheap answer), so a tier-0 miss costs one judged
single-shot attempt — it never silently picks the wrong final path.

Conservative default — tier-0 signal set pending E2 validation
(PLAN-v0.29 §E2). Only the high-confidence positive signal is wired:
structured-operation lexis (enumeration / aggregation / comparison
filter / temporal filter) **and** the corpus actually has typed table
nodes. Score-based signals (``search_scores``) and anchor signals are
recorded in ``RouteDecision.signals`` for diagnostics but do NOT
influence the route until the E2 AUC harness validates them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from synaptic.agent_loop import _is_enumeration_query

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend


# Single-shot synthesis prompt for ``ask()``'s cheap path. Ported verbatim
# from ``examples/ablation/rag_vs_agent_answer.py`` (``_RAG_SYSTEM``) so the
# product path and the measured naive-RAG arm share one and the same prompt.
RAG_SYNTHESIS_SYSTEM = (
    "Answer the question using ONLY the provided context. Be concise and factual. "
    "If the context lacks the answer, say what you can from it."
)


# --- Structured-operation lexis -------------------------------------
#
# Substring tokens matched against the lowercased query. Deliberately
# conservative: these are the phrasings of the assort/X2BEE Hard
# structured types (aggregation*, filter, temporal, exhaustive, ...)
# where single-shot retrieval is measured at ~0. Tokens that overfire on
# the Easy keyword-lookup sets are excluded on purpose — a tier-0 miss
# is recoverable by the tier-1 gate, a tier-0 false promotion is not.

_AGGREGATION_TOKENS = (
    # Korean superlative / ranking
    "가장 많",
    "가장 적",
    "가장 높",
    "가장 낮",
    "가장 빠",
    "가장 늦",
    "1위",
    "최대",
    "최소",
    "최저",
    "최고",
    # Korean count / metric nouns
    "몇 개",
    "몇 건",
    "몇 명",
    "몇 종류",
    "개수",
    "건수",
    "총합",
    "합계",
    "평균",
    "비율",
    # English
    "how many",
    "count of",
    "sum of",
    "average",
    "fewest",
    "highest",
    "lowest",
)

# "top 3" / "TOP5" — anchored on the digit so "laptop" never matches.
_TOP_N_RE = re.compile(r"\btop\s*\d+")

# Comparison filters only count when the query carries a number to
# compare against ("9만원 이상") — bare "이상" appears in plain prose.
_COMPARISON_TOKENS = (
    "이상",
    "이하",
    "초과",
    "미만",
    "보다 큰",
    "보다 작은",
    "보다 비싼",
    "보다 싼",
    "or more",
    "or less",
    "at least",
    "at most",
    "more than",
    "less than",
)

_DIGIT_RE = re.compile(r"\d")

# Year/month filter phrasing ("2024년 11월에 방송된 상품"). No \b — Korean
# counters sit flush against the following particle (\b never fires there).
_TEMPORAL_RE = re.compile(r"\d{4}\s*년|\d{1,2}\s*월")


@dataclass(slots=True)
class RouteDecision:
    """Outcome of one tier-0 :func:`decide_route` call.

    Attributes:
        route: ``"agent"`` to go straight to the multi-turn agent loop,
            ``"single_shot"`` to try the cheap path first (the tier-1
            sufficiency gate in ``ask()`` may still escalate it).
        reasons: Human-readable explanation of the decision, one entry
            per contributing factor — surfaced in ``AskResult`` so the
            route is auditable.
        signals: Every signal evaluated (fired or not), for diagnostics
            and the E2 AUC harness. Keys are stable strings.
    """

    route: Literal["single_shot", "agent"]
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AskResult:
    """Final output of one :meth:`SynapticGraph.ask` call.

    Attributes:
        answer: The synthesised answer text (never empty when the LLM
            produced any text on either path — an escalated agent run
            that comes back blank falls back to the cheap synthesis).
        route: The path that produced ``answer`` — ``"single_shot"``
            (cheap retrieval + one synthesis call) or ``"agent"``
            (multi-turn loop, whether tier-0 routed or tier-1 escalated).
        route_reasons: Why this route was taken (tier-0 reasons plus a
            tier-1 entry when the sufficiency judge escalated).
        escalated: True only when the cheap path ran first AND the
            tier-1 sufficiency gate promoted the query to the agent
            loop. Tier-0 direct-to-agent and ``mode="agent"`` are not
            escalations.
        prompt_tokens: Total prompt tokens across every LLM call this
            ask made (cheap synthesis + sufficiency judge + the full
            agent loop when escalated). 0 when responses carry no
            ``usage`` (fail-open).
        completion_tokens: Same accumulation for completion tokens.
        evidence: The nodes backing the answer — search results on the
            cheap path, the agent's resolved nodes on the agent path.
    """

    answer: str
    route: Literal["single_shot", "agent"]
    route_reasons: list[str] = field(default_factory=list)
    escalated: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    evidence: list[Any] = field(default_factory=list)


async def corpus_has_table_nodes(backend: StorageBackend, *, sample_limit: int = 5000) -> bool:
    """True when the corpus contains table-ingested rows (typed nodes).

    Samples up to ``sample_limit`` ENTITY nodes and looks for the
    ``_table_name`` property the table ingester stamps on every row —
    the same sniff ``run_agent_loop`` uses for composite-ID synthesis.
    A false negative only costs tier-0 promotion (the tier-1 gate still
    applies), so sampling is acceptable.
    """
    from synaptic.models import NodeKind

    nodes = await backend.list_nodes(kind=NodeKind.ENTITY, limit=sample_limit)
    return any((n.properties or {}).get("_table_name") for n in nodes)


def decide_route(
    query: str,
    *,
    has_table_nodes: bool,
    anchor: Any | None = None,
    search_scores: list[float] | None = None,
) -> RouteDecision:
    """Tier-0 deterministic route decision — pure function, zero LLM calls.

    Conservative default pending E2 validation (PLAN-v0.29 §E2): the
    only promoting signal is structured-operation lexis (enumeration /
    aggregation / comparison filter / temporal filter) over a corpus
    that has typed table nodes — the measured agent-required class
    (single-shot 0.0 → agent 91 % on assort Hard). Every other query
    stays ``single_shot`` and is delegated to the tier-1 sufficiency
    gate inside ``ask()``.

    Args:
        query: The user question.
        has_table_nodes: Whether the corpus contains table-ingested
            typed nodes (see :func:`corpus_has_table_nodes`).
        anchor: Optional pre-computed query anchor — recorded in
            ``signals`` only; not yet a routing input.
        search_scores: Optional ranked top-k scores from a prior search —
            recorded in ``signals`` only (top1 / margin); not yet a
            routing input.

    Returns:
        :class:`RouteDecision` with the route, the contributing
        reasons, and every evaluated signal.
    """
    q = (query or "").lower().strip()

    enumeration = _is_enumeration_query(q)
    aggregation = any(tok in q for tok in _AGGREGATION_TOKENS) or bool(_TOP_N_RE.search(q))
    comparison = bool(_DIGIT_RE.search(q)) and any(tok in q for tok in _COMPARISON_TOKENS)
    temporal = bool(_TEMPORAL_RE.search(q))
    structured_lexis = enumeration or aggregation or comparison or temporal

    signals: dict[str, Any] = {
        "enumeration": enumeration,
        "aggregation": aggregation,
        "comparison_filter": comparison,
        "temporal_filter": temporal,
        "structured_lexis": structured_lexis,
        "has_table_nodes": has_table_nodes,
        "anchor_present": anchor is not None,
    }
    if search_scores:
        # Diagnostic only — score signals are embedder-dependent
        # (deadzone precedent) and stay un-routed until E2 validates them.
        signals["top1_score"] = search_scores[0]
        if len(search_scores) >= 2:
            signals["score_margin"] = search_scores[0] - search_scores[1]

    if structured_lexis and has_table_nodes:
        fired = [
            name
            for name, on in (
                ("enumeration", enumeration),
                ("aggregation", aggregation),
                ("comparison-filter", comparison),
                ("temporal-filter", temporal),
            )
            if on
        ]
        return RouteDecision(
            route="agent",
            reasons=[
                "structured-operation lexis (" + ", ".join(fired) + ") over a corpus "
                "with typed table nodes — single-shot retrieval cannot "
                "aggregate/filter rows"
            ],
            signals=signals,
        )

    if structured_lexis:
        reason = (
            "structured lexis present but the corpus has no typed table nodes — "
            "staying single-shot (tier-1 gate may still escalate)"
        )
    else:
        reason = (
            "no high-confidence tier-0 signal — single-shot first, "
            "tier-1 sufficiency gate decides escalation"
        )
    return RouteDecision(route="single_shot", reasons=[reason], signals=signals)
