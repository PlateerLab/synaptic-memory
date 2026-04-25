"""Unified validation scorer — single source of truth for ship/no-ship.

Why this exists
===============
Per-corpus benchmarks (KRRA Hard / assort Hard / X2BEE / MuSiQue / ...)
each stress 1-2 dimensions of GraphRAG quality. Iterating on a feature
that improves one dimension while regressing another is hard to detect
when each bench reports a single MRR / hit number. Concrete recent
example:

    v0.19 → v0.20 (Phase A: pagination + adaptive enumeration budget)
       enumeration queries: improved (h012 used 12 turns vs prior 5)
       broad-topical queries: regressed (deterministic prompt-shift
                              rerouted some queries down wrong paths)
       net per-bench: KRRA Hard -3, assort Hard -1
       net by dimension: enumeration +2, broad-topical -6

Without per-dimension scoring, the regression is invisible until
multiple benches return -N and we have no story for what changed.

This module classifies each query by (domain, language, hop_count,
recall_type, structured_pct, enumeration), then aggregates results
into a weighted UnifiedScore with per-dimension breakdown. The score
is the single number every Phase decision should compete against.

CLI
===
    uv run python eval/unified.py
        Score the latest agent run results across all benches.

    uv run python eval/unified.py --compare prev_unified.json
        Diff against a saved baseline; show per-dimension trend +
        ship recommendation.

    uv run python eval/unified.py --weights config.toml
        Override default dimension weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).parent
QUERIES_DIR = EVAL_DIR / "data" / "queries"
RESULTS_DIR = EVAL_DIR / "baselines"


# --- Dimension schema ----------------------------------------------


class Language(StrEnum):
    KO = "ko"
    EN = "en"
    MIXED = "mixed"


class RecallType(StrEnum):
    SINGLE_LOOKUP = "single"  # one specific item
    TOP_N = "top_n"  # ranked few
    ENUMERATION = "enumeration"  # complete set ("모두/전체/list all")
    SUMMARIZATION = "summarization"  # synthesize across many
    MULTI_HOP = "multi_hop"  # chain across entities


@dataclass(slots=True)
class QueryDimensions:
    """Tags applied to each query for cross-cutting evaluation.

    Mostly inferred from query text + GT shape (number of relevant
    docs, structure of corpus). Manual override in query JSON via
    ``dimensions: {...}``.
    """

    domain: str = ""  # "krra", "assort", "x2bee", "musique", ...
    language: str = Language.KO.value  # ko / en / mixed
    hop_count: int = 1  # 1 = direct lookup, 2+ = chain
    recall_type: str = RecallType.SINGLE_LOOKUP.value
    structured_pct: float = 0.0  # 0=text-only, 1=table-only
    enumeration: bool = False  # explicit "all X" requested
    cross_domain: bool = False  # requires combining 2+ domains
    cross_language: bool = False  # query language ≠ doc language

    def tags(self) -> list[str]:
        """Flat tag list for grouping/filtering."""
        out = [
            f"domain:{self.domain}",
            f"lang:{self.language}",
            f"hop:{self.hop_count}",
            f"recall:{self.recall_type}",
        ]
        if self.enumeration:
            out.append("enumeration")
        if self.cross_domain:
            out.append("cross_domain")
        if self.cross_language:
            out.append("cross_language")
        if self.structured_pct >= 0.5:
            out.append("structured")
        return out


# --- Auto-classification heuristics --------------------------------

# Korean enumeration markers (kept in sync with
# src/synaptic/agent_loop.py:_ENUMERATION_TOKENS so a single query
# can't be classified as enumeration here but missed there).
_ENUMERATION_HINTS = (
    "모두",
    "전체",
    "목록",
    "리스트",
    "전수",
    "list all",
    "every ",
    "all of the ",
    "all the ",
    "show me all",
    "모든",
)

_MULTI_HOP_HINTS = (
    # Possessive / filter chain markers
    "의 ",  # "X의 Y" possessive often signals chain
    " 중 ",  # "X 중 Y" filter then operate
    " 에서 ",  # "X에서 Y" location/source then attribute
    # Korean composition markers (KRRA multihop style)
    "와 ",  # "X와 Y" — pairs / co-occurrence
    "과 ",  # "X과 Y" — same, after consonant
    "수립에서",  # "establishment in" — composition
    "구축과",  # "construction with" — composition
    "관련된",  # "related" — link
    "현황과",  # "status with" — composition
    "교차",  # "intersection" — explicit cross-cutting
    # English connectors
    "and the ",
    "whose ",
    "which has ",
    "combined with",
    "in relation to",
)

# Query file stems whose every query is multi-hop by construction.
# These are explicitly authored as multi-hop benchmarks; the classifier
# heuristics underestimate them because the connector vocabulary is
# academic ("교차" / "수립에서") rather than colloquial ("의/중/에서").
_MULTI_HOP_FILES: frozenset[str] = frozenset(
    {
        "krra_multihop",
        "musique",
        "musique_ans",
        "hotpotqa",
        "hotpotqa_24",
        "2wikimultihop",
    }
)

# Primary corpus language per query file stem. Used to infer
# ``cross_language`` when query language ≠ corpus language. A KRRA
# (pure Korean corpus) query in English IS cross-language; an X2BEE
# query in English is partially cross-language (the corpus mixes
# both); a HotPotQA query in English is NOT cross-language. Without
# this map, the classifier can't distinguish.
_FILE_CORPUS_LANG: dict[str, str] = {
    # Pure Korean corpora
    "krra": "ko",
    "krra_hard": "ko",
    "krra_conversational": "ko",
    "krra_multihop": "ko",
    "krra_graph": "ko",
    "assort": "ko",
    "assort_hard": "ko",
    "assort_conversational": "ko",
    "allganize_rag_ko": "ko",
    "allganize_rag_eval": "ko",
    "publichealthqa_ko": "ko",
    "autorag_retrieval": "ko",
    # Mixed Korean + English corpora
    "x2bee": "mixed",
    "x2bee_hard": "mixed",
    "x2bee_conversational": "mixed",
    # Pure English corpora
    "hotpotqa": "en",
    "hotpotqa_24": "en",
    "musique": "en",
    "musique_ans": "en",
    "2wikimultihop": "en",
    "klue_mrc": "ko",  # KLUE is Korean
}

_SUMMARIZATION_HINTS = (
    "요약",
    "summary",
    "summarize",
    "주요 내용",
    "main points",
    "테마",
    "themes",
)


def _has_korean(s: str) -> bool:
    return any("가" <= ch <= "힯" for ch in s)


def _has_latin(s: str) -> bool:
    return any(("a" <= ch.lower() <= "z") for ch in s)


def classify_query(
    query_text: str,
    *,
    domain: str = "",
    relevant_docs: list[str] | None = None,
    explicit: dict[str, Any] | None = None,
) -> QueryDimensions:
    """Auto-tag a query. ``explicit`` overrides any inferred tag."""
    explicit = explicit or {}
    relevant_docs = relevant_docs or []
    q_lower = query_text.lower().strip()

    # language
    has_ko, has_en = _has_korean(query_text), _has_latin(query_text)
    if has_ko and has_en:
        lang = Language.MIXED.value
    elif has_ko:
        lang = Language.KO.value
    elif has_en:
        lang = Language.EN.value
    else:
        lang = Language.KO.value  # default for ambiguous

    # enumeration — match the exact list used by the agent loop so
    # that an enumeration-classified query is *also* the one that
    # triggers the adaptive turn budget upstream
    is_enum = any(tok in q_lower for tok in _ENUMERATION_HINTS)

    # recall_type — enumeration wins over multi-hop wins over single
    if any(tok in q_lower for tok in _SUMMARIZATION_HINTS):
        recall = RecallType.SUMMARIZATION.value
    elif is_enum:
        recall = RecallType.ENUMERATION.value
    elif any(tok in query_text for tok in _MULTI_HOP_HINTS):
        recall = RecallType.MULTI_HOP.value
    elif len(relevant_docs) >= 4:
        # Many GT docs without explicit "all" marker → still effectively
        # an enumeration / summarization-style query
        recall = RecallType.ENUMERATION.value
    elif len(relevant_docs) >= 2:
        recall = RecallType.TOP_N.value
    else:
        recall = RecallType.SINGLE_LOOKUP.value

    # hop_count — heuristic from connector count
    hop = 1
    for marker in (" 의 ", " 중 ", " whose ", "which has "):
        hop += q_lower.count(marker)
    hop = min(hop, 4)

    # structured_pct — inferred from GT id shape: "table:pk" pattern
    structured = 0.0
    if relevant_docs:
        struct_n = sum(1 for d in relevant_docs if ":" in d and not d.startswith("doc_"))
        structured = struct_n / len(relevant_docs)

    base = QueryDimensions(
        domain=domain,
        language=lang,
        hop_count=hop,
        recall_type=recall,
        structured_pct=structured,
        enumeration=is_enum,
    )

    # Apply explicit overrides last
    for k, v in explicit.items():
        if hasattr(base, k):
            setattr(base, k, v)
    return base


# --- Score computation ---------------------------------------------


@dataclass(slots=True)
class DimensionScore:
    """Aggregate hit-rate over the queries matching a dimension slice."""

    name: str
    n_queries: int = 0
    n_hits: int = 0

    @property
    def hit_rate(self) -> float:
        return self.n_hits / self.n_queries if self.n_queries else 0.0


@dataclass(slots=True)
class UnifiedReport:
    """Output of one scoring run — JSON-serialisable.

    Always contains:
      ``per_dimension``: hit-rate per slice (lang, recall_type, hop, ...)
      ``per_bench``: raw hit-rate per source bench (legacy compatibility)
      ``unified_score``: weighted composite [0.0, 1.0]
      ``query_count``: total queries scored
      ``weights``: the weight vector used
    """

    unified_score: float = 0.0
    query_count: int = 0
    n_hits: int = 0
    per_dimension: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_bench: dict[str, dict[str, Any]] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# Default weights — biased toward Synaptic's product positioning.
# Korean is primary (40%), structured 20%, multi-hop and cross-domain
# track HippoRAG2 / multi-domain trajectory (15% each). Enumeration is
# rare in current corpora so weighted modestly (10%).
DEFAULT_WEIGHTS: dict[str, float] = {
    "lang:ko": 0.30,
    "lang:en": 0.10,
    "lang:mixed": 0.05,
    "recall:multi_hop": 0.15,
    "recall:enumeration": 0.10,
    "structured": 0.10,
    "cross_domain": 0.10,
    "cross_language": 0.10,
}


def _slice_key(dim: QueryDimensions, axis: str) -> str | None:
    """Map a dimension axis name to the matching slice key for ``dim``."""
    if axis.startswith("lang:"):
        return f"lang:{dim.language}" if axis == f"lang:{dim.language}" else None
    if axis.startswith("recall:"):
        return f"recall:{dim.recall_type}" if axis == f"recall:{dim.recall_type}" else None
    if axis == "structured":
        return "structured" if dim.structured_pct >= 0.5 else None
    if axis == "cross_domain":
        return "cross_domain" if dim.cross_domain else None
    if axis == "cross_language":
        return "cross_language" if dim.cross_language else None
    if axis == "enumeration":
        return "enumeration" if dim.enumeration else None
    return None


def score(
    items: Iterable[tuple[QueryDimensions, bool, str]],
    *,
    weights: dict[str, float] | None = None,
) -> UnifiedReport:
    """Compute a UnifiedReport from per-query (dimensions, hit, bench) tuples.

    ``items``: iterable of (QueryDimensions, hit_bool, source_bench_name).
    ``weights``: dimension axis → weight (must sum to 1.0; normalised
    automatically if not).
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    total = sum(w.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        w = {k: v / total for k, v in w.items()}

    by_axis: dict[str, DimensionScore] = {axis: DimensionScore(name=axis) for axis in w}
    by_bench: dict[str, DimensionScore] = {}
    n_total = 0
    n_hit = 0
    items_list = list(items)
    for dim, hit, bench_name in items_list:
        n_total += 1
        if hit:
            n_hit += 1
        if bench_name:
            by_bench.setdefault(bench_name, DimensionScore(name=bench_name))
            by_bench[bench_name].n_queries += 1
            if hit:
                by_bench[bench_name].n_hits += 1
        for axis in w:
            if _slice_key(dim, axis):
                by_axis[axis].n_queries += 1
                if hit:
                    by_axis[axis].n_hits += 1

    # Weighted score: each axis contributes (weight * hit_rate). Axes
    # with 0 queries contribute 0 (they pull the max-attainable score
    # down — flagged in notes so the user can rebalance weights or add
    # queries for that slice).
    unified = sum(w[axis] * by_axis[axis].hit_rate for axis in w)

    notes: list[str] = []
    for axis, ds in by_axis.items():
        if ds.n_queries == 0:
            notes.append(f"NO_COVERAGE axis={axis} weight={w[axis]:.2f}")

    return UnifiedReport(
        unified_score=unified,
        query_count=n_total,
        n_hits=n_hit,
        per_dimension={
            axis: {
                "n_queries": ds.n_queries,
                "n_hits": ds.n_hits,
                "hit_rate": round(ds.hit_rate, 4),
                "weight": w[axis],
            }
            for axis, ds in by_axis.items()
        },
        per_bench={
            name: {
                "n_queries": ds.n_queries,
                "n_hits": ds.n_hits,
                "hit_rate": round(ds.hit_rate, 4),
            }
            for name, ds in sorted(by_bench.items())
        },
        weights=w,
        notes=notes,
    )


# --- Result loaders ------------------------------------------------


def load_query_files(directory: Path = QUERIES_DIR) -> dict[str, list[dict]]:
    """Load every query JSON into a dict keyed by short corpus name."""
    out: dict[str, list[dict]] = {}
    for f in sorted(directory.glob("*.json")):
        with f.open() as fh:
            data = json.load(fh)
        qs = data.get("queries", data) if isinstance(data, dict) else data
        if isinstance(qs, list):
            out[f.stem] = qs
    return out


def _classify_qfile(qfile_stem: str, queries: list[dict]) -> dict[str, QueryDimensions]:
    """Classify every query in a file. Return qid → dimensions.

    File-level defaults (e.g. ``krra_multihop`` → ``recall_type=multi_hop``)
    are applied before per-query overrides so an academically-phrased
    multi-hop query whose connectors don't match the regex still gets
    counted in the right slice.
    """
    domain_map = {
        "krra": "krra",
        "krra_hard": "krra",
        "krra_conversational": "krra",
        "krra_multihop": "krra",
        "krra_graph": "krra",
        "assort": "assort",
        "assort_hard": "assort",
        "assort_conversational": "assort",
        "x2bee": "x2bee",
        "x2bee_hard": "x2bee",
        "x2bee_conversational": "x2bee",
        # cross_domain.json carries the cross_domain=true flag in
        # per-query dimensions block; domain stays generic
        "cross_domain": "multi",
    }
    domain = domain_map.get(qfile_stem, qfile_stem)

    # File-level default dimensions — applied as base, then per-query
    # ``dimensions: {...}`` block (if present) overrides individual fields.
    file_defaults: dict[str, Any] = {}
    if qfile_stem in _MULTI_HOP_FILES:
        file_defaults["recall_type"] = RecallType.MULTI_HOP.value
        # Use ``hop_count`` floor of 2 — these benchmarks are authored
        # to require 2+ hops by construction.
        file_defaults["hop_count"] = 2

    corpus_lang = _FILE_CORPUS_LANG.get(qfile_stem, "")

    out: dict[str, QueryDimensions] = {}
    for q in queries:
        qid = q.get("qid", "")
        if not qid:
            continue
        explicit = {**file_defaults, **q.get("dimensions", {})}
        dim = classify_query(
            q.get("query", ""),
            domain=domain,
            relevant_docs=q.get("relevant_docs", []),
            explicit=explicit,
        )
        # Cross-language: query language ≠ corpus primary language.
        # Skip if explicit override already set it.
        if "cross_language" not in q.get("dimensions", {}) and corpus_lang:
            if corpus_lang == "ko" and dim.language in ("en", "mixed"):
                dim.cross_language = True
            elif corpus_lang == "en" and dim.language in ("ko", "mixed"):
                dim.cross_language = True
            elif corpus_lang == "mixed" and dim.language == "en":
                # An English query against a mixed corpus is partially
                # cross-language; tag it so the slice has coverage.
                dim.cross_language = True
        out[qid] = dim
    return out


def load_bench_log(log_path: Path) -> list[tuple[str, str, bool]]:
    """Parse a bench .log file → list of (bench_name, qid, hit)."""
    import re

    out: list[tuple[str, str, bool]] = []
    bench_name = log_path.stem
    pat = re.compile(r"\[(?P<qid>[a-z]\d{3})\]\s+turns=\d+\s+found=\d+\s+hit=(?P<hit>True|False)")
    with log_path.open() as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                out.append((bench_name, m.group("qid"), m.group("hit") == "True"))
    return out


# --- CLI -----------------------------------------------------------


def _format_report(rep: UnifiedReport) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(
        f"  UnifiedScore: {rep.unified_score:.4f}    ({rep.n_hits}/{rep.query_count}"
        f" raw hit, {100 * rep.n_hits / rep.query_count if rep.query_count else 0:.1f} %)"
    )
    lines.append("=" * 70)
    lines.append("")
    lines.append("  Per dimension:")
    for axis, info in rep.per_dimension.items():
        lines.append(
            f"    {axis:<26s}  hit={info['hit_rate']:.3f}  "
            f"n={info['n_queries']:>3d}  w={info['weight']:.2f}"
        )
    if rep.notes:
        lines.append("")
        lines.append("  Notes:")
        for n in rep.notes:
            lines.append(f"    ! {n}")
    if rep.per_bench:
        lines.append("")
        lines.append("  Per bench (raw hit-rates, no weighting):")
        for name, info in rep.per_bench.items():
            lines.append(
                f"    {name:<28s}  {info['n_hits']:>3d}/{info['n_queries']:<3d}"
                f"  ({info['hit_rate']:.3f})"
            )
    lines.append("")
    return "\n".join(lines)


def _format_diff(prev: UnifiedReport, curr: UnifiedReport) -> str:
    lines: list[str] = []
    delta = curr.unified_score - prev.unified_score
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
    lines.append("")
    lines.append("=" * 70)
    lines.append(
        f"  UnifiedScore: {prev.unified_score:.4f} → {curr.unified_score:.4f} "
        f"({delta:+.4f}) {arrow}"
    )
    lines.append("=" * 70)
    lines.append("")
    lines.append("  Per-dimension Δ:")
    for axis, curr_info in curr.per_dimension.items():
        prev_info = prev.per_dimension.get(axis, {"hit_rate": 0.0})
        d = curr_info["hit_rate"] - prev_info.get("hit_rate", 0.0)
        marker = "↑" if d > 0.005 else ("↓" if d < -0.005 else "·")
        lines.append(
            f"    {axis:<26s}  {prev_info.get('hit_rate', 0.0):.3f} → "
            f"{curr_info['hit_rate']:.3f}  ({d:+.3f}) {marker}"
        )
    rec = "SHIP" if delta >= 0 else ("NEEDS_REVIEW" if delta > -0.01 else "REGRESSION")
    lines.append("")
    lines.append(f"  Recommendation: {rec}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("/tmp/syn-bench-logs"),  # noqa: S108 — convention used by eval/run_all
        help="Directory containing per-bench .log files (with [qid] hit=True/False lines).",
    )
    p.add_argument(
        "--logs-pattern",
        default="*.log",
        help="Glob for bench log files inside --logs-dir (default *.log).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional: write JSON report to this path.",
    )
    p.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional: prior UnifiedReport JSON to diff against.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    queries = load_query_files()
    qid_dims: dict[str, QueryDimensions] = {}
    qid_to_bench_hint: dict[str, str] = {}
    for stem, qs in queries.items():
        for qid, dim in _classify_qfile(stem, qs).items():
            # If a qid appears in multiple files, the first one wins —
            # benchmark logs are unambiguous by file name so this only
            # matters for qid → dim lookup.
            qid_dims.setdefault(qid, dim)
            qid_to_bench_hint.setdefault(qid, stem)

    if not args.logs_dir.exists():
        print(f"!! logs dir not found: {args.logs_dir}", file=sys.stderr)
        return 2

    items: list[tuple[QueryDimensions, bool, str]] = []
    matched_files = sorted(args.logs_dir.glob(args.logs_pattern))
    if not matched_files:
        print(f"!! no log files matched {args.logs_pattern} in {args.logs_dir}", file=sys.stderr)
        return 2
    for f in matched_files:
        for bench_name, qid, hit in load_bench_log(f):
            dim = qid_dims.get(qid) or QueryDimensions(domain="unknown")
            # Override domain from log filename if available — log names
            # like "v020-krra-hard.log" have higher fidelity than the
            # qid → file mapping for runs that deliberately limit to one
            # bench at a time.
            for tok in ("krra", "assort", "x2bee", "musique", "hotpot", "allganize"):
                if tok in bench_name.lower():
                    dim = QueryDimensions(**{**asdict(dim), "domain": tok})
                    break
            items.append((dim, hit, bench_name))

    rep = score(items)
    print(_format_report(rep))

    if args.compare and args.compare.exists():
        with args.compare.open() as fh:
            prev_data = json.load(fh)
        # Reconstruct a minimal report for diffing
        prev = UnifiedReport(
            unified_score=prev_data["unified_score"],
            query_count=prev_data["query_count"],
            n_hits=prev_data.get("n_hits", 0),
            per_dimension=prev_data.get("per_dimension", {}),
            per_bench=prev_data.get("per_bench", {}),
            weights=prev_data.get("weights", {}),
        )
        print(_format_diff(prev, rep))

    if args.out:
        args.out.write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
        print(f"  Written: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
