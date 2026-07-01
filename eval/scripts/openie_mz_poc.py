"""Run a small OpenIE PoC on the local Korea Racing Authority corpus.

The script builds a fresh chunk graph from ``~/synaptic-eval/mz_chunks.jsonl``,
scores it, copies it, applies the opt-in OpenIE semantic layer to the copy,
and scores again. It keeps the existing ``mz_full.db`` untouched.

Example:

    uv run --extra sqlite python eval/scripts/openie_mz_poc.py \
        --llm-base-url http://localhost:8000/v1 \
        --llm-model /home/son/xgen-models/huggingface/Qwen3.6-27B

Use ``--skip-openie`` to verify ingest/scoring without an LLM server.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.document_ingester import (
    ChunkRecord,
    DocumentIngester,
    DocumentRecord,
    InMemoryDocumentSource,
)
from synaptic.extensions.domain_profile import DomainProfile
from synaptic.extensions.embedder import OpenAIEmbeddingProvider
from synaptic.extensions.entity_extractor_openie import (
    LLMOpenIEExtractor,
    OpenIELinker,
    purge_openie_artifacts,
)
from synaptic.extensions.evidence_search import EvidenceSearch
from synaptic.extensions.llm_provider import OpenAILLMProvider
from synaptic.graph import SynapticGraph
from synaptic.models import Edge, EdgeKind, Node, NodeKind

HOME = Path.home() / "synaptic-eval"
DEFAULT_CHUNKS = HOME / "mz_chunks.jsonl"
DEFAULT_QUERIES = [HOME / "mz_queries.json", HOME / "multihop_queries.json"]
DEFAULT_BASELINE_DB = HOME / "mz_openie_poc_base.db"
DEFAULT_OPENIE_DB = HOME / "mz_openie_poc_openie.db"
DEFAULT_CACHE = HOME / "openie_cache_mz.jsonl"
DEFAULT_RESULTS = HOME / "mz_openie_poc_results.json"
STRONG_RELATIONS = frozenset(
    {
        "depends_on",
        "is_a",
        "produced",
        "caused",
        "supersedes",
        "contradicts",
    }
)


class _CacheOnlyLLMProvider:
    """LLMProvider that makes cache misses explicit during eval smokes."""

    def __init__(self, *, model: str = "cache-only") -> None:
        self._model = model or "cache-only"

    async def generate(self, **_: object) -> str:
        raise RuntimeError("OpenIE cache miss in --openie-cache-only mode")


@dataclass(slots=True)
class ScoreSummary:
    name: str
    n: int = 0
    r1: int = 0
    r5: int = 0
    r10: int = 0
    by_type: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    search_times_ms: list[float] = field(default_factory=list)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    def add(self, query_type: str, docs: list[str], gold_files: list[str]) -> None:
        gold = {reg_name(path) for path in gold_files}
        ranked = [reg_name(path) for path in docs]
        hit1 = int(any(g in ranked[:1] for g in gold))
        hit5 = int(any(g in ranked[:5] for g in gold))
        hit10 = int(any(g in ranked[:10] for g in gold))
        self.n += 1
        self.r1 += hit1
        self.r5 += hit5
        self.r10 += hit10
        bucket = self.by_type[query_type or "unknown"]
        bucket[0] += 1
        bucket[1] += hit5

    def record_timing(self, search_time_ms: float, timings_ms: dict[str, float]) -> None:
        self.search_times_ms.append(float(search_time_ms))
        for key, value in timings_ms.items():
            self.stage_timings_ms[key] = self.stage_timings_ms.get(key, 0.0) + float(value)

    def line(self) -> str:
        if self.n == 0:
            return f"{self.name:14} no scored queries"
        return (
            f"{self.name:14}"
            f" R@1={self.r1 / self.n:6.1%}"
            f" R@5={self.r5 / self.n:6.1%}"
            f" R@10={self.r10 / self.n:6.1%}"
            f" n={self.n}"
        )

    def recall_at(self, k: int) -> float:
        if self.n == 0:
            return 0.0
        if k == 1:
            return self.r1 / self.n
        if k == 5:
            return self.r5 / self.n
        if k == 10:
            return self.r10 / self.n
        raise ValueError(f"unsupported k={k}")

    def to_dict(self) -> dict[str, object]:
        by_type: dict[str, dict[str, object]] = {}
        for key, (n, hits) in self.by_type.items():
            by_type[key] = {
                "n": n,
                "r5": hits / n if n else 0.0,
                "hits_at_5": hits,
            }
        return {
            "name": self.name,
            "n": self.n,
            "hits_at_1": self.r1,
            "hits_at_5": self.r5,
            "hits_at_10": self.r10,
            "r1": self.recall_at(1),
            "r5": self.recall_at(5),
            "r10": self.recall_at(10),
            "by_type": by_type,
            "timing": _timing_summary(self.search_times_ms, self.stage_timings_ms),
        }

    def clone_as(self, name: str) -> ScoreSummary:
        clone = ScoreSummary(name=name, n=self.n, r1=self.r1, r5=self.r5, r10=self.r10)
        for key, value in self.by_type.items():
            clone.by_type[key] = list(value)
        clone.search_times_ms = list(self.search_times_ms)
        clone.stage_timings_ms = dict(self.stage_timings_ms)
        return clone


def _timing_summary(
    search_times_ms: list[float],
    stage_timings_ms: dict[str, float],
) -> dict[str, object]:
    n = len(search_times_ms)
    if n == 0:
        return {"n": 0, "total_ms": 0.0, "avg_ms": 0.0, "p95_ms": 0.0, "stages": {}}
    total = sum(search_times_ms)
    ordered = sorted(search_times_ms)
    p95_index = max(0, min(n - 1, int((n - 1) * 0.95)))
    stages = {
        key: {
            "total_ms": round(value, 3),
            "avg_ms": round(value / n, 3),
        }
        for key, value in sorted(stage_timings_ms.items())
    }
    return {
        "n": n,
        "total_ms": round(total, 3),
        "avg_ms": round(total / n, 3),
        "p95_ms": round(ordered[p95_index], 3),
        "stages": stages,
    }


@dataclass(slots=True)
class OpenIERunSummary:
    skipped: bool = False
    chunks_scanned: int = 0
    chunks_selected: int = 0
    entity_nodes_touched: int = 0
    artifacts_created: int = 0
    relation_edges_created: int = 0
    openie_nodes_embedded: int = 0
    extraction_failures: int = 0
    gated: bool = False
    gate_reason: str = ""
    artifacts_purged_before: int = 0
    cache_checked_chunks: int = 0
    cache_eligible_chunks: int = 0
    cache_skipped_chunks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_entries: int = 0
    cache_missing_rows: int = 0
    cache_missing_output: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "skipped": self.skipped,
            "chunks_scanned": self.chunks_scanned,
            "chunks_selected": self.chunks_selected,
            "entity_nodes_touched": self.entity_nodes_touched,
            "artifacts_created": self.artifacts_created,
            "relation_edges_created": self.relation_edges_created,
            "openie_nodes_embedded": self.openie_nodes_embedded,
            "extraction_failures": self.extraction_failures,
            "gated": self.gated,
            "gate_reason": self.gate_reason,
            "artifacts_purged_before": self.artifacts_purged_before,
            "cache_checked_chunks": self.cache_checked_chunks,
            "cache_eligible_chunks": self.cache_eligible_chunks,
            "cache_skipped_chunks": self.cache_skipped_chunks,
            "cache_coverage_rate": (
                self.cache_eligible_chunks / self.cache_checked_chunks
                if self.cache_checked_chunks
                else 0.0
            ),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_entries": self.cache_entries,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if self.cache_hits + self.cache_misses
                else 0.0
            ),
            "cache_missing_rows": self.cache_missing_rows,
            "cache_missing_output": self.cache_missing_output,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(slots=True)
class CacheWarmSummary:
    """Summary for warming an OpenIE JSONL cache from missing chunk rows."""

    input: str = ""
    cache_path: str = ""
    dry_run: bool = False
    pending_output: str = ""
    failure_output: str = ""
    rows_loaded: int = 0
    rows_pending: int = 0
    rows_deferred_limit: int = 0
    rows_attempted: int = 0
    rows_succeeded: int = 0
    rows_skipped_cached: int = 0
    extraction_failures: int = 0
    coverage_projection_total_chunks: int = 0
    coverage_projection_existing_covered_chunks: int = 0
    coverage_projection_added_chunks: int = 0
    coverage_projection_after_batch_chunks: int = 0
    coverage_projection_after_batch_rate: float = 0.0
    coverage_projection_target_rate: float = 0.0
    coverage_projection_target_chunks: int = 0
    coverage_projection_rows_needed_for_target: int = 0
    coverage_projection_batches_needed_at_limit: int = 0
    coverage_projection_target_reachable: bool | None = None
    entities: int = 0
    triples: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_entries: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "input": self.input,
            "cache_path": self.cache_path,
            "dry_run": self.dry_run,
            "pending_output": self.pending_output,
            "failure_output": self.failure_output,
            "rows_loaded": self.rows_loaded,
            "rows_pending": self.rows_pending,
            "rows_deferred_limit": self.rows_deferred_limit,
            "rows_attempted": self.rows_attempted,
            "rows_succeeded": self.rows_succeeded,
            "rows_skipped_cached": self.rows_skipped_cached,
            "extraction_failures": self.extraction_failures,
            "coverage_projection_total_chunks": self.coverage_projection_total_chunks,
            "coverage_projection_existing_covered_chunks": (
                self.coverage_projection_existing_covered_chunks
            ),
            "coverage_projection_added_chunks": self.coverage_projection_added_chunks,
            "coverage_projection_after_batch_chunks": self.coverage_projection_after_batch_chunks,
            "coverage_projection_after_batch_rate": self.coverage_projection_after_batch_rate,
            "coverage_projection_target_rate": self.coverage_projection_target_rate,
            "coverage_projection_target_chunks": self.coverage_projection_target_chunks,
            "coverage_projection_rows_needed_for_target": (
                self.coverage_projection_rows_needed_for_target
            ),
            "coverage_projection_batches_needed_at_limit": (
                self.coverage_projection_batches_needed_at_limit
            ),
            "coverage_projection_target_reachable": self.coverage_projection_target_reachable,
            "entities": self.entities,
            "triples": self.triples,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_entries": self.cache_entries,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if self.cache_hits + self.cache_misses
                else 0.0
            ),
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(slots=True)
class CacheAuditSummary:
    """Read-only health summary for an OpenIE JSONL cache."""

    cache_path: str = ""
    bad_rows_output: str = ""
    compact_output: str = ""
    lines_total: int = 0
    valid_record_lines: int = 0
    invalid_json_lines: int = 0
    invalid_record_lines: int = 0
    duplicate_keys: int = 0
    unique_keys: int = 0
    parseable_records: int = 0
    unparseable_records: int = 0
    empty_records: int = 0
    compacted_records: int = 0
    entities: int = 0
    triples: int = 0

    @property
    def passed(self) -> bool:
        return (
            self.invalid_json_lines == 0
            and self.invalid_record_lines == 0
            and self.unparseable_records == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_path": self.cache_path,
            "bad_rows_output": self.bad_rows_output,
            "compact_output": self.compact_output,
            "lines_total": self.lines_total,
            "valid_record_lines": self.valid_record_lines,
            "invalid_json_lines": self.invalid_json_lines,
            "invalid_record_lines": self.invalid_record_lines,
            "duplicate_keys": self.duplicate_keys,
            "unique_keys": self.unique_keys,
            "parseable_records": self.parseable_records,
            "unparseable_records": self.unparseable_records,
            "empty_records": self.empty_records,
            "compacted_records": self.compacted_records,
            "entities": self.entities,
            "triples": self.triples,
            "passed": self.passed,
        }


@dataclass(slots=True)
class GateSummary:
    no_regress_r5: bool
    min_delta_r5: bool
    scored_queries_ok: bool
    openie_applied: bool
    openie_extraction_ok: bool
    revertible: bool | None
    delta_r5: float
    required_delta_r5: float
    baseline_fingerprint: str = ""
    purged_fingerprint: str = ""
    purged_artifacts: int = 0
    relation_probe_ok: bool | None = None
    relation_expanded_lift_ok: bool | None = None
    relation_evidence_lift_ok: bool | None = None
    strong_relation_evidence_ok: bool | None = None
    required_relation_expanded_lift: int = 0
    required_relation_evidence_lift: int = 0
    required_strong_relation_evidence_rate: float = 0.0
    cache_coverage_ok: bool | None = None
    cache_coverage_rate: float = 0.0
    required_cache_coverage_rate: float = 0.0

    @property
    def passed(self) -> bool:
        return (
            self.no_regress_r5
            and self.min_delta_r5
            and self.scored_queries_ok
            and self.openie_applied
            and self.openie_extraction_ok
            and (self.revertible is not False)
            and (self.relation_probe_ok is not False)
            and (self.cache_coverage_ok is not False)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "no_regress_r5": self.no_regress_r5,
            "min_delta_r5": self.min_delta_r5,
            "scored_queries_ok": self.scored_queries_ok,
            "openie_applied": self.openie_applied,
            "openie_extraction_ok": self.openie_extraction_ok,
            "revertible": self.revertible,
            "delta_r5": self.delta_r5,
            "required_delta_r5": self.required_delta_r5,
            "baseline_fingerprint": self.baseline_fingerprint,
            "purged_fingerprint": self.purged_fingerprint,
            "purged_artifacts": self.purged_artifacts,
            "relation_probe_ok": self.relation_probe_ok,
            "relation_expanded_lift_ok": self.relation_expanded_lift_ok,
            "relation_evidence_lift_ok": self.relation_evidence_lift_ok,
            "strong_relation_evidence_ok": self.strong_relation_evidence_ok,
            "required_relation_expanded_lift": self.required_relation_expanded_lift,
            "required_relation_evidence_lift": self.required_relation_evidence_lift,
            "required_strong_relation_evidence_rate": self.required_strong_relation_evidence_rate,
            "cache_coverage_ok": self.cache_coverage_ok,
            "cache_coverage_rate": self.cache_coverage_rate,
            "required_cache_coverage_rate": self.required_cache_coverage_rate,
        }


@dataclass(slots=True)
class RelationProbeSummary:
    """Probe whether OpenIE relation edges surface graph-only targets."""

    n: int = 0
    graph_expanded_hits: int = 0
    graph_evidence_hits: int = 0
    no_graph_expanded_hits: int = 0
    no_graph_evidence_hits: int = 0
    by_relation: dict[str, dict[str, int]] = field(default_factory=dict)
    probes: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        by_relation: dict[str, dict[str, object]] = {}
        groups = _empty_relation_groups()
        for relation, raw in sorted(self.by_relation.items()):
            n = raw.get("n", 0)
            graph_expanded = raw.get("graph_expanded_hits", 0)
            graph_evidence = raw.get("graph_evidence_hits", 0)
            no_graph_expanded = raw.get("no_graph_expanded_hits", 0)
            no_graph_evidence = raw.get("no_graph_evidence_hits", 0)
            group = "strong" if relation in STRONG_RELATIONS else "weak"
            _add_relation_group(groups[group], raw)
            by_relation[relation] = {
                **raw,
                "graph_expanded_rate": graph_expanded / n if n else 0.0,
                "graph_evidence_rate": graph_evidence / n if n else 0.0,
                "no_graph_expanded_rate": no_graph_expanded / n if n else 0.0,
                "no_graph_evidence_rate": no_graph_evidence / n if n else 0.0,
                "expanded_lift": graph_expanded - no_graph_expanded,
                "evidence_lift": graph_evidence - no_graph_evidence,
            }
        return {
            "n": self.n,
            "graph_expanded_hits": self.graph_expanded_hits,
            "graph_evidence_hits": self.graph_evidence_hits,
            "no_graph_expanded_hits": self.no_graph_expanded_hits,
            "no_graph_evidence_hits": self.no_graph_evidence_hits,
            "graph_expanded_rate": self.graph_expanded_hits / self.n if self.n else 0.0,
            "graph_evidence_rate": self.graph_evidence_hits / self.n if self.n else 0.0,
            "no_graph_expanded_rate": self.no_graph_expanded_hits / self.n if self.n else 0.0,
            "no_graph_evidence_rate": self.no_graph_evidence_hits / self.n if self.n else 0.0,
            "expanded_lift": self.graph_expanded_hits - self.no_graph_expanded_hits,
            "evidence_lift": self.graph_evidence_hits - self.no_graph_evidence_hits,
            "by_relation": by_relation,
            "relation_groups": _relation_group_summaries(groups),
            "probes": self.probes,
        }


def _empty_relation_groups() -> dict[str, dict[str, int]]:
    return {
        "strong": {
            "n": 0,
            "graph_expanded_hits": 0,
            "graph_evidence_hits": 0,
            "no_graph_expanded_hits": 0,
            "no_graph_evidence_hits": 0,
        },
        "weak": {
            "n": 0,
            "graph_expanded_hits": 0,
            "graph_evidence_hits": 0,
            "no_graph_expanded_hits": 0,
            "no_graph_evidence_hits": 0,
        },
    }


def _add_relation_group(group: dict[str, int], raw: dict[str, int]) -> None:
    for key in (
        "n",
        "graph_expanded_hits",
        "graph_evidence_hits",
        "no_graph_expanded_hits",
        "no_graph_evidence_hits",
    ):
        group[key] += int(raw.get(key, 0))


def _relation_group_summaries(groups: dict[str, dict[str, int]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name, raw in groups.items():
        n = raw.get("n", 0)
        graph_expanded = raw.get("graph_expanded_hits", 0)
        graph_evidence = raw.get("graph_evidence_hits", 0)
        no_graph_expanded = raw.get("no_graph_expanded_hits", 0)
        no_graph_evidence = raw.get("no_graph_evidence_hits", 0)
        out[name] = {
            **raw,
            "graph_expanded_rate": graph_expanded / n if n else 0.0,
            "graph_evidence_rate": graph_evidence / n if n else 0.0,
            "no_graph_expanded_rate": no_graph_expanded / n if n else 0.0,
            "no_graph_evidence_rate": no_graph_evidence / n if n else 0.0,
            "expanded_lift": graph_expanded - no_graph_expanded,
            "evidence_lift": graph_evidence - no_graph_evidence,
        }
    return out


def reg_name(filename: str) -> str:
    value = re.sub(r"\.hwp$", "", filename or "")
    value = re.sub(r"^\d{4}년도_", "", value)
    value = re.sub(r"\[붙임[^\]]*\]|\(붙임[^)]*\)", "", value)
    value = re.sub(r"\(제\d+-\d+호\)|제\d+-\d+호|\(\?임\)|\(\uFF1F임\)", "", value)
    value = re.sub(r"\(\d{2}-\d{2}-\d{2}\)", "", value)
    value = re.sub(r"^\s*\d+\s*[.\uFF0E]\s*", "", value)
    value = re.sub(r"(일부|전부)?\s*개정.*$|제정.*$|전문.*$|_수정.*$", "", value)
    value = re.sub(r"\(최종\).*?$", "", value)
    value = re.sub(r"[「」｢｣\uFF08\uFF09()\[\]<>·•★]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_queries(paths: list[Path]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("queries", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not row.get("query"):
                continue
            gold = row.get("gold_files") or row.get("relevant_docs")
            if not gold:
                continue
            normalized = dict(row)
            normalized["gold_files"] = list(gold)
            queries.append(normalized)
    return queries


def load_documents(chunks_path: Path, *, max_input_chunks: int = 0) -> list[DocumentRecord]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id") or row.get("source") or row.get("title") or "")
            content = str(row.get("content") or "")
            if not doc_id or not content.strip():
                continue
            grouped.setdefault(doc_id, []).append(row)

    selected_rows: list[dict[str, Any]] = []
    if max_input_chunks <= 0:
        for rows in grouped.values():
            selected_rows.extend(rows)
    else:
        positions = {doc_id: 0 for doc_id in grouped}
        doc_ids = sorted(grouped)
        while len(selected_rows) < max_input_chunks:
            added = False
            for doc_id in doc_ids:
                idx = positions[doc_id]
                rows = grouped[doc_id]
                if idx >= len(rows):
                    continue
                selected_rows.append(rows[idx])
                positions[doc_id] += 1
                added = True
                if len(selected_rows) >= max_input_chunks:
                    break
            if not added:
                break

    docs_by_id: OrderedDict[str, DocumentRecord] = OrderedDict()
    chunk_counts: dict[str, int] = defaultdict(int)
    for row in selected_rows:
        doc_id = str(row["doc_id"])
        title = str(row.get("title") or doc_id)
        source = str(row.get("source") or title)
        doc = docs_by_id.get(doc_id)
        if doc is None:
            doc = DocumentRecord(doc_id=doc_id, title=title, source=source, chunks=[])
            docs_by_id[doc_id] = doc
        idx = chunk_counts[doc_id]
        chunk_counts[doc_id] += 1
        doc.chunks.append(
            ChunkRecord(
                chunk_id=f"{doc_id}:{idx:05d}",
                doc_id=doc_id,
                text=str(row.get("content") or ""),
                index=idx,
            )
        )
    return list(docs_by_id.values())


def profile_from_args(args: argparse.Namespace, *, openie_enabled: bool) -> DomainProfile:
    whitelist = tuple(
        item.strip() for item in args.openie_relation_whitelist.split(",") if item.strip()
    )
    model_profile = args.openie_model_profile or _infer_openie_model_profile(args.llm_model)
    max_output_tokens = args.openie_max_output_tokens
    if model_profile == "deepseek_v4_flash" and max_output_tokens == 1024:
        max_output_tokens = 4096
    return DomainProfile(
        name="mz_openie_poc",
        locale="ko",
        openie_enabled=openie_enabled,
        openie_relation_whitelist=whitelist,
        openie_min_candidate_entities=args.openie_min_candidate_entities,
        openie_max_candidate_df_ratio=args.openie_max_candidate_df_ratio,
        openie_sample_rate=args.openie_sample_rate,
        openie_max_chunks=args.openie_max_chunks,
        openie_max_concurrency=args.openie_max_concurrency,
        openie_model_profile=model_profile,
        openie_max_output_tokens=max_output_tokens,
        openie_max_triples_per_chunk=args.openie_max_triples_per_chunk,
    )


def _infer_openie_model_profile(model: str) -> str:
    model_l = (model or "").lower()
    if "deepseek-v4-flash" in model_l or "deepseek_v4_flash" in model_l:
        return "deepseek_v4_flash"
    if "qwen3.6" in model_l or "qwen36" in model_l:
        return "qwen36_local"
    return ""


async def count_openie_artifacts(backend: object) -> tuple[int, int]:
    """Return total OpenIE artifacts and non-MENTIONS OpenIE relation edges."""
    nodes = await backend.list_nodes(limit=1_000_000)
    edge_ids: set[str] = set()
    relation_edge_ids: set[str] = set()
    node_count = 0
    for node in nodes:
        if "_openie" in (node.tags or []):
            node_count += 1
        for edge in await backend.get_edges(node.id, direction="both"):
            if not edge.id.startswith("openie_"):
                continue
            edge_ids.add(edge.id)
            if str(edge.kind) != "mentions":
                relation_edge_ids.add(edge.id)
    return node_count + len(edge_ids), len(relation_edge_ids)


@dataclass(slots=True)
class _CacheOnlySelector:
    """Limit replay evals to chunks already covered by the OpenIE cache."""

    extractor: LLMOpenIEExtractor
    collect_missing: bool = False
    checked_chunks: int = 0
    eligible_chunks: int = 0
    missing_rows: list[dict[str, object]] = field(default_factory=list)

    def __call__(self, node: Node) -> bool:
        self.checked_chunks += 1
        cached = self.extractor.has_cached_for_linking(node.title, node.content)
        if cached:
            self.eligible_chunks += 1
        elif self.collect_missing:
            self.missing_rows.append(_cache_missing_row(node))
        return cached

    @property
    def skipped_chunks(self) -> int:
        return self.checked_chunks - self.eligible_chunks


def _cache_only_selector(
    extractor: LLMOpenIEExtractor,
    *,
    collect_missing: bool = False,
) -> _CacheOnlySelector:
    return _CacheOnlySelector(extractor, collect_missing=collect_missing)


def _cache_missing_row(node: Node) -> dict[str, object]:
    return {
        "node_id": node.id,
        "title": node.title,
        "source": node.source,
        "doc_id": node.properties.get("doc_id", ""),
        "chunk_index": node.properties.get("chunk_index", ""),
        "content_length": len(node.content or ""),
        "content": node.content,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _cache_warm_failure_row(row: dict[str, object], exc: Exception) -> dict[str, object]:
    failed = dict(row)
    failed["error_type"] = type(exc).__name__
    failed["error"] = str(exc)[:300]
    return failed


def _apply_cache_warm_projection(
    summary: CacheWarmSummary,
    *,
    total_chunks: int,
    target_coverage: float = 0.0,
    batch_limit: int = 0,
) -> None:
    total = max(0, int(total_chunks))
    summary.coverage_projection_total_chunks = total
    if total <= 0:
        return
    existing = max(0, total - summary.rows_loaded + summary.rows_skipped_cached)
    added = summary.rows_pending if summary.dry_run else summary.rows_succeeded
    after = min(total, existing + added)
    summary.coverage_projection_existing_covered_chunks = existing
    summary.coverage_projection_added_chunks = added
    summary.coverage_projection_after_batch_chunks = after
    summary.coverage_projection_after_batch_rate = after / total if total else 0.0
    target = max(0.0, min(1.0, float(target_coverage or 0.0)))
    summary.coverage_projection_target_rate = target
    if target <= 0.0:
        return
    target_chunks = min(total, math.ceil(total * target))
    available_uncached = summary.rows_pending + summary.rows_deferred_limit
    rows_needed = max(0, target_chunks - existing)
    limit = max(0, int(batch_limit or 0))
    if rows_needed == 0:
        batches = 0
    elif limit > 0:
        batches = math.ceil(rows_needed / limit)
    else:
        batches = 1
    summary.coverage_projection_target_chunks = target_chunks
    summary.coverage_projection_rows_needed_for_target = rows_needed
    summary.coverage_projection_batches_needed_at_limit = batches
    summary.coverage_projection_target_reachable = rows_needed <= available_uncached


def _cache_audit_bad_row(
    *,
    line_number: int,
    reason: str,
    error: str = "",
    key: str = "",
) -> dict[str, object]:
    return {
        "line_number": line_number,
        "reason": reason,
        "error": error[:300],
        "key": key,
    }


def load_cache_warm_rows(path: Path, *, limit: int = 0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            title = str(payload.get("title") or "")
            content = str(payload.get("content") or "")
            if not (title or content.strip()):
                continue
            row = dict(payload)
            row["title"] = title
            row["content"] = content
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _openie_llm_provider(args: argparse.Namespace) -> object:
    return OpenAILLMProvider(
        api_base=args.llm_base_url,
        model=args.llm_model,
        api_key=os.environ.get(args.llm_api_key_env, ""),
        timeout=args.llm_timeout,
    )


def _llm_api_key_required(base_url: str) -> bool:
    parsed = urlparse(base_url)
    hostname = parsed.hostname or ""
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return False
    return parsed.scheme == "https"


async def embed_missing_openie_nodes(
    backend: object,
    embedder: object,
    *,
    batch_size: int = 32,
) -> int:
    """Embed OpenIE-created entity hubs that were added after baseline indexing."""
    nodes = await backend.list_nodes(kind=NodeKind.ENTITY, limit=1_000_000)
    pending = [
        node for node in nodes if not node.embedding and "_openie_entity" in (node.tags or [])
    ]
    embedded = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        texts = [f"{node.title}\n{(node.content or '')[:300]}" for node in batch]
        try:
            vecs_raw = await embedder.embed_batch(texts)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"[openie] embedding batch failed: {exc}")
            continue
        vecs = list(vecs_raw or [])
        if len(vecs) != len(batch):
            print(
                f"[openie] embedding batch failed: expected {len(batch)} vectors, got {len(vecs)}"
            )
            continue
        changed = []
        for node, vec in zip(batch, vecs, strict=True):
            if not vec:
                continue
            node.embedding = vec
            changed.append(node)
        if not changed:
            continue
        save_batch = getattr(backend, "save_nodes_batch", None)
        if callable(save_batch):
            await save_batch(changed)
        else:
            for node in changed:
                await backend.save_node(node)
        embedded += len(changed)
    return embedded


async def build_baseline_db(args: argparse.Namespace) -> None:
    if args.baseline_db.exists() and args.reuse_baseline:
        print(f"[build] reuse baseline DB: {args.baseline_db}")
        return
    for suffix in ("", ".hnsw", ".hnsw.meta.json"):
        Path(f"{args.baseline_db}{suffix}").unlink(missing_ok=True)

    docs = load_documents(args.chunks, max_input_chunks=args.max_input_chunks)
    n_chunks = sum(len(doc.chunks) for doc in docs)
    print(f"[build] docs={len(docs)} chunks={n_chunks} -> {args.baseline_db}")

    backend = SqliteGraphBackend(str(args.baseline_db))
    await backend.connect()
    try:
        ingester = DocumentIngester(
            profile=profile_from_args(args, openie_enabled=False),
            backend=backend,
        )
        stats = await ingester.ingest(InMemoryDocumentSource(docs))
        print(
            "[build] "
            f"documents={stats.documents_ingested} chunks={stats.chunks_created} "
            f"edges={stats.edges_created}"
        )

        if args.embed_base_url:
            embedder = OpenAIEmbeddingProvider(
                api_base=args.embed_base_url,
                model=args.embed_model,
                timeout=args.embed_timeout,
            )
            t0 = time.perf_counter()
            await SynapticGraph._embed_all_nodes(backend, embedder)
            print(f"[build] embedded nodes in {time.perf_counter() - t0:.1f}s")
    finally:
        await backend.close()


def copy_graph(src: Path, dst: Path) -> None:
    for suffix in ("", ".hnsw", ".hnsw.meta.json"):
        Path(f"{dst}{suffix}").unlink(missing_ok=True)
    shutil.copy2(src, dst)
    for suffix in (".hnsw", ".hnsw.meta.json"):
        sidecar = Path(f"{src}{suffix}")
        if sidecar.exists():
            shutil.copy2(sidecar, Path(f"{dst}{suffix}"))


async def apply_openie(args: argparse.Namespace) -> OpenIERunSummary:
    copy_graph(args.baseline_db, args.openie_db)
    if args.skip_openie:
        print("[openie] skipped; copied baseline only")
        return OpenIERunSummary(skipped=True)
    if not args.openie_cache_only and (not args.llm_base_url or not args.llm_model):
        msg = "--llm-base-url and --llm-model are required unless --skip-openie is set"
        raise SystemExit(msg)

    profile = profile_from_args(args, openie_enabled=True)
    if args.openie_cache_only:
        provider = _CacheOnlyLLMProvider(model=args.llm_model or "cache-only")
    else:
        provider = _openie_llm_provider(args)
    extractor = LLMOpenIEExtractor(
        provider,
        seed=args.openie_seed,
        relation_whitelist=profile.openie_relation_whitelist,
        max_output_tokens=profile.openie_max_output_tokens,
        max_triples_per_chunk=profile.openie_max_triples_per_chunk,
        cache_path=args.openie_cache,
        fail_open=False,
    )

    backend = SqliteGraphBackend(str(args.openie_db))
    await backend.connect()
    try:
        deleted = 0
        if args.reset_openie:
            deleted = await purge_openie_artifacts(backend)
            print(f"[openie] purged existing artifacts={deleted}")
        t0 = time.perf_counter()
        selector = (
            _cache_only_selector(
                extractor,
                collect_missing=bool(args.openie_cache_missing_output),
            )
            if args.openie_cache_only
            else None
        )
        stats = await OpenIELinker(extractor, profile=profile, selector=selector).link(
            backend,
            source_limit=args.openie_source_limit,
        )
        cache_checked_chunks = selector.checked_chunks if selector is not None else 0
        cache_eligible_chunks = selector.eligible_chunks if selector is not None else 0
        cache_skipped_chunks = selector.skipped_chunks if selector is not None else 0
        cache_missing_rows = len(selector.missing_rows) if selector is not None else 0
        cache_missing_output = ""
        if selector is not None and args.openie_cache_missing_output:
            cache_missing_output = str(args.openie_cache_missing_output)
            write_jsonl(args.openie_cache_missing_output, selector.missing_rows)
        openie_nodes_embedded = 0
        if args.embed_base_url:
            embedder = OpenAIEmbeddingProvider(
                api_base=args.embed_base_url,
                model=args.embed_model,
                timeout=args.embed_timeout,
            )
            openie_nodes_embedded = await embed_missing_openie_nodes(backend, embedder)
        artifacts_created, relation_edges_created = await count_openie_artifacts(backend)
        cache_stats = extractor.cache_stats()
        elapsed = time.perf_counter() - t0
        print(
            "[openie] "
            f"scanned={stats.chunks_scanned} selected={stats.chunks_selected} "
            f"touched={stats.entity_nodes_touched} failures={stats.extraction_failures} "
            f"embedded_openie_nodes={openie_nodes_embedded} "
            f"artifacts={artifacts_created} relation_edges={relation_edges_created} "
            f"cache_eligible={cache_eligible_chunks}/{cache_checked_chunks} "
            f"cache_hits={cache_stats['hits']} cache_misses={cache_stats['misses']} "
            f"cache_missing_rows={cache_missing_rows} "
            f"gated={stats.gated} reason={stats.gate_reason!r} "
            f"elapsed={elapsed:.1f}s"
        )
        return OpenIERunSummary(
            chunks_scanned=stats.chunks_scanned,
            chunks_selected=stats.chunks_selected,
            entity_nodes_touched=stats.entity_nodes_touched,
            artifacts_created=artifacts_created,
            relation_edges_created=relation_edges_created,
            openie_nodes_embedded=openie_nodes_embedded,
            extraction_failures=stats.extraction_failures,
            gated=stats.gated,
            gate_reason=stats.gate_reason,
            artifacts_purged_before=deleted,
            cache_checked_chunks=cache_checked_chunks,
            cache_eligible_chunks=cache_eligible_chunks,
            cache_skipped_chunks=cache_skipped_chunks,
            cache_hits=cache_stats["hits"],
            cache_misses=cache_stats["misses"],
            cache_entries=cache_stats["entries"],
            cache_missing_rows=cache_missing_rows,
            cache_missing_output=cache_missing_output,
            elapsed_seconds=elapsed,
        )
    finally:
        await backend.close()


async def warm_openie_cache(args: argparse.Namespace) -> CacheWarmSummary:
    if not args.openie_cache_warm_input:
        return CacheWarmSummary()
    if not args.openie_cache_warm_input.exists():
        raise SystemExit(f"cache warm input not found: {args.openie_cache_warm_input}")
    if not args.openie_cache_warm_dry_run and (not args.llm_base_url or not args.llm_model):
        msg = "--llm-base-url and --llm-model are required for --openie-cache-warm-input"
        raise SystemExit(msg)

    rows = load_cache_warm_rows(args.openie_cache_warm_input)
    profile = profile_from_args(args, openie_enabled=True)
    provider = (
        _CacheOnlyLLMProvider(model=args.llm_model or "cache-warm-dry-run")
        if args.openie_cache_warm_dry_run
        else _openie_llm_provider(args)
    )
    extractor = LLMOpenIEExtractor(
        provider,
        seed=args.openie_seed,
        relation_whitelist=profile.openie_relation_whitelist,
        max_output_tokens=profile.openie_max_output_tokens,
        max_triples_per_chunk=profile.openie_max_triples_per_chunk,
        cache_path=args.openie_cache,
        fail_open=False,
    )

    summary = CacheWarmSummary(
        input=str(args.openie_cache_warm_input),
        cache_path=str(args.openie_cache),
        dry_run=bool(args.openie_cache_warm_dry_run),
        pending_output=str(args.openie_cache_warm_pending_output or ""),
        failure_output=str(args.openie_cache_warm_failure_output or ""),
        rows_loaded=len(rows),
    )
    t0 = time.perf_counter()
    pending_limit = max(0, int(args.openie_cache_warm_limit or 0))
    pending_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    for row in rows:
        title = str(row["title"])
        content = str(row["content"])
        text = f"{title}\n{content}" if content else title
        if extractor.has_cached(text, title=title):
            summary.rows_skipped_cached += 1
            continue

        if pending_limit > 0 and summary.rows_pending >= pending_limit:
            summary.rows_deferred_limit += 1
            continue

        summary.rows_pending += 1
        pending_rows.append(dict(row))
        if args.openie_cache_warm_dry_run:
            continue

        summary.rows_attempted += 1
        try:
            result = await extractor.extract(text, title=title)
        except Exception as exc:
            summary.extraction_failures += 1
            failure_rows.append(_cache_warm_failure_row(row, exc))
            print(f"[cache-warm] failed row={summary.rows_attempted}: {exc}")
            continue
        summary.rows_succeeded += 1
        summary.entities += len(result.entities)
        summary.triples += len(result.triples)

    if args.openie_cache_warm_pending_output:
        write_jsonl(args.openie_cache_warm_pending_output, pending_rows)
    if args.openie_cache_warm_failure_output:
        write_jsonl(args.openie_cache_warm_failure_output, failure_rows)

    _apply_cache_warm_projection(
        summary,
        total_chunks=getattr(args, "openie_cache_warm_total_chunks", 0) or 0,
        target_coverage=getattr(args, "openie_cache_warm_target_coverage", 0.0) or 0.0,
        batch_limit=pending_limit,
    )

    cache_stats = extractor.cache_stats()
    summary.cache_hits = cache_stats["hits"]
    summary.cache_misses = cache_stats["misses"]
    summary.cache_entries = cache_stats["entries"]
    summary.elapsed_seconds = time.perf_counter() - t0
    print(
        "[cache-warm] "
        f"loaded={summary.rows_loaded} pending={summary.rows_pending} "
        f"deferred_limit={summary.rows_deferred_limit} "
        f"attempted={summary.rows_attempted} "
        f"succeeded={summary.rows_succeeded} skipped_cached={summary.rows_skipped_cached} "
        f"failures={summary.extraction_failures} "
        f"entities={summary.entities} triples={summary.triples} "
        f"cache_hits={summary.cache_hits} cache_misses={summary.cache_misses} "
        f"cache_entries={summary.cache_entries} "
        f"projected_coverage={summary.coverage_projection_after_batch_rate:.1%} "
        f"elapsed={summary.elapsed_seconds:.1f}s"
    )
    return summary


def audit_openie_cache(args: argparse.Namespace) -> CacheAuditSummary:
    path = args.openie_cache
    summary = CacheAuditSummary(
        cache_path=str(path),
        bad_rows_output=str(args.openie_cache_audit_bad_output or ""),
        compact_output=str(args.openie_cache_compact_output or ""),
    )
    if not path.exists():
        raise SystemExit(f"OpenIE cache not found: {path}")

    seen: set[str] = set()
    bad_rows: list[dict[str, object]] = []
    compact_records: OrderedDict[str, str] = OrderedDict()
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            summary.lines_total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                summary.invalid_json_lines += 1
                bad_rows.append(
                    _cache_audit_bad_row(
                        line_number=line_number,
                        reason="invalid_json_line",
                        error=str(exc),
                    )
                )
                continue

            key = item.get("key") if isinstance(item, dict) else None
            raw = item.get("raw") if isinstance(item, dict) else None
            if not isinstance(key, str) or not isinstance(raw, str):
                summary.invalid_record_lines += 1
                bad_rows.append(
                    _cache_audit_bad_row(
                        line_number=line_number,
                        reason="invalid_record_shape",
                    )
                )
                continue

            summary.valid_record_lines += 1
            if key in seen:
                summary.duplicate_keys += 1
            else:
                seen.add(key)
            try:
                result = LLMOpenIEExtractor._parse_response(raw)
            except Exception as exc:
                summary.unparseable_records += 1
                bad_rows.append(
                    _cache_audit_bad_row(
                        line_number=line_number,
                        reason="unparseable_raw",
                        error=str(exc),
                        key=key,
                    )
                )
                continue
            summary.parseable_records += 1
            compact_records[key] = raw
            compact_records.move_to_end(key)
            summary.entities += len(result.entities)
            summary.triples += len(result.triples)
            if not result.entities and not result.triples:
                summary.empty_records += 1

    summary.unique_keys = len(seen)
    if args.openie_cache_audit_bad_output:
        write_jsonl(args.openie_cache_audit_bad_output, bad_rows)
    if args.openie_cache_compact_output:
        compact_rows = [{"key": key, "raw": raw} for key, raw in compact_records.items()]
        write_jsonl(args.openie_cache_compact_output, compact_rows)
        summary.compacted_records = len(compact_rows)

    print(
        "[cache-audit] "
        f"lines={summary.lines_total} unique_keys={summary.unique_keys} "
        f"duplicates={summary.duplicate_keys} parseable={summary.parseable_records} "
        f"unparseable={summary.unparseable_records} invalid_json={summary.invalid_json_lines} "
        f"invalid_record={summary.invalid_record_lines} empty={summary.empty_records} "
        f"entities={summary.entities} triples={summary.triples} "
        f"compacted={summary.compacted_records} "
        f"passed={summary.passed}"
    )
    return summary


async def score_db(
    name: str,
    db_path: Path,
    queries: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    query_embeddings: dict[str, list[float]] | None = None,
) -> ScoreSummary:
    present_docs = await list_doc_ids(db_path)
    scored = [
        q for q in queries if any(str(gold) in present_docs for gold in q.get("gold_files", []))
    ]
    summary = ScoreSummary(name=name)
    if not scored:
        return summary

    backend = SqliteGraphBackend(str(db_path))
    await backend.connect()
    embedder = (
        OpenAIEmbeddingProvider(
            api_base=args.embed_base_url,
            model=args.embed_model,
            timeout=args.embed_timeout,
        )
        if args.embed_base_url
        else None
    )
    graph = SynapticGraph(backend, embedder=embedder)
    try:
        for row in scored:
            result = await graph.search(
                str(row["query"]),
                limit=args.search_limit,
                embedding=query_embeddings.get(str(row["query"])) if query_embeddings else None,
                rerank=False,
                per_document_cap=1,
            )
            docs = hits_to_doc_ids(result, limit=10)
            summary.add(
                str(row.get("type") or row.get("src") or "unknown"), docs, row["gold_files"]
            )
            summary.record_timing(
                float(getattr(result, "search_time_ms", 0.0) or 0.0),
                dict(getattr(result, "timings_ms", {}) or {}),
            )
    finally:
        await graph.close()
    return summary


async def relation_probe_for_db(
    db_path: Path,
    *,
    limit: int = 20,
    search_k: int = 10,
    max_per_source_relation: int = 3,
) -> RelationProbeSummary:
    """Measure whether graph expansion surfaces OpenIE relation targets.

    Each probe uses a query built from the subject entity and relation label,
    deliberately omitting the object entity title. The no-graph search should
    mostly see only lexical seeds; the graph search can follow the OpenIE edge
    to the target. This is a compact eval for information a plain chunk RAG
    path cannot reach through text matching alone.
    """
    summary = RelationProbeSummary()
    if limit <= 0:
        return summary

    backend = SqliteGraphBackend(str(db_path))
    await backend.connect()
    try:
        probes = await _openie_relation_probes(
            backend,
            limit=limit,
            max_per_source_relation=max_per_source_relation,
        )
        search_backend = _RelationProbeCachedBackend(backend)
        no_graph = EvidenceSearch(backend=search_backend, graph_expansion=False)
        with_graph = EvidenceSearch(backend=search_backend)
        for source, target, edge in probes:
            query = _relation_probe_query(source.title, edge.kind)
            if _contains_title(query, target.title):
                continue
            no_graph_result = await no_graph.search(query, k=search_k)
            graph_result = await with_graph.search(query, k=search_k)

            no_graph_expanded = _node_id_in_expanded(no_graph_result.expanded, target.id)
            graph_expanded = _node_id_in_expanded(graph_result.expanded, target.id)
            no_graph_evidence = _node_id_in_evidence(no_graph_result.evidence, target.id)
            graph_evidence = _node_id_in_evidence(graph_result.evidence, target.id)
            relation = str(edge.kind)

            summary.n += 1
            summary.no_graph_expanded_hits += int(no_graph_expanded)
            summary.graph_expanded_hits += int(graph_expanded)
            summary.no_graph_evidence_hits += int(no_graph_evidence)
            summary.graph_evidence_hits += int(graph_evidence)
            relation_bucket = summary.by_relation.setdefault(
                relation,
                {
                    "n": 0,
                    "graph_expanded_hits": 0,
                    "graph_evidence_hits": 0,
                    "no_graph_expanded_hits": 0,
                    "no_graph_evidence_hits": 0,
                },
            )
            relation_bucket["n"] += 1
            relation_bucket["no_graph_expanded_hits"] += int(no_graph_expanded)
            relation_bucket["graph_expanded_hits"] += int(graph_expanded)
            relation_bucket["no_graph_evidence_hits"] += int(no_graph_evidence)
            relation_bucket["graph_evidence_hits"] += int(graph_evidence)
            summary.probes.append(
                {
                    "query": query,
                    "edge_id": edge.id,
                    "relation": relation,
                    "confidence": _probe_edge_confidence(edge),
                    "source_id": source.id,
                    "source_title": source.title,
                    "target_id": target.id,
                    "target_title": target.title,
                    "no_graph_expanded": no_graph_expanded,
                    "graph_expanded": graph_expanded,
                    "no_graph_evidence": no_graph_evidence,
                    "graph_evidence": graph_evidence,
                }
            )
    finally:
        await backend.close()
    return summary


class _RelationProbeCachedBackend:
    """Read-through cache shared by no-graph and graph relation-probe searches."""

    __slots__ = ("_backend", "_edges", "_fts", "_neighbors", "_nodes")

    def __init__(self, backend: SqliteGraphBackend) -> None:
        self._backend = backend
        self._nodes: dict[str, Node | None] = {}
        self._edges: dict[tuple[str, str], list[Edge]] = {}
        self._fts: dict[tuple[str, int, bool], object] = {}
        self._neighbors: dict[tuple[str, int], object] = {}

    def __getattr__(self, name: str) -> object:
        return getattr(self._backend, name)

    async def get_node(self, node_id: str) -> Node | None:
        if node_id in self._nodes:
            return self._nodes[node_id]
        node = await self._backend.get_node(node_id)
        self._nodes[node_id] = node
        return node

    async def get_edges(self, node_id: str, *, direction: str = "both") -> list[Edge]:
        key = (node_id, direction)
        if key not in self._edges:
            self._edges[key] = await self._backend.get_edges(node_id, direction=direction)
        return self._edges[key]

    async def get_neighbors(self, node_id: str, *, depth: int = 1) -> object:
        key = (node_id, depth)
        if key not in self._neighbors:
            self._neighbors[key] = await self._backend.get_neighbors(node_id, depth=depth)
        return self._neighbors[key]

    async def search_fts(self, query: str, *, limit: int = 20, with_scores: bool = False) -> object:
        key = (query, limit, with_scores)
        if key not in self._fts:
            self._fts[key] = await self._backend.search_fts(
                query,
                limit=limit,
                with_scores=with_scores,
            )
        return self._fts[key]


async def _openie_relation_probes(
    backend: SqliteGraphBackend,
    *,
    limit: int,
    max_per_source_relation: int = 3,
) -> list[tuple[Node, Node, Edge]]:
    nodes = await backend.list_nodes(kind=NodeKind.ENTITY, limit=1_000_000)
    by_id = {node.id: node for node in nodes}
    out: list[tuple[Node, Node, Edge]] = []
    seen_edges: set[str] = set()
    source_relation_counts: dict[tuple[str, str], int] = {}
    for source in nodes:
        if len(out) >= limit:
            break
        try:
            edges = await backend.get_edges(source.id, direction="outgoing")
        except Exception:
            continue
        for edge in edges:
            if len(out) >= limit:
                break
            if edge.id in seen_edges or edge.kind == EdgeKind.MENTIONS:
                continue
            props = edge.properties or {}
            if props.get("is_openie") != "true":
                continue
            relation_key = str(edge.kind.value if isinstance(edge.kind, EdgeKind) else edge.kind)
            pair_key = (source.id, relation_key)
            if max_per_source_relation > 0 and (
                source_relation_counts.get(pair_key, 0) >= max_per_source_relation
            ):
                continue
            target = by_id.get(edge.target_id)
            if target is None:
                target = await backend.get_node(edge.target_id)
            if target is None or not source.title or not target.title:
                continue
            seen_edges.add(edge.id)
            source_relation_counts[pair_key] = source_relation_counts.get(pair_key, 0) + 1
            out.append((source, target, edge))
    return out


def _relation_probe_query(subject_title: str, kind: EdgeKind | str) -> str:
    relation = str(kind.value if isinstance(kind, EdgeKind) else kind)
    phrase = {
        "depends_on": "depends_on 의존 관계",
        "part_of": "part_of 포함 관계",
        "is_a": "is_a 유형 관계",
        "related": "related 관련 관계",
        "caused": "caused 원인 관계",
        "produced": "produced 산출 관계",
        "contradicts": "contradicts 상충 관계",
        "supersedes": "supersedes 대체 관계",
    }.get(relation, f"{relation} 관계")
    return f"{subject_title} {phrase}".strip()


def _probe_edge_confidence(edge: Edge) -> float:
    raw = (edge.properties or {}).get("confidence", edge.weight)
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 1.0


def _contains_title(query: str, title: str) -> bool:
    needle = (title or "").strip().casefold()
    if len(needle) < 2:
        return False
    return needle in query.casefold()


def _node_id_in_expanded(expanded: list[object], node_id: str) -> bool:
    for item in expanded:
        node = getattr(item, "node", None)
        if getattr(node, "id", "") == node_id:
            return True
    return False


def _node_id_in_evidence(evidence: list[object], node_id: str) -> bool:
    for item in evidence:
        node = getattr(item, "node", None)
        if getattr(node, "id", "") == node_id:
            return True
    return False


async def precompute_query_embeddings(
    queries: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    batch_size: int = 1,
) -> dict[str, list[float]]:
    if not args.embed_base_url:
        return {}
    texts = sorted({str(row["query"]) for row in queries if row.get("query")})
    if not texts:
        return {}
    embedder = OpenAIEmbeddingProvider(
        api_base=args.embed_base_url,
        model=args.embed_model,
        timeout=args.embed_timeout,
    )
    out: dict[str, list[float]] = {}
    # Default to one query per request so the eval uses the same vector
    # path as graph.search(embedder.embed). Some local embedding servers
    # show tiny batch-size-dependent differences that can move R@1.
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            vectors = await embedder.embed_batch(batch)
        except Exception as exc:
            print(f"[score] query embedding batch failed: {exc}")
            continue
        for text, vector in zip(batch, vectors, strict=False):
            if vector:
                out[text] = vector
    print(f"[score] precomputed query embeddings={len(out)}/{len(texts)}")
    return out


async def list_doc_ids(db_path: Path) -> set[str]:
    backend = SqliteGraphBackend(str(db_path))
    await backend.connect()
    try:
        nodes = await backend.list_nodes(kind=NodeKind.ENTITY, limit=1_000_000)
        return {
            node.properties.get("doc_id", "") for node in nodes if node.properties.get("doc_id")
        }
    finally:
        await backend.close()


def hits_to_doc_ids(search_result: object, *, limit: int) -> list[str]:
    out: list[str] = []
    for hit in getattr(search_result, "nodes", []):
        node = getattr(hit, "node", hit)
        props = getattr(node, "properties", {}) or {}
        doc_id = props.get("doc_id") or getattr(node, "title", "")
        if not doc_id or doc_id in out:
            continue
        out.append(doc_id)
        if len(out) >= limit:
            break
    return out


def print_comparison(base: ScoreSummary, openie: ScoreSummary) -> None:
    print("\n=== Retrieval comparison ===")
    print(base.line())
    print(openie.line())
    if base.n and openie.n:
        print(f"delta R@5={(openie.r5 / openie.n) - (base.r5 / base.n):+.1%}")
    print("\nR@5 by query type/source:")
    keys = sorted(set(base.by_type) | set(openie.by_type))
    for key in keys:
        bn, bh = base.by_type.get(key, [0, 0])
        on, oh = openie.by_type.get(key, [0, 0])
        bscore = f"{bh / bn:.1%}" if bn else "n/a"
        oscore = f"{oh / on:.1%}" if on else "n/a"
        print(f"  {key:32} baseline={bscore:>7} openie={oscore:>7} n={max(bn, on)}")


def graph_fingerprint(db_path: Path) -> str:
    """Logical graph fingerprint over nodes/edges, independent of SQLite pages."""
    h = hashlib.sha256()
    con = sqlite3.connect(db_path)
    try:
        for row in con.execute(
            "SELECT id, kind, title, content, tags_json, level, vitality, "
            "access_count, success_count, failure_count, source, properties_json, "
            "embedding_json FROM syn_nodes ORDER BY id"
        ):
            h.update(b"N")
            h.update(_stable_row_json(row).encode())
        for row in con.execute(
            "SELECT id, source_id, target_id, kind, weight, properties_json "
            "FROM syn_edges ORDER BY id"
        ):
            h.update(b"E")
            h.update(_stable_row_json(row).encode())
    finally:
        con.close()
    return h.hexdigest()


def _stable_row_json(row: tuple[object, ...]) -> str:
    values: list[object] = []
    for value in row:
        if isinstance(value, str) and value[:1] in {"[", "{"}:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                values.append(value)
            else:
                values.append(parsed)
        else:
            values.append(value)
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


async def verify_revertibility(args: argparse.Namespace) -> tuple[bool, str, str, int]:
    baseline_fp = graph_fingerprint(args.baseline_db)
    if args.skip_openie:
        return True, baseline_fp, baseline_fp, 0

    verify_db = args.openie_db.with_name(f"{args.openie_db.stem}.purgecheck{args.openie_db.suffix}")
    copy_graph(args.openie_db, verify_db)
    backend = SqliteGraphBackend(str(verify_db))
    await backend.connect()
    try:
        deleted = await purge_openie_artifacts(backend)
    finally:
        await backend.close()
    purged_fp = graph_fingerprint(verify_db)
    for suffix in ("", ".hnsw", ".hnsw.meta.json"):
        Path(f"{verify_db}{suffix}").unlink(missing_ok=True)
    return baseline_fp == purged_fp, baseline_fp, purged_fp, deleted


async def evaluate_gates(
    args: argparse.Namespace,
    *,
    base: ScoreSummary,
    openie: ScoreSummary,
    openie_run: OpenIERunSummary,
    relation_probe: RelationProbeSummary | None = None,
) -> GateSummary:
    delta_r5 = openie.recall_at(5) - base.recall_at(5)
    no_regress = delta_r5 >= 0.0
    min_delta = delta_r5 >= args.min_delta_r5
    scored_queries_ok = base.n > 0 and openie.n > 0
    openie_applied = args.skip_openie or (
        openie_run.chunks_selected > 0
        and openie_run.entity_nodes_touched > 0
        and openie_run.relation_edges_created > 0
        and not openie_run.gated
    )
    openie_extraction_ok = args.skip_openie or openie_run.extraction_failures == 0
    required_cache_coverage = float(getattr(args, "min_openie_cache_coverage", 0.0) or 0.0)
    cache_coverage_rate = (
        openie_run.cache_eligible_chunks / openie_run.cache_checked_chunks
        if openie_run.cache_checked_chunks
        else 0.0
    )
    cache_coverage_ok: bool | None = None
    if required_cache_coverage > 0.0:
        cache_coverage_ok = cache_coverage_rate >= required_cache_coverage
    relation_expanded_lift_ok: bool | None = None
    relation_evidence_lift_ok: bool | None = None
    strong_relation_evidence_ok: bool | None = None
    relation_probe_ok: bool | None = None
    required_expanded_lift = int(getattr(args, "min_relation_expanded_lift", 0) or 0)
    required_evidence_lift = int(getattr(args, "min_relation_evidence_lift", 0) or 0)
    required_strong_rate = float(getattr(args, "min_strong_relation_evidence_rate", 0.0) or 0.0)
    relation_threshold_requested = (
        required_expanded_lift > 0 or required_evidence_lift > 0 or required_strong_rate > 0.0
    )
    if not args.skip_openie and relation_probe is not None and relation_probe.n > 0:
        relation_payload = relation_probe.to_dict()
        expanded_lift = int(relation_payload["expanded_lift"])
        evidence_lift = int(relation_payload["evidence_lift"])
        strong = relation_payload["relation_groups"]["strong"]
        strong_rate = float(strong["graph_evidence_rate"])
        relation_expanded_lift_ok = expanded_lift >= required_expanded_lift
        relation_evidence_lift_ok = evidence_lift >= required_evidence_lift
        strong_relation_evidence_ok = (
            strong_rate >= required_strong_rate
            if int(strong["n"]) > 0
            else required_strong_rate <= 0.0
        )
        relation_probe_ok = (
            relation_expanded_lift_ok and relation_evidence_lift_ok and strong_relation_evidence_ok
        )
    elif relation_threshold_requested:
        relation_expanded_lift_ok = False
        relation_evidence_lift_ok = False
        strong_relation_evidence_ok = False
        relation_probe_ok = False
    revertible: bool | None = None
    baseline_fp = ""
    purged_fp = ""
    purged_artifacts = 0
    if args.verify_revertibility:
        revertible, baseline_fp, purged_fp, purged_artifacts = await verify_revertibility(args)
        print(
            "[gate] revertibility="
            f"{'PASS' if revertible else 'FAIL'} purged_artifacts={purged_artifacts}"
        )
    print(
        "[gate] "
        f"scored_queries={'PASS' if scored_queries_ok else 'FAIL'} "
        f"no_regress_r5={'PASS' if no_regress else 'FAIL'} "
        f"min_delta_r5={'PASS' if min_delta else 'FAIL'} "
        f"delta_r5={delta_r5:+.1%} required={args.min_delta_r5:+.1%}"
    )
    if not args.skip_openie:
        print(
            "[gate] "
            f"openie_applied={'PASS' if openie_applied else 'FAIL'} "
            f"openie_extraction_ok={'PASS' if openie_extraction_ok else 'FAIL'} "
            f"touched={openie_run.entity_nodes_touched} "
            f"relation_edges={openie_run.relation_edges_created} "
            f"failures={openie_run.extraction_failures}"
        )
    if relation_probe_ok is not None:
        print(
            "[gate] "
            f"relation_probe={'PASS' if relation_probe_ok else 'FAIL'} "
            f"expanded_lift={'PASS' if relation_expanded_lift_ok else 'FAIL'} "
            f"evidence_lift={'PASS' if relation_evidence_lift_ok else 'FAIL'} "
            f"strong_evidence={'PASS' if strong_relation_evidence_ok else 'FAIL'}"
        )
    if cache_coverage_ok is not None:
        print(
            "[gate] "
            f"cache_coverage={'PASS' if cache_coverage_ok else 'FAIL'} "
            f"coverage={cache_coverage_rate:.1%} required={required_cache_coverage:.1%}"
        )
    return GateSummary(
        no_regress_r5=no_regress,
        min_delta_r5=min_delta,
        scored_queries_ok=scored_queries_ok,
        openie_applied=openie_applied,
        openie_extraction_ok=openie_extraction_ok,
        revertible=revertible,
        delta_r5=delta_r5,
        required_delta_r5=args.min_delta_r5,
        baseline_fingerprint=baseline_fp,
        purged_fingerprint=purged_fp,
        purged_artifacts=purged_artifacts,
        relation_probe_ok=relation_probe_ok,
        relation_expanded_lift_ok=relation_expanded_lift_ok,
        relation_evidence_lift_ok=relation_evidence_lift_ok,
        strong_relation_evidence_ok=strong_relation_evidence_ok,
        required_relation_expanded_lift=required_expanded_lift,
        required_relation_evidence_lift=required_evidence_lift,
        required_strong_relation_evidence_rate=required_strong_rate,
        cache_coverage_ok=cache_coverage_ok,
        cache_coverage_rate=cache_coverage_rate,
        required_cache_coverage_rate=required_cache_coverage,
    )


def write_results(
    args: argparse.Namespace,
    *,
    base: ScoreSummary,
    openie: ScoreSummary,
    openie_run: OpenIERunSummary,
    gates: GateSummary,
    memory_health: dict[str, object] | None,
    relation_probe: RelationProbeSummary | None = None,
    scores_reused: bool = False,
) -> None:
    if not args.results:
        return
    args.results.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": int(time.time()),
        "chunks": str(args.chunks),
        "queries": [str(p) for p in args.queries],
        "baseline_db": str(args.baseline_db),
        "openie_db": str(args.openie_db),
        "skip_openie": args.skip_openie,
        "scores_reused": scores_reused,
        "max_input_chunks": args.max_input_chunks,
        "search_limit": args.search_limit,
        "openie": openie_run.to_dict(),
        "scores": {
            "baseline": base.to_dict(),
            "openie": openie.to_dict(),
        },
        "gates": gates.to_dict(),
        "memory_health": memory_health or {},
        "relation_probe": (relation_probe or RelationProbeSummary()).to_dict(),
    }
    with args.results.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[results] wrote {args.results}")


def write_cache_warm_results(args: argparse.Namespace, summary: CacheWarmSummary) -> None:
    payload = {
        "timestamp": int(time.time()),
        "openie_cache_warm": summary.to_dict(),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[results] wrote {args.results}")


def write_cache_audit_results(args: argparse.Namespace, summary: CacheAuditSummary) -> None:
    payload = {
        "timestamp": int(time.time()),
        "openie_cache_audit": summary.to_dict(),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[results] wrote {args.results}")


async def memory_health_for_db(path: Path) -> dict[str, object]:
    backend = SqliteGraphBackend(str(path))
    await backend.connect()
    try:
        graph = SynapticGraph(backend)
        return asdict(await graph.memory_health(persist_signals=False))
    finally:
        await backend.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--queries", type=Path, nargs="*", default=DEFAULT_QUERIES)
    parser.add_argument("--baseline-db", type=Path, default=DEFAULT_BASELINE_DB)
    parser.add_argument("--openie-db", type=Path, default=DEFAULT_OPENIE_DB)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reuse-baseline", action="store_true")
    parser.add_argument("--max-input-chunks", type=int, default=0)
    parser.add_argument("--skip-openie", action="store_true")
    parser.add_argument("--reset-openie", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--embed-base-url", default="http://localhost:18013/v1")
    parser.add_argument("--embed-model", default="Qwen3-Embedding-4B")
    parser.add_argument("--embed-timeout", type=int, default=120)
    parser.add_argument("--openie-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--openie-cache-audit", action="store_true")
    parser.add_argument("--openie-cache-audit-bad-output", type=Path, default=None)
    parser.add_argument("--openie-cache-compact-output", type=Path, default=None)
    parser.add_argument("--openie-cache-only", action="store_true")
    parser.add_argument("--openie-cache-missing-output", type=Path, default=None)
    parser.add_argument("--openie-cache-warm-input", type=Path, default=None)
    parser.add_argument("--openie-cache-warm-limit", type=int, default=0)
    parser.add_argument("--openie-cache-warm-total-chunks", type=int, default=0)
    parser.add_argument("--openie-cache-warm-target-coverage", type=float, default=0.0)
    parser.add_argument("--openie-cache-warm-dry-run", action="store_true")
    parser.add_argument("--openie-cache-warm-pending-output", type=Path, default=None)
    parser.add_argument("--openie-cache-warm-failure-output", type=Path, default=None)
    parser.add_argument(
        "--openie-model-profile",
        choices=("", "qwen36_local", "deepseek_v4_flash", "generic_openai_compatible"),
        default="",
    )
    parser.add_argument("--openie-seed", type=int, default=42)
    parser.add_argument("--openie-source-limit", type=int, default=1_000_000)
    parser.add_argument("--openie-max-chunks", type=int, default=200)
    parser.add_argument("--openie-min-candidate-entities", type=int, default=2)
    parser.add_argument("--openie-max-candidate-df-ratio", type=float, default=0.3)
    parser.add_argument("--openie-sample-rate", type=float, default=1.0)
    parser.add_argument("--openie-max-concurrency", type=int, default=4)
    parser.add_argument("--openie-max-output-tokens", type=int, default=1024)
    parser.add_argument("--openie-max-triples-per-chunk", type=int, default=24)
    parser.add_argument(
        "--openie-relation-whitelist",
        default="depends_on,part_of,is_a,related,caused,produced,contradicts,supersedes",
    )
    parser.add_argument("--search-limit", type=int, default=30)
    parser.add_argument("--relation-probe-limit", type=int, default=20)
    parser.add_argument("--relation-probe-max-per-source-relation", type=int, default=3)
    parser.add_argument("--min-relation-expanded-lift", type=int, default=0)
    parser.add_argument("--min-relation-evidence-lift", type=int, default=0)
    parser.add_argument("--min-strong-relation-evidence-rate", type=float, default=0.0)
    parser.add_argument("--min-openie-cache-coverage", type=float, default=0.0)
    parser.add_argument("--min-delta-r5", type=float, default=0.0)
    parser.add_argument(
        "--verify-revertibility", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fail-on-gate", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def preflight_dependencies(args: argparse.Namespace) -> None:
    """Fail early when HTTP-backed providers are requested without aiohttp."""
    warming_cache = bool(args.openie_cache_warm_input)
    warming_dry_run = bool(args.openie_cache_warm_dry_run)
    llm_http_requested = (
        (warming_cache or not args.skip_openie)
        and args.llm_base_url
        and not args.openie_cache_only
        and not warming_dry_run
    )
    embed_requested = bool(args.embed_base_url) and not warming_cache
    if not (embed_requested or llm_http_requested):
        return
    try:
        import aiohttp  # noqa: F401
    except ModuleNotFoundError as exc:
        msg = (
            "aiohttp is required for --embed-base-url or --llm-base-url. "
            "Run with `uv run --extra sqlite --extra embedding ...`, "
            "or pass `--embed-base-url ''` for an LLM-free/local-index smoke."
        )
        raise SystemExit(msg) from exc
    if llm_http_requested and _llm_api_key_required(args.llm_base_url):
        key_name = str(args.llm_api_key_env or "")
        if not key_name or not os.environ.get(key_name):
            msg = (
                f"{key_name or '--llm-api-key-env'} is required for remote LLM "
                f"base URL {args.llm_base_url!r}. Export it in the shell, or use "
                "`--openie-cache-warm-dry-run` to plan coverage without API calls."
            )
            raise SystemExit(msg)


async def main() -> int:
    args = parse_args()
    if args.openie_cache_audit:
        audit_summary = audit_openie_cache(args)
        write_cache_audit_results(args, audit_summary)
        if args.fail_on_gate and not audit_summary.passed:
            return 2
        return 0

    if args.openie_cache_warm_input:
        preflight_dependencies(args)
        warm_summary = await warm_openie_cache(args)
        write_cache_warm_results(args, warm_summary)
        if args.fail_on_gate and warm_summary.extraction_failures:
            return 2
        return 0

    if not args.chunks.exists():
        print(f"ERROR: chunks file not found: {args.chunks}")
        return 1
    queries = load_queries(args.queries)
    if not queries:
        print("ERROR: no queries loaded")
        return 1
    preflight_dependencies(args)

    await build_baseline_db(args)
    openie_run = await apply_openie(args)
    query_embeddings = await precompute_query_embeddings(queries, args)
    base = await score_db(
        "baseline",
        args.baseline_db,
        queries,
        args,
        query_embeddings=query_embeddings or None,
    )
    scores_reused = False
    if args.skip_openie:
        openie = base.clone_as("openie")
        scores_reused = True
        print("[score] skip-openie: reused baseline scores for copied openie DB")
    else:
        openie = await score_db(
            "openie",
            args.openie_db,
            queries,
            args,
            query_embeddings=query_embeddings or None,
        )
    relation_probe = await relation_probe_for_db(
        args.openie_db,
        limit=0 if args.skip_openie else args.relation_probe_limit,
        search_k=min(10, max(1, args.search_limit)),
        max_per_source_relation=args.relation_probe_max_per_source_relation,
    )
    if relation_probe.n:
        print(
            "[probe] relation targets "
            f"expanded={relation_probe.graph_expanded_hits}/{relation_probe.n} "
            f"(no_graph={relation_probe.no_graph_expanded_hits}/{relation_probe.n}) "
            f"evidence={relation_probe.graph_evidence_hits}/{relation_probe.n} "
            f"(no_graph={relation_probe.no_graph_evidence_hits}/{relation_probe.n})"
        )
    print_comparison(base, openie)
    gates = await evaluate_gates(
        args,
        base=base,
        openie=openie,
        openie_run=openie_run,
        relation_probe=relation_probe,
    )
    memory_health = await memory_health_for_db(args.openie_db)
    write_results(
        args,
        base=base,
        openie=openie,
        openie_run=openie_run,
        gates=gates,
        memory_health=memory_health,
        relation_probe=relation_probe,
        scores_reused=scores_reused,
    )
    if args.fail_on_gate and not gates.passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
