"""Auto graph snapshot — markdown report of corpus structure.

Why this exists
---------------
v0.18-α1-2 KRRA Conv diagnostic (and the broader 6-bench agent
measurement) showed that LLM agents waste their first 1-2 turns on
*exploration*: probing for which categories exist, what tables are
available, what the dominant entities look like. Each turn costs latency
and budget; an exploration turn that produces no answer is a turn the
agent can't use to actually reason.

This module pre-computes a compact markdown summary of the graph that
can be injected into the system prompt at the start of a session, so
the agent already knows:

- Document scale: how many documents / chunks / structured rows
- Categories: the human-readable category tree (with doc counts)
- Top phrase hubs: the 10-15 most-mentioned entities (DF rank)
- Tables: structured-data tables, row counts, sample columns + FKs
- Edge statistics: count by EdgeKind so the agent knows what `follow`
  edges to expect
- Sample queries: 2-3 short illustrations of the kind of question this
  corpus can answer

All stats are computed from direct backend reads — **no LLM calls**.
This preserves Synaptic's "indexing-time LLM = 0" principle. Cost is
~5 backend calls + N (entities, capped at 50k) on the slowest path.

Compared to ``synaptic.search_session.build_graph_context`` (which
exists and is already used in the agent loop), the snapshot is:
- **Markdown-shaped** (heading hierarchy, tables) for human + LLM
  readability — `build_graph_context` is bracket-tagged terse output
- **Adds top-entities + edge-kind stats** for cold-start priming —
  `build_graph_context` only covers categories + tables
- **Standalone CLI / MCP-tool emittable** so users can preview what
  their graph "looks like" before hooking up an agent
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("snapshot")


@dataclass(slots=True)
class SnapshotStats:
    """Numeric snapshot of graph structure — useful for tests / programmatic checks."""

    n_documents: int = 0
    n_chunks: int = 0
    n_entities_phrase: int = 0
    n_entities_structured: int = 0
    n_categories: int = 0
    n_edges_total: int = 0
    edges_by_kind: dict[str, int] | None = None
    tables: dict[str, int] | None = None
    top_phrase_hubs: list[tuple[str, int]] | None = None  # (title, mention_count)
    categories: list[tuple[str, int]] | None = None  # (title, doc_count)


async def collect_stats(
    backend: StorageBackend,
    *,
    max_entities_scanned: int = 5_000,
    top_n_phrase_hubs: int = 15,
    top_n_categories: int = 30,
    edge_sample_size: int = 50,
    hub_probe_size: int = 50,
) -> SnapshotStats:
    """Compute the numeric statistics that the markdown snapshot reports.

    Separated from ``generate_snapshot`` so callers (tests, MCP tools
    that want JSON) can get the raw numbers without rendering markdown.

    Performance budget: target < 5 s on a 100k-node SQLite graph. The
    expensive operations are per-node ``get_edges`` round-trips, so
    they're sampled — exact totals matter less than structural shape
    for a priming snapshot.
    """
    from synaptic.models import EdgeKind, NodeKind

    # "Document" is not a single NodeKind — different ingesters write
    # different kinds (RULE for policy docs, ARTIFACT for files, plain
    # passes through as ENTITY/CONCEPT). Sum what's plausibly a document.
    n_docs = 0
    n_chunks = 0
    for kind, target in (
        (NodeKind.RULE, "n_documents"),
        (NodeKind.ARTIFACT, "n_documents"),
        (NodeKind.LESSON, "n_documents"),
        (NodeKind.DECISION, "n_documents"),
        (NodeKind.CHUNK, "n_chunks"),
    ):
        try:
            c = await backend.count_nodes(kind=kind)
        except Exception:
            c = 0
        if target == "n_documents":
            n_docs += c
        else:
            n_chunks += c

    # Categories — CONCEPT nodes tagged "category". Doc count via
    # incoming PART_OF edges (one round-trip per category — bounded
    # because typical corpora have ≤ 30 categories).
    n_categories = 0
    cat_counts: list[tuple[str, int]] = []
    try:
        cats = await backend.list_nodes(kind=NodeKind.CONCEPT, limit=200)
    except Exception:
        cats = []
    for cat in cats:
        if "category" not in (cat.tags or []):
            continue
        n_categories += 1
        try:
            edges = await backend.get_edges(cat.id, direction="incoming")
            doc_count = sum(1 for e in edges if str(e.kind) == str(EdgeKind.PART_OF))
        except Exception:
            doc_count = 0
        cat_counts.append((cat.title, doc_count))
    cat_counts.sort(key=lambda x: -x[1])
    cat_counts = cat_counts[:top_n_categories]

    # Entities: split phrase-hub vs structured-row. Both share
    # NodeKind.ENTITY but structured ones carry ``_table_name`` property.
    # Capped scan: 5k entities is enough to characterise the table
    # distribution and surface common phrase hubs for any reasonable
    # corpus. Larger graphs lose precision on phrase-hub ranking but
    # snapshot stays bounded in time.
    n_phrase = 0
    n_structured = 0
    table_counts: Counter[str] = Counter()
    try:
        entities = await backend.list_nodes(kind=NodeKind.ENTITY, limit=max_entities_scanned)
    except Exception:
        entities = []
    phrase_candidates = []
    for e in entities:
        props = e.properties or {}
        tbl = props.get("_table_name")
        if tbl:
            n_structured += 1
            table_counts[tbl] += 1
        else:
            n_phrase += 1
            if e.title and len(e.title) <= 30:
                phrase_candidates.append(e)

    # Top phrase hubs by mention count — probe only ``hub_probe_size``
    # candidates (default 50) to keep wall-time under a second. The
    # rendered hubs are "good search anchors", not a precise top-N, so
    # missing one or two doesn't hurt agent priming.
    phrase_hubs: list[tuple[str, int]] = []
    for hub in phrase_candidates[:hub_probe_size]:
        try:
            edges = await backend.get_edges(hub.id, direction="incoming")
            mentions = sum(1 for ed in edges if str(ed.kind) == str(EdgeKind.MENTIONS))
        except Exception:
            mentions = 0
        if mentions > 0:
            phrase_hubs.append((hub.title, mentions))
    phrase_hubs.sort(key=lambda x: -x[1])
    phrase_hubs = phrase_hubs[:top_n_phrase_hubs]

    # Edge kind sample — ``edge_sample_size`` nodes (default 50). We
    # report counts as "(sampled)" so callers know not to treat as
    # exact graph-wide totals.
    edges_by_kind: Counter[str] = Counter()
    try:
        sample_nodes_for_edges = await backend.list_nodes(limit=edge_sample_size)
    except Exception:
        sample_nodes_for_edges = []
    for n in sample_nodes_for_edges:
        try:
            edges = await backend.get_edges(n.id, direction="outgoing")
        except Exception:
            edges = []
        for ed in edges:
            edges_by_kind[str(ed.kind)] += 1
    n_edges_total = sum(edges_by_kind.values())

    return SnapshotStats(
        n_documents=n_docs,
        n_chunks=n_chunks,
        n_entities_phrase=n_phrase,
        n_entities_structured=n_structured,
        n_categories=n_categories,
        n_edges_total=n_edges_total,
        edges_by_kind=dict(edges_by_kind),
        tables=dict(table_counts),
        top_phrase_hubs=phrase_hubs,
        categories=cat_counts,
    )


def render_markdown(
    stats: SnapshotStats,
    *,
    include_sample_queries: bool = True,
    title: str = "Knowledge Graph Snapshot",
) -> str:
    """Render a ``SnapshotStats`` instance as a markdown report.

    Caps output around ~3-5k tokens by default — within easy injection
    range for any modern LLM context window. Sections that have no data
    (e.g. no structured tables) are silently omitted.
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")
    lines: list[str] = [f"# {title}", "", f"> Generated {ts} · Synaptic Memory", ""]

    # --- Scale ---
    lines.append("## Scale")
    lines.append("")
    if stats.n_documents:
        lines.append(f"- **Documents**: {stats.n_documents:,}")
    if stats.n_chunks:
        lines.append(f"- **Chunks**: {stats.n_chunks:,}")
    if stats.n_entities_phrase:
        lines.append(f"- **Phrase hubs**: {stats.n_entities_phrase:,}")
    if stats.n_entities_structured:
        lines.append(f"- **Structured rows**: {stats.n_entities_structured:,}")
    if stats.n_categories:
        lines.append(f"- **Categories**: {stats.n_categories}")
    if stats.n_edges_total:
        lines.append(f"- **Edges (sampled)**: {stats.n_edges_total:,}")
    lines.append("")

    # --- Categories ---
    if stats.categories:
        lines.append("## Categories")
        lines.append("")
        lines.append("Use these names as the ``category`` parameter in ``deep_search``.")
        lines.append("")
        for cat_title, doc_count in stats.categories:
            lines.append(f"- {cat_title} ({doc_count} docs)")
        lines.append("")

    # --- Tables (structured data) ---
    if stats.tables:
        lines.append("## Tables")
        lines.append("")
        lines.append(
            "Use ``filter_nodes`` / ``aggregate_nodes`` / ``join_related`` "
            "for exact queries on these tables."
        )
        lines.append("")
        for tbl, count in sorted(stats.tables.items(), key=lambda kv: -kv[1]):
            lines.append(f"- ``{tbl}`` ({count:,} rows)")
        lines.append("")

    # --- Top phrase hubs ---
    if stats.top_phrase_hubs:
        lines.append("## Top phrase hubs (by mention count)")
        lines.append("")
        lines.append("Frequently-mentioned terms — likely good search anchors.")
        lines.append("")
        for title_, mentions in stats.top_phrase_hubs:
            lines.append(f"- {title_} ({mentions} mentions)")
        lines.append("")

    # --- Edge kinds ---
    if stats.edges_by_kind:
        lines.append("## Edge types (sampled)")
        lines.append("")
        lines.append("Use these as ``edge_kind`` in ``follow``:")
        lines.append("")
        for kind, count in sorted(stats.edges_by_kind.items(), key=lambda kv: -kv[1]):
            lines.append(f"- ``{kind}`` ({count:,})")
        lines.append("")

    # --- Sample query hints ---
    if include_sample_queries:
        hints: list[str] = []
        if stats.tables:
            largest_tbl = max(stats.tables.items(), key=lambda kv: kv[1])[0]
            hints.append(
                f'Aggregate / filter: ``aggregate_nodes(table="{largest_tbl}", '
                f'group_by="<column>", metric="count")``'
            )
        if stats.top_phrase_hubs:
            top_hub = stats.top_phrase_hubs[0][0]
            hints.append(f'Topic search: ``deep_search(query="{top_hub}")``')
        if stats.categories:
            first_cat = stats.categories[0][0]
            hints.append(
                f'Category-scoped search: ``deep_search(query="<your topic>", '
                f'category="{first_cat}")``'
            )
        if hints:
            lines.append("## Sample queries")
            lines.append("")
            for h in hints:
                lines.append(f"- {h}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


async def generate_snapshot(
    backend: StorageBackend,
    *,
    max_entities_scanned: int = 5_000,
    top_n_phrase_hubs: int = 15,
    top_n_categories: int = 30,
    include_sample_queries: bool = True,
    title: str = "Knowledge Graph Snapshot",
) -> str:
    """One-shot snapshot generator: collect stats + render markdown.

    Returns the markdown string. For programmatic callers that want
    raw numbers, use :func:`collect_stats` directly.
    """
    stats = await collect_stats(
        backend,
        max_entities_scanned=max_entities_scanned,
        top_n_phrase_hubs=top_n_phrase_hubs,
        top_n_categories=top_n_categories,
    )
    return render_markdown(
        stats,
        include_sample_queries=include_sample_queries,
        title=title,
    )
