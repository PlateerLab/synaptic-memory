"""V2 agent tools — compound tools that chain multiple primitives.

The v1 tools (``agent_tools.py``) are atomic: one tool = one operation.
The agent has to orchestrate them across multiple turns. This works but
is expensive — a complex query like "인권경영과 예산 편성의 관계" takes
6-10 turns because the agent has to:

    search → list_categories → search(cat) → get_doc → search(cat2) → get_doc2

The v2 tools solve this by **chaining internally**:

- ``deep_search``: search → auto-expand top hits → read relevant chunks
  → return a comprehensive result in ONE turn.
- ``compare``: decompose a multi-entity query → parallel search each →
  merge results.

These compound tools call the v1 primitives internally via
``asyncio.gather`` for parallelism. The agent sees them as single tools
that return richer results.

Typical turn reduction: 6-10 turns → 1-3 turns.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from synaptic.agent_tools import (
    Hint,
    ToolResult,
    _budget_check,
    expand_tool,
    get_document_tool,
    search_tool,
)
from synaptic.search_session import SearchSession

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("agent-tools-v2")


async def deep_search_tool(
    backend: StorageBackend,
    session: SearchSession,
    query: str,
    *,
    limit: int = 5,
    category: str | None = None,
    read_top_k: int = 2,
    embedder: object | None = None,
    reranker: object | None = None,
) -> ToolResult:
    """One-turn deep search: search → expand → read documents.

    Chains three v1 primitives internally:
    1. ``search`` with the query (+ optional category filter)
    2. ``expand`` the top hit to discover neighbours
    3. ``get_document`` on the top-k results with query-aware chunking

    Returns a single consolidated result: evidence list + expanded
    neighbours + document excerpts — all in one turn instead of 3-5.

    Args:
        backend: Storage backend.
        session: Active search session.
        query: User query.
        limit: Max evidence items from the initial search.
        category: Optional category filter.
        read_top_k: How many top documents to read in full.
        embedder: Optional embedder for EvidenceSearch.
        reranker: Optional cross-encoder reranker.
    """
    budget = _budget_check(session, "deep_search")
    if budget is not None:
        return budget

    session.record_query(query)

    # Step 1: search
    search_result = await search_tool(
        backend,
        session,
        query,
        limit=limit,
        category=category,
        embedder=embedder,
    )
    evidence = search_result.data.get("evidence", [])

    # Step 2: expand top hit (parallel with step 3)
    expanded_neighbours: list[dict] = []
    doc_excerpts: list[dict] = []

    if evidence:
        top_node_id = evidence[0].get("id", "")
        top_doc_ids = list(
            dict.fromkeys(e.get("document_id", "") for e in evidence if e.get("document_id"))
        )[:read_top_k]

        # Parallel: expand + get_documents
        tasks = []
        # Expand top hit
        if top_node_id:
            tasks.append(_safe_expand(backend, session, top_node_id))
        # Read top documents
        for doc_id in top_doc_ids:
            tasks.append(_safe_get_doc(backend, session, doc_id, query))

        results = await asyncio.gather(*tasks)

        for r in results:
            if r is None:
                continue
            if r.tool == "expand" and r.ok:
                expanded_neighbours = r.data.get("neighbours", [])
            elif r.tool == "get_document" and r.ok:
                doc_excerpts.append(
                    {
                        "document": r.data.get("document", {}),
                        "relevant_chunks": [
                            c for c in r.data.get("chunks", []) if c.get("relevant")
                        ],
                        "total_chunks": r.data.get("chunk_count", 0),
                    }
                )

    # Build consolidated response
    hints: list[Hint] = []
    if not evidence:
        # Decompose the query into its first content word and suggest
        # a FTS fallback. "try a different category" as a literal arg
        # (the prior hint) was being copied verbatim by the LLM and
        # failing — executable hints work, meta-hints don't.
        tokens = [t for t in query.split() if len(t) >= 2]
        if tokens:
            hints.append(
                Hint(
                    action="search",
                    args={"query": tokens[0]},
                    reason=(
                        "deep_search found nothing — retry plain FTS on the "
                        "first keyword alone; often the full question phrase "
                        "over-constrains BM25"
                    ),
                )
            )
        hints.append(
            Hint(
                action="list_categories",
                args={},
                reason="inspect available categories, then retry deep_search with category= filter",
            )
        )

    return ToolResult(
        tool="deep_search",
        ok=True,
        data={
            "evidence": evidence,
            "expanded_neighbours": expanded_neighbours[:5],
            "document_excerpts": doc_excerpts,
            "search_anchors": search_result.data.get("anchors", {}),
        },
        hints=hints,
        session=session.summary(),
    )


async def compare_search_tool(
    backend: StorageBackend,
    session: SearchSession,
    query: str,
    *,
    embedder: object | None = None,
) -> ToolResult:
    """Decompose a multi-topic query and search each in parallel.

    Splits queries containing "과", "와", "및", "관련", "관계" into
    sub-queries, searches each with category filtering, and merges
    results. Solves cross-document queries in 1 turn instead of 4-6.

    Example:
        "인권경영과 예산 편성의 관계"
        → sub1: search("인권경영")
        → sub2: search("예산 편성")
        → merge: both result sets with cross-references
    """
    budget = _budget_check(session, "compare_search")
    if budget is not None:
        return budget

    session.record_query(query)

    # Decompose query
    sub_queries = _decompose_query(query)

    if len(sub_queries) <= 1:
        # Not decomposable — fall back to regular deep_search
        return await deep_search_tool(backend, session, query, embedder=embedder)

    # Parallel search for each sub-query
    tasks = [search_tool(backend, session, sq, limit=5, embedder=embedder) for sq in sub_queries]
    results = await asyncio.gather(*tasks)

    # Merge results
    all_evidence: list[dict] = []
    sub_results: list[dict] = []
    for sq, r in zip(sub_queries, results):
        evidence = r.data.get("evidence", []) if r.ok else []
        sub_results.append(
            {
                "sub_query": sq,
                "evidence_count": len(evidence),
                "top_result": evidence[0] if evidence else None,
            }
        )
        all_evidence.extend(evidence)

    # Deduplicate by node id
    seen_ids: set[str] = set()
    unique_evidence: list[dict] = []
    for e in all_evidence:
        eid = e.get("id", "")
        if eid not in seen_ids:
            seen_ids.add(eid)
            unique_evidence.append(e)

    return ToolResult(
        tool="compare_search",
        ok=True,
        data={
            "original_query": query,
            "sub_queries": sub_results,
            "merged_evidence": unique_evidence[:10],
        },
        hints=[],
        session=session.summary(),
    )


# --- Helpers ---


def _decompose_query(query: str) -> list[str]:
    """Split a compound query into sub-queries by Korean conjunctions.

    "인권경영과 예산 편성의 관계" → ["인권경영", "예산 편성"]
    "승마 행사 및 대회 계획" → ["승마 행사", "대회 계획"]

    Returns the original query as a single-element list if no
    conjunction is found — the caller treats it as non-decomposable.
    """
    import re

    # Korean conjunctions that signal multi-topic queries
    parts = re.split(r"(?:과|와|및|그리고)\s+", query)
    # Clean up trailing particles
    cleaned = []
    for p in parts:
        p = re.sub(
            r"(의\s+관계|의\s+연관|에\s+대해|에\s+미치는|을\s+비교|를\s+비교)$", "", p
        ).strip()
        if len(p) >= 2:
            cleaned.append(p)

    if len(cleaned) < 2:
        return [query]
    return cleaned


async def _safe_expand(
    backend: StorageBackend,
    session: SearchSession,
    node_id: str,
) -> ToolResult | None:
    """Expand with error swallowing."""
    try:
        return await expand_tool(backend, session, node_id, limit=5)
    except Exception:
        return None


async def _safe_get_doc(
    backend: StorageBackend,
    session: SearchSession,
    doc_id: str,
    query: str,
) -> ToolResult | None:
    """Get document with error swallowing."""
    try:
        return await get_document_tool(
            backend,
            session,
            doc_id,
            query=query,
            max_full_chunks=3,
        )
    except Exception:
        return None
