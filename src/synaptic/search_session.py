"""SearchSession — stateful context for multi-turn agent search.

The synaptic-memory tool layer is designed for **LLM agents**: the
same agent will call `search`, look at the results, decide what to
explore next, call `expand` or `get_document`, and loop. Across those
calls we need to remember:

- Which nodes the agent has already seen (don't serve them again).
- Which queries have been tried (so the agent can tell when it's
  exhausted obvious angles).
- How much budget is left (so the agent knows when to stop exploring
  and synthesise an answer).
- Which categories / anchors have been touched (so the agent can
  reason about coverage — "have I checked every category that could
  hold the answer yet?").

A ``SearchSession`` is a thin data bag carrying exactly that state.
Every tool in :mod:`synaptic.agent_tools` takes a session as its
first argument, reads from it to personalise its response, and writes
back the new state. Without sessions the tool layer would collapse
to single-shot retrieval and lose the whole point of the agent loop.

The session is **in-process** — if you need persistence across
process restarts, wrap it in your own storage. For the typical
"one process per agent conversation" deployment the in-memory
``SessionStore`` here is enough.

Example::

    from synaptic.search_session import SessionStore

    store = SessionStore()
    session = store.create()

    # First turn — search for something
    session.record_query("인권경영 기본계획")
    session.mark_seen(["chunk_a", "chunk_b"])

    # Second turn — the agent can see what's been done
    print(session.summary())
    # {
    #   "session_id": "...",
    #   "seen_nodes": 2,
    #   "queries_tried": 1,
    #   "budget_remaining": 19,
    #   ...
    # }
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("search-session")


@dataclass(slots=True)
class SearchSession:
    """Per-agent conversation state for multi-turn tool use.

    Attributes:
        session_id: Stable identifier. Generated if not supplied.
            Keep it short (UUID4 hex truncated to 12 chars) so LLM
            prompts stay compact when the agent is asked to carry
            the id across turns.
        budget_tool_calls: How many tool calls the agent is allowed
            to make in this session. The tool layer decrements
            ``tool_calls_used`` every time a tool runs. When the
            budget is exhausted the tools start refusing with a
            structured "budget_exceeded" hint so the LLM knows to
            stop exploring and answer.
        tool_calls_used: Running counter of tool invocations. Always
            monotonically increases.
        seen_node_ids: Set of node ids the agent has already received
            in a result. Tools filter these out by default so the
            agent never reads the same chunk twice. Call ``.clear()``
            explicitly if you want a fresh lap.
        queries_tried: Ordered list of query strings that have been
            sent to a search-style tool. Used for the "have I tried
            this already?" check in the LLM prompt.
        categories_explored: Set of category labels the agent has
            pulled into its context via ``list_categories`` or
            through anchor extraction. Lets the LLM reason about
            topical coverage.
        facts: Free-form scratch dict for the tools to stash small
            pieces of derived state. Current examples:
            - ``"last_query_anchors"``: the QueryAnchors dict from
              the most recent search call (so follow-up tools can
              reuse them without re-extracting).
            - ``"last_evidence_ids"``: ids of the last evidence set
              returned, useful when the agent asks the model to
              "expand the top result".
        created_at: Unix timestamp at creation. Read-only for users;
            the session store uses it for TTL eviction.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    budget_tool_calls: int = 20
    tool_calls_used: int = 0
    seen_node_ids: set[str] = field(default_factory=set)
    queries_tried: list[str] = field(default_factory=list)
    categories_explored: set[str] = field(default_factory=set)
    expanded_nodes: set[str] = field(default_factory=set)
    facts: dict[str, Any] = field(default_factory=dict)
    # Per-session memoization of (tool, args) → result. Caps redundant
    # API/tool calls when the agent re-issues the same request, and
    # cheaply detects accidental loops (same call hash 3+ times in a
    # row signals trouble — caller can short-circuit). Values are kept
    # as the raw ``ToolResult`` dicts so they can be served back to the
    # LLM without re-running the underlying scan.
    tool_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # --- Mutation helpers ---

    def record_call(self) -> None:
        """Increment the tool-call counter.

        Every atomic tool calls this once at the start of its run.
        The counter is independent of whether the tool found results
        or not — a fruitless call still burns budget, matching how an
        agent actually perceives exploration cost.
        """
        self.tool_calls_used += 1

    def record_query(self, query: str) -> None:
        """Append a query to the history if it isn't already there.

        Deduplication is intentional: if the agent re-tries the same
        query verbatim the entry isn't duplicated, which keeps the
        summary readable. The tools also use this to skip repeated
        FTS round-trips.
        """
        query = (query or "").strip()
        if not query or query in self.queries_tried:
            return
        self.queries_tried.append(query)

    def mark_seen(self, node_ids) -> None:
        """Remember that the agent has received these node ids.

        Accepts any iterable — the tools pass lists or generators
        depending on what's cheap at the call site. Empty / falsy
        entries are ignored so callers can pass raw tool output
        without prefiltering.
        """
        for nid in node_ids:
            if nid:
                self.seen_node_ids.add(nid)

    def mark_categories(self, labels) -> None:
        """Remember category labels the agent has touched."""
        for label in labels:
            if label:
                self.categories_explored.add(label)

    def set_fact(self, key: str, value: Any) -> None:
        """Stash a small bit of derived state keyed by tool convention."""
        self.facts[key] = value

    def cache_key(self, tool_name: str, args: dict[str, Any]) -> str:
        """Compute a stable hash for a (tool, args) pair.

        Used by the dispatcher / agent loop to memoize tool results
        within a session. Keys are deterministic across runs so a
        replayed session sees the same caching behaviour. ``None``
        and falsy values are dropped from the args before hashing so
        ``filter(table='', limit=None)`` and ``filter()`` collide.
        """
        import hashlib
        import json as _json

        clean = {k: v for k, v in (args or {}).items() if v not in (None, "", [], {})}
        payload = _json.dumps([tool_name, clean], sort_keys=True, ensure_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def cache_get(self, key: str) -> dict[str, Any] | None:
        """Return a cached tool result for ``key`` or ``None``."""
        return self.tool_cache.get(key)

    def cache_put(self, key: str, result: dict[str, Any]) -> None:
        """Store a tool result under ``key`` for later memoized lookup."""
        # Cap cache size to prevent unbounded growth in long sessions.
        if len(self.tool_cache) >= 256:
            # Drop the oldest 64 entries (insertion-order in py3.7+ dict)
            for stale_key in list(self.tool_cache.keys())[:64]:
                del self.tool_cache[stale_key]
        self.tool_cache[key] = result

    # --- Query helpers ---

    def has_seen(self, node_id: str) -> bool:
        """``True`` if the agent has already received this node id."""
        return node_id in self.seen_node_ids

    def filter_unseen(self, node_ids) -> list[str]:
        """Return only the node ids that the agent has not seen yet.

        Order-preserving so the caller's original ranking is
        respected — seen entries are simply dropped, not reshuffled.
        """
        return [nid for nid in node_ids if nid and nid not in self.seen_node_ids]

    def budget_remaining(self) -> int:
        """How many tool calls the agent may still make."""
        return max(0, self.budget_tool_calls - self.tool_calls_used)

    def is_exhausted(self) -> bool:
        """``True`` when no more tool calls are allowed.

        Tools check this at the top of every run and short-circuit
        with a ``budget_exceeded`` hint when it returns ``True``.
        """
        return self.budget_remaining() <= 0

    def summary(self) -> dict[str, Any]:
        """Compact snapshot suitable for logging or LLM prompts.

        Returns a plain dict (not a dataclass) so it's trivially
        JSON-serialisable. The size is kept small on purpose —
        the agent prompt has a cost and dumping every seen node id
        would blow the context.
        """
        return {
            "session_id": self.session_id,
            "tool_calls_used": self.tool_calls_used,
            "budget_remaining": self.budget_remaining(),
            "seen_nodes": len(self.seen_node_ids),
            "queries_tried": len(self.queries_tried),
            "last_queries": self.queries_tried[-3:],
            "categories_explored": sorted(self.categories_explored),
            "expanded_nodes": len(self.expanded_nodes),
        }


async def build_graph_context(backend: StorageBackend) -> str:
    """Build a compact graph metadata string for system prompt injection.

    Calling ``list_categories`` every session wastes 1-2 turns. This
    function pre-builds the metadata once and injects it into the
    system prompt so the agent already knows the graph structure.

    Returns a string like:
        [Graph metadata]
        Categories (10): 규정 및 지침(235), 운영계획(315), ...
        Total: 1,110 documents, 18,600 chunks
        Node types: RULE, DECISION, OBSERVATION, ...
    """
    try:
        from synaptic.models import EdgeKind, NodeKind

        # Categories + doc counts
        cats = await backend.list_nodes(kind=NodeKind.CONCEPT, limit=100)
        cat_entries = []
        for cat in cats:
            if "category" not in (cat.tags or []):
                continue
            try:
                edges = await backend.get_edges(cat.id, direction="incoming")
                doc_count = sum(1 for e in edges if str(e.kind) == str(EdgeKind.PART_OF))
            except Exception:
                doc_count = 0
            cat_entries.append(f"{cat.title}({doc_count})")

        # Total counts
        total_docs = await backend.count_nodes(kind=None)

        # Count nodes by kind to distinguish document vs structured graphs.
        # Structured entities are identified by the ``_table_name`` property
        # stamped by TableIngester / DbIngester — raw ENTITY nodes from
        # phrase extraction don't count as structured.
        try:
            doc_count = await backend.count_nodes(kind=NodeKind.DOCUMENT)
        except Exception:
            doc_count = 0
        try:
            chunk_count = await backend.count_nodes(kind=NodeKind.CHUNK)
        except Exception:
            chunk_count = 0

        lines = [
            "[Graph metadata — use this instead of calling list_categories]",
            f"Categories ({len(cat_entries)}): {', '.join(cat_entries)}",
            f"Total nodes: {total_docs}",
            "Use category names above as the 'category' parameter in search.",
        ]

        # --- Structured data: table schemas ---
        # Detect tables from _table_name property and sample columns.
        structured_row_count = 0
        try:
            sample_nodes = await backend.list_nodes(kind=NodeKind.ENTITY, limit=50_000)
            tables: dict[str, dict[str, set[str]]] = {}  # table -> col -> sample values
            table_counts: dict[str, int] = {}
            for n in sample_nodes:
                props = n.properties or {}
                tbl = props.get("_table_name")
                if not tbl:
                    continue
                table_counts[tbl] = table_counts.get(tbl, 0) + 1
                structured_row_count += 1
                if tbl not in tables:
                    tables[tbl] = {}
                for k, v in props.items():
                    if k.startswith("_"):
                        continue
                    if k not in tables[tbl]:
                        tables[tbl][k] = set()
                    if len(tables[tbl][k]) < 3 and v:
                        tables[tbl][k].add(str(v)[:50])

            if tables:
                lines.append("")
                lines.append("[Structured data — tables and columns for filter/aggregate/join]")
                for tbl in sorted(tables, key=lambda t: -table_counts.get(t, 0)):
                    cnt = table_counts.get(tbl, 0)
                    cols = tables[tbl]
                    col_desc = []
                    for col, samples in sorted(cols.items()):
                        sample_str = ", ".join(sorted(samples)[:3])
                        col_desc.append(f"  {col}: e.g. {sample_str}")
                    lines.append(f"Table: {tbl} ({cnt} rows)")
                    lines.extend(col_desc)
            # --- FK relationships ---
            # Detect FK columns (_id, _no, _code suffix) and match values to PK tables.
            if tables:
                # Also collect actual PK from _primary_key property
                pk_tables: dict[str, str] = {}  # table -> pk column
                for n in sample_nodes:
                    props = n.properties or {}
                    tbl = props.get("_table_name")
                    pk_col = props.get("_primary_key")
                    if tbl and pk_col and tbl not in pk_tables:
                        pk_tables[tbl] = pk_col

                # For each table's FK-looking columns, find which table they reference
                fk_lines: list[str] = []
                for tbl, cols in tables.items():
                    for col in cols:
                        if not (
                            col.endswith("_no") or col.endswith("_id") or col.endswith("_code")
                        ):
                            continue
                        # Skip if this is the table's own PK
                        if pk_tables.get(tbl) == col:
                            continue
                        # Find target table where this column is the PK
                        for target_tbl, target_pk in pk_tables.items():
                            if target_tbl == tbl:
                                continue
                            if target_pk == col:
                                fk_lines.append(f"  {tbl}.{col} → {target_tbl}")
                                break

                if fk_lines:
                    lines.append("")
                    lines.append("[Foreign key relationships — use for join_related]")
                    lines.extend(fk_lines)

        except Exception:
            pass  # structured metadata is optional

        # Graph composition hint — tells the agent which tools fit which data.
        # Placed at the end so table schemas are already known.
        has_documents = doc_count > 0 or chunk_count > 0
        has_structured = structured_row_count > 0
        if has_documents or has_structured:
            lines.append("")
            lines.append("[Graph composition — match tool to data type]")
            if has_documents:
                lines.append(
                    f"- Document nodes: {doc_count} docs, {chunk_count} chunks "
                    f"→ use search/deep_search/get_document (NOT filter_nodes)"
                )
            if has_structured:
                lines.append(
                    f"- Structured rows: {structured_row_count} "
                    f"→ use filter_nodes/aggregate_nodes/join_related"
                )
            if has_documents and has_structured:
                lines.append(
                    "- MIXED GRAPH: Pick the tool that matches your query's data type. "
                    "Document questions need text search; row/column questions need structured tools."
                )

        return "\n".join(lines)
    except Exception as exc:
        logger.warning("build_graph_context failed: %s", exc)
        return ""


class SessionStore:
    """In-memory ``{session_id: SearchSession}`` map with TTL eviction.

    Deliberately simple: a dict behind a couple of helper methods.
    Production deployments that need cross-process sessions can
    subclass this or write their own store with the same two
    methods — ``create`` and ``get`` — and pass it to the MCP server.

    Sessions older than ``ttl_seconds`` are evicted lazily on
    ``create`` / ``get_or_create`` calls — we don't run a background
    thread. This prevents memory leaks on long-running MCP servers
    where clients open sessions and never close them.
    """

    __slots__ = ("_sessions", "_ttl")

    def __init__(self, *, ttl_seconds: float = 3600) -> None:
        self._sessions: dict[str, SearchSession] = {}
        self._ttl = ttl_seconds

    def _evict_expired(self) -> None:
        """Drop sessions older than TTL. Called lazily on access."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if (now - s.created_at) > self._ttl]
        for sid in expired:
            del self._sessions[sid]
            logger.debug("session-store: evicted expired session %s", sid)

    def create(
        self,
        *,
        session_id: str | None = None,
        budget_tool_calls: int = 20,
    ) -> SearchSession:
        """Create a new session and register it in the store.

        If ``session_id`` is supplied and already exists, the existing
        session is returned unchanged — this lets the agent keep
        using the same id across reconnects without accidentally
        resetting its own state.
        """
        self._evict_expired()
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        session = SearchSession(
            session_id=session_id or uuid.uuid4().hex[:12],
            budget_tool_calls=budget_tool_calls,
        )
        self._sessions[session.session_id] = session
        logger.debug("session-store: created %s", session.session_id)
        return session

    def get(self, session_id: str) -> SearchSession | None:
        """Look up a session by id, returning ``None`` if missing."""
        return self._sessions.get(session_id)

    def get_or_create(
        self,
        session_id: str | None = None,
        *,
        budget_tool_calls: int = 20,
    ) -> SearchSession:
        """Convenience method — get or make, never returns ``None``."""
        self._evict_expired()
        if session_id:
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing
        return self.create(
            session_id=session_id,
            budget_tool_calls=budget_tool_calls,
        )

    def delete(self, session_id: str) -> None:
        """Drop a session. No-op if the id isn't present."""
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions
