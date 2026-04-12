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
from typing import Any

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
    facts: dict[str, Any] = field(default_factory=dict)
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
        }


class SessionStore:
    """In-memory ``{session_id: SearchSession}`` map.

    Deliberately simple: a dict behind a couple of helper methods.
    Production deployments that need cross-process sessions can
    subclass this or write their own store with the same two
    methods — ``create`` and ``get`` — and pass it to the MCP server.

    The store never evicts sessions automatically. Caller decides
    when to call ``delete``; most agent sessions live for a single
    conversation and are dropped when the process exits.
    """

    __slots__ = ("_sessions",)

    def __init__(self) -> None:
        self._sessions: dict[str, SearchSession] = {}

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
