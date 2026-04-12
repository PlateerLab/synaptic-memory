"""SqliteGraphBackend — SQLite backend with graph traversal extensions.

Phase D of the Graphiti-style pluggable backend strategy. Subclasses
:class:`SQLiteBackend` and adds the :class:`GraphTraversal` protocol
methods (``shortest_path``, ``find_by_type_hierarchy``, ``pattern_match``)
so the zero-infra SQLite store can serve as the new default backend.

Storage is inherited unchanged — same ``syn_nodes`` / ``syn_edges``
tables, same FTS5 virtual table, same recursive-CTE ``get_neighbors``.
Use this class when you want graph traversal on top of SQLite; use the
parent :class:`SQLiteBackend` when CRUD + FTS is enough.

Why subclass rather than add to SQLiteBackend directly:

1. Keeps the minimal CRUD backend minimal for users who don't need
   traversal (smaller mental surface, no accidental recursion cost).
2. Makes intent explicit at construction time — ``SqliteGraphBackend``
   tells the reader "this instance is expected to support multi-hop
   reasoning".
3. Lets us iterate on the traversal implementation without risking
   the stability of the base class that's been shipping since v0.5.

Kuzu parity notes:

- :meth:`shortest_path` uses Python-level BFS calling ``get_neighbors``
  once per depth layer. For mid-sized corpora (~20K nodes, depth ≤ 3)
  this is fast enough. Future work can push it into a single recursive
  CTE if measurement shows it's a hotspot.
- :meth:`find_by_type_hierarchy` is currently a flat ``list_nodes(kind=)``
  call, matching Kuzu's behaviour ("hierarchy expansion TBD").
- :meth:`pattern_match` raises ``NotImplementedError``. Cypher pattern
  matching is not meaningfully expressible in SQL without a full
  parser; callers who need it should use :class:`KuzuBackend`.

Example::

    from synaptic.backends.sqlite_graph import SqliteGraphBackend

    backend = SqliteGraphBackend("graph.db")
    await backend.connect()

    # All SQLiteBackend methods work
    await backend.save_node(node)

    # Plus GraphTraversal extensions
    path = await backend.shortest_path(a.id, b.id, max_depth=4)
"""

from __future__ import annotations

from synaptic.backends.sqlite import SQLiteBackend
from synaptic.models import Edge, Node


class SqliteGraphBackend(SQLiteBackend):
    """SQLite storage + GraphTraversal protocol.

    See module docstring for rationale. Construction, schema, and all
    CRUD methods are inherited from :class:`SQLiteBackend`.
    """

    __slots__ = ()  # inherits _conn, _path from SQLiteBackend

    # --- GraphTraversal protocol ---

    async def shortest_path(
        self,
        from_id: str,
        to_id: str,
        *,
        max_depth: int = 5,
    ) -> list[tuple[Node, Edge]]:
        """BFS shortest path from ``from_id`` to ``to_id``.

        Returns the ordered sequence of ``(node, edge)`` tuples along
        the path, **excluding the start node**. Matches the Kuzu
        backend's return contract so callers can swap backends without
        changing their code.

        Args:
            from_id: Source node id.
            to_id: Target node id.
            max_depth: Maximum edges to traverse. Defaults to 5.

        Returns:
            A list of ``(node, edge)`` pairs describing the path. Empty
            list if ``from_id == to_id`` or if no path exists within
            ``max_depth``.
        """
        if from_id == to_id:
            return []

        depth_cap = max(1, int(max_depth))
        visited: set[str] = {from_id}
        # BFS queue entries are (current_node_id, path_so_far).
        frontier: list[tuple[str, list[tuple[Node, Edge]]]] = [(from_id, [])]

        for _ in range(depth_cap):
            next_frontier: list[tuple[str, list[tuple[Node, Edge]]]] = []
            for current, path in frontier:
                hops = await self.get_neighbors(current, depth=1)
                for node, edge in hops:
                    if node.id in visited:
                        continue
                    new_path = [*path, (node, edge)]
                    if node.id == to_id:
                        return new_path
                    visited.add(node.id)
                    next_frontier.append((node.id, new_path))
            if not next_frontier:
                break
            frontier = next_frontier

        return []

    async def find_by_type_hierarchy(
        self,
        type_name: str,
        *,
        limit: int = 50,
    ) -> list[Node]:
        """Find nodes whose ``kind`` equals ``type_name``.

        Currently flat — does not walk an is-a hierarchy. This matches
        Kuzu's current behaviour; hierarchy expansion is planned once
        the runtime ontology registry wiring stabilizes.
        """
        return await self.list_nodes(kind=type_name, limit=limit)

    async def pattern_match(
        self,
        pattern: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        """Cypher pattern matching is not supported on SQLite.

        Raises ``NotImplementedError`` with a clear redirect. For
        multi-hop reasoning on SQLite, combine :meth:`get_neighbors` and
        :meth:`shortest_path` with Python-level filtering. If you need
        full openCypher, use :class:`synaptic.backends.kuzu.KuzuBackend`.

        Args:
            pattern: Ignored.
            limit: Ignored.
        """
        msg = (
            "SqliteGraphBackend does not support Cypher pattern_match. "
            "Use get_neighbors + shortest_path for traversal, or switch "
            "to KuzuBackend for Cypher queries."
        )
        raise NotImplementedError(msg)
