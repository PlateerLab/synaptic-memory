"""Tests for ``eval/build_metacorpus.py``.

Phase 1.4 of the v0.20+ track. The combiner is one-shot tooling but
its output is the data foundation for cross-domain evaluation, so
the contract is worth locking:

  1. Every node must carry properties._domain_id so the bench harness
     can score per-domain coverage.
  2. Pre-existing _domain_id (nested combines) must NOT be overwritten.
  3. INSERT OR IGNORE behaviour: collisions counted, not silent.
  4. Edges copied verbatim — combiner doesn't invent cross-domain
     edges; that comes later.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.build_metacorpus import build


def _make_source(path: Path, nodes: list[tuple], edges: list[tuple] | None = None) -> None:
    """Build a tiny syn_nodes / syn_edges sqlite for testing."""
    edges = edges or []
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE syn_nodes (
            id TEXT PRIMARY KEY, kind TEXT, title TEXT, content TEXT,
            tags_json TEXT, level TEXT, vitality REAL,
            access_count INTEGER, success_count INTEGER, failure_count INTEGER,
            source TEXT, properties_json TEXT, embedding_json TEXT,
            created_at REAL, updated_at REAL
        );
        CREATE TABLE syn_edges (
            id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT,
            kind TEXT, weight REAL, created_at REAL
        );
        """
    )
    for nid, kind, title, props in nodes:
        cur.execute(
            "INSERT INTO syn_nodes(id, kind, title, content, tags_json, level, "
            "vitality, access_count, success_count, failure_count, source, "
            "properties_json, embedding_json, created_at, updated_at) "
            "VALUES (?, ?, ?, '', '[]', 'L0', 1.0, 0, 0, 0, '', ?, '[]', 0, 0)",
            (nid, kind, title, json.dumps(props or {})),
        )
    for eid, sid, tid, kind in edges:
        cur.execute(
            "INSERT INTO syn_edges(id, source_id, target_id, kind, weight, created_at) "
            "VALUES (?, ?, ?, ?, 1.0, 0)",
            (eid, sid, tid, kind),
        )
    conn.commit()
    conn.close()


def test_every_node_gets_domain_id_tag(tmp_path):
    src = tmp_path / "a.sqlite"
    _make_source(src, [("n1", "concept", "X", {}), ("n2", "entity", "Y", {})])
    out = tmp_path / "out.sqlite"
    stats = build({"foo": src}, out)
    assert stats[0].nodes_inserted == 2
    conn = sqlite3.connect(out)
    rows = list(
        conn.execute(
            "SELECT id, json_extract(properties_json, '$._domain_id') FROM syn_nodes"
        )
    )
    assert {r[1] for r in rows} == {"foo"}, rows


def test_existing_domain_id_preserved(tmp_path):
    """Nested combines: if a node already has _domain_id (e.g. from a
    prior MetaCorpus build), the combiner must NOT overwrite it. Tests
    setdefault semantics — needed for incremental aggregation."""
    src = tmp_path / "a.sqlite"
    _make_source(src, [("n1", "concept", "X", {"_domain_id": "originally_legal"})])
    out = tmp_path / "out.sqlite"
    build({"new_domain": src}, out)
    conn = sqlite3.connect(out)
    row = conn.execute(
        "SELECT json_extract(properties_json, '$._domain_id') FROM syn_nodes"
    ).fetchone()
    assert row[0] == "originally_legal"


def test_node_id_collision_counted_not_silent(tmp_path):
    """Two sources both have node id 'shared' — second insert should be
    skipped + counted, not silently overwrite. We rely on this loud
    behaviour to know if phrase-hub collisions become a real problem
    at scale."""
    a = tmp_path / "a.sqlite"
    b = tmp_path / "b.sqlite"
    _make_source(a, [("shared", "entity", "from_a", {})])
    _make_source(b, [("shared", "entity", "from_b", {})])
    out = tmp_path / "out.sqlite"
    stats = build({"a": a, "b": b}, out)
    a_stats = next(s for s in stats if s.domain == "a")
    b_stats = next(s for s in stats if s.domain == "b")
    assert a_stats.nodes_inserted == 1
    assert b_stats.nodes_skipped == 1
    assert b_stats.other_collisions == 1
    # First-write-wins
    conn = sqlite3.connect(out)
    title = conn.execute("SELECT title FROM syn_nodes WHERE id='shared'").fetchone()[0]
    assert title == "from_a"


def test_phrase_hub_collisions_counted_separately(tmp_path):
    """Phrase hub IDs use the form ``phrase_<hash>`` derived from
    phrase text only. If two domains hash the same phrase ("operations")
    the collision is expected and recoverable (Phase 1.2 namespacing).
    Counted separately from ``other_collisions`` so the alarm bell
    only fires for real collisions."""
    a = tmp_path / "a.sqlite"
    b = tmp_path / "b.sqlite"
    _make_source(a, [("phrase_abc123", "entity", "operations", {})])
    _make_source(b, [("phrase_abc123", "entity", "operations", {})])
    out = tmp_path / "out.sqlite"
    stats = build({"a": a, "b": b}, out)
    b_stats = next(s for s in stats if s.domain == "b")
    assert b_stats.phrase_collisions == 1
    assert b_stats.other_collisions == 0


def test_edges_copied_verbatim(tmp_path):
    src = tmp_path / "a.sqlite"
    _make_source(
        src,
        nodes=[("n1", "entity", "A", {}), ("n2", "entity", "B", {})],
        edges=[("e1", "n1", "n2", "related")],
    )
    out = tmp_path / "out.sqlite"
    build({"foo": src}, out)
    conn = sqlite3.connect(out)
    rows = list(conn.execute("SELECT source_id, target_id, kind FROM syn_edges"))
    assert rows == [("n1", "n2", "related")]


def test_destination_overwritten_on_rebuild(tmp_path):
    """Re-running the combiner must produce a clean output, not an
    incremental merge — partial state is too easy to misread."""
    src = tmp_path / "a.sqlite"
    out = tmp_path / "out.sqlite"
    _make_source(src, [("n1", "concept", "first", {})])
    build({"foo": src}, out)
    # Mutate source
    src.unlink()
    _make_source(src, [("n2", "concept", "second", {})])
    build({"foo": src}, out)
    conn = sqlite3.connect(out)
    ids = [r[0] for r in conn.execute("SELECT id FROM syn_nodes")]
    assert ids == ["n2"], "rebuild should not retain n1 from prior run"


def test_per_domain_node_count_queryable(tmp_path):
    """End-to-end: after combining two domains, a single SQL query
    must accurately partition the nodes by domain. This is the access
    pattern the unified scorer / domain_coverage validator depends on."""
    a = tmp_path / "a.sqlite"
    b = tmp_path / "b.sqlite"
    _make_source(a, [(f"a{i}", "entity", f"A{i}", {}) for i in range(3)])
    _make_source(b, [(f"b{i}", "entity", f"B{i}", {}) for i in range(5)])
    out = tmp_path / "out.sqlite"
    build({"alpha": a, "beta": b}, out)
    conn = sqlite3.connect(out)
    rows = dict(
        conn.execute(
            "SELECT json_extract(properties_json, '$._domain_id'), COUNT(*) "
            "FROM syn_nodes GROUP BY 1"
        )
    )
    assert rows == {"alpha": 3, "beta": 5}
