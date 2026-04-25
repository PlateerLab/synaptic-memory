"""Combine N domain-specific sqlite corpora into one MetaCorpus.

Phase 1.4 of the v0.20+ track. Output: one sqlite file containing every
node + every edge from each source corpus, with each node tagged
``properties._domain_id = <domain>`` so cross-domain queries can score
per-domain coverage without changing the Node schema.

Why combine instead of federate at query time?
  - Existing backends are single-DB. Federated query = new code path.
  - Combining is one-time, side-effect-free, doesn't touch runtime.
  - Once cross-domain queries actually work on the combined DB, that
    proves the value before we invest in federation.

Node ID collision risk: source IDs are mostly 16-char MD5 hashes
(`doc_<hash>` / `chunk_<hash>` / `phrase_<hash>`). Across all 3
corpora (~150K nodes total), birthday-paradox collision probability
is ~10⁻¹². If a collision DOES happen at runtime, the second insert
will overwrite the first — loud rather than silent because the
combiner checks via INSERT OR IGNORE and reports duplicates.

Phrase hub IDs (``phrase_<md5>``) are the highest collision risk
because the same word ("operations") in two domains produces the same
hash. Phase 1.2 (deferred) namespaces phrase hubs as
``phrase_{domain}_{md5}``; until then, the combiner reports phrase
collisions so we know the magnitude.

CLI
===
    uv run python eval/build_metacorpus.py
        Default: combine krra + assort + x2bee → eval/data/metacorpus.sqlite

    uv run python eval/build_metacorpus.py --out custom.sqlite \\
        --source krra=eval/data/krra_graph.sqlite \\
        --source assort=eval/data/assort_graph.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"

DEFAULT_SOURCES: dict[str, Path] = {
    "krra": DATA_DIR / "krra_graph.sqlite",
    "assort": DATA_DIR / "assort_graph.sqlite",
    "x2bee": DATA_DIR / "x2bee_graph.sqlite",
}

DEFAULT_OUT = DATA_DIR / "metacorpus.sqlite"


@dataclass(slots=True)
class MergeStats:
    """Per-domain merge accounting — surfaced so collisions are visible."""

    domain: str
    nodes_read: int = 0
    nodes_inserted: int = 0
    nodes_skipped: int = 0
    edges_read: int = 0
    edges_inserted: int = 0
    edges_skipped: int = 0
    phrase_collisions: int = 0
    other_collisions: int = 0


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the same syn_nodes / syn_edges schema used by SqliteGraphBackend.

    Mirrors ``src/synaptic/backends/sqlite.py`` so this combined file
    drops in as a normal SqliteGraphBackend corpus. Indexes match what
    the FTS / vector index would otherwise rebuild on first open.
    """
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS syn_nodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL DEFAULT 'concept',
            title TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL DEFAULT '',
            tags_json TEXT NOT NULL DEFAULT '[]',
            level TEXT NOT NULL DEFAULT 'L0',
            vitality REAL NOT NULL DEFAULT 1.0,
            access_count INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            failure_count INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT '',
            properties_json TEXT NOT NULL DEFAULT '{}',
            embedding_json TEXT NOT NULL DEFAULT '[]',
            created_at REAL,
            updated_at REAL
        );
        CREATE TABLE IF NOT EXISTS syn_edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'related',
            weight REAL NOT NULL DEFAULT 1.0,
            created_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_nodes_kind ON syn_nodes(kind);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON syn_edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON syn_edges(target_id);
        """
    )
    conn.commit()


def _merge_one(
    src: Path,
    domain: str,
    dst: sqlite3.Connection,
) -> MergeStats:
    """Merge one source sqlite into the destination connection.

    Every node gets ``properties_json._domain_id = domain`` injected
    before insert. Pre-existing _domain_id (e.g. user already partitioned
    a single corpus) is preserved.

    Edges copied verbatim. Cross-corpus edges don't exist yet by
    construction — sources are independent before the combiner runs.
    """
    stats = MergeStats(domain=domain)
    src_conn = sqlite3.connect(src)
    src_conn.row_factory = sqlite3.Row
    src_cur = src_conn.cursor()
    dst_cur = dst.cursor()

    # Nodes — read full row, mutate properties_json, insert.
    for row in src_cur.execute("SELECT * FROM syn_nodes"):
        stats.nodes_read += 1
        try:
            props = json.loads(row["properties_json"] or "{}")
        except json.JSONDecodeError:
            props = {}
        if not isinstance(props, dict):
            props = {}
        # Don't clobber an explicit _domain_id (allow nested combines)
        props.setdefault("_domain_id", domain)
        new_props_json = json.dumps(props, ensure_ascii=False)

        try:
            dst_cur.execute(
                """
                INSERT INTO syn_nodes (
                    id, kind, title, content, tags_json, level, vitality,
                    access_count, success_count, failure_count, source,
                    properties_json, embedding_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["kind"],
                    row["title"],
                    row["content"],
                    row["tags_json"],
                    row["level"],
                    row["vitality"],
                    row["access_count"],
                    row["success_count"],
                    row["failure_count"],
                    row["source"],
                    new_props_json,
                    row["embedding_json"],
                    row["created_at"],
                    row["updated_at"],
                ),
            )
            stats.nodes_inserted += 1
        except sqlite3.IntegrityError:
            # Collision — same node id already inserted by an earlier
            # source. Track which kind so we know if it's a phrase-hub
            # collision (expected, fixed by Phase 1.2) or something
            # actually concerning.
            stats.nodes_skipped += 1
            if str(row["id"]).startswith("phrase_"):
                stats.phrase_collisions += 1
            else:
                stats.other_collisions += 1

    # Edges — copy as-is.
    for row in src_cur.execute("SELECT * FROM syn_edges"):
        stats.edges_read += 1
        try:
            dst_cur.execute(
                """
                INSERT INTO syn_edges (
                    id, source_id, target_id, kind, weight, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["source_id"],
                    row["target_id"],
                    row["kind"],
                    row["weight"],
                    row["created_at"],
                ),
            )
            stats.edges_inserted += 1
        except sqlite3.IntegrityError:
            stats.edges_skipped += 1

    src_conn.close()
    dst.commit()
    return stats


def build(sources: dict[str, Path], out: Path) -> list[MergeStats]:
    """Build a MetaCorpus at ``out`` from each (domain → sqlite) source."""
    if out.exists():
        out.unlink()  # always start clean — partial combines are confusing
    out.parent.mkdir(parents=True, exist_ok=True)
    dst = sqlite3.connect(out)
    _ensure_schema(dst)
    all_stats: list[MergeStats] = []
    for domain, src in sources.items():
        if not src.exists():
            print(f"!! source missing: {domain} → {src}", file=sys.stderr)
            continue
        all_stats.append(_merge_one(src, domain, dst))
    dst.close()
    return all_stats


def _format(stats: list[MergeStats], out: Path) -> str:
    lines = ["", "MetaCorpus build report", "=" * 50, ""]
    total_n = total_e = 0
    for s in stats:
        lines.append(
            f"  {s.domain:<10s} → nodes {s.nodes_inserted}/{s.nodes_read}"
            f" (+{s.nodes_skipped} skipped)  edges {s.edges_inserted}/{s.edges_read}"
            f" (+{s.edges_skipped} skipped)"
        )
        if s.phrase_collisions:
            lines.append(
                f"             phrase hub collisions: {s.phrase_collisions}"
                f" (Phase 1.2 will namespace these)"
            )
        if s.other_collisions:
            lines.append(f"             ⚠ OTHER collisions: {s.other_collisions} (investigate)")
        total_n += s.nodes_inserted
        total_e += s.edges_inserted
    lines.append("")
    lines.append(f"  TOTAL: {total_n} nodes, {total_e} edges → {out}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output sqlite path (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source as 'domain=path/to/sqlite'. Repeatable. "
        "If omitted, defaults to krra+assort+x2bee from eval/data/.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.source:
        sources: dict[str, Path] = {}
        for spec in args.source:
            if "=" not in spec:
                print(f"!! bad --source: {spec!r} (need domain=path)", file=sys.stderr)
                return 2
            domain, path = spec.split("=", 1)
            sources[domain.strip()] = Path(path.strip())
    else:
        sources = DEFAULT_SOURCES
    stats = build(sources, args.out)
    print(_format(stats, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
