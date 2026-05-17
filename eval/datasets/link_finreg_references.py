"""Add explicit cross-reference edges to the financial-statute graph.

finreg articles cite each other constantly ("제15조제2항에 따라", "제30조를
준용한다"). After ``ingest_finreg.py`` the graph only has ``part_of`` edges
(article → law); the cross-references live as plain prose inside article
text, invisible to graph traversal.

This pass scans every article's text, resolves each in-law "제N조" citation
to the cited article node, and writes a ``REFERENCES`` edge. The agent can
then walk A → B in one structural hop via ``follow(node, "references")``
instead of re-parsing prose and gambling on a second search.

Unlike the v0.23 ReferenceLinker (measured negative — KRRA's 70k ENTITY
phrase-hubs were too noisy to resolve targets against), finreg has a
pristine target inventory: every article carries a canonical ``article_no``.

Usage:
    uv run python eval/datasets/link_finreg_references.py
    uv run python eval/datasets/link_finreg_references.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.models import Edge, EdgeKind, NodeKind

GRAPH = REPO_ROOT / "eval" / "data" / "finreg_graph.sqlite"
_REF_RE = re.compile(r"제(\d+)조(?:의(\d+))?")


async def main() -> int:
    ap = argparse.ArgumentParser(description="Link finreg cross-references")
    ap.add_argument("--dry-run", action="store_true", help="Count edges, do not write")
    args = ap.parse_args()

    if not GRAPH.exists():
        print(f"ERROR: {GRAPH} not found — run ingest_finreg.py first.")
        return 1

    from synaptic.backends.sqlite_graph import SqliteGraphBackend

    backend = SqliteGraphBackend(str(GRAPH))
    await backend.connect()

    articles = await backend.list_nodes(kind=NodeKind.ENTITY, limit=500_000)
    # (law, article_no) -> node_id
    index: dict[tuple[str, str], str] = {}
    for n in articles:
        p = n.properties or {}
        law, art = p.get("law"), p.get("article_no")
        if law and art:
            index[(law, art)] = n.id
    print(f"{len(articles)} article nodes, {len(index)} indexed")

    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    citing = 0
    for n in articles:
        p = n.properties or {}
        law = p.get("law", "")
        self_no = p.get("article_no", "")
        refs = set()
        for m in _REF_RE.finditer(n.content or ""):
            ref = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
            if ref != self_no and (law, ref) in index:
                refs.add(ref)
        if refs:
            citing += 1
        for ref in refs:
            tgt = index[(law, ref)]
            key = (n.id, tgt)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                Edge(source_id=n.id, target_id=tgt, kind=EdgeKind.REFERENCES, weight=1.0)
            )

    print(
        f"{len(edges)} REFERENCES edges from {citing} citing articles "
        f"(avg {len(edges) / max(citing, 1):.1f} refs/article)"
    )
    if args.dry_run:
        print("dry-run — nothing written")
    else:
        await backend.save_edges_batch(edges)
        print(f"wrote {len(edges)} edges -> {GRAPH}")

    await backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
