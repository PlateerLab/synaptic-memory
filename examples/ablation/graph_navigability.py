"""Graph navigability reality-check (LLM-free, corpus-agnostic).

Answers the make-or-break question from the structure-for-agent-navigation
research: can an agent reach any node from any node in FEW hops on synaptic's
actual graph? If the graph is fragmented (islands) or not small-world, no
LLM-free structure can make it navigable (the O(log log n) doubling-dimension
wall) and the whole "fewer-hops" thesis is dead for that corpus.

Pure structural metrics on the existing syn_edges (no LLM, no embedder):
  - connectivity: # components, largest-component fraction, isolated nodes
  - hubs: degree distribution (mean/max/p99) — heavy tails accelerate greedy routing
  - small-world: sampled BFS effective hop-distance on the largest component
    (median / p90 hops between random node pairs)

Run:  uv run python examples/ablation/graph_navigability.py eval/data/krra_graph.sqlite
"""

from __future__ import annotations

import sqlite3
import sys
from collections import deque


def _load_adjacency(db_path: str) -> tuple[dict[str, list[str]], int]:
    con = sqlite3.connect(db_path)
    adj: dict[str, list[str]] = {}
    n_nodes = con.execute("SELECT COUNT(*) FROM syn_nodes").fetchone()[0]
    # ensure every node is present (isolated nodes have no edges)
    for (nid,) in con.execute("SELECT id FROM syn_nodes"):
        adj[nid] = []
    n_edges = 0
    for s, t in con.execute("SELECT source_id, target_id FROM syn_edges"):
        if s in adj and t in adj:
            adj[s].append(t)
            adj[t].append(s)  # undirected for reachability
            n_edges += 1
    con.close()
    return adj, n_edges


def _components(adj: dict[str, list[str]]) -> list[int]:
    seen: set[str] = set()
    sizes: list[int] = []
    for start in adj:
        if start in seen:
            continue
        size = 0
        q = deque([start])
        seen.add(start)
        while q:
            u = q.popleft()
            size += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        sizes.append(size)
    return sizes


def _bfs_hops(adj: dict[str, list[str]], src: str, cap: int = 200_000) -> list[int]:
    dist = {src: 0}
    q = deque([src])
    out: list[int] = []
    while q and len(dist) < cap:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                out.append(dist[v])
                q.append(v)
    return out


def _pct(xs: list[int], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else "eval/data/krra_graph.sqlite"
    adj, n_edges = _load_adjacency(db)
    n = len(adj)
    degs = [len(v) for v in adj.values()]
    isolated = sum(1 for d in degs if d == 0)
    sizes = _components(adj)
    sizes.sort(reverse=True)
    largest = sizes[0] if sizes else 0

    print(f"=== navigability: {db} ===")
    print(f"nodes={n}  edges={n_edges}  avg_degree={2 * n_edges / n:.2f}")
    print(f"degree: max={max(degs)}  p99={_pct(degs, 99)}  median={_pct(degs, 50)}")
    print(f"isolated_nodes={isolated} ({100 * isolated / n:.1f}%)")
    print(f"components={len(sizes)}  largest={largest} ({100 * largest / n:.1f}% of graph)")

    # small-world hop distances: BFS from a few seeds inside the largest component
    import random as _r  # noqa: ASYNC — script, deterministic seed

    _r.seed(0)
    big = [u for u in adj if adj[u]]  # non-isolated
    seeds = _r.sample(big, min(5, len(big))) if big else []
    all_hops: list[int] = []
    for s in seeds:
        all_hops.extend(_bfs_hops(adj, s))
    if all_hops:
        reach = len(all_hops) / (len(seeds) * (largest - 1)) if largest > 1 else 0
        print(
            f"hop-distance (BFS from {len(seeds)} seeds): "
            f"median={_pct(all_hops, 50)}  p90={_pct(all_hops, 90)}  max={max(all_hops)}  "
            f"avg_reach_per_seed={reach:.2f}"
        )

    # verdict
    frag = largest / n if n else 0
    print("--- verdict ---")
    print(f"  reachable-as-one-graph: {'YES' if frag > 0.9 else 'NO — fragmented'} ({frag:.0%})")
    hubness = max(degs) / (2 * n_edges / n) if n_edges else 0
    print(f"  hub structure (max/avg degree ratio): {hubness:.0f}x {'(heavy-tail, good)' if hubness > 20 else ''}")


if __name__ == "__main__":
    main()
