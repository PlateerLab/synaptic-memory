"""Personalized PageRank (PPR) engine — zero-dependency, dict-based sparse implementation.

Power iteration:
    r(t+1) = (1 - damping) * personalization + damping * A_norm * r(t)

Where A_norm is a column-normalized adjacency matrix built from the graph edges.
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

from synaptic.extensions.graph_read_cache import GraphReadCache
from synaptic.models import EdgeKind, NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
    from synaptic.protocols import StorageBackend

# Edge type별 PPR 전파 가중치 — 의미 있는 관계일수록 더 강하게 전파
_EDGE_TYPE_WEIGHTS: dict[EdgeKind, float] = {
    EdgeKind.CAUSED: 1.0,  # 인과 관계 — 강한 전파
    EdgeKind.RESULTED_IN: 1.0,  # 결과 — 강한 전파
    EdgeKind.DEPENDS_ON: 0.9,  # 의존 — 강한 전파
    EdgeKind.LEARNED_FROM: 0.8,  # 교훈 — 중간
    EdgeKind.PRODUCED: 0.8,  # 생산 — 중간
    EdgeKind.PART_OF: 0.7,  # 부분-전체 — 중간
    EdgeKind.CONTAINS: 0.6,  # 포함 (phrase) — 약한 전파
    EdgeKind.RELATED: 0.4,  # 일반 관련 — 약한 (노이즈 방지)
    EdgeKind.CONTRADICTS: 0.2,  # 모순 — 최소 전파
    EdgeKind.SUPERSEDES: 0.3,  # 대체 — 약한
    EdgeKind.IS_A: 0.5,  # 타입 계층 — 중간
    EdgeKind.INVOKED: 0.6,  # 호출 — 중간
    EdgeKind.FOLLOWED_BY: 0.7,  # 순서 — 중간
    # v1.0: chunk-entity graph
    EdgeKind.MENTIONS: 0.8,  # chunk → entity — 강한
    EdgeKind.EXTRACTED_FROM: 0.8,  # entity → chunk — 강한
    EdgeKind.NEXT_CHUNK: 0.3,  # chunk → chunk 순서 — 약한 (청크 간 직접 전파 제한)
}


async def personalized_pagerank(
    backend: StorageBackend,
    seed_scores: dict[str, float],
    *,
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-6,
    top_k: int = 20,
    bfs_depth: int = 2,
    read_cache: GraphReadCache | None = None,
    timings_ms: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Perform PPR and return top-k (node_id, score) pairs.

    The graph is discovered incrementally via BFS from seed nodes so that
    only the reachable subgraph is materialized — no need to enumerate all
    nodes/edges in the backend.

    Args:
        backend: Storage backend implementing the StorageBackend protocol.
        seed_scores: {node_id: weight} — search result scores as personalization.
        damping: Probability of following an edge (vs teleporting back to seeds).
        max_iter: Maximum power-iteration steps.
        tol: Convergence threshold (L1 norm of rank change).
        top_k: Number of top-ranked nodes to return.
        bfs_depth: How many graph layers to materialize from the seeds.

    Returns:
        List of (node_id, ppr_score) sorted descending by score.
    """
    if not seed_scores:
        return []
    reads = read_cache or GraphReadCache(backend)

    # --- 1. BFS to discover the reachable subgraph (depth 2 from seeds) ---
    stage_t0 = time()
    # adjacency: source -> [(target, weight), ...]
    adj: dict[str, list[tuple[str, float]]] = {}
    visited: set[str] = set()
    frontier = set(seed_scores.keys())

    for _ in range(max(0, bfs_depth)):
        if not frontier:
            break
        next_frontier: set[str] = set()
        frontier_nodes = [nid for nid in frontier if nid not in visited]
        frontier_edges = await reads.get_edges_many(frontier_nodes, direction="both")
        for nid in frontier_nodes:
            if nid in visited:
                continue
            visited.add(nid)
            if nid not in adj:
                adj[nid] = []
            edges = frontier_edges.get(nid, [])
            for edge in edges:
                # Determine the neighbor
                if edge.source_id == nid:
                    neighbor_id = edge.target_id
                else:
                    neighbor_id = edge.source_id

                # Edge type weighting: meaningful relations spread more
                edge_type_weight = _EDGE_TYPE_WEIGHTS.get(edge.kind, 0.5)
                effective_weight = edge.weight * edge_type_weight

                # Add edge in both directions (undirected for PPR spreading)
                adj[nid].append((neighbor_id, effective_weight))
                if neighbor_id not in adj:
                    adj[neighbor_id] = []
                adj[neighbor_id].append((nid, effective_weight))

                if neighbor_id not in visited:
                    next_frontier.add(neighbor_id)
        frontier = next_frontier

    # Mark remaining frontier nodes as visited (leaf nodes with no outgoing expansion)
    visited.update(frontier)
    for nid in frontier:
        if nid not in adj:
            adj[nid] = []

    all_nodes = set(adj.keys()) | set(seed_scores.keys())
    _record_timing(timings_ms, "bfs", stage_t0)

    # No edges at all — return seed scores directly (sorted)
    if not any(adj.values()):
        stage_t0 = time()
        sorted_seeds = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)
        _record_timing(timings_ms, "sort", stage_t0)
        return sorted_seeds[:top_k]

    # --- 2. Normalize personalization vector ---
    total_seed = sum(seed_scores.values())
    if total_seed == 0:
        return []
    personalization: dict[str, float] = {nid: s / total_seed for nid, s in seed_scores.items()}

    rank = _power_iteration(
        adj,
        personalization,
        all_nodes,
        damping=damping,
        max_iter=max_iter,
        tol=tol,
        timings_ms=timings_ms,
    )

    # --- 3. Return top-k ---
    stage_t0 = time()
    sorted_results = sorted(
        ((nid, rank.get(nid, 0.0)) for nid in all_nodes),
        key=lambda x: x[1],
        reverse=True,
    )
    _record_timing(timings_ms, "sort", stage_t0)
    return sorted_results[:top_k]


async def personalized_pagerank_v2(
    backend: StorageBackend,
    seed_scores: dict[str, float],
    *,
    chunk_entity_index: ChunkEntityIndex | None = None,
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-6,
    top_k: int = 20,
    bfs_depth: int = 2,
    edge_weight_floor: float = 0.15,
    passage_boost: float = 1.5,
    read_cache: GraphReadCache | None = None,
    timings_ms: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """HippoRAG2-inspired PPR v2 with noise reduction.

    Key improvements over v1:
      1. CHUNK seed boost: passage nodes get higher teleport weight (more grounded)
      2. Entity-mediated spreading: CHUNK→CHUNK direct propagation blocked,
         spreading only through ENTITY nodes (reduces noise)
      3. Weak edge zeroing: edges with weight < edge_weight_floor are ignored

    Args:
        backend: Storage backend.
        seed_scores: {node_id: weight} — search result scores.
        chunk_entity_index: Bidirectional index (enables chunk/entity awareness).
        damping: Edge-follow probability.
        max_iter: Max iterations.
        tol: Convergence threshold.
        top_k: Top results to return.
        bfs_depth: How many graph layers to materialize from the seeds.
        edge_weight_floor: Zero out edges below this weight.
        passage_boost: Teleport weight multiplier for CHUNK nodes.

    Returns:
        List of (node_id, ppr_score) sorted descending.
    """
    if not seed_scores:
        return []
    reads = read_cache or GraphReadCache(backend)

    # --- 1. BFS subgraph discovery (depth 2) ---
    stage_t0 = time()
    adj: dict[str, list[tuple[str, float]]] = {}
    node_kinds: dict[str, str] = {}  # node_id → kind
    visited: set[str] = set()
    frontier = set(seed_scores.keys())

    for _ in range(max(0, bfs_depth)):
        if not frontier:
            break
        next_frontier: set[str] = set()
        frontier_nodes = [nid for nid in frontier if nid not in visited]
        frontier_edges = await reads.get_edges_many(frontier_nodes, direction="both")
        for nid in frontier_nodes:
            if nid in visited:
                continue
            visited.add(nid)
            if nid not in adj:
                adj[nid] = []

            # Track node kind for chunk/entity awareness
            if nid not in node_kinds:
                node = await reads.get_node(nid)
                if node:
                    node_kinds[nid] = str(node.kind)

            edges = frontier_edges.get(nid, [])
            for edge in edges:
                if edge.source_id == nid:
                    neighbor_id = edge.target_id
                else:
                    neighbor_id = edge.source_id

                # Track neighbor kind
                if neighbor_id not in node_kinds:
                    neighbor_node = await reads.get_node(neighbor_id)
                    if neighbor_node:
                        node_kinds[neighbor_id] = str(neighbor_node.kind)

                edge_type_weight = _EDGE_TYPE_WEIGHTS.get(edge.kind, 0.5)
                effective_weight = edge.weight * edge_type_weight

                # Weak edge zeroing
                if effective_weight < edge_weight_floor:
                    continue

                # Entity-mediated spreading: block CHUNK→CHUNK direct propagation
                src_kind = node_kinds.get(nid, "")
                dst_kind = node_kinds.get(neighbor_id, "")
                if src_kind == NodeKind.CHUNK and dst_kind == NodeKind.CHUNK:
                    # Only allow NEXT_CHUNK (sequential reading), heavily dampened
                    if edge.kind != EdgeKind.NEXT_CHUNK:
                        continue

                adj[nid].append((neighbor_id, effective_weight))
                if neighbor_id not in adj:
                    adj[neighbor_id] = []
                adj[neighbor_id].append((nid, effective_weight))

                if neighbor_id not in visited:
                    next_frontier.add(neighbor_id)
        frontier = next_frontier

    visited.update(frontier)
    for nid in frontier:
        if nid not in adj:
            adj[nid] = []

    all_nodes = set(adj.keys()) | set(seed_scores.keys())
    _record_timing(timings_ms, "bfs", stage_t0)

    if not any(adj.values()):
        stage_t0 = time()
        sorted_seeds = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)
        _record_timing(timings_ms, "sort", stage_t0)
        return sorted_seeds[:top_k]

    # --- 2. Personalization with passage boost ---
    boosted_seeds: dict[str, float] = {}
    for nid, score in seed_scores.items():
        if node_kinds.get(nid) == NodeKind.CHUNK:
            boosted_seeds[nid] = score * passage_boost
        else:
            boosted_seeds[nid] = score

    total_seed = sum(boosted_seeds.values())
    if total_seed == 0:
        return []
    personalization = {nid: s / total_seed for nid, s in boosted_seeds.items()}

    rank = _power_iteration(
        adj,
        personalization,
        all_nodes,
        damping=damping,
        max_iter=max_iter,
        tol=tol,
        timings_ms=timings_ms,
    )

    # --- 3. Return top-k ---
    stage_t0 = time()
    sorted_results = sorted(
        ((nid, rank.get(nid, 0.0)) for nid in all_nodes),
        key=lambda x: x[1],
        reverse=True,
    )
    _record_timing(timings_ms, "sort", stage_t0)
    return sorted_results[:top_k]


def _power_iteration(
    adj: dict[str, list[tuple[str, float]]],
    personalization: dict[str, float],
    all_nodes: set[str],
    *,
    damping: float,
    max_iter: int,
    tol: float,
    timings_ms: dict[str, float] | None = None,
) -> dict[str, float]:
    """Run sparse PPR power iteration over a pre-discovered adjacency graph."""
    stage_t0 = time()
    node_ids = list(all_nodes)
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    node_count = len(node_ids)
    transitions: list[tuple[int, list[tuple[int, float]]]] = []
    for src, neighbors in adj.items():
        if not neighbors:
            continue
        src_idx = node_index.get(src)
        if src_idx is None:
            continue
        total = sum(w for _, w in neighbors)
        src_out = total if total > 0 else 1.0
        indexed_neighbors: list[tuple[int, float]] = []
        for tgt, weight in neighbors:
            tgt_idx = node_index.get(tgt)
            if tgt_idx is None:
                continue
            indexed_neighbors.append((tgt_idx, damping * weight / src_out))
        if indexed_neighbors:
            transitions.append((src_idx, indexed_neighbors))

    teleport_coeff = 1.0 - damping
    teleport = [0.0] * node_count
    rank = [0.0] * node_count
    for nid, score in personalization.items():
        idx = node_index.get(nid)
        if idx is None:
            continue
        teleport[idx] = teleport_coeff * score
        rank[idx] = score
    _record_timing(timings_ms, "prepare", stage_t0)

    stage_t0 = time()
    for _ in range(max_iter):
        new_rank = teleport.copy()

        for src, neighbors in transitions:
            src_rank = rank[src]
            if src_rank == 0.0:
                continue
            for tgt_idx, coefficient in neighbors:
                contribution = src_rank * coefficient
                new_rank[tgt_idx] += contribution

        diff = sum(abs(new_rank[i] - rank[i]) for i in range(node_count))
        rank = new_rank
        if diff < tol:
            break
    _record_timing(timings_ms, "iterate", stage_t0)
    return dict(zip(node_ids, rank, strict=True))


def _record_timing(timings_ms: dict[str, float] | None, key: str, started_at: float) -> None:
    if timings_ms is None:
        return
    timings_ms[key] = timings_ms.get(key, 0.0) + (time() - started_at) * 1000
