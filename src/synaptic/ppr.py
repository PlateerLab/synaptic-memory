"""Personalized PageRank (PPR) engine — zero-dependency, dict-based sparse implementation.

Power iteration:
    r(t+1) = (1 - damping) * personalization + damping * A_norm * r(t)

Where A_norm is a column-normalized adjacency matrix built from the graph edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    Returns:
        List of (node_id, ppr_score) sorted descending by score.
    """
    if not seed_scores:
        return []

    # --- 1. BFS to discover the reachable subgraph (depth 2 from seeds) ---
    # adjacency: source -> [(target, weight), ...]
    adj: dict[str, list[tuple[str, float]]] = {}
    visited: set[str] = set()
    frontier = set(seed_scores.keys())
    bfs_depth = 2

    for _ in range(bfs_depth):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for nid in frontier:
            if nid in visited:
                continue
            visited.add(nid)
            if nid not in adj:
                adj[nid] = []
            edges = await backend.get_edges(nid, direction="both")
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

    # No edges at all — return seed scores directly (sorted)
    if not any(adj.values()):
        sorted_seeds = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_seeds[:top_k]

    # --- 2. Build column-normalized adjacency (as sparse dicts) ---
    # out_weight[node] = sum of weights of outgoing edges
    out_weight: dict[str, float] = {}
    for src, neighbors in adj.items():
        total = sum(w for _, w in neighbors)
        out_weight[src] = total if total > 0 else 1.0

    # --- 3. Normalize personalization vector ---
    total_seed = sum(seed_scores.values())
    if total_seed == 0:
        return []
    personalization: dict[str, float] = {nid: s / total_seed for nid, s in seed_scores.items()}

    # --- 4. Power iteration ---
    # Initialize rank vector = personalization
    rank: dict[str, float] = {}
    for nid in all_nodes:
        rank[nid] = personalization.get(nid, 0.0)

    teleport_coeff = 1.0 - damping

    for _ in range(max_iter):
        new_rank: dict[str, float] = {}
        # Initialize with teleport (personalization)
        for nid in all_nodes:
            new_rank[nid] = teleport_coeff * personalization.get(nid, 0.0)

        # Distribute rank along edges
        for src, neighbors in adj.items():
            if not neighbors:
                continue
            src_rank = rank[src]
            src_out = out_weight[src]
            for tgt, w in neighbors:
                # Column-normalized: edge_weight / total_out_weight * src_rank
                contribution = damping * src_rank * w / src_out
                new_rank[tgt] = new_rank.get(tgt, 0.0) + contribution

        # Check convergence (L1 norm)
        diff = sum(abs(new_rank.get(nid, 0.0) - rank.get(nid, 0.0)) for nid in all_nodes)
        rank = new_rank
        if diff < tol:
            break

    # --- 5. Return top-k ---
    sorted_results = sorted(rank.items(), key=lambda x: x[1], reverse=True)
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
    edge_weight_floor: float = 0.15,
    passage_boost: float = 1.5,
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
        edge_weight_floor: Zero out edges below this weight.
        passage_boost: Teleport weight multiplier for CHUNK nodes.

    Returns:
        List of (node_id, ppr_score) sorted descending.
    """
    if not seed_scores:
        return []

    # --- 1. BFS subgraph discovery (depth 2) ---
    adj: dict[str, list[tuple[str, float]]] = {}
    node_kinds: dict[str, str] = {}  # node_id → kind
    visited: set[str] = set()
    frontier = set(seed_scores.keys())
    bfs_depth = 2

    for _ in range(bfs_depth):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for nid in frontier:
            if nid in visited:
                continue
            visited.add(nid)
            if nid not in adj:
                adj[nid] = []

            # Track node kind for chunk/entity awareness
            if nid not in node_kinds:
                node = await backend.get_node(nid)
                if node:
                    node_kinds[nid] = str(node.kind)

            edges = await backend.get_edges(nid, direction="both")
            for edge in edges:
                if edge.source_id == nid:
                    neighbor_id = edge.target_id
                else:
                    neighbor_id = edge.source_id

                # Track neighbor kind
                if neighbor_id not in node_kinds:
                    neighbor_node = await backend.get_node(neighbor_id)
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

    if not any(adj.values()):
        sorted_seeds = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_seeds[:top_k]

    # --- 2. Column-normalized adjacency ---
    out_weight: dict[str, float] = {}
    for src, neighbors in adj.items():
        total = sum(w for _, w in neighbors)
        out_weight[src] = total if total > 0 else 1.0

    # --- 3. Personalization with passage boost ---
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

    # --- 4. Power iteration ---
    rank: dict[str, float] = {nid: personalization.get(nid, 0.0) for nid in all_nodes}
    teleport_coeff = 1.0 - damping

    for _ in range(max_iter):
        new_rank = {nid: teleport_coeff * personalization.get(nid, 0.0) for nid in all_nodes}

        for src, neighbors in adj.items():
            if not neighbors:
                continue
            src_rank = rank[src]
            src_out = out_weight[src]
            for tgt, w in neighbors:
                contribution = damping * src_rank * w / src_out
                new_rank[tgt] = new_rank.get(tgt, 0.0) + contribution

        diff = sum(abs(new_rank.get(nid, 0.0) - rank.get(nid, 0.0)) for nid in all_nodes)
        rank = new_rank
        if diff < tol:
            break

    # --- 5. Return top-k ---
    sorted_results = sorted(rank.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
