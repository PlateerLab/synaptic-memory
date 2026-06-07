"""Connectivity backbone — make ANY corpus navigable (LLM-free, corpus-agnostic).

Real corpora fragment. The relation-free graph only links containment
(Category→Document→Chunk), FK, and entity co-occurrence, so a node with none of
those becomes an unreachable island — measured: KRRA 28.9 % isolated nodes,
x2bee shattered into 1 099 components. An agent cannot traverse an edge that does
not exist, so fragmentation is a hard cap on graph navigation, independent of how
good the retrieval or the agent loop is.

This adds a *minimal, high-quality* semantic backbone that guarantees global
connectivity, using only stored embeddings (no LLM, no per-domain logic):

  1. **Components** — union-find over existing edges → the mainland (largest
     component) plus the islands.
  2. **Candidate bridges** — for each island node, the HNSW top-k nearest
     neighbours (``search_vector`` on its own stored embedding); any neighbour in
     a *different* component is a candidate bridge, weighted by cosine
     similarity. O(islands·k), not the O(N²) of all-pairs.
  3. **Max-Spanning-Forest** — Kruskal over the candidate bridges by similarity
     descending: accept a bridge iff it merges two components. This is the
     *optimal* way to connect components — fewest edges, highest total edge
     quality — not an arbitrary similarity threshold. Each accepted bridge
     becomes a ``RELATED`` edge.

Only island nodes are queried, so the dense mainland is left untouched (this
avoids the measured net-negative of blanket k-NN expansion on ranking); bridges
add REACH exactly where it was absent, which is where the agent's graph traverse
pays off. Opt-in — call :func:`bridge_components` after ingest; not auto-run.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("connectivity")


@dataclass(slots=True)
class BridgeStats:
    """Outcome of one :func:`bridge_components` pass.

    ``isolated`` counts size-1 components; ``components`` counts all connected
    components. A fully-navigable corpus has ``components_after == 1`` (or a
    small constant) and ``isolated_after == 0``.
    """

    nodes: int = 0
    components_before: int = 0
    components_after: int = 0
    isolated_before: int = 0
    isolated_after: int = 0
    bridges_added: int = 0
    bridges_vector: int = 0
    bridges_lexical: int = 0
    skipped_no_signal: int = 0

    def summary(self) -> str:
        return (
            f"{self.nodes} nodes: components {self.components_before}→"
            f"{self.components_after}, isolated {self.isolated_before}→"
            f"{self.isolated_after}, +{self.bridges_added} bridges "
            f"({self.bridges_vector} vector / {self.bridges_lexical} lexical)"
            f" ({self.skipped_no_signal} skipped: no signal)"
        )


class _UnionFind:
    """Union-find with path compression — components in near-linear time."""

    __slots__ = ("parent",)

    def __init__(self, ids: list[str]) -> None:
        self.parent: dict[str, str] = {i: i for i in ids}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


_TOKEN_RE = None


def _tokenize(text: str, *, cap: int = 100) -> list[str]:
    """Lowercased alphanumeric/CJK tokens (len ≥ 2), deduped, capped.

    Cheap and language-neutral — good enough for the lexical bridge's
    token-overlap signal (entity title ↔ the chunks that mention it)."""
    global _TOKEN_RE
    if _TOKEN_RE is None:
        import re

        _TOKEN_RE = re.compile(r"[0-9a-z가-힣]+")
    out: list[str] = []
    seen: set[str] = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) >= 2 and tok not in seen:
            seen.add(tok)
            out.append(tok)
            if len(out) >= cap:
                break
    return out


def _build_token_index(
    nodes: list, *, df_max: int, max_tokens_per_node: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """In-memory inverted index for the lexical bridge.

    Returns ``(token→node_ids, node_id→tokens)``. Tokens appearing in more than
    ``df_max`` nodes are dropped as too generic to be a distinguishing bridge
    signal (a DF filter — the same idea that keeps TF-IDF / SPRIG honest), which
    also bounds the postings we scan per island node.
    """
    node_tokens: dict[str, list[str]] = {}
    postings: dict[str, list[str]] = {}
    for n in nodes:
        toks = _tokenize(f"{n.title or ''} {n.content or ''}", cap=max_tokens_per_node)
        node_tokens[n.id] = toks
        for t in toks:
            postings.setdefault(t, []).append(n.id)
    # DF filter: drop overly-common tokens (kept index is the bridge signal).
    token_index = {t: ids for t, ids in postings.items() if len(ids) <= df_max}
    return token_index, node_tokens


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


async def bridge_components(
    backend: StorageBackend,
    *,
    k: int = 10,
    min_similarity: float = 0.0,
    max_bridges: int | None = None,
    node_limit: int = 1_000_000,
    df_max: int = 2000,
    max_tokens_per_node: int = 100,
) -> BridgeStats:
    """Connect a fragmented graph into a navigable structure (LLM-free).

    Adds the minimal high-quality semantic backbone (Max-Spanning-Forest over
    HNSW-nearest cross-component pairs) so every embedded node is reachable.
    Mutates the graph by inserting ``RELATED`` bridge edges; deterministic
    (candidates ordered by similarity then id), idempotent (stable edge ids).

    Args:
        backend: Storage backend (needs ``search_vector`` + stored embeddings).
        k: Nearest neighbours queried per island node — the bridge candidate
            pool. Larger k finds better cross-component bridges at more cost.
        min_similarity: Drop candidate bridges below this cosine. 0.0 keeps all
            (guarantees connectivity where any cross-component neighbour exists).
        max_bridges: Optional cap on edges added (safety valve on huge graphs).
        node_limit: Max nodes to load.

    Returns:
        :class:`BridgeStats` — components / isolated before-and-after + bridges.
    """
    from synaptic.models import Edge, EdgeKind

    nodes = await backend.list_nodes(kind=None, limit=node_limit)
    ids = [n.id for n in nodes]
    id_set = set(ids)
    by_id = {n.id: n for n in nodes}
    uf = _UnionFind(ids)

    # 1. existing edges → components (outgoing only avoids double counting)
    for n in nodes:
        for e in await backend.get_edges(n.id, direction="outgoing"):
            if e.target_id in id_set:
                uf.union(n.id, e.target_id)

    def _component_stats() -> tuple[int, int]:
        size: dict[str, int] = {}
        for i in ids:
            r = uf.find(i)
            size[r] = size.get(r, 0) + 1
        n_components = len(size)
        n_isolated = sum(1 for i in ids if size[uf.find(i)] == 1)
        return n_components, n_isolated

    stats = BridgeStats(nodes=len(ids))
    stats.components_before, stats.isolated_before = _component_stats()

    if not ids:
        stats.components_after, stats.isolated_after = 0, 0
        return stats

    # mainland = largest component; islands are everything else.
    comp_size: dict[str, int] = {}
    for i in ids:
        r = uf.find(i)
        comp_size[r] = comp_size.get(r, 0) + 1
    main_root = max(comp_size, key=lambda r: comp_size[r])

    # In-memory inverted index for the lexical bridge signal (built once; avoids
    # a per-island-node DB query — the bottleneck that timed out on KRRA's 26k
    # entity islands).
    token_index, node_tokens = _build_token_index(
        nodes, df_max=df_max, max_tokens_per_node=max_tokens_per_node
    )

    # 2. candidate bridges from island nodes only (mainland left untouched).
    #    Use the right signal per node: embedded nodes (chunks) bridge by vector
    #    cosine; non-embedded nodes (entity / phrase hubs — 78 % of KRRA) bridge
    #    LEXICALLY to the chunks that mention their title (the MENTIONS link that
    #    never formed). One framework, two signals, no LLM, no embedder.
    candidates: list[tuple[float, str, str, bool]] = []  # (score, src, dst, is_vector)
    for n in nodes:
        if uf.find(n.id) == main_root:
            continue
        cand: list[tuple[float, str, bool]] = []  # (score, other_id, is_vector)
        if n.embedding:
            try:
                hits = await backend.search_vector(n.embedding, limit=k + 1)
            except Exception as exc:
                logger.debug("search_vector failed for %s: %s", n.id, exc)
                hits = []
            for hit in hits:
                other = by_id.get(hit.id)
                if other is not None and other.id != n.id and other.embedding:
                    cand.append((_cosine(n.embedding, other.embedding), other.id, True))
        else:
            # In-memory token-overlap (no per-node DB query — that was the 26k-
            # FTS bottleneck). Score candidates by shared DF-filtered tokens.
            toks = node_tokens.get(n.id, ())
            if not toks:
                stats.skipped_no_signal += 1
                continue
            overlap: dict[str, int] = {}
            for t in toks:
                for other_id in token_index.get(t, ()):  # DF-capped postings
                    if other_id != n.id:
                        overlap[other_id] = overlap.get(other_id, 0) + 1
            if not overlap:
                continue
            denom = float(len(toks))
            # keep the top-k best-overlapping candidates
            best = sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
            for other_id, shared in best:
                cand.append((shared / denom, other_id, False))  # normalized overlap

        for score, other_id, is_vec in cand:
            if uf.find(other_id) == uf.find(n.id):  # same component already
                continue
            if score < min_similarity:
                continue
            candidates.append((score, n.id, other_id, is_vec))

    # 3. Max-Spanning-Forest (Kruskal): highest-similarity bridge first, accept
    #    iff it merges two still-separate components → minimal, best-quality set.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    new_edges: list[Edge] = []
    for sim, src, dst, is_vec in candidates:
        if max_bridges is not None and stats.bridges_added >= max_bridges:
            break
        if uf.union(src, dst):
            new_edges.append(
                Edge(
                    id=f"bridge_{src}_{dst}",
                    source_id=src,
                    target_id=dst,
                    kind=EdgeKind.RELATED,
                    weight=round(float(sim), 4),
                )
            )
            stats.bridges_added += 1
            if is_vec:
                stats.bridges_vector += 1
            else:
                stats.bridges_lexical += 1

    if new_edges:
        await backend.save_edges_batch(new_edges)

    stats.components_after, stats.isolated_after = _component_stats()
    return stats
