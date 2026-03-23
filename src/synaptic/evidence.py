"""EvidenceAssembler — converts SearchResult into an LLM-optimized evidence chain."""

from __future__ import annotations

import re
from collections import deque
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

from synaptic.models import (
    Edge,
    EdgeKind,
    EvidenceChain,
    EvidenceStep,
    Node,
    SearchResult,
)

# Directed edge kinds used for topological sorting
_DIRECTED_KINDS = frozenset(
    {
        EdgeKind.CAUSED,
        EdgeKind.RESULTED_IN,
        EdgeKind.DEPENDS_ON,
        EdgeKind.FOLLOWED_BY,
        EdgeKind.LEARNED_FROM,
    }
)

# Stop words (excluded from term overlap calculation)
_STOPWORDS = frozenset(
    {
        # English
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        "not",
        "with",
        "by",
        "from",
        "that",
        "this",
        "it",
        "its",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "why",
        # Korean
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "에",
        "의",
        "와",
        "과",
        "도",
        "에서",
        "로",
        "으로",
        "하는",
        "있는",
        "하고",
        "하면",
        "에게",
    }
)

# Fact extraction patterns
_FACT_PATTERNS = [
    # Numbers + units
    re.compile(
        r"\d[\d,.]*\s*(%|만|억|원|달러|km|kg|GB|MB|TB|명|건|개|년|월|일|시간|분|초|percent|million|billion|thousand)",
        re.IGNORECASE,
    ),
    # Dates (2024-01-01, 2024년, January 2024, 15 March 1990)
    re.compile(r"\b\d{4}[-/년.]\d{1,2}[-/월.]?\d{0,2}일?\b"),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s*\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}\b",
        re.IGNORECASE,
    ),
    # Numbers only (years, population, etc.) - 4+ digits
    re.compile(r"\b\d{4,}\b"),
]


class EvidenceAssembler:
    """Converts SearchResult into an LLM-optimized evidence chain."""

    __slots__ = ("_max_sentences", "_max_tokens", "_relevance_threshold")

    def __init__(
        self,
        *,
        max_sentences_per_node: int = 5,
        relevance_threshold: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        self._max_sentences = max_sentences_per_node
        self._relevance_threshold = relevance_threshold
        self._max_tokens = max_tokens

    async def assemble(
        self,
        backend: StorageBackend,
        query: str,
        search_result: SearchResult,
        *,
        max_steps: int = 8,
    ) -> EvidenceChain:
        """Assemble search results into an evidence chain."""
        t0 = time()

        if not search_result.nodes:
            return EvidenceChain(query=query, assembly_time_ms=(time() - t0) * 1000)

        # 1. Extract seed nodes (top max_steps)
        seed_nodes = search_result.nodes[:max_steps]
        seed_ids = [a.node.id for a in seed_nodes]
        seed_map: dict[str, Node] = {a.node.id: a.node for a in seed_nodes}

        # 2. BFS to find bridge nodes
        bridge_paths = await self._find_bridge_paths(backend, seed_ids)

        # Collect new nodes discovered via bridge paths
        all_ids: list[str] = list(seed_ids)
        for path in bridge_paths:
            for nid in path:
                if nid not in seed_map:
                    node = await backend.get_node(nid)
                    if node:
                        seed_map[nid] = node
                        if nid not in all_ids:
                            all_ids.append(nid)

        # 3. Collect edges (for topological sorting)
        all_edges: list[Edge] = []
        id_set = set(all_ids)
        for nid in all_ids:
            edges = await backend.get_edges(nid)
            for e in edges:
                other = e.target_id if e.source_id == nid else e.source_id
                if other in id_set:
                    all_edges.append(e)

        # 4. Topological sort
        sorted_ids = self._topological_sort(all_ids, all_edges, seed_ids)

        # 5. Generate steps
        steps: list[EvidenceStep] = []
        all_facts: list[str] = []
        seed_id_set = set(seed_ids)

        for i, nid in enumerate(sorted_ids[:max_steps]):
            node = seed_map.get(nid)
            if not node:
                continue

            role = "seed" if nid in seed_id_set else "bridge"
            compressed = self._compress_content(node.content, query)
            facts = self._extract_facts(node.content)
            all_facts.extend(facts)

            # Connection description to the next step
            conn = ""
            if i < len(sorted_ids) - 1:
                next_id = sorted_ids[i + 1]
                for e in all_edges:
                    if (e.source_id == nid and e.target_id == next_id) or (
                        e.target_id == nid and e.source_id == next_id
                    ):
                        conn = e.kind.value
                        break

            steps.append(
                EvidenceStep(
                    node=node,
                    role=role,
                    connection_to_next=conn,
                    compressed_content=compressed,
                    facts=facts,
                )
            )

        # 6. Final context formatting
        context = self._format_context(steps)

        # Approximate token count
        tokens = len(context.split())

        return EvidenceChain(
            query=query,
            steps=steps,
            compressed_context=context,
            facts=list(dict.fromkeys(all_facts)),  # deduplicate, preserve order
            total_tokens_approx=tokens,
            assembly_time_ms=(time() - t0) * 1000,
        )

    async def _find_bridge_paths(
        self,
        backend: StorageBackend,
        seed_ids: list[str],
    ) -> list[list[str]]:
        """BFS shortest path search between seed nodes."""
        paths: list[list[str]] = []
        max_depth = 3

        # Only top 5 seeds (to avoid O(N^2))
        seeds = seed_ids[:5]

        for i in range(len(seeds) - 1):
            src, dst = seeds[i], seeds[i + 1]
            path = await self._bfs_shortest(backend, src, dst, max_depth)
            if path and len(path) > 2:  # only when bridge nodes exist
                paths.append(path)

        return paths

    async def _bfs_shortest(
        self,
        backend: StorageBackend,
        src: str,
        dst: str,
        max_depth: int,
    ) -> list[str] | None:
        """BFS shortest path from src to dst."""
        if src == dst:
            return [src]

        queue: deque[tuple[str, list[str]]] = deque([(src, [src])])
        visited: set[str] = {src}

        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth + 1:
                break

            edges = await backend.get_edges(current)
            for edge in edges:
                neighbor = edge.target_id if edge.source_id == current else edge.source_id
                if neighbor == dst:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def _topological_sort(
        self,
        node_ids: list[str],
        edges: list[Edge],
        seed_ids: list[str],
    ) -> list[str]:
        """Topological sort. Uses directed edges only; falls back to original order on failure."""
        id_set = set(node_ids)

        # Filter to directed edges only
        directed = [
            e
            for e in edges
            if e.kind in _DIRECTED_KINDS and e.source_id in id_set and e.target_id in id_set
        ]

        if not directed:
            return list(node_ids)  # original order (by activation)

        # Kahn's algorithm
        in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
        adj: dict[str, list[str]] = {nid: [] for nid in node_ids}

        for e in directed:
            adj[e.source_id].append(e.target_id)
            in_degree[e.target_id] = in_degree.get(e.target_id, 0) + 1

        queue: deque[str] = deque(nid for nid in node_ids if in_degree.get(nid, 0) == 0)
        result: list[str] = []

        while queue:
            nid = queue.popleft()
            result.append(nid)
            for neighbor in adj.get(nid, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Append nodes missed due to cycles, etc. (in original order)
        remaining = [nid for nid in node_ids if nid not in set(result)]
        result.extend(remaining)

        return result

    def _compress_content(self, content: str, query: str) -> str:
        """Select and compress only query-relevant sentences."""
        if not content:
            return ""

        # Sentence splitting — after period/question mark/exclamation mark + whitespace
        sentences = re.split(r"(?<=[.!?。])\s+", content.strip())
        if not sentences:
            return content[:500]

        # Extract query terms
        query_terms = {
            t.lower()
            for t in re.split(r"[\s,;:!?()\[\]]+", query)
            if t.lower() not in _STOPWORDS and len(t) >= 2
        }

        if not query_terms:
            # No terms extracted from query — return first N sentences
            return " ".join(sentences[: self._max_sentences])

        # Score each sentence by relevance (with position bias for first sentence)
        scored: list[tuple[int, str, float]] = []
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            sent_terms = set(re.split(r"[\s,;:!?()\[\]]+", sent_lower))
            overlap = len(query_terms & sent_terms)
            relevance = overlap / len(query_terms)
            # Position bias: first sentence gets +0.1 bonus
            if i == 0:
                relevance += 0.1
            scored.append((i, sent, relevance))

        # Select sentences above threshold
        selected = [(i, s) for i, s, r in scored if r >= self._relevance_threshold]

        # Fall back to top N if none selected
        if not selected:
            scored.sort(key=lambda x: x[2], reverse=True)
            selected = [(i, s) for i, s, _ in scored[: self._max_sentences]]

        # Preserve original order
        selected.sort(key=lambda x: x[0])

        # Limit count
        selected = selected[: self._max_sentences]

        return " ".join(s for _, s in selected)

    def _extract_facts(self, content: str) -> list[str]:
        """Extract sentences containing key facts (numbers, dates, proper nouns) via regex."""
        if not content:
            return []

        sentences = re.split(r"(?<=[.!?。])\s+", content.strip())
        facts: list[str] = []
        seen: set[str] = set()

        for sent in sentences:
            for pattern in _FACT_PATTERNS:
                if pattern.search(sent):
                    normalized = sent.strip()
                    if normalized and normalized not in seen:
                        facts.append(normalized)
                        seen.add(normalized)
                    break

        return facts

    def _format_context(self, steps: list[EvidenceStep]) -> str:
        """Assemble steps into a final context string for LLM consumption."""
        parts: list[str] = []

        for i, step in enumerate(steps):
            # Role + title
            title = step.node.title or "Untitled"
            parts.append(f"[{step.role.upper()}] {title}")

            # Compressed content
            if step.compressed_content:
                parts.append(step.compressed_content)

            # Key facts (max 3)
            if step.facts:
                facts_text = " | ".join(step.facts[:3])
                parts.append(f"Key facts: {facts_text}")

            # Connection to next step
            if step.connection_to_next and i < len(steps) - 1:
                parts.append(f"→ {step.connection_to_next}")

            parts.append("")  # separator

        context = "\n".join(parts).strip()

        # Token limit
        words = context.split()
        if len(words) > self._max_tokens:
            context = " ".join(words[: self._max_tokens])

        return context
