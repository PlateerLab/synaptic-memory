"""EvidenceAssembler — SearchResult를 LLM-optimized evidence chain으로 변환."""

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


# 위상 정렬에 사용할 방향성 edge kinds
_DIRECTED_KINDS = frozenset({
    EdgeKind.CAUSED,
    EdgeKind.RESULTED_IN,
    EdgeKind.DEPENDS_ON,
    EdgeKind.FOLLOWED_BY,
    EdgeKind.LEARNED_FROM,
})

# 불용어 (term overlap 계산에서 제외)
_STOPWORDS = frozenset({
    # 영어
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
    "to", "for", "of", "and", "or", "but", "not", "with", "by", "from",
    "that", "this", "it", "its", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "what", "which", "who", "when", "where", "how", "why",
    # 한국어
    "은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "도",
    "에서", "로", "으로", "하는", "있는", "하고", "하면", "에게",
})

# Fact 추출 패턴
_FACT_PATTERNS = [
    # 숫자 + 단위
    re.compile(
        r'\d[\d,.]*\s*(%|만|억|원|달러|km|kg|GB|MB|TB|명|건|개|년|월|일|시간|분|초|percent|million|billion|thousand)',
        re.IGNORECASE,
    ),
    # 날짜 (2024-01-01, 2024년, January 2024, 15 March 1990)
    re.compile(r'\b\d{4}[-/년.]\d{1,2}[-/월.]?\d{0,2}일?\b'),
    re.compile(
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+\d{1,2},?\s*\d{4}\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b\d{1,2}\s+'
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+\d{4}\b',
        re.IGNORECASE,
    ),
    # 숫자만 (연도, 인구 등) - 4자리 이상
    re.compile(r'\b\d{4,}\b'),
]


class EvidenceAssembler:
    """SearchResult를 LLM-optimized evidence chain으로 변환."""

    __slots__ = ("_max_sentences", "_relevance_threshold", "_max_tokens")

    def __init__(
        self,
        *,
        max_sentences_per_node: int = 5,
        relevance_threshold: float = 0.2,
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
        """Search 결과를 evidence chain으로 조립."""
        t0 = time()

        if not search_result.nodes:
            return EvidenceChain(query=query, assembly_time_ms=(time() - t0) * 1000)

        # 1. Seed 노드 추출 (상위 max_steps개)
        seed_nodes = search_result.nodes[:max_steps]
        seed_ids = [a.node.id for a in seed_nodes]
        seed_map: dict[str, Node] = {a.node.id: a.node for a in seed_nodes}

        # 2. BFS로 bridge 노드 탐색
        bridge_paths = await self._find_bridge_paths(backend, seed_ids)

        # bridge에서 발견된 새 노드 수집
        all_ids: list[str] = list(seed_ids)
        for path in bridge_paths:
            for nid in path:
                if nid not in seed_map:
                    node = await backend.get_node(nid)
                    if node:
                        seed_map[nid] = node
                        if nid not in all_ids:
                            all_ids.append(nid)

        # 3. 엣지 수집 (위상 정렬용)
        all_edges: list[Edge] = []
        id_set = set(all_ids)
        for nid in all_ids:
            edges = await backend.get_edges(nid)
            for e in edges:
                other = e.target_id if e.source_id == nid else e.source_id
                if other in id_set:
                    all_edges.append(e)

        # 4. 위상 정렬
        sorted_ids = self._topological_sort(all_ids, all_edges, seed_ids)

        # 5. Step 생성
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

            # 다음 step으로의 연결 설명
            conn = ""
            if i < len(sorted_ids) - 1:
                next_id = sorted_ids[i + 1]
                for e in all_edges:
                    if (e.source_id == nid and e.target_id == next_id) or \
                       (e.target_id == nid and e.source_id == next_id):
                        conn = e.kind.value
                        break

            steps.append(EvidenceStep(
                node=node,
                role=role,
                connection_to_next=conn,
                compressed_content=compressed,
                facts=facts,
            ))

        # 6. 최종 context 포맷팅
        context = self._format_context(steps)

        # 토큰 근사
        tokens = len(context.split())

        return EvidenceChain(
            query=query,
            steps=steps,
            compressed_context=context,
            facts=list(dict.fromkeys(all_facts)),  # 중복 제거, 순서 유지
            total_tokens_approx=tokens,
            assembly_time_ms=(time() - t0) * 1000,
        )

    async def _find_bridge_paths(
        self,
        backend: StorageBackend,
        seed_ids: list[str],
    ) -> list[list[str]]:
        """Seed 노드 간 BFS shortest path 탐색."""
        paths: list[list[str]] = []
        max_depth = 3

        # 상위 5개 seed만 (O(N²) 방지)
        seeds = seed_ids[:5]

        for i in range(len(seeds) - 1):
            src, dst = seeds[i], seeds[i + 1]
            path = await self._bfs_shortest(backend, src, dst, max_depth)
            if path and len(path) > 2:  # bridge가 있는 경우만
                paths.append(path)

        return paths

    async def _bfs_shortest(
        self,
        backend: StorageBackend,
        src: str,
        dst: str,
        max_depth: int,
    ) -> list[str] | None:
        """BFS로 src → dst 최단 경로."""
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
        """위상 정렬. 방향성 edge만 사용, 실패 시 원래 순서 폴백."""
        id_set = set(node_ids)

        # 방향성 edge 필터
        directed = [
            e for e in edges
            if e.kind in _DIRECTED_KINDS
            and e.source_id in id_set
            and e.target_id in id_set
        ]

        if not directed:
            return list(node_ids)  # 원래 순서 (activation 순)

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

        # 순환 등으로 누락된 노드 추가 (원래 순서)
        remaining = [nid for nid in node_ids if nid not in set(result)]
        result.extend(remaining)

        return result

    def _compress_content(self, content: str, query: str) -> str:
        """Query 관련 문장만 선택하여 압축."""
        if not content:
            return ""

        # 문장 분리 — 마침표/물음표/느낌표 뒤 공백 + 다음 문자
        sentences = re.split(r'(?<=[.!?。])\s+', content.strip())
        if not sentences:
            return content[:500]

        # query term 추출
        query_terms = {
            t.lower() for t in re.split(r'[\s,;:!?()\[\]]+', query)
            if t.lower() not in _STOPWORDS and len(t) >= 2
        }

        if not query_terms:
            # query에서 term을 못 뽑으면 처음 N문장 반환
            return " ".join(sentences[:self._max_sentences])

        # 각 문장의 relevance
        scored: list[tuple[int, str, float]] = []
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            sent_terms = set(re.split(r'[\s,;:!?()\[\]]+', sent_lower))
            overlap = len(query_terms & sent_terms)
            relevance = overlap / len(query_terms)
            scored.append((i, sent, relevance))

        # threshold 이상 선택
        selected = [(i, s) for i, s, r in scored if r >= self._relevance_threshold]

        # 없으면 상위 N개 폴백
        if not selected:
            scored.sort(key=lambda x: x[2], reverse=True)
            selected = [(i, s) for i, s, _ in scored[:self._max_sentences]]

        # 원래 순서 유지
        selected.sort(key=lambda x: x[0])

        # 개수 제한
        selected = selected[:self._max_sentences]

        return " ".join(s for _, s in selected)

    def _extract_facts(self, content: str) -> list[str]:
        """정규식으로 핵심 사실(숫자, 날짜, 고유명사) 포함 문장 추출."""
        if not content:
            return []

        sentences = re.split(r'(?<=[.!?。])\s+', content.strip())
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
        """Steps를 LLM에게 전달할 최종 context 문자열로 조립."""
        parts: list[str] = []

        for i, step in enumerate(steps):
            # 역할 + 제목
            title = step.node.title or "Untitled"
            parts.append(f"[{step.role.upper()}] {title}")

            # 압축된 content
            if step.compressed_content:
                parts.append(step.compressed_content)

            # 핵심 facts (최대 3개)
            if step.facts:
                facts_text = " | ".join(step.facts[:3])
                parts.append(f"Key facts: {facts_text}")

            # 다음 step 연결
            if step.connection_to_next and i < len(steps) - 1:
                parts.append(f"→ {step.connection_to_next}")

            parts.append("")  # 구분

        context = "\n".join(parts).strip()

        # 토큰 제한
        words = context.split()
        if len(words) > self._max_tokens:
            context = " ".join(words[:self._max_tokens])

        return context
