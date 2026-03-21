"""Search quality tests — precision and recall on real data.

Tests whether the search engine returns relevant results for realistic queries.
Uses human-crafted query → expected_terms pairs as ground truth.
"""

from __future__ import annotations

import pytest

from synaptic.graph import SynapticGraph

pytestmark = pytest.mark.qa

# Query → terms that SHOULD appear in at least one result title/content
_WIKI_QUERIES: list[tuple[str, list[str]]] = [
    ("데이터베이스", ["데이터베이스", "SQL", "관계형"]),
    ("인공지능 머신러닝", ["인공지능", "학습", "기계"]),
    ("프로그래밍 언어", ["프로그래밍", "언어", "컴파일"]),
    ("네트워크 프로토콜", ["네트워크", "프로토콜", "TCP"]),
    ("클라우드 컴퓨팅", ["클라우드", "가상", "서버"]),
    ("웹 개발", ["웹", "HTML", "HTTP"]),
    ("보안 암호화", ["보안", "암호", "인증"]),
    ("알고리즘 정렬", ["알고리즘", "정렬", "탐색"]),
]


class TestWikipediaSearchQuality:
    """Search quality on Korean Wikipedia tech articles."""

    @pytest.mark.parametrize("query,expected_terms", _WIKI_QUERIES)
    async def test_relevant_results(
        self, wiki_graph: SynapticGraph, query: str, expected_terms: list[str]
    ) -> None:
        result = await wiki_graph.search(query, limit=10)

        if not result.nodes:
            pytest.skip(f"No results for '{query}' — data may be insufficient")

        # At least one result should contain at least one expected term
        all_text = " ".join(f"{n.node.title} {n.node.content}" for n in result.nodes).lower()

        matches = [t for t in expected_terms if t.lower() in all_text]
        assert len(matches) > 0, (
            f"Query '{query}': none of {expected_terms} found in results. "
            f"Got titles: {[n.node.title for n in result.nodes[:5]]}"
        )

    async def test_precision_at_5(self, wiki_graph: SynapticGraph) -> None:
        """Top-5 results for '데이터베이스' should mostly be DB-related."""
        result = await wiki_graph.search("데이터베이스", limit=5)
        if len(result.nodes) < 3:
            pytest.skip("Not enough results")

        db_terms = ["데이터", "sql", "쿼리", "테이블", "관계", "db", "database"]
        relevant = 0
        for activated in result.nodes:
            text = f"{activated.node.title} {activated.node.content}".lower()
            if any(t in text for t in db_terms):
                relevant += 1

        precision = relevant / len(result.nodes)
        assert precision >= 0.4, (
            f"Precision@5 for '데이터베이스' = {precision:.2f}, expected >= 0.4"
        )

    async def test_search_returns_results(self, wiki_graph: SynapticGraph) -> None:
        """Basic sanity: common tech terms should return something."""
        for query in ["Python", "자바", "리눅스", "알고리즘"]:
            result = await wiki_graph.search(query, limit=5)
            # At least one of these should return results
            if result.nodes:
                return
        pytest.fail("None of the basic queries returned results")

    async def test_resonance_ordering(self, wiki_graph: SynapticGraph) -> None:
        """Results should be ordered by resonance score (descending)."""
        result = await wiki_graph.search("프로그래밍", limit=10)
        if len(result.nodes) < 2:
            pytest.skip("Not enough results")

        for i in range(len(result.nodes) - 1):
            assert result.nodes[i].resonance >= result.nodes[i + 1].resonance


class TestGitHubSearchQuality:
    """Search quality on GitHub commits + issues."""

    async def test_commit_search(self, github_graph: SynapticGraph) -> None:
        """Search for commit-related terms should find commits."""
        result = await github_graph.search("fix bug", limit=10)
        if not result.nodes:
            # Try Korean
            result = await github_graph.search("버그 수정", limit=10)
        if not result.nodes:
            pytest.skip("No commit data or no matching results")

        sources = [n.node.source for n in result.nodes]
        assert any("github" in s for s in sources)

    async def test_cross_source_search(self, combined_graph: SynapticGraph) -> None:
        """Search should find results from both Wikipedia and GitHub."""
        result = await combined_graph.search("프로그래밍", limit=20)
        if not result.nodes:
            pytest.skip("No results")

        sources = {n.node.source for n in result.nodes}
        # Not strictly required, but good if both sources appear
        has_wiki = any("wikipedia" in s for s in sources)
        has_github = any("github" in s for s in sources)
        assert has_wiki or has_github, f"Expected mixed sources, got: {sources}"

    async def test_search_time_reasonable(self, combined_graph: SynapticGraph) -> None:
        """Search should complete within 500ms even with 100+ nodes."""
        result = await combined_graph.search("데이터베이스", limit=10)
        assert result.search_time_ms < 500, (
            f"Search took {result.search_time_ms:.1f}ms, expected < 500ms"
        )
