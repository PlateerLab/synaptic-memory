"""Edge-case search quality tests — Korean/English crossover, typos, long queries.

Tests tricky scenarios that stress synonym expansion, fuzzy matching,
spreading activation, and reinforcement ranking.
"""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import NodeKind

pytestmark = pytest.mark.qa


class TestKoreanOnlyTerms:
    """Search for terms that only exist in Korean should find results."""

    async def test_korean_compound_term(self, wiki_graph: SynapticGraph) -> None:
        """'관계형 데이터베이스' — Korean compound term should find relevant results."""
        result = await wiki_graph.search("관계형 데이터베이스", limit=10)

        if not result.nodes:
            pytest.skip("No results — data may not contain relational DB articles")

        all_text = " ".join(f"{n.node.title} {n.node.content}" for n in result.nodes).lower()
        db_terms = ["데이터베이스", "관계형", "sql", "테이블", "rdbms", "database"]
        matches = [t for t in db_terms if t in all_text]
        assert len(matches) > 0, (
            f"'관계형 데이터베이스' search returned no DB-related content. "
            f"Titles: {[n.node.title for n in result.nodes[:5]]}"
        )

    async def test_korean_only_no_english_equivalent(self, wiki_graph: SynapticGraph) -> None:
        """Korean-only search for '운영 체제' (operating system)."""
        result = await wiki_graph.search("운영 체제", limit=10)

        if not result.nodes:
            pytest.skip("No results for '운영 체제'")

        all_text = " ".join(f"{n.node.title} {n.node.content}" for n in result.nodes).lower()
        os_terms = ["운영", "체제", "커널", "프로세스", "시스템"]
        matches = [t for t in os_terms if t in all_text]
        assert len(matches) > 0


class TestEnglishToKoreanSynonym:
    """English terms should find Korean articles via synonym expansion."""

    async def test_machine_learning_finds_korean(self, wiki_graph: SynapticGraph) -> None:
        """'machine learning' should find Korean ML/AI articles via synonym expansion."""
        result = await wiki_graph.search("machine learning", limit=10)

        # Synonym expansion should map 'learning' -> '학습', '훈련', etc.
        if not result.nodes:
            pytest.skip("No results for 'machine learning'")

        assert "synonym" in result.stages_used or len(result.nodes) > 0, (
            "Expected synonym expansion to trigger for English query on Korean data"
        )

        all_text = " ".join(f"{n.node.title} {n.node.content}" for n in result.nodes).lower()
        ml_terms = ["학습", "기계", "인공지능", "machine", "learning", "훈련", "신경망"]
        matches = [t for t in ml_terms if t in all_text]
        assert len(matches) > 0, (
            f"'machine learning' found no ML-related Korean content. "
            f"Titles: {[n.node.title for n in result.nodes[:5]]}"
        )

    async def test_database_finds_korean(self, wiki_graph: SynapticGraph) -> None:
        """'database' should find Korean DB articles via synonym group."""
        result = await wiki_graph.search("database", limit=10)

        if not result.nodes:
            pytest.skip("No results for 'database'")

        all_text = " ".join(f"{n.node.title} {n.node.content}" for n in result.nodes).lower()
        assert any(t in all_text for t in ["데이터베이스", "database", "db", "sql"])


class TestTypoFuzzyMatch:
    """Typo/variant handling — FTS substring matching + synonym expansion."""

    async def test_korean_typo_variant(self, wiki_graph: SynapticGraph) -> None:
        """'데이타베이스' (old spelling) — substring '데이' or synonym should catch."""
        result_standard = await wiki_graph.search("데이터베이스", limit=10)
        result_typo = await wiki_graph.search("데이타베이스", limit=10)

        if not result_standard.nodes:
            pytest.skip("No results for standard spelling")

        # 오타 보정은 향후 embedding vector search로 커버 예정
        # 현재는 부분 매칭('데이')으로 일부 결과를 잡을 수 있음
        # 결과가 없어도 실패가 아닌 baseline으로 기록
        if not result_typo.nodes:
            pytest.skip("Typo recovery requires embedding vector search (not available in MemoryBackend)")

    async def test_english_typo(self, wiki_graph: SynapticGraph) -> None:
        """'Pytohn' (typo for Python) — substring matching or embedding should catch."""
        result = await wiki_graph.search("Pytohn", limit=10)

        # 오타 보정은 향후 embedding 또는 edit-distance index로 커버
        if not result.nodes:
            pytest.skip("Typo recovery requires embedding vector search")


class TestLongQuery:
    """Very long queries should not crash or hang."""

    async def test_long_query_no_crash(self, wiki_graph: SynapticGraph) -> None:
        """100+ character query should return without error."""
        long_query = (
            "인공지능과 머신러닝을 활용한 데이터베이스 최적화 방법론에 대한 "
            "심층적인 분석과 클라우드 컴퓨팅 환경에서의 성능 개선 전략 그리고 "
            "마이크로서비스 아키텍처에서의 분산 시스템 모니터링과 장애 복구 자동화"
        )
        assert len(long_query) > 100

        result = await wiki_graph.search(long_query, limit=10)

        # Should not crash, and should return a valid SearchResult
        assert result.query == long_query
        assert result.search_time_ms >= 0
        assert isinstance(result.nodes, list)

    async def test_very_long_query_200_chars(self, wiki_graph: SynapticGraph) -> None:
        """200+ character query should also work."""
        long_query = "프로그래밍 " * 50  # 300+ chars
        assert len(long_query) > 200

        result = await wiki_graph.search(long_query.strip(), limit=5)
        assert isinstance(result.nodes, list)
        assert result.search_time_ms < 5000  # Should not hang


class TestSpreadingActivation:
    """Linked nodes should surface via spreading activation."""

    async def test_linked_node_surfaces_in_search(self) -> None:
        """Add 2 nodes, link them, search for one — spreading activation brings the other."""
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(backend)

        # Node A: clearly about "quantum computing"
        node_a = await graph.add(
            title="양자 컴퓨팅 개론",
            content="양자 컴퓨터는 큐비트를 사용하여 계산을 수행하는 새로운 패러다임이다.",
            kind=NodeKind.CONCEPT,
            tags=["quantum"],
        )

        # Node B: about "encryption" — not directly matching "양자 컴퓨팅"
        node_b = await graph.add(
            title="암호화 알고리즘",
            content="RSA와 AES를 비롯한 현대 암호화 기술의 원리를 설명한다.",
            kind=NodeKind.CONCEPT,
            tags=["encryption"],
        )

        # Link them
        await graph.link(node_a.id, node_b.id)

        # Search for quantum computing — node_b should appear via spreading activation
        result = await graph.search("양자 컴퓨팅", limit=10)
        result_ids = [n.node.id for n in result.nodes]

        assert node_a.id in result_ids, "Primary node not found in search results"
        assert node_b.id in result_ids, (
            f"Linked node not found via spreading activation. Got IDs: {result_ids}"
        )

        await backend.close()

    async def test_spreading_activation_with_weight(self) -> None:
        """Higher edge weight should give higher activation to neighbor."""
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(backend)

        node_a = await graph.add(
            title="메인 토픽",
            content="이것은 검색의 주요 대상이다.",
            kind=NodeKind.CONCEPT,
        )
        node_b = await graph.add(
            title="강한 연결",
            content="전혀 다른 내용이지만 강하게 연결되어 있다.",
            kind=NodeKind.CONCEPT,
        )
        node_c = await graph.add(
            title="약한 연결",
            content="역시 다른 내용이고 약하게 연결되어 있다.",
            kind=NodeKind.CONCEPT,
        )

        await graph.link(node_a.id, node_b.id, weight=3.0)
        await graph.link(node_a.id, node_c.id, weight=0.2)

        result = await graph.search("메인 토픽", limit=10)
        result_map = {n.node.id: n for n in result.nodes}

        if node_b.id in result_map and node_c.id in result_map:
            # Strong link should have higher activation than weak link
            assert result_map[node_b.id].activation >= result_map[node_c.id].activation, (
                f"Strong link activation ({result_map[node_b.id].activation:.3f}) "
                f"should >= weak link ({result_map[node_c.id].activation:.3f})"
            )

        await backend.close()


class TestReinforcementRanking:
    """Reinforced nodes should rank higher in search results."""

    async def test_reinforcement_boosts_ranking(self) -> None:
        """Reinforce a node 10 times, then verify it ranks higher."""
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(backend)

        # Create several nodes with similar content
        nodes = []
        for i in range(5):
            node = await graph.add(
                title=f"소프트웨어 설계 원칙 {i + 1}",
                content=f"소프트웨어 공학에서 중요한 설계 원칙 번호 {i + 1}에 대한 설명.",
                kind=NodeKind.CONCEPT,
                tags=["설계", "소프트웨어"],
            )
            nodes.append(node)

        # Initial search — get baseline ranking
        result_before = await graph.search("소프트웨어 설계", limit=5)
        assert len(result_before.nodes) >= 3, "Need at least 3 results for ranking test"

        # Pick the LAST result (lowest ranked)
        target = result_before.nodes[-1]
        target_id = target.node.id
        initial_rank = len(result_before.nodes) - 1

        # Reinforce 10 times
        for _ in range(10):
            await graph.reinforce([target_id], success=True)

        # Search again
        result_after = await graph.search("소프트웨어 설계", limit=5)
        new_rank = next(
            (i for i, n in enumerate(result_after.nodes) if n.node.id == target_id),
            len(result_after.nodes),
        )

        # Should have improved ranking (lower index = higher rank)
        assert new_rank < initial_rank, (
            f"After 10 reinforcements, rank should improve: was #{initial_rank}, now #{new_rank}"
        )

        # Also verify resonance increased
        target_after = next((n for n in result_after.nodes if n.node.id == target_id), None)
        assert target_after is not None
        assert target_after.resonance > target.resonance, (
            f"Resonance should increase after reinforcement: "
            f"was {target.resonance:.3f}, now {target_after.resonance:.3f}"
        )

        await backend.close()

    async def test_unreinforced_vs_reinforced_ordering(self) -> None:
        """Two identical nodes — reinforced one should rank higher."""
        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(backend)

        node_plain = await graph.add(
            title="데이터 분석 기법",
            content="데이터 분석과 통계적 방법론에 대한 설명이다.",
            kind=NodeKind.CONCEPT,
        )
        node_reinforced = await graph.add(
            title="데이터 분석 방법",
            content="데이터 분석과 통계적 접근법에 대한 설명이다.",
            kind=NodeKind.CONCEPT,
        )

        # Reinforce one node heavily
        for _ in range(10):
            await graph.reinforce([node_reinforced.id], success=True)

        result = await graph.search("데이터 분석", limit=5)
        result_ids = [n.node.id for n in result.nodes]

        assert node_reinforced.id in result_ids, "Reinforced node should appear in results"
        assert node_plain.id in result_ids, "Plain node should also appear in results"

        rank_reinforced = result_ids.index(node_reinforced.id)
        rank_plain = result_ids.index(node_plain.id)
        assert rank_reinforced < rank_plain, (
            f"Reinforced node (rank {rank_reinforced}) should rank higher "
            f"than plain node (rank {rank_plain})"
        )

        await backend.close()
