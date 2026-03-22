"""qrels 기반 에이전트 세션 시뮬레이션 — Hebbian co-activation 학습."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from synaptic import SynapticGraph


@dataclass
class SimulationStats:
    total_sessions: int = 0
    success_sessions: int = 0
    failure_sessions: int = 0
    total_reinforcements: int = 0


class SessionSimulator:
    """qrels를 순회하며 에이전트 세션을 시뮬레이션하고 Hebbian 학습을 수행."""

    def __init__(self, graph: SynapticGraph) -> None:
        self.graph = graph

    async def simulate_sessions(
        self,
        qrels: dict[str, dict[str, int]],
        id_map: dict[str, str],
        *,
        success_rate: float = 0.8,
        max_sessions: int = 50,
    ) -> SimulationStats:
        """qrels를 순회하면서 에이전트 세션 시뮬레이션.

        각 query = 1 세션:
        1. 해당 query의 relevant document ID들을 가져옴
        2. id_map으로 corpus_id → graph_node_id 변환
        3. graph.reinforce(node_ids, success=True/False) 호출
        4. success_rate 확률로 성공/실패 분배

        Args:
            qrels: query_id → {corpus_id: relevance_score} 매핑.
            id_map: corpus_id → graph_node_id 매핑.
            success_rate: 세션 성공 확률 (0.0~1.0).
            max_sessions: 최대 세션 수.

        Returns:
            시뮬레이션 통계.
        """
        rng = random.Random(42)
        stats = SimulationStats()

        for query_id, doc_rels in list(qrels.items())[:max_sessions]:
            # corpus_id → graph node id 변환 (id_map에 있는 것만)
            node_ids = [
                id_map[cid] for cid in doc_rels if cid in id_map
            ]

            # co-activation에는 최소 2개 노드 필요
            if len(node_ids) < 2:
                continue

            success = rng.random() < success_rate
            await self.graph.reinforce(node_ids, success=success)

            stats.total_sessions += 1
            stats.total_reinforcements += len(node_ids)
            if success:
                stats.success_sessions += 1
            else:
                stats.failure_sessions += 1

        return stats
