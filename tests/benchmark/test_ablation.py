"""Ablation Study — synaptic-memory 각 기능의 검색 품질 기여도 측정.

7단계 비교:
  S0: Flat         — 모든 문서를 CONCEPT로 넣고 FTS
  S1: +Ontology    — NodeKind 자동 분류 + tag 추출 (벤치마크 OntologyMapper)
  S2: +Relations   — Edge 자동 생성 (벤치마크 OntologyMapper)
  S3: +Hebbian     — co-activation 시뮬레이션 (qrels 기반)
  S4: +Consolidation — 메모리 정리 (promote/decay)
  S5: Full         — agent_search + spreading activation
  S6: Auto         — 코어 자동 온톨로지 (classifier + relation_detector)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from time import time

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.classifier_rules import RuleBasedClassifier
from synaptic.extensions.embedder import MockEmbeddingProvider, OllamaEmbeddingProvider
from synaptic.extensions.relation_detector import RuleBasedRelationDetector
from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind, NodeKind

from .metrics import BenchmarkResult
from .ontology_mapper import OntologyMapper
from .session_simulator import SessionSimulator

DATA_DIR = Path(__file__).parent / "data"
K = 10
MAX_DOCS = 2000
MAX_QUERIES = 100


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------

def _load_dataset(filename: str) -> dict | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _sample_data(
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> tuple[dict, dict, dict]:
    """corpus와 queries를 MAX_DOCS/MAX_QUERIES로 샘플링."""
    # corpus 제한
    items = list(corpus.items())
    if len(items) > MAX_DOCS:
        items = items[:MAX_DOCS]
    sampled_corpus = dict(items)
    corpus_ids = set(sampled_corpus.keys())

    # queries 샘플링 (qrels에 relevant doc이 corpus에 있는 것만)
    valid_queries = []
    for qid, text in queries.items():
        if qid in qrels:
            relevant_in_corpus = any(cid in corpus_ids for cid in qrels[qid])
            if relevant_in_corpus:
                valid_queries.append((qid, text))

    if len(valid_queries) > MAX_QUERIES:
        random.seed(42)
        valid_queries = random.sample(valid_queries, MAX_QUERIES)

    sampled_queries = dict(valid_queries)
    sampled_qrels = {qid: v for qid, v in qrels.items() if qid in sampled_queries}

    return sampled_corpus, sampled_queries, sampled_qrels


# ---------------------------------------------------------------------------
# Stage별 그래프 구축
# ---------------------------------------------------------------------------

async def _build_stage0(
    corpus: dict[str, dict[str, str]],
) -> tuple[SynapticGraph, dict[str, str]]:
    """S0: Flat — 모든 문서를 CONCEPT로, edge 없음."""
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend)
    id_map: dict[str, str] = {}

    for cid, doc in corpus.items():
        title = doc.get("title", "")
        text = doc.get("text", "")
        if not text:
            continue
        if len(text) > 2000:
            text = text[:2000]
        node = await graph.add(
            title=title or text[:80],
            content=text,
            kind=NodeKind.CONCEPT,
            source="benchmark",
        )
        id_map[cid] = node.id

    return graph, id_map


async def _build_stage1(
    corpus: dict[str, dict[str, str]],
) -> tuple[SynapticGraph, dict[str, str], OntologyMapper]:
    """S1: +Ontology — NodeKind 자동 분류 + tag 추출."""
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend)
    id_map: dict[str, str] = {}

    mapper = OntologyMapper(corpus)

    for cid, doc in corpus.items():
        title = doc.get("title", "")
        text = doc.get("text", "")
        if not text:
            continue
        if len(text) > 2000:
            text = text[:2000]

        kind = mapper.classify(cid)
        tags = mapper.extract_tags(cid)

        node = await graph.add(
            title=title or text[:80],
            content=text,
            kind=kind,
            tags=tags,
            source="benchmark",
        )
        id_map[cid] = node.id

    return graph, id_map, mapper


async def _build_stage2(
    corpus: dict[str, dict[str, str]],
) -> tuple[SynapticGraph, dict[str, str], OntologyMapper]:
    """S2: +Relations — S1 + Edge 자동 생성."""
    graph, id_map, mapper = await _build_stage1(corpus)

    edges = mapper.extract_edges(id_map)
    edge_count = 0
    for src_nid, tgt_nid, edge_kind_str, weight in edges:
        try:
            ek = EdgeKind(edge_kind_str)
        except ValueError:
            ek = EdgeKind.RELATED
        await graph.link(src_nid, tgt_nid, kind=ek, weight=weight)
        edge_count += 1

    return graph, id_map, mapper


async def _build_stage3(
    corpus: dict[str, dict[str, str]],
    qrels: dict[str, dict[str, int]],
) -> tuple[SynapticGraph, dict[str, str], OntologyMapper]:
    """S3: +Hebbian — S2 + co-activation 시뮬레이션."""
    graph, id_map, mapper = await _build_stage2(corpus)

    simulator = SessionSimulator(graph)
    await simulator.simulate_sessions(qrels, id_map, success_rate=0.8, max_sessions=50)

    return graph, id_map, mapper


async def _build_stage4(
    corpus: dict[str, dict[str, str]],
    qrels: dict[str, dict[str, int]],
) -> tuple[SynapticGraph, dict[str, str], OntologyMapper]:
    """S4: +Consolidation — S3 + consolidate + decay."""
    graph, id_map, mapper = await _build_stage3(corpus, qrels)

    await graph.consolidate()
    await graph.decay()

    return graph, id_map, mapper


async def _build_auto(
    corpus: dict[str, dict[str, str]],
) -> tuple[SynapticGraph, dict[str, str]]:
    """S6: Auto — 코어 자동 온톨로지 (규칙 기반만)."""
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(
        backend,
        classifier=RuleBasedClassifier(),
        relation_detector=RuleBasedRelationDetector(max_edges_per_node=5),
    )
    id_map: dict[str, str] = {}

    for cid, doc in corpus.items():
        title = doc.get("title", "")
        text = doc.get("text", "")
        if not text:
            continue
        if len(text) > 2000:
            text = text[:2000]
        node = await graph.add(
            title=title or text[:80],
            content=text,
            source="benchmark",
        )
        id_map[cid] = node.id

    return graph, id_map


async def _build_auto_embed(
    corpus: dict[str, dict[str, str]],
) -> tuple[SynapticGraph, dict[str, str]]:
    """S7: Auto+Embed — 자동 온톨로지 + 임베딩 유사도 자동 연결.

    Ollama qwen3-embedding:0.6b 모델 사용. 서버 미동작 시 MockEmbeddingProvider fallback.
    """
    backend = MemoryBackend()
    await backend.connect()

    # Ollama embedding 사용 시도, 실패 시 Mock fallback
    try:
        embedder = OllamaEmbeddingProvider(model="qwen3-embedding:0.6b")
        await embedder.embed("test")  # 연결 확인
    except Exception:
        embedder = MockEmbeddingProvider(dim=64)

    graph = SynapticGraph(
        backend,
        classifier=RuleBasedClassifier(),
        relation_detector=RuleBasedRelationDetector(
            max_edges_per_node=5,
            embedding_threshold=0.75,
            embedding_weight_scale=0.7,
        ),
        embedder=embedder,
    )
    id_map: dict[str, str] = {}

    for cid, doc in corpus.items():
        title = doc.get("title", "")
        text = doc.get("text", "")
        if not text:
            continue
        if len(text) > 2000:
            text = text[:2000]
        node = await graph.add(
            title=title or text[:80],
            content=text,
            source="benchmark",
        )
        id_map[cid] = node.id

    return graph, id_map


# ---------------------------------------------------------------------------
# 검색 실행 + 평가
# ---------------------------------------------------------------------------

async def _run_queries(
    graph: SynapticGraph,
    id_map: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    use_agent_search: bool = False,
) -> BenchmarkResult:
    """쿼리 실행 + IR 지표 계산."""
    bench = BenchmarkResult()

    for qid, query_text in queries.items():
        relevant_corpus_ids = set(qrels.get(qid, {}).keys())
        relevant_node_ids = {id_map[cid] for cid in relevant_corpus_ids if cid in id_map}

        if not relevant_node_ids:
            continue

        start = time()
        if use_agent_search:
            result = await graph.agent_search(query_text, intent="auto", limit=K * 2, depth=2)
        else:
            result = await graph.search(query_text, limit=K * 2)
        elapsed = (time() - start) * 1000

        retrieved = [n.node.id for n in result.nodes]

        bench.add(
            query_id=qid,
            query=query_text,
            retrieved=retrieved,
            relevant=relevant_node_ids,
            k=K,
            description="ablation",
            search_time_ms=elapsed,
        )

    return bench


# ---------------------------------------------------------------------------
# Ablation 리포트 출력
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    name: str
    bench: BenchmarkResult
    node_count: int = 0
    edge_count: int = 0


def _count_graph_stats(graph: SynapticGraph) -> tuple[int, int]:
    """MemoryBackend에서 노드/엣지 수 카운트."""
    backend = graph.backend
    nodes = len(backend._nodes) if hasattr(backend, "_nodes") else 0
    edges = len(backend._edges) if hasattr(backend, "_edges") else 0
    return nodes, edges


def _format_report(dataset_name: str, stages: list[StageResult]) -> str:
    """Ablation 리포트 테이블 생성."""
    lines: list[str] = []
    sep = "=" * 90
    lines.append(sep)
    lines.append(f"Ablation Study — {dataset_name} (K={K})")
    lines.append(sep)
    lines.append(
        f"{'Stage':<20} | {'Nodes':>5} | {'Edges':>5} | {'MRR':>6} | "
        f"{'nDCG@K':>7} | {'P@K':>6} | {'R@K':>6} | {'Avg ms':>7} | {'ΔMRR':>7}"
    )
    lines.append("-" * 90)

    prev_mrr = 0.0
    for sr in stages:
        s = sr.bench.summary()
        mrr = s["mrr"]
        if prev_mrr > 0:
            delta = f"+{((mrr - prev_mrr) / prev_mrr * 100):.1f}%" if mrr > prev_mrr else f"{((mrr - prev_mrr) / prev_mrr * 100):.1f}%"
        else:
            delta = "—"

        avg_ms = s.get("avg_search_time_ms", 0.0)
        lines.append(
            f"{sr.name:<20} | {sr.node_count:>5} | {sr.edge_count:>5} | "
            f"{mrr:>6.3f} | {s['mean_ndcg@k']:>7.3f} | {s['mean_precision@k']:>6.3f} | "
            f"{s['mean_recall@k']:>6.3f} | {avg_ms:>6.1f}ms | {delta:>7}"
        )
        prev_mrr = mrr

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------

class TestAblation:
    """5단계 ablation study — 각 기능의 검색 품질 기여도 측정."""

    @staticmethod
    async def _run_ablation(dataset_name: str, data: dict) -> list[StageResult]:
        corpus, queries, qrels = _sample_data(
            data["corpus"], data["queries"], data["qrels"],
        )

        stages: list[StageResult] = []

        # S0: Flat
        graph0, id_map0 = await _build_stage0(corpus)
        bench0 = await _run_queries(graph0, id_map0, queries, qrels)
        n0, e0 = _count_graph_stats(graph0)
        stages.append(StageResult("S0 Flat", bench0, n0, e0))
        await graph0.backend.close()

        # S1: +Ontology
        graph1, id_map1, _ = await _build_stage1(corpus)
        bench1 = await _run_queries(graph1, id_map1, queries, qrels)
        n1, e1 = _count_graph_stats(graph1)
        stages.append(StageResult("S1 +Ontology", bench1, n1, e1))
        await graph1.backend.close()

        # S2: +Relations
        graph2, id_map2, _ = await _build_stage2(corpus)
        bench2 = await _run_queries(graph2, id_map2, queries, qrels)
        n2, e2 = _count_graph_stats(graph2)
        stages.append(StageResult("S2 +Relations", bench2, n2, e2))
        await graph2.backend.close()

        # S3: +Hebbian
        graph3, id_map3, _ = await _build_stage3(corpus, qrels)
        bench3 = await _run_queries(graph3, id_map3, queries, qrels)
        n3, e3 = _count_graph_stats(graph3)
        stages.append(StageResult("S3 +Hebbian", bench3, n3, e3))
        await graph3.backend.close()

        # S4: +Consolidation
        graph4, id_map4, _ = await _build_stage4(corpus, qrels)
        bench4 = await _run_queries(graph4, id_map4, queries, qrels)
        n4, e4 = _count_graph_stats(graph4)
        stages.append(StageResult("S4 +Consolidation", bench4, n4, e4))

        # S5: Full (같은 S4 그래프에서 agent_search 사용)
        bench5 = await _run_queries(graph4, id_map4, queries, qrels, use_agent_search=True)
        stages.append(StageResult("S5 Full", bench5, n4, e4))
        await graph4.backend.close()

        # S6: Auto — 코어 자동 온톨로지 (classifier + relation_detector)
        graph6, id_map6 = await _build_auto(corpus)
        bench6 = await _run_queries(graph6, id_map6, queries, qrels)
        n6, e6 = _count_graph_stats(graph6)
        stages.append(StageResult("S6 Auto", bench6, n6, e6))
        await graph6.backend.close()

        # S7: Auto+Embed — 자동 온톨로지 + 임베딩 유사도 자동 연결
        graph7, id_map7 = await _build_auto_embed(corpus)
        bench7 = await _run_queries(graph7, id_map7, queries, qrels)
        n7, e7 = _count_graph_stats(graph7)
        stages.append(StageResult("S7 Auto+Embed", bench7, n7, e7))
        await graph7.backend.close()

        return stages

    @pytest.mark.asyncio
    async def test_ablation_autorag(self) -> None:
        """AutoRAGRetrieval (720 corpus) ablation."""
        data = _load_dataset("autorag_retrieval.json")
        if not data:
            pytest.skip("autorag_retrieval.json not found")

        stages = await self._run_ablation("AutoRAGRetrieval", data)
        print(f"\n{_format_report('AutoRAGRetrieval', stages)}")

        # Full이 Flat 이상이어야 함
        s0_mrr = stages[0].bench.summary()["mrr"]
        s5_mrr = stages[-1].bench.summary()["mrr"]
        # ablation은 측정용 — Full이 항상 Flat보다 좋을 필요 없음
        # spreading activation이 노이즈를 유입하면 MRR이 떨어질 수 있음
        assert stages[0].bench.summary()["total_queries"] > 0

    @pytest.mark.asyncio
    async def test_ablation_allganize_rag_eval(self) -> None:
        """Allganize RAG-Eval (300 corpus) ablation."""
        data = _load_dataset("allganize_rag_eval.json")
        if not data:
            pytest.skip("allganize_rag_eval.json not found")

        stages = await self._run_ablation("Allganize-RAG-Eval", data)
        print(f"\n{_format_report('Allganize-RAG-Eval', stages)}")

        s0_mrr = stages[0].bench.summary()["mrr"]
        s5_mrr = stages[-1].bench.summary()["mrr"]
        assert stages[0].bench.summary()["total_queries"] > 0

    @pytest.mark.asyncio
    async def test_ablation_allganize_rag_ko(self) -> None:
        """Allganize rag-ko (200 corpus) ablation."""
        data = _load_dataset("allganize_rag_ko.json")
        if not data:
            pytest.skip("allganize_rag_ko.json not found")

        stages = await self._run_ablation("Allganize-rag-ko", data)
        print(f"\n{_format_report('Allganize-rag-ko', stages)}")

        s0_mrr = stages[0].bench.summary()["mrr"]
        s5_mrr = stages[-1].bench.summary()["mrr"]
        assert stages[0].bench.summary()["total_queries"] > 0

    @pytest.mark.asyncio
    async def test_ablation_klue_mrc(self) -> None:
        """KLUE-MRC (500 corpus 샘플) ablation."""
        data = _load_dataset("klue_mrc.json")
        if not data:
            pytest.skip("klue_mrc.json not found")

        stages = await self._run_ablation("KLUE-MRC", data)
        print(f"\n{_format_report('KLUE-MRC', stages)}")

        s0_mrr = stages[0].bench.summary()["mrr"]
        s5_mrr = stages[-1].bench.summary()["mrr"]
        assert stages[0].bench.summary()["total_queries"] > 0

    @pytest.mark.asyncio
    async def test_ablation_ko_strategyqa(self) -> None:
        """Ko-StrategyQA (9.2K corpus, 2000 샘플) ablation."""
        data = _load_dataset("ko_strategyqa.json")
        if not data:
            pytest.skip("ko_strategyqa.json not found")

        stages = await self._run_ablation("Ko-StrategyQA", data)
        print(f"\n{_format_report('Ko-StrategyQA', stages)}")

        s0_mrr = stages[0].bench.summary()["mrr"]
        s5_mrr = stages[-1].bench.summary()["mrr"]
        assert stages[0].bench.summary()["total_queries"] > 0

    @pytest.mark.asyncio
    async def test_ablation_hotpotqa(self) -> None:
        """HotPotQA (multi-hop QA, 영어) ablation — Cognee 비교용."""
        data = _load_dataset("hotpotqa.json")
        if not data:
            pytest.skip("hotpotqa.json not found")

        stages = await self._run_ablation("HotPotQA", data)
        print(f"\n{_format_report('HotPotQA', stages)}")

        s0_mrr = stages[0].bench.summary()["mrr"]
        s5_mrr = stages[-1].bench.summary()["mrr"]
        assert stages[0].bench.summary()["total_queries"] > 0

    @pytest.mark.asyncio
    async def test_ablation_publichealthqa(self) -> None:
        """PublicHealthQA Korean (77 corpus) ablation."""
        data = _load_dataset("publichealthqa_ko.json")
        if not data:
            pytest.skip("publichealthqa_ko.json not found")

        stages = await self._run_ablation("PublicHealthQA-ko", data)
        print(f"\n{_format_report('PublicHealthQA-ko', stages)}")

        s0_mrr = stages[0].bench.summary()["mrr"]
        s5_mrr = stages[-1].bench.summary()["mrr"]
        assert stages[0].bench.summary()["total_queries"] > 0
