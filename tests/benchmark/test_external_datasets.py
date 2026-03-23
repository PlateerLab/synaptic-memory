"""외부 데이터셋 벤치마크 — HuggingFace IR/QA 데이터셋으로 검색 품질 평가.

데이터셋 (14종):
  한국어: Ko-StrategyQA, AutoRAGRetrieval, KLUE-MRC, Allganize (2종),
          PublicHealthQA, MIRACLRetrieval, MultiLongDocRetrieval, XPQARetrieval
  영어: HotPotQA (24/200), NFCorpus, SciFact, FiQA

각 데이터셋별:
  1. corpus를 SynapticGraph에 인덱싱
  2. 쿼리로 검색
  3. qrels 기준으로 MRR, nDCG, P@K, R@K 평가
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from time import time

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import NodeKind

from .metrics import BenchmarkResult

DATA_DIR = Path(__file__).parent / "data"
K = 10  # 평가 기준


def _load_dataset(filename: str) -> dict | None:
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


async def _build_graph(
    corpus: dict[str, dict[str, str]],
    *,
    max_docs: int = 0,
) -> tuple[SynapticGraph, dict[str, str]]:
    """corpus를 SynapticGraph에 인덱싱. FTS only (embedding은 모델 품질에 의존)."""
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend)
    id_map: dict[str, str] = {}

    items = list(corpus.items())
    if max_docs > 0 and len(items) > max_docs:
        items = items[:max_docs]

    for corpus_id, doc in items:
        title = doc.get("title", "")
        text = doc.get("text", "")
        if not text:
            continue
        # 긴 문서 잘라내기 (메모리 + 성능)
        if len(text) > 2000:
            text = text[:2000]
        node = await graph.add(
            title=title or text[:80],
            content=text,
            kind=NodeKind.CONCEPT,
            source=f"benchmark",
        )
        id_map[corpus_id] = node.id

    return graph, id_map


async def _run_benchmark(
    dataset_name: str,
    graph: SynapticGraph,
    id_map: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    max_queries: int = 0,
) -> BenchmarkResult:
    """쿼리 실행 + 평가 지표 계산."""
    bench = BenchmarkResult()

    # qrels가 있는 쿼리만 대상으로 샘플링
    query_items = [(qid, qt) for qid, qt in queries.items() if qid in qrels]
    if max_queries > 0 and len(query_items) > max_queries:
        random.seed(42)
        query_items = random.sample(query_items, max_queries)

    for qid, query_text in query_items:
        relevant_corpus_ids = set(qrels.get(qid, {}).keys())
        relevant_node_ids = {id_map[cid] for cid in relevant_corpus_ids if cid in id_map}

        if not relevant_node_ids:
            continue  # ground truth가 인덱싱 안 된 경우 skip

        start = time()
        result = await graph.search(query_text, limit=K * 2)
        elapsed = (time() - start) * 1000

        retrieved = [n.node.id for n in result.nodes]

        bench.add(
            query_id=qid,
            query=query_text,
            retrieved=retrieved,
            relevant=relevant_node_ids,
            k=K,
            description=dataset_name,
            search_time_ms=elapsed,
        )

    return bench


class TestKoStrategyQA:
    """Ko-StrategyQA 벤치마크 — MTEB 공식 한국어 retrieval task."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("ko_strategyqa.json")
        if not data:
            pytest.skip("ko_strategyqa.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        # corpus 전체 인덱싱 (9.2K)
        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "Ko-StrategyQA",
            graph, id_map,
            data["queries"], data["qrels"],
            max_queries=100,  # 시간 절약을 위해 100개 샘플
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        # baseline 기록용 — 현재 수준 확인
        assert s["total_queries"] > 0, "평가 가능한 쿼리가 없음"
        print(f"\n[Ko-StrategyQA] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()


class TestAutoRAGRetrieval:
    """AutoRAGRetrieval 벤치마크 — 엔터프라이즈 5개 도메인."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("autorag_retrieval.json")
        if not data:
            pytest.skip("autorag_retrieval.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        # corpus 전체 인덱싱 (720)
        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "AutoRAGRetrieval",
            graph, id_map,
            data["queries"], data["qrels"],
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0, "평가 가능한 쿼리가 없음"
        print(f"\n[AutoRAGRetrieval] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()


class TestKlueMRC:
    """KLUE-MRC 벤치마크 — QA→IR 변환, 샘플링."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("klue_mrc.json")
        if not data:
            pytest.skip("klue_mrc.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        # 전체 5.8K는 느리므로 500개 샘플링
        corpus_items = list(data["corpus"].items())
        random.seed(42)
        sampled_ids = set(k for k, _ in random.sample(corpus_items, min(500, len(corpus_items))))

        sampled_corpus = {k: v for k, v in data["corpus"].items() if k in sampled_ids}
        sampled_queries = {k: v for k, v in data["queries"].items() if k.replace("klue_", "klue_doc_") in sampled_ids}
        sampled_qrels = {k: v for k, v in data["qrels"].items() if k in sampled_queries}

        graph, id_map = await _build_graph(sampled_corpus)

        bench = await _run_benchmark(
            "KLUE-MRC",
            graph, id_map,
            sampled_queries, sampled_qrels,
            max_queries=100,
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0, "평가 가능한 쿼리가 없음"
        print(f"\n[KLUE-MRC] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()


class TestAllganizeRAGEval:
    """Allganize RAG-Evaluation-Dataset-KO — 엔터프라이즈 5개 도메인 RAG 평가."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("allganize_rag_eval.json")
        if not data:
            pytest.skip("allganize_rag_eval.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "Allganize-RAG-Eval",
            graph, id_map,
            data["queries"], data["qrels"],
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0
        print(f"\n[Allganize-RAG-Eval] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()


class TestAllganizeRagKo:
    """Allganize rag-ko — Golden/Negative context RAG 평가."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("allganize_rag_ko.json")
        if not data:
            pytest.skip("allganize_rag_ko.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "Allganize-rag-ko",
            graph, id_map,
            data["queries"], data["qrels"],
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0
        print(f"\n[Allganize-rag-ko] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()


class TestHotPotQA:
    """HotPotQA 벤치마크 — multi-hop QA retrieval (영어, Cognee 비교용).

    두 가지 서브셋:
      - 24 queries: Cognee와 동일 규모 비교
      - 200 queries: 더 큰 규모 평가

    HotPotQA 특성:
      - 각 question에 relevant 문서 2개 (multi-hop)
      - corpus에 distractor 문서 포함 → 검색 난이도 높음
    """

    @pytest.mark.asyncio
    async def test_benchmark_24(self) -> None:
        """Cognee 비교용 24-question 서브셋."""
        data = _load_dataset("hotpotqa_24.json")
        if not data:
            pytest.skip("hotpotqa_24.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "HotPotQA-24",
            graph, id_map,
            data["queries"], data["qrels"],
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0, "평가 가능한 쿼리가 없음"

        # Supporting Facts Retrieval 분석
        sf_hits = 0
        sf_total = 0
        for q in bench.queries:
            relevant_count = len(q["relevant"])
            retrieved_set = set(q["retrieved_top_k"])
            relevant_set = set(q["relevant"])
            sf_total += relevant_count
            sf_hits += len(retrieved_set & relevant_set)

        sf_recall = sf_hits / sf_total if sf_total > 0 else 0.0

        print(f"\n[HotPotQA-24 — Cognee Comparison]")
        print(f"  MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")
        print(f"  Supporting Facts Recall@{K}={sf_recall:.3f} "
              f"({sf_hits}/{sf_total} supporting docs retrieved)")
        if data.get("metadata", {}).get("cognee_human_correctness"):
            print(f"  [Reference] Cognee Human-like Correctness: "
                  f"{data['metadata']['cognee_human_correctness']}")

        await graph.backend.close()

    @pytest.mark.asyncio
    async def test_benchmark_200(self) -> None:
        """200-question 서브셋 (더 큰 규모 평가)."""
        data = _load_dataset("hotpotqa.json")
        if not data:
            pytest.skip("hotpotqa.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "HotPotQA-200",
            graph, id_map,
            data["queries"], data["qrels"],
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0, "평가 가능한 쿼리가 없음"

        # Supporting Facts Retrieval 분석
        sf_hits = 0
        sf_total = 0
        for q in bench.queries:
            relevant_count = len(q["relevant"])
            retrieved_set = set(q["retrieved_top_k"])
            relevant_set = set(q["relevant"])
            sf_total += relevant_count
            sf_hits += len(retrieved_set & relevant_set)

        sf_recall = sf_hits / sf_total if sf_total > 0 else 0.0

        print(f"\n[HotPotQA-200]")
        print(f"  MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")
        print(f"  Supporting Facts Recall@{K}={sf_recall:.3f} "
              f"({sf_hits}/{sf_total} supporting docs retrieved)")

        await graph.backend.close()


class TestPublicHealthQA:
    """PublicHealthQA Korean — 의료/공중보건 도메인."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("publichealthqa_ko.json")
        if not data:
            pytest.skip("publichealthqa_ko.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "PublicHealthQA-ko",
            graph, id_map,
            data["queries"], data["qrels"],
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0
        print(f"\n[PublicHealthQA-ko] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()


# ── BeIR 영문 데이터셋 ──


class TestNFCorpus:
    """NFCorpus — 의료/영양 도메인 (BeIR, 영어)."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("nfcorpus.json")
        if not data:
            pytest.skip("nfcorpus.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "NFCorpus",
            graph, id_map,
            data["queries"], data["qrels"],
            max_queries=100,
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0
        print(f"\n[NFCorpus] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()


class TestSciFact:
    """SciFact — 과학적 주장 검증 (BeIR, 영어)."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("scifact.json")
        if not data:
            pytest.skip("scifact.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "SciFact",
            graph, id_map,
            data["queries"], data["qrels"],
            max_queries=100,
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0
        print(f"\n[SciFact] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()


class TestFiQA:
    """FiQA — 금융 QA (BeIR, 영어, 57K corpus)."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("fiqa.json")
        if not data:
            pytest.skip("fiqa.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        # 57K corpus — 전체 인덱싱 (FTS 스케일 테스트)
        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "FiQA",
            graph, id_map,
            data["queries"], data["qrels"],
            max_queries=100,
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0
        print(f"\n[FiQA] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()


# ── MTEB 한국어 데이터셋 ──


class TestMIRACLRetrieval:
    """MIRACLRetrieval Korean — MTEB 핵심 한국어 검색 (샘플링된 corpus)."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("miracl_retrieval_ko.json")
        if not data:
            pytest.skip("miracl_retrieval_ko.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "MIRACLRetrieval-ko",
            graph, id_map,
            data["queries"], data["qrels"],
            max_queries=100,
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0
        print(f"\n[MIRACLRetrieval-ko] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()


class TestMultiLongDocRetrieval:
    """MultiLongDocRetrieval Korean — 장문서 검색."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("multilongdoc_ko.json")
        if not data:
            pytest.skip("multilongdoc_ko.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "MultiLongDocRetrieval-ko",
            graph, id_map,
            data["queries"], data["qrels"],
            max_queries=100,
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0
        print(f"\n[MultiLongDocRetrieval-ko] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()


class TestXPQARetrieval:
    """XPQARetrieval Korean — 다도메인 한국어 검색."""

    @pytest.mark.asyncio
    async def test_benchmark(self) -> None:
        data = _load_dataset("xpqa_ko.json")
        if not data:
            pytest.skip("xpqa_ko.json not found. Run: uv run python tests/benchmark/download_datasets.py")

        graph, id_map = await _build_graph(data["corpus"])

        bench = await _run_benchmark(
            "XPQARetrieval-ko",
            graph, id_map,
            data["queries"], data["qrels"],
        )

        print(f"\n{bench.report(k=K)}")

        s = bench.summary()
        assert s["total_queries"] > 0
        print(f"\n[XPQARetrieval-ko] MRR={s['mrr']:.3f}, nDCG@{K}={s['mean_ndcg@k']:.3f}, "
              f"P@{K}={s['mean_precision@k']:.3f}, R@{K}={s['mean_recall@k']:.3f}")

        await graph.backend.close()
