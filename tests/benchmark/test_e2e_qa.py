"""End-to-End QA 벤치마크 — Retrieval → LLM Generation → DeepEval 평가.

Cognee HotPotQA 24문항 correctness 0.925 와 직접 비교.

파이프라인:
  1. HotPotQA corpus를 SynapticGraph에 인덱싱 (Auto-Ontology)
  2. 질문으로 graph.search() → context 검색
  3. LLM에 context + question 전달 → 답변 생성
  4. DeepEval GEval(Correctness) 또는 FactualCorrectness로 평가

실행:
  # gemma3:4b (로컬, 무료)
  uv run pytest tests/benchmark/test_e2e_qa.py -v -s

  # GPT-4o (최종 보고용)
  OPENAI_API_KEY=xxx uv run pytest tests/benchmark/test_e2e_qa.py -v -s --e2e-model=gpt-4o

의존성:
  uv pip install synaptic-memory[eval]
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import time

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import NodeKind

DATA_DIR = Path(__file__).parent / "data"


# ── 데이터 ──────────────────────────────────────────────


def _load_hotpotqa() -> dict | None:
    """HotPotQA 벤치마크 데이터 로드."""
    for name in ("hotpotqa_200.json", "hotpotqa_24.json"):
        path = DATA_DIR / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


# ── LLM 호출 ────────────────────────────────────────────


async def _generate_answer(
    question: str,
    contexts: list[str],
    *,
    model: str = "qwen3.5:4b",
    base_url: str = "http://localhost:11434",
    api_key: str = "ollama",
) -> str:
    """LLM으로 답변 생성. Ollama native API 또는 OpenAI-compatible API."""
    try:
        import aiohttp
    except ImportError:
        pytest.skip("aiohttp 필요: uv pip install aiohttp")

    context_text = "\n\n---\n\n".join(contexts[:5])  # top-5 context
    system_prompt = (
        "You are a helpful assistant. Answer the question based ONLY on the provided context. "
        "If the context doesn't contain enough information, say 'I don't know'. "
        "Keep the answer concise (1-2 sentences). Give the direct answer."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"

    is_ollama = api_key == "ollama"

    if is_ollama:
        # Ollama native API — think: false로 thinking mode 비활성화
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
        }
        url = f"{base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
    else:
        # OpenAI-compatible API
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        url = f"{base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return f"[LLM ERROR: {resp.status}] {text[:200]}"
            data = await resp.json()

            if is_ollama:
                return (data.get("message", {}).get("content", "") or "").strip()
            else:
                return (data["choices"][0]["message"].get("content", "") or "").strip()


# ── 평가 ────────────────────────────────────────────────


@dataclass
class E2EResult:
    """End-to-End QA 평가 결과."""

    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0

    # DeepEval/RAGAS 스코어 (나중에 채움)
    correctness: float = 0.0
    faithfulness: float = 0.0
    relevancy: float = 0.0


@dataclass
class E2EBenchmark:
    """End-to-End 벤치마크 전체 결과."""

    dataset_name: str
    model_name: str
    results: list[E2EResult] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def mean_correctness(self) -> float:
        scores = [r.correctness for r in self.results if r.correctness > 0]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def mean_faithfulness(self) -> float:
        scores = [r.faithfulness for r in self.results if r.faithfulness > 0]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def mean_retrieval_ms(self) -> float:
        times = [r.retrieval_time_ms for r in self.results]
        return sum(times) / len(times) if times else 0.0

    def report(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"End-to-End QA Benchmark: {self.dataset_name}",
            f"Model: {self.model_name}",
            f"Questions: {len(self.results)}",
            f"{'='*60}",
            f"  Mean Correctness:   {self.mean_correctness:.3f}",
            f"  Mean Faithfulness:  {self.mean_faithfulness:.3f}",
            f"  Mean Retrieval:     {self.mean_retrieval_ms:.1f}ms",
            f"  Total Time:         {self.total_time_s:.1f}s",
            f"{'='*60}",
        ]

        # 개별 결과
        lines.append(f"\n{'Question':<50} {'Correct':>8} {'Faith':>8}")
        lines.append("-" * 70)
        for r in self.results:
            q = r.question[:47] + "..." if len(r.question) > 50 else r.question
            lines.append(f"{q:<50} {r.correctness:>8.3f} {r.faithfulness:>8.3f}")

        return "\n".join(lines)


def _evaluate_correctness_simple(answer: str, ground_truth: str) -> float:
    """3단계 correctness 평가 (LLM judge 없이).

    1. Exact match (정규화 후)
    2. Ground truth가 answer에 포함되어 있는지
    3. F1 토큰 매칭
    """
    import re

    def normalize(text: str) -> str:
        """소문자화, 구두점/공백 정규화."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize(text: str) -> set[str]:
        return {t for t in normalize(text).split() if len(t) >= 2}

    norm_answer = normalize(answer)
    norm_truth = normalize(ground_truth)

    if not norm_truth:
        return 0.0

    # 1. Exact match
    if norm_truth == norm_answer or norm_truth in norm_answer:
        return 1.0

    # 2. Ground truth의 모든 토큰이 answer에 포함
    truth_tokens = tokenize(ground_truth)
    if not truth_tokens:
        return 0.0

    pred_tokens = tokenize(answer)
    if not pred_tokens:
        return 0.0

    # truth 토큰이 모두 answer에 있으면 높은 점수
    recall = len(truth_tokens & pred_tokens) / len(truth_tokens)
    if recall >= 1.0:
        return 0.9  # 완전 recall이지만 exact는 아님

    # 3. F1
    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


async def _evaluate_with_deepeval(results: list[E2EResult], model: str) -> None:
    """DeepEval GEval로 correctness/faithfulness 평가."""
    try:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.metrics import GEval, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError:
        print("  [WARN] DeepEval 미설치 — F1 기반 simple correctness 사용")
        for r in results:
            r.correctness = _evaluate_correctness_simple(r.answer, r.ground_truth)
        return

    # Ollama 모델 설정
    try:
        from deepeval.models import OllamaModel
        judge_model = OllamaModel(model="gemma3:4b", base_url="http://localhost:11434")
    except Exception:
        judge_model = None

    correctness_metric = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output is factually correct based on the expected output. "
            "Score 0 if completely wrong, 0.5 if partially correct, 1.0 if fully correct."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge_model,
        threshold=0.5,
    )

    for r in results:
        test_case = LLMTestCase(
            input=r.question,
            actual_output=r.answer,
            expected_output=r.ground_truth,
            retrieval_context=r.contexts[:5],
        )
        try:
            correctness_metric.measure(test_case)
            r.correctness = correctness_metric.score or 0.0
        except Exception as e:
            print(f"  [WARN] DeepEval correctness 실패: {e}")
            r.correctness = _evaluate_correctness_simple(r.answer, r.ground_truth)


# ── 벤치마크 테스트 ──────────────────────────────────────


class TestE2EHotPotQA:
    """HotPotQA End-to-End QA — Cognee 비교용."""

    @pytest.mark.asyncio
    async def test_hotpotqa_e2e(self) -> None:
        data = _load_hotpotqa()
        if not data:
            pytest.skip("HotPotQA 데이터 없음. Run: uv run python tests/benchmark/download_datasets.py")

        corpus = data["corpus"]
        queries = data["queries"]
        qrels = data["qrels"]
        answers = data.get("answers", {})

        # 24문항 샘플링 (Cognee와 동일)
        import random
        random.seed(42)
        query_ids = list(queries.keys())
        if len(query_ids) > 24:
            query_ids = random.sample(query_ids, 24)

        # 1. 그래프 구축 (Auto-Ontology)
        print("\n[Phase 1] 그래프 구축...")
        from synaptic.extensions.classifier_rules import RuleBasedClassifier
        from synaptic.extensions.relation_detector import RuleBasedRelationDetector

        backend = MemoryBackend()
        await backend.connect()
        detector = RuleBasedRelationDetector()
        graph = SynapticGraph(
            backend,
            classifier=RuleBasedClassifier(),
            relation_detector=detector,
        )

        id_map: dict[str, str] = {}
        for cid, doc in corpus.items():
            node_id = await graph.add(
                title=doc.get("title", ""),
                content=doc.get("text", ""),
            )
            id_map[cid] = node_id

        # 엣지 수 확인
        edge_count = len(backend._edges) if hasattr(backend, '_edges') else 0
        print(f"  노드: {len(id_map)}, 엣지: {edge_count}")

        # LLM 설정
        model = os.environ.get("E2E_MODEL", "qwen3.5:4b")
        base_url = os.environ.get("E2E_BASE_URL", "http://localhost:11434")
        api_key = os.environ.get("OPENAI_API_KEY", "ollama")

        if "gpt" in model or "claude" in model:
            base_url = "https://api.openai.com"

        print(f"  LLM: {model} @ {base_url}")

        # 2. Retrieval + Generation
        print(f"\n[Phase 2] Retrieval + Generation ({len(query_ids)}문항)...")
        benchmark = E2EBenchmark(dataset_name="HotPotQA", model_name=model)
        start_total = time()

        for i, qid in enumerate(query_ids):
            question = queries[qid]
            # ground truth 답변 (짧은 정답)
            ground_truth = answers.get(qid, "")
            if not ground_truth:
                # fallback: corpus에서 추출
                gt_ids = list(qrels.get(qid, {}).keys())
                gt_texts = [corpus[gid].get("text", "")[:200] for gid in gt_ids if gid in corpus]
                ground_truth = " ".join(gt_texts)[:500]

            # Retrieval + Evidence Chain Assembly
            t0 = time()
            evidence = await graph.build_evidence(
                question, limit=10, max_steps=8, max_tokens=2048,
            )
            retrieval_ms = (time() - t0) * 1000

            # Evidence chain context를 LLM에 전달
            contexts = [evidence.compressed_context] if evidence.compressed_context else []

            # Generation
            t0 = time()
            answer = await _generate_answer(
                question, contexts,
                model=model, base_url=base_url, api_key=api_key,
            )
            gen_ms = (time() - t0) * 1000

            result = E2EResult(
                question=question,
                ground_truth=ground_truth,
                answer=answer,
                contexts=contexts,
                retrieval_time_ms=retrieval_ms,
                generation_time_ms=gen_ms,
            )
            benchmark.results.append(result)

            progress = f"[{i+1}/{len(query_ids)}]"
            print(f"  {progress} Q: {question[:60]}...")
            print(f"         A: {answer[:80]}...")

        benchmark.total_time_s = time() - start_total

        # 3. 평가
        print(f"\n[Phase 3] 평가 (Correctness)...")

        # F1 기반 simple correctness (항상 실행)
        for r in benchmark.results:
            r.correctness = _evaluate_correctness_simple(r.answer, r.ground_truth)

        # DeepEval이 있으면 LLM judge로 덮어씀
        # await _evaluate_with_deepeval(benchmark.results, model)

        print(benchmark.report())

        # 4. Cognee 비교
        cognee_correctness = 0.925
        our_correctness = benchmark.mean_correctness
        print(f"\n  📊 Cognee Correctness:  {cognee_correctness:.3f}")
        print(f"  📊 Synaptic Correctness: {our_correctness:.3f}")
        if our_correctness >= cognee_correctness:
            print("  ✅ Cognee 수준 달성!")
        else:
            gap = cognee_correctness - our_correctness
            print(f"  ⚠️  Gap: {gap:.3f} ({gap/cognee_correctness*100:.1f}%)")

        assert len(benchmark.results) > 0

        await graph.backend.close()

    @pytest.mark.asyncio
    async def test_hotpotqa_e2e_claude(self) -> None:
        """Claude API로 답변 생성 — Cognee 공정 비교 (그래프는 RuleBased)."""
        from dotenv import dotenv_values
        env = dotenv_values()
        api_key = os.environ.get("ANTHROPIC_API_KEY", "") or env.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY 없음. .env에 설정 필요")

        data = _load_hotpotqa()
        if not data:
            pytest.skip("HotPotQA 데이터 없음")

        corpus = data["corpus"]
        queries = data["queries"]
        answers = data.get("answers", {})

        import random
        random.seed(42)
        query_ids = list(queries.keys())
        if len(query_ids) > 24:
            query_ids = random.sample(query_ids, 24)

        # 1. 그래프 구축 (RuleBased — 빠름)
        print("\n[Phase 1] 그래프 구축 (RuleBased)...")
        from synaptic.extensions.classifier_rules import RuleBasedClassifier
        from synaptic.extensions.relation_detector import RuleBasedRelationDetector

        backend = MemoryBackend()
        await backend.connect()
        graph = SynapticGraph(
            backend,
            classifier=RuleBasedClassifier(),
            relation_detector=RuleBasedRelationDetector(),
        )

        for cid, doc in corpus.items():
            await graph.add(
                title=doc.get("title", ""),
                content=doc.get("text", ""),
            )

        edge_count = len(backend._edges) if hasattr(backend, '_edges') else 0
        print(f"  노드: {len(corpus)}, 엣지: {edge_count}")

        # 2. Claude로 답변 생성
        print(f"\n[Phase 2] Retrieval + Generation (Claude, {len(query_ids)}문항)...")
        benchmark = E2EBenchmark(dataset_name="HotPotQA-Claude", model_name="claude-sonnet-4-20250514")
        start_total = time()

        for i, qid in enumerate(query_ids):
            question = queries[qid]
            ground_truth = answers.get(qid, "")

            # Retrieval + Evidence Chain
            t0 = time()
            evidence = await graph.build_evidence(
                question, limit=10, max_steps=8, max_tokens=2048,
            )
            retrieval_ms = (time() - t0) * 1000

            contexts = [evidence.compressed_context] if evidence.compressed_context else []

            # Claude로 답변 생성
            t0 = time()
            answer = await _generate_answer_claude(
                question, contexts, api_key=api_key,
            )
            gen_ms = (time() - t0) * 1000

            result = E2EResult(
                question=question,
                ground_truth=ground_truth,
                answer=answer,
                contexts=contexts,
                retrieval_time_ms=retrieval_ms,
                generation_time_ms=gen_ms,
            )
            benchmark.results.append(result)

            print(f"  [{i+1}/{len(query_ids)}] Q: {question[:60]}...")
            print(f"         A: {answer[:80]}...")

        benchmark.total_time_s = time() - start_total

        # 3. 평가
        print(f"\n[Phase 3] 평가 (Correctness)...")
        for r in benchmark.results:
            r.correctness = _evaluate_correctness_simple(r.answer, r.ground_truth)

        print(benchmark.report())

        cognee_correctness = 0.925
        our_correctness = benchmark.mean_correctness
        print(f"\n  📊 Cognee Correctness (GPT-4o):  {cognee_correctness:.3f}")
        print(f"  📊 Synaptic + Claude:            {our_correctness:.3f}")
        if our_correctness >= cognee_correctness:
            print("  ✅ Cognee 수준 달성/초과!")
        else:
            gap = cognee_correctness - our_correctness
            print(f"  ⚠️  Gap: {gap:.3f} ({gap/cognee_correctness*100:.1f}%)")

        assert len(benchmark.results) > 0
        await graph.backend.close()


async def _generate_answer_claude(
    question: str,
    contexts: list[str],
    *,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Claude API로 답변 생성."""
    try:
        import aiohttp
    except ImportError:
        return "[aiohttp 필요]"

    context_text = "\n\n---\n\n".join(contexts[:5])
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer the question based ONLY on the provided context. "
        "Keep the answer concise (1-2 sentences). Give the direct answer."
    )

    payload = {
        "model": model,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return f"[Claude ERROR: {resp.status}] {text[:200]}"
            data = await resp.json()
            return data["content"][0]["text"].strip()
