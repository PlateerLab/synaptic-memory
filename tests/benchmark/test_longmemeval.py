"""LongMemEval 벤치마크 — 장기 메모리 평가 (ICLR 2025).

500문항, 6유형: single-session-user, single-session-assistant,
single-session-preference, multi-session, temporal-reasoning, knowledge-update.

실행:
  # 데이터셋 다운로드
  wget -O tests/benchmark/data/longmemeval_s.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

  # 벤치마크 실행 (Ollama qwen3.5:4b)
  uv run pytest tests/benchmark/test_longmemeval.py -v -s

  # 외부 LLM 서버 사용
  LONGMEM_LLM_BASE=http://118.223.251.22:10051/v1 \
  LONGMEM_LLM_MODEL=Qwen3.5-27B-BF16-00001-of-00002.gguf \
  uv run pytest tests/benchmark/test_longmemeval.py -v -s
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from time import time

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.classifier_rules import RuleBasedClassifier
from synaptic.extensions.relation_detector import RuleBasedRelationDetector
from synaptic.graph import SynapticGraph
from synaptic.models import NodeKind

DATA_DIR = Path(__file__).parent / "data"
MAX_QUESTIONS = int(os.environ.get("LONGMEM_MAX_Q", "50"))

# LLM 설정
LLM_BASE = os.environ.get("LONGMEM_LLM_BASE", "http://localhost:11434")
LLM_MODEL = os.environ.get("LONGMEM_LLM_MODEL", "qwen3.5:4b")


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------


def _load_longmemeval() -> list[dict] | None:
    path = DATA_DIR / "longmemeval_s.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 세션 → 그래프 인덱싱
# ---------------------------------------------------------------------------


async def _index_sessions(
    graph: SynapticGraph,
    sessions: list[list[dict]],
    session_ids: list[str],
    session_dates: list[str],
) -> dict[str, list[str]]:
    """대화 세션들을 SynapticGraph에 turn-pair 단위로 인덱싱.

    각 user→assistant 쌍을 하나의 노드로 만들어 세밀한 검색 가능.
    """
    id_map: dict[str, list[str]] = {}  # session_id → [node_ids]

    for session, sid, date in zip(sessions, session_ids, session_dates):
        node_ids: list[str] = []

        # user→assistant 쌍으로 묶기
        i = 0
        pair_idx = 0
        while i < len(session):
            turn = session[i]
            role = turn.get("role", "user")
            content = turn.get("content", "")

            if role == "user":
                # user + 다음 assistant 응답을 페어링
                user_text = content
                asst_text = ""
                if i + 1 < len(session) and session[i + 1].get("role") == "assistant":
                    asst_text = session[i + 1].get("content", "")
                    i += 2
                else:
                    i += 1

                pair_text = f"[User] {user_text}"
                if asst_text:
                    pair_text += f"\n[Assistant] {asst_text}"

                # 너무 긴 turn은 잘라내기
                if len(pair_text) > 2000:
                    pair_text = pair_text[:2000]

                title = user_text[:80]
                node = await graph.add(
                    title=title,
                    content=pair_text,
                    kind=NodeKind.CONCEPT,
                    tags=[f"session:{sid}", f"date:{date}", f"turn:{pair_idx}"],
                    source=f"longmemeval:{sid}:{pair_idx}",
                )
                node_ids.append(node.id)
                pair_idx += 1
            else:
                # assistant-only turn (세션 시작이 assistant인 경우)
                if content.strip():
                    node = await graph.add(
                        title=content[:80],
                        content=f"[Assistant] {content[:2000]}",
                        kind=NodeKind.CONCEPT,
                        tags=[f"session:{sid}", f"date:{date}", f"turn:{pair_idx}"],
                        source=f"longmemeval:{sid}:{pair_idx}",
                    )
                    node_ids.append(node.id)
                    pair_idx += 1
                i += 1

        id_map[sid] = node_ids

    return id_map


# ---------------------------------------------------------------------------
# LLM 답변 생성
# ---------------------------------------------------------------------------


async def _generate_answer(
    question: str,
    contexts: list[str],
    *,
    model: str = LLM_MODEL,
    base_url: str = LLM_BASE,
) -> str:
    """LLM으로 답변 생성."""
    try:
        import aiohttp
    except ImportError:
        pytest.skip("aiohttp 필요: uv pip install aiohttp")

    context_text = "\n\n---\n\n".join(contexts[:10])
    system_prompt = (
        "You are a helpful personal assistant with perfect memory. "
        "Answer the question based ONLY on the provided conversation history. "
        "Be concise and specific. If the information is not available, say 'I don't know'. "
        "Give the direct answer without explanation."
    )
    user_prompt = f"Conversation History:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"

    is_ollama = "11434" in base_url and "/v1" not in base_url

    if is_ollama:
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
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        url = f"{base_url}/chat/completions" if "/v1" in base_url else f"{base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return f"[LLM ERROR: {resp.status}] {text[:200]}"
            data = await resp.json()

            if is_ollama:
                return (data.get("message", {}).get("content", "") or "").strip()
            else:
                return (data["choices"][0]["message"].get("content", "") or "").strip()


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------


def _evaluate_correctness(answer: str, ground_truth: str | int | float) -> float:
    """F1 기반 correctness 평가."""
    answer = str(answer)
    ground_truth = str(ground_truth)

    def normalize(text: str) -> str:
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

    # Exact match
    if norm_truth == norm_answer or norm_truth in norm_answer:
        return 1.0

    truth_tokens = tokenize(ground_truth)
    pred_tokens = tokenize(answer)

    if not truth_tokens or not pred_tokens:
        return 0.0

    # Full recall
    recall = len(truth_tokens & pred_tokens) / len(truth_tokens)
    if recall >= 1.0:
        return 0.9

    # F1
    common = pred_tokens & truth_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


@dataclass
class LongMemResult:
    question_id: str
    question_type: str
    question: str
    ground_truth: str
    answer: str
    correctness: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    # retrieval metrics
    retrieved_session_ids: list[str] = field(default_factory=list)
    answer_session_ids: list[str] = field(default_factory=list)
    session_recall: float = 0.0


@dataclass
class LongMemBenchmark:
    results: list[LongMemResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correctness >= 0.5) / len(self.results)

    @property
    def mean_correctness(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.correctness for r in self.results) / len(self.results)

    @property
    def mean_session_recall(self) -> float:
        valid = [r for r in self.results if r.answer_session_ids]
        if not valid:
            return 0.0
        return sum(r.session_recall for r in valid) / len(valid)

    def by_type(self) -> dict[str, dict]:
        types: dict[str, list[LongMemResult]] = {}
        for r in self.results:
            types.setdefault(r.question_type, []).append(r)

        result = {}
        for qtype, items in sorted(types.items()):
            acc = sum(1 for r in items if r.correctness >= 0.5) / len(items)
            mean_c = sum(r.correctness for r in items) / len(items)
            result[qtype] = {"count": len(items), "accuracy": acc, "mean_correctness": mean_c}
        return result

    def report(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"LongMemEval Benchmark Results",
            f"{'='*70}",
            f"  Total Questions:    {len(self.results)}",
            f"  Accuracy (≥0.5):    {self.accuracy:.3f}",
            f"  Mean Correctness:   {self.mean_correctness:.3f}",
            f"  Mean Session Recall:{self.mean_session_recall:.3f}",
            f"{'='*70}",
            f"  {'Type':<30} {'Count':>5} {'Acc':>7} {'Correct':>8}",
            f"  {'-'*55}",
        ]
        for qtype, stats in self.by_type().items():
            lines.append(
                f"  {qtype:<30} {stats['count']:>5} {stats['accuracy']:>7.3f} {stats['mean_correctness']:>8.3f}"
            )
        lines.append(f"{'='*70}")

        # Comparison
        lines.append(f"\n  📊 Supermemory ASMR: 98.60% (8-variant ensemble)")
        lines.append(f"  📊 GPT-4o (full context): ~64%")
        lines.append(f"  📊 Synaptic Memory: {self.accuracy:.1%}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 벤치마크 테스트
# ---------------------------------------------------------------------------


class TestLongMemEval:
    """LongMemEval-S 벤치마크 — 장기 메모리 평가."""

    @pytest.mark.asyncio
    async def test_longmemeval_s(self) -> None:
        data = _load_longmemeval()
        if not data:
            pytest.skip(
                "longmemeval_s.json not found. Download: "
                "wget -O tests/benchmark/data/longmemeval_s.json "
                "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
            )

        import random
        random.seed(42)

        # 유형별 균등 샘플링
        by_type: dict[str, list[dict]] = {}
        for d in data:
            by_type.setdefault(d["question_type"], []).append(d)

        sampled: list[dict] = []
        per_type = max(1, MAX_QUESTIONS // len(by_type))
        for qtype, items in by_type.items():
            random.shuffle(items)
            sampled.extend(items[:per_type])

        if len(sampled) > MAX_QUESTIONS:
            sampled = sampled[:MAX_QUESTIONS]

        print(f"\n[LongMemEval-S] {len(sampled)} questions sampled from {len(data)}")
        print(f"  LLM: {LLM_MODEL} @ {LLM_BASE}")
        type_counts = Counter(d["question_type"] for d in sampled)
        for t, c in type_counts.most_common():
            print(f"    {t}: {c}")

        benchmark = LongMemBenchmark()

        for i, instance in enumerate(sampled):
            qid = instance["question_id"]
            qtype = instance["question_type"]
            question = instance["question"]
            ground_truth = str(instance["answer"])
            sessions = instance["haystack_sessions"]
            session_ids = instance["haystack_session_ids"]
            session_dates = instance["haystack_dates"]
            answer_sids = set(instance["answer_session_ids"])

            # 1. 세션 인덱싱
            backend = MemoryBackend()
            await backend.connect()
            # relation_detector 없이 순수 FTS — O(n²) 후보 검색 회피하여 속도 최적화
            graph = SynapticGraph(backend)

            id_map = await _index_sessions(graph, sessions, session_ids, session_dates)

            # 2. 검색 — turn-pair 단위로 검색하고 원본 context 전달
            t0 = time()
            result = await graph.search(question, limit=15)
            retrieval_ms = (time() - t0) * 1000

            # 검색 결과에서 context 구성 (날짜 포함)
            contexts = []
            retrieved_sids: list[str] = []
            for n in result.nodes:
                date_tag = ""
                for tag in n.node.tags:
                    if tag.startswith("session:"):
                        retrieved_sids.append(tag.replace("session:", ""))
                    if tag.startswith("date:"):
                        date_tag = tag.replace("date:", "")
                ctx = f"[Date: {date_tag}]\n{n.node.content}" if date_tag else n.node.content
                contexts.append(ctx)

            # 세션 recall 계산 (중복 제거)
            retrieved_sids = list(dict.fromkeys(retrieved_sids))  # 순서 유지 중복 제거
            if answer_sids:
                hits = len(answer_sids & set(retrieved_sids))
                session_recall = hits / len(answer_sids)
            else:
                session_recall = 0.0

            # 3. LLM 답변 생성
            t0 = time()
            answer = await _generate_answer(question, contexts, model=LLM_MODEL, base_url=LLM_BASE)
            gen_ms = (time() - t0) * 1000

            # 4. 평가
            correctness = _evaluate_correctness(answer, ground_truth)

            result_item = LongMemResult(
                question_id=qid,
                question_type=qtype,
                question=question,
                ground_truth=ground_truth,
                answer=answer,
                correctness=correctness,
                retrieval_time_ms=retrieval_ms,
                generation_time_ms=gen_ms,
                retrieved_session_ids=retrieved_sids,
                answer_session_ids=list(answer_sids),
                session_recall=session_recall,
            )
            benchmark.results.append(result_item)

            status = "✓" if correctness >= 0.5 else "✗"
            print(f"  [{i+1}/{len(sampled)}] {status} [{qtype[:15]}] {question[:50]}...")
            print(f"    GT: {ground_truth[:60]} | A: {answer[:60]}")

            await graph.backend.close()

        # 결과 출력
        print(benchmark.report())

        assert len(benchmark.results) > 0
