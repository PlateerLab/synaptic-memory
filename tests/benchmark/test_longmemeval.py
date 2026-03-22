"""LongMemEval 벤치마크 — 장기 메모리 평가 (ICLR 2025).

500문항, 6유형: single-session-user, single-session-assistant,
single-session-preference, multi-session, temporal-reasoning, knowledge-update.

Phase 1 개선:
  - 다중 검색 (원본 쿼리 + 키워드 쿼리 + 날짜 쿼리)
  - 시간순 context 정렬 + recency boost
  - 유형별 특화 프롬프트

실행:
  wget -O tests/benchmark/data/longmemeval_s.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

  # 50문항 (기본)
  uv run pytest tests/benchmark/test_longmemeval.py -v -s

  # 외부 LLM 서버
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
from synaptic.graph import SynapticGraph
from synaptic.models import NodeKind

DATA_DIR = Path(__file__).parent / "data"
MAX_QUESTIONS = int(os.environ.get("LONGMEM_MAX_Q", "50"))

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
# 세션 인덱싱 (turn-pair 단위)
# ---------------------------------------------------------------------------


async def _index_sessions(
    graph: SynapticGraph,
    sessions: list[list[dict]],
    session_ids: list[str],
    session_dates: list[str],
) -> dict[str, list[str]]:
    """대화 세션들을 turn-pair 단위로 인덱싱."""
    id_map: dict[str, list[str]] = {}

    for session, sid, date in zip(sessions, session_ids, session_dates):
        node_ids: list[str] = []
        i = 0
        pair_idx = 0
        while i < len(session):
            turn = session[i]
            role = turn.get("role", "user")
            content = turn.get("content", "")

            if role == "user":
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
# Phase 1: 다중 검색 전략
# ---------------------------------------------------------------------------

# 시간 관련 키워드
_TEMPORAL_KEYWORDS = {
    "when", "date", "day", "days", "month", "months", "year", "years",
    "before", "after", "between", "ago", "last", "first", "recent",
    "how long", "how many days", "how many months",
}


def _is_temporal_question(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in _TEMPORAL_KEYWORDS)


def _is_counting_question(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in ["how many", "how much", "total", "count", "number of"])


def _extract_key_phrases(question: str) -> list[str]:
    """질문에서 핵심 구절 추출 — 다중 검색용."""
    # 따옴표 내 구절
    quoted = re.findall(r'"([^"]+)"', question)
    # 고유명사 (연속 대문자 단어)
    proper = re.findall(r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
    # stopword 제거한 핵심 단어
    stopwords = {
        "i", "me", "my", "we", "you", "the", "a", "an", "is", "are", "was", "were",
        "did", "do", "does", "have", "has", "had", "what", "which", "who", "how",
        "many", "much", "when", "where", "that", "this", "for", "with", "from",
        "about", "into", "during", "of", "in", "on", "at", "to", "and", "or",
    }
    words = [w for w in re.findall(r'\b\w+\b', question.lower()) if w not in stopwords and len(w) >= 3]

    # 핵심 구절 조합 (2-3 단어 조합)
    sub_queries = []
    if quoted:
        sub_queries.extend(quoted)
    if proper:
        sub_queries.extend(proper)
    if len(words) >= 4:
        # 긴 질문이면 키워드 조합으로 서브 쿼리
        sub_queries.append(" ".join(words[:3]))
        sub_queries.append(" ".join(words[-3:]))

    return sub_queries


async def _multi_search(
    graph: SynapticGraph,
    question: str,
    question_type: str,
    *,
    limit: int = 20,
) -> list[tuple[str, str, float]]:
    """다중 검색 — 원본 + 키워드 + 서브쿼리로 검색 후 합산.

    Returns: [(date, content, score), ...] 날짜순 정렬.
    """
    seen_ids: set[str] = set()
    results: list[tuple[str, str, str, float]] = []  # (node_id, date, content, score)

    async def _collect(query: str, boost: float = 1.0):
        try:
            sr = await graph.search(query, limit=limit)
        except Exception:
            return
        for n in sr.nodes:
            if n.node.id in seen_ids:
                continue
            seen_ids.add(n.node.id)
            date_tag = ""
            for tag in n.node.tags:
                if tag.startswith("date:"):
                    date_tag = tag.replace("date:", "")
            results.append((n.node.id, date_tag, n.node.content, n.activation * boost))

    # 1차: 원본 쿼리
    await _collect(question, boost=1.0)

    # 2차: 서브 쿼리 (핵심 구절)
    sub_queries = _extract_key_phrases(question)
    for sq in sub_queries[:3]:
        await _collect(sq, boost=0.8)

    # 3차: temporal 질문이면 날짜 관련 재검색
    if _is_temporal_question(question):
        # 날짜/시간 관련 키워드 강조 쿼리
        date_words = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{4}|\d{1,2}(?:st|nd|rd|th)?)\b', question, re.IGNORECASE)
        if date_words:
            await _collect(" ".join(date_words), boost=0.7)

    # 4차: counting 질문이면 관련 엔티티 재검색
    if _is_counting_question(question):
        # "How many X" → X로 검색
        match = re.search(r'how many (\w+(?:\s+\w+)?)', question, re.IGNORECASE)
        if match:
            await _collect(match.group(1), boost=0.7)

    # 날짜순 정렬 (knowledge-update에서 최신 우선)
    results.sort(key=lambda x: x[1])  # 날짜 오름차순

    return [(date, content, score) for _, date, content, score in results]


# ---------------------------------------------------------------------------
# 유형별 특화 프롬프트
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS = {
    "single-session-user": (
        "You are a personal assistant with perfect memory of past conversations. "
        "The user is asking about something THEY mentioned in a past conversation. "
        "Find the exact answer from the user's own words in the history. "
        "Be precise and concise. Give ONLY the direct answer."
    ),
    "single-session-assistant": (
        "You are a personal assistant with perfect memory. "
        "The user is asking about something YOU (the assistant) said in a past conversation. "
        "Find your exact response from the conversation history. "
        "Be precise and concise. Give ONLY the direct answer."
    ),
    "single-session-preference": (
        "You are a personal assistant who remembers the user's preferences and personal details. "
        "The user is asking you to use their preferences to help them. "
        "Find relevant personal information from past conversations and use it to give a helpful, personalized response. "
        "Be specific about the user's stated preferences."
    ),
    "multi-session": (
        "You are a personal assistant with perfect memory across ALL conversations. "
        "The user is asking a question that requires combining information from MULTIPLE past conversations. "
        "Carefully search through ALL provided conversation excerpts, gather relevant pieces, and combine them. "
        "If the question asks for a count or list, be thorough — check every excerpt. "
        "Give a precise, complete answer."
    ),
    "temporal-reasoning": (
        "You are a personal assistant who can reason about time and dates. "
        "Pay close attention to the [Date: ...] timestamps on each conversation. "
        "The user is asking about timing, duration, or the order of events. "
        "Calculate date differences carefully. Use the provided dates to answer precisely. "
        "If asked 'how many days', compute the exact difference between the relevant dates."
    ),
    "knowledge-update": (
        "You are a personal assistant with perfect memory. "
        "IMPORTANT: The user's information may have CHANGED over time. "
        "When the same topic appears in multiple conversations, ALWAYS use the MOST RECENT information (latest date). "
        "Earlier statements may be outdated. "
        "Give ONLY the current/latest answer."
    ),
}

_DEFAULT_SYSTEM = (
    "You are a helpful personal assistant with perfect memory. "
    "Answer the question based ONLY on the provided conversation history. "
    "Be concise and specific. Give the direct answer without explanation."
)


def _build_prompt(question: str, contexts: list[tuple[str, str, float]], question_type: str) -> tuple[str, str]:
    """유형별 특화 프롬프트 생성."""
    system = _SYSTEM_PROMPTS.get(question_type, _DEFAULT_SYSTEM)

    # knowledge-update: 최신 먼저 (역순)
    if question_type == "knowledge-update":
        contexts = list(reversed(contexts))

    # context 조립 (날짜 포함)
    context_parts = []
    for date, content, _score in contexts[:8]:
        if date:
            context_parts.append(f"[Conversation Date: {date}]\n{content}")
        else:
            context_parts.append(content)

    context_text = "\n\n---\n\n".join(context_parts)

    if question_type == "multi-session":
        user = (
            f"Below are excerpts from multiple past conversations. "
            f"Carefully review ALL of them to answer the question.\n\n"
            f"Conversation History:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Important: Check every excerpt and combine information across conversations.\n"
            f"Answer:"
        )
    elif question_type == "temporal-reasoning":
        user = (
            f"Below are past conversations with their dates. "
            f"Use the dates to reason about timing.\n\n"
            f"Conversation History:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Think step by step about the dates, then give a precise answer.\n"
            f"Answer:"
        )
    elif question_type == "knowledge-update":
        user = (
            f"Below are past conversations ordered from NEWEST to OLDEST. "
            f"If information conflicts, trust the NEWEST conversation.\n\n"
            f"Conversation History:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer (use the most recent information):"
        )
    else:
        user = f"Conversation History:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"

    return system, user


# ---------------------------------------------------------------------------
# LLM 답변 생성
# ---------------------------------------------------------------------------


async def _generate_answer(
    question: str,
    contexts: list[tuple[str, str, float]],
    question_type: str,
    *,
    model: str = LLM_MODEL,
    base_url: str = LLM_BASE,
) -> str:
    """LLM으로 답변 생성 — 유형별 특화 프롬프트 사용."""
    try:
        import aiohttp
    except ImportError:
        pytest.skip("aiohttp 필요: uv pip install aiohttp")

    system_prompt, user_prompt = _build_prompt(question, contexts, question_type)

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
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        url = f"{base_url}/chat/completions" if "/v1" in base_url else f"{base_url}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=300),
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
            f"LongMemEval Benchmark Results (Phase 1)",
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
        lines.append(f"\n  📊 Supermemory ASMR: 98.60% (8-variant ensemble)")
        lines.append(f"  📊 GPT-4o (full context): ~64%")
        lines.append(f"  📊 Synaptic Memory: {self.accuracy:.1%}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 벤치마크 테스트
# ---------------------------------------------------------------------------


class TestLongMemEval:
    """LongMemEval-S 벤치마크 — Phase 1 개선."""

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

        print(f"\n[LongMemEval-S Phase 1] {len(sampled)} questions from {len(data)}")
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
            graph = SynapticGraph(backend)
            id_map = await _index_sessions(graph, sessions, session_ids, session_dates)

            # 2. 다중 검색 (Phase 1)
            t0 = time()
            search_results = await _multi_search(graph, question, qtype, limit=20)
            retrieval_ms = (time() - t0) * 1000

            # 세션 recall
            retrieved_sids: list[str] = []
            for date, content, _ in search_results:
                # date tag에서 session id는 직접 매핑 불가하므로 별도 검색
                pass
            # 별도 검색으로 session recall 계산
            sr = await graph.search(question, limit=20)
            for n in sr.nodes:
                for tag in n.node.tags:
                    if tag.startswith("session:"):
                        sid = tag.replace("session:", "")
                        if sid not in retrieved_sids:
                            retrieved_sids.append(sid)

            session_recall = 0.0
            if answer_sids:
                hits = len(answer_sids & set(retrieved_sids))
                session_recall = hits / len(answer_sids)

            # 3. LLM 답변 생성 (유형별 특화 프롬프트)
            t0 = time()
            answer = await _generate_answer(
                question, search_results, qtype,
                model=LLM_MODEL, base_url=LLM_BASE,
            )
            gen_ms = (time() - t0) * 1000

            # 4. 평가
            correctness = _evaluate_correctness(answer, ground_truth)

            benchmark.results.append(LongMemResult(
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
            ))

            status = "✓" if correctness >= 0.5 else "✗"
            print(f"  [{i+1}/{len(sampled)}] {status} [{qtype[:15]}] {question[:50]}...")
            print(f"    GT: {ground_truth[:60]} | A: {answer[:60]}")

            await graph.backend.close()

        print(benchmark.report())
        assert len(benchmark.results) > 0
