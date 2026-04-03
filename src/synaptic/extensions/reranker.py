"""LLM-based reranking — "이 질문에 이 정보가 답이 되는가?" 판단.

Only applied to top-N candidates (default 5) to minimize LLM cost.
Single LLM call with all candidates in one prompt.

Usage::

    reranker = LLMReranker(llm=OllamaLLMProvider(...), max_candidates=5)
    graph = SynapticGraph(backend, reranker=reranker)
    # search() automatically applies reranking at the end

Without LLM, use NoOpReranker (pass-through, default).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Protocol

from synaptic.models import ActivatedNode

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider

logger = logging.getLogger("reranker")

_RERANK_SYSTEM_PROMPT = """당신은 검색 결과의 관련성을 판단하는 전문가입니다.

사용자 질문과 후보 문서들이 주어집니다.
각 후보에 대해 질문에 대한 관련성 점수를 0-10으로 매기세요.

JSON 배열로 응답하세요:
[{"index": 0, "score": 8}, {"index": 1, "score": 3}, ...]

점수 기준:
- 9-10: 질문에 직접적으로 답하는 정보
- 6-8: 관련된 유용한 정보
- 3-5: 간접적으로 관련
- 0-2: 관련 없음"""


class Reranker(Protocol):
    """Reranks search candidates for final output."""

    async def rerank(
        self, query: str, candidates: list[ActivatedNode], *, top_k: int = 10
    ) -> list[ActivatedNode]: ...


class NoOpReranker:
    """Pass-through reranker — returns candidates unchanged."""

    async def rerank(
        self, query: str, candidates: list[ActivatedNode], *, top_k: int = 10
    ) -> list[ActivatedNode]:
        return candidates[:top_k]


class LLMReranker:
    """LLM-as-judge reranker.

    Sends top-N candidates to LLM in a single call.
    LLM scores each candidate 0-10 for relevance.
    Results re-sorted by LLM score.

    Example::

        reranker = LLMReranker(llm, max_candidates=5)
        reranked = await reranker.rerank("결제 장애 대응", candidates)
    """

    __slots__ = ("_llm", "_max_candidates")

    def __init__(self, llm: LLMProvider, *, max_candidates: int = 5) -> None:
        self._llm = llm
        self._max_candidates = max_candidates

    async def rerank(
        self, query: str, candidates: list[ActivatedNode], *, top_k: int = 10
    ) -> list[ActivatedNode]:
        if not candidates:
            return []

        # Only rerank top-N (LLM cost control)
        to_rerank = candidates[: self._max_candidates]
        remainder = candidates[self._max_candidates :]

        # Build candidate descriptions for LLM
        candidate_texts: list[str] = []
        for i, an in enumerate(to_rerank):
            title = an.node.title or "Untitled"
            content = an.node.content[:200] if an.node.content else ""
            candidate_texts.append(f"[{i}] {title}: {content}")

        user_prompt = (
            f"질문: {query}\n\n"
            f"후보 문서:\n" + "\n".join(candidate_texts)
        )

        try:
            response = await self._llm.generate(
                system=_RERANK_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=256,
            )

            scores = json.loads(response)
            if not isinstance(scores, list):
                return candidates[:top_k]

            # Map LLM scores to candidates
            score_map: dict[int, float] = {}
            for item in scores:
                if isinstance(item, dict) and "index" in item and "score" in item:
                    idx = int(item["index"])
                    score = float(item["score"]) / 10.0  # Normalize to 0-1
                    if 0 <= idx < len(to_rerank):
                        score_map[idx] = score

            # Create reranked list
            reranked: list[ActivatedNode] = []
            for i, an in enumerate(to_rerank):
                llm_score = score_map.get(i, an.resonance)
                # Blend: 60% LLM score + 40% original resonance
                blended = 0.6 * llm_score + 0.4 * an.resonance
                reranked.append(
                    ActivatedNode(
                        node=an.node,
                        activation=an.activation,
                        resonance=blended,
                        path=an.path,
                    )
                )

            # Sort by new score
            reranked.sort(key=lambda a: a.resonance, reverse=True)

            # Append remainder (not reranked)
            reranked.extend(remainder)

            return reranked[:top_k]

        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
            logger.warning(f"LLM reranking failed: {e}")
            return candidates[:top_k]
        except Exception as e:
            logger.warning(f"LLM reranking error: {e}")
            return candidates[:top_k]
