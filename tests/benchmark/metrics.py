"""IR 평가 지표 — MRR, nDCG, Precision@K, Recall@K."""

from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """상위 K개 중 관련 결과 비율."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """전체 관련 결과 중 상위 K개 안에 잡힌 비율."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """첫 번째 관련 결과의 역순위. MRR 계산용."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain @ K."""
    total = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = 1.0 if doc_id in relevant else 0.0
        total += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    return total


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized DCG @ K."""
    actual = dcg_at_k(retrieved, relevant, k)
    # Ideal: all relevant docs at the top
    ideal_retrieved = sorted(retrieved[:k], key=lambda d: d in relevant, reverse=True)
    # But ideal should have min(len(relevant), k) hits at top
    n_ideal = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """F1 score @ K."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


class BenchmarkResult:
    """벤치마크 결과 집계."""

    def __init__(self) -> None:
        self.queries: list[dict[str, object]] = []

    def add(
        self,
        query_id: str,
        query: str,
        retrieved: list[str],
        relevant: set[str],
        *,
        k: int = 5,
        description: str = "",
        search_time_ms: float = 0.0,
    ) -> dict[str, object]:
        result = {
            "query_id": query_id,
            "query": query,
            "description": description,
            "retrieved_top_k": retrieved[:k],
            "relevant": sorted(relevant),
            "precision@k": precision_at_k(retrieved, relevant, k),
            "recall@k": recall_at_k(retrieved, relevant, k),
            "f1@k": f1_at_k(retrieved, relevant, k),
            "mrr": reciprocal_rank(retrieved, relevant),
            "ndcg@k": ndcg_at_k(retrieved, relevant, k),
            "search_time_ms": search_time_ms,
        }
        self.queries.append(result)
        return result

    def summary(self) -> dict[str, float]:
        """전체 쿼리에 대한 평균 지표."""
        if not self.queries:
            return {}
        n = len(self.queries)
        return {
            "mean_precision@k": sum(q["precision@k"] for q in self.queries) / n,  # type: ignore[arg-type]
            "mean_recall@k": sum(q["recall@k"] for q in self.queries) / n,  # type: ignore[arg-type]
            "mean_f1@k": sum(q["f1@k"] for q in self.queries) / n,  # type: ignore[arg-type]
            "mrr": sum(q["mrr"] for q in self.queries) / n,  # type: ignore[arg-type]
            "mean_ndcg@k": sum(q["ndcg@k"] for q in self.queries) / n,  # type: ignore[arg-type]
            "mean_search_time_ms": sum(q["search_time_ms"] for q in self.queries) / n,  # type: ignore[arg-type]
            "total_queries": n,
        }

    def report(self, *, k: int = 5) -> str:
        """사람이 읽을 수 있는 리포트."""
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("Synaptic Memory Benchmark Report")
        lines.append("=" * 80)

        for q in self.queries:
            hit = "✓" if q["mrr"] > 0 else "✗"  # type: ignore[operator]
            lines.append(
                f'  {hit} [{q["query_id"]}] "{q["query"]}"  '
                f"P@{k}={q['precision@k']:.2f}  R@{k}={q['recall@k']:.2f}  "
                f"MRR={q['mrr']:.2f}  nDCG={q['ndcg@k']:.2f}  "
                f"{q['search_time_ms']:.1f}ms"
            )
            if q["description"]:
                lines.append(f"    → {q['description']}")

        s = self.summary()
        lines.append("-" * 80)
        lines.append(
            f"  Mean P@{k}={s['mean_precision@k']:.3f}  "
            f"Mean R@{k}={s['mean_recall@k']:.3f}  "
            f"MRR={s['mrr']:.3f}  "
            f"Mean nDCG@{k}={s['mean_ndcg@k']:.3f}  "
            f"Avg latency={s['mean_search_time_ms']:.1f}ms"
        )
        lines.append(f"  Total queries: {s['total_queries']:.0f}")
        lines.append("=" * 80)
        return "\n".join(lines)
