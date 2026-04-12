# Evaluation Baselines

버전별 평가 결과를 추적합니다. 개발 후 `eval/run_all.py` 실행 시 자동 저장됩니다.

## 사용법

```bash
# 평가 실행 + baselines에 저장
uv run python eval/run_all.py --quick \
    --embed-url http://14.6.220.78:11434/v1 \
    --reranker-url http://14.6.220.78:8180

# 이전 결과와 비교 (회귀 감지)
uv run python eval/run_all.py --quick \
    --embed-url http://14.6.220.78:11434/v1 \
    --compare eval/baselines/qa_latest.json
```

## 현재 베이스라인

### v0.12.0 (2026-04-12)

| Dataset | Corpus | MRR | P@10 | R@10 | nDCG | Hit |
|---------|--------|-----|------|------|------|-----|
| KRRA Easy | 20 | **0.967** | 0.503 | 0.893 | 0.901 | 20/20 |
| KRRA Hard | 15 | 0.507 | 0.156 | 0.633 | 0.504 | 11/15 |
| assort Easy | 15 | **0.933** | 0.093 | 0.900 | 0.908 | 14/15 |
| assort Hard | 15 | 0.000 | 0.000 | 0.000 | 0.000 | 0/15 |
| HotPotQA-24 | 226 | **0.727** | 0.163 | 0.812 | 0.677 | 24/24 |
| Allganize RAG-ko | 200 | **0.621** | 0.109 | 0.900 | 0.693 | 180/200 |
| Allganize RAG-Eval | 300 | **0.615** | 0.095 | 0.880 | 0.684 | 264/300 |
| PublicHealthQA | 77 | 0.318 | 0.062 | 0.584 | 0.382 | 45/77 |
| AutoRAG | 720 | **0.592** | 0.086 | 0.860 | 0.659 | 98/114 |

**조건**: FTS(Kiwi) + Embedding(qwen3-embedding:4b) + Reranker(bge-reranker-v2-m3)

### Multi-turn Agent (별도 측정)

| Dataset | 난이도 | Single-shot | Multi-turn | LLM |
|---------|-------|-------------|------------|-----|
| KRRA | Hard 15q | 11/15 | **15/15** | Claude Sonnet |
| KRRA | Hard 4q | 0/4 | **4/4** | GPT-4o-mini |
| assort | Hard 6q | 0/6 | **5/6** | GPT-4o-mini |

## 파일 목록

- `qa_latest.json` — 가장 최근 결과 (run_all.py가 자동 덮어씀)
- `v0.12.0_YYYYMMDD.json` — 버전별 스냅샷 (수동 복사)
