#!/usr/bin/env bash
# P0 full measurement — bge-m3 + bge-reranker-v2-m3 baseline.
# Plan C: MuSiQue 500q (decision-critical) + HotPotQA/2Wiki 100q + Allganize.
set -e

cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="examples/ablation/diagnostics"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/p0_full_$(date +%Y%m%d_%H%M%S).log"

echo "==> P0 full measurement starting — log: $LOG"
{
echo "==== Environment ===="
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
date

echo
echo "==== Stage 1/3: MuSiQue 500q (decision-critical) ===="
date
uv run python examples/ablation/run_tier1_benchmarks.py \
    --only musique --subset 500 \
    --local-bge --use-sqlite-graph --embed-batch 64

echo
echo "==== Stage 2/3: HotPotQA + 2Wiki @ 100q each ===="
date
uv run python examples/ablation/run_tier1_benchmarks.py \
    --only hotpotqa,2wiki --subset 100 \
    --local-bge --use-sqlite-graph --embed-batch 64

echo
echo "==== Stage 3/3: Allganize RAG-ko + RAG-Eval ===="
date
uv run python examples/benchmark_allganize.py --local-bge

echo
echo "==== DONE ===="
date
} 2>&1 | tee "$LOG"
