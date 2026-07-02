"""Tests for Tier-1 benchmark runner helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "ablation" / "run_tier1_benchmarks.py"
)
SPEC = importlib.util.spec_from_file_location("run_tier1_benchmarks", RUNNER_PATH)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


@pytest.mark.asyncio
async def test_corpus_limit_keeps_selected_query_gold_docs(tmp_path):
    path = tmp_path / "tiny_bench.json"
    path.write_text(
        json.dumps(
            {
                "corpus": {
                    "filler_a": {"title": "Filler A", "text": "unrelated alpha"},
                    "filler_b": {"title": "Filler B", "text": "unrelated beta"},
                    "gold_doc": {"title": "Gold", "text": "needle targetterm"},
                },
                "queries": {"q1": "targetterm"},
                "qrels": {"q1": {"gold_doc": 1}},
            }
        ),
        encoding="utf-8",
    )

    report = await runner.run_one(
        runner.Dataset(name="Tiny", path=path, reference="unit"),
        subset=1,
        corpus_limit=2,
    )

    assert report.n_docs == 2
    assert report.hit_at_10 == 1
    assert report.recall_at_10 == 1.0
