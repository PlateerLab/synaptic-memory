"""Tests for benchmark downloader helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

DOWNLOADER_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "ablation" / "download_benchmarks.py"
)
SPEC = importlib.util.spec_from_file_location("download_benchmarks", DOWNLOADER_PATH)
assert SPEC is not None
assert SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


class _FakeDataset(list):
    pass


def test_msmarco_jsonl_shard_preserves_gold_before_filler(monkeypatch, tmp_path):
    corpus = _FakeDataset(
        [
            {"_id": "0", "title": "", "text": "filler zero"},
            {"_id": "1", "title": "", "text": "filler one"},
            {"_id": "2", "title": "", "text": "filler two"},
            {"_id": "3", "title": "", "text": "needle gold"},
        ]
    )

    def fake_load_dataset(repo, config=None, *, split, streaming=False):
        if repo == "BeIR/msmarco" and config == "queries":
            return _FakeDataset([{"_id": "q1", "text": "needle"}])
        if repo == "BeIR/msmarco-qrels":
            return _FakeDataset([{"query-id": "q1", "corpus-id": "3", "score": 1}])
        if repo == "BeIR/msmarco" and config == "corpus":
            assert streaming is False
            return corpus
        raise AssertionError((repo, config, split, streaming))

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )

    manifest_path = tmp_path / "msmarco_passage.json"
    downloader.build_msmarco_passage(manifest_path, corpus_limit=3)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (tmp_path / "msmarco_passage.corpus.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["schema"] == "beir_jsonl_v1"
    assert manifest["corpus_size"] == 3
    assert manifest["preserved_gold_docs"] == 1
    assert [row["_id"] for row in rows] == ["3", "0", "1"]


def test_large_output_suffix_keeps_default_shard(monkeypatch, tmp_path):
    calls: list[Path] = []
    monkeypatch.setattr(downloader, "OUT_DIR", tmp_path)
    monkeypatch.setitem(
        downloader.LARGE_BUILDERS,
        "msmarco_passage",
        (lambda out_path, *, corpus_limit: calls.append(out_path), "msmarco_passage.json"),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_benchmarks.py",
            "--only",
            "msmarco_passage",
            "--large-corpus-limit",
            "5000000",
            "--large-output-suffix",
            "_5m",
        ],
    )

    downloader.main()

    assert calls == [tmp_path / "msmarco_passage_5m.json"]


def test_large_scale_tier_full_uses_complete_corpus_and_safe_suffix(
    monkeypatch,
    tmp_path,
):
    calls: list[tuple[Path, int]] = []
    monkeypatch.setattr(downloader, "OUT_DIR", tmp_path)
    monkeypatch.setitem(
        downloader.LARGE_BUILDERS,
        "msmarco_passage",
        (
            lambda out_path, *, corpus_limit: calls.append((out_path, corpus_limit)),
            "msmarco_passage.json",
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_benchmarks.py",
            "--only",
            "msmarco_passage",
            "--large-scale-tier",
            "full",
        ],
    )

    downloader.main()

    assert calls == [
        (
            tmp_path / "msmarco_passage_full.json",
            downloader.MSMARCO_FULL_CORPUS_SIZE,
        )
    ]


def test_large_scale_tier_overrides_limit_but_keeps_explicit_suffix(
    monkeypatch,
    tmp_path,
):
    calls: list[tuple[Path, int]] = []
    monkeypatch.setattr(downloader, "OUT_DIR", tmp_path)
    monkeypatch.setitem(
        downloader.LARGE_BUILDERS,
        "msmarco_passage",
        (
            lambda out_path, *, corpus_limit: calls.append((out_path, corpus_limit)),
            "msmarco_passage.json",
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_benchmarks.py",
            "--only",
            "msmarco_passage",
            "--large-corpus-limit",
            "123",
            "--large-scale-tier",
            "5m",
            "--large-output-suffix",
            "_custom",
        ],
    )

    downloader.main()

    assert calls == [(tmp_path / "msmarco_passage_custom.json", 5_000_000)]
