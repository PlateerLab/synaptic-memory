"""Tests for the tier-0 routing-signal AUC harness (``eval/routing_signal_auc.py``).

v0.29 E2. Locks the measurement discipline the harness exists to
enforce:

1. AUC math is exact (rank-based Mann-Whitney, tie-corrected) —
   a wrong AUC silently corrupts the go/no-go gate.
2. Threshold tuning is structurally confined to split=train and the
   verdict to split=heldout — leakage must raise, not warn.
3. NaN signals (no retrieval pass) are dropped from AUC, never fire
   in the combo, and are called out as "requires retrieval pass".
4. AUC / recall use confirmed-tier labels only; ``unsolved`` records
   carry no routing information and must not enter either class.
5. The held-out verdict fails closed: no confirmed positives OR an
   absent escalation-budget corpus forces NO-GO — a budget that can't
   be checked must never pass silently.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.routing_signal_auc import (
    ComboThresholds,
    GTRecord,
    RetrievalInfo,
    SignalContext,
    budget_key,
    build_contexts,
    compute_signal_values,
    evaluate_heldout,
    evaluate_signal,
    generate_report,
    load_retrieval_results,
    load_routing_gt,
    main,
    precision_recall_curve,
    rank_auc,
    recall_at_precision,
    s1_structured_lexicon,
    s2_margin_deficit,
    s2_score_flatness,
    s2_zero_results,
    s3_table_row_in_topk,
    tune_thresholds,
)


def _rec(
    qid: str,
    *,
    label: str = "cheap_sufficient",
    tier: str = "confirmed",
    split: str = "train",
    corpus: str = "finreg",
    query: str = "",
    has_table_nodes: bool | None = None,
) -> GTRecord:
    return GTRecord(
        qid=qid,
        query=query,
        label=label,
        tier=tier,
        split=split,
        corpus=corpus,
        has_table_nodes=has_table_nodes,
    )


# --- rank_auc -----------------------------------------------------


def test_rank_auc_known_value():
    # pairs: (3>2)✓ (3>0)✓ (1<2)✗ (1>0)✓ → 3/4
    assert rank_auc([3.0, 1.0], [2.0, 0.0]) == pytest.approx(0.75)


def test_rank_auc_perfect_separation_is_one():
    assert rank_auc([0.9, 0.8], [0.7, 0.1]) == pytest.approx(1.0)


def test_rank_auc_inverted_is_zero():
    assert rank_auc([0.1], [0.9]) == pytest.approx(0.0)


def test_rank_auc_all_tied_is_half():
    assert rank_auc([1.0, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(0.5)


def test_rank_auc_random_signal_near_half():
    rng = random.Random(7)
    pos = [rng.random() for _ in range(400)]
    neg = [rng.random() for _ in range(400)]
    assert 0.45 < rank_auc(pos, neg) < 0.55


def test_rank_auc_empty_class_raises():
    with pytest.raises(ValueError):
        rank_auc([], [1.0])
    with pytest.raises(ValueError):
        rank_auc([1.0], [])


# --- loaders ------------------------------------------------------


def test_load_routing_gt_is_lenient(tmp_path):
    lines = [
        json.dumps(
            {
                "qid": "q1",
                "query": "전체 상품 목록",
                "label": "agent_required",
                "tier": "confirmed",
                "split": "train",
                "corpus": "assort_hard",
                "extra_key": "ignored",
            }
        ),
        json.dumps({"qid": "q2", "label": "cheap_sufficient", "tier": "hit-only", "split": "held-out", "corpus": "AutoRAG"}),
        json.dumps({"qid": "q3", "label": "both"}),  # missing tier/split/corpus
        json.dumps({"label": "cheap_sufficient"}),  # no qid → skipped
        json.dumps({"qid": "q4"}),  # no label → skipped
        "",  # blank line
        "not json at all",  # junk line
    ]
    p = tmp_path / "gt.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    records = load_routing_gt(p)
    assert [r.qid for r in records] == ["q1", "q2", "q3"]
    assert records[1].tier == "hit_only"
    assert records[1].split == "heldout"
    assert records[1].corpus == "autorag"
    assert records[2].tier == "unmeasured"
    assert records[2].split == ""


def test_load_routing_gt_normalises_label_spellings(tmp_path):
    lines = [
        json.dumps({"qid": "q1", "label": "agent-required"}),
        json.dumps({"qid": "q2", "label": "Cheap Sufficient"}),
        json.dumps({"qid": "q3", "label": "single-shot-hit"}),
    ]
    p = tmp_path / "gt.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    records = load_routing_gt(p)
    assert [r.label for r in records] == [
        "agent_required",
        "cheap_sufficient",
        "single_shot_hit",
    ]


def test_load_routing_gt_warns_and_dedupes_duplicate_qids(tmp_path, capsys):
    # contexts/values key by qid (last wins) — the record list must
    # match, or the duplicate is double-counted in recall/escalation
    lines = [
        json.dumps({"qid": "q1", "label": "cheap_sufficient", "split": "train"}),
        json.dumps({"qid": "q1", "label": "agent_required", "split": "heldout"}),
        json.dumps({"qid": "q2", "label": "both"}),
    ]
    p = tmp_path / "gt.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    records = load_routing_gt(p)
    assert [r.qid for r in records] == ["q1", "q2"]
    assert records[0].label == "agent_required"  # last occurrence wins
    err = capsys.readouterr().err
    assert "duplicate" in err and "q1" in err


def test_load_retrieval_results_lenient_keys(tmp_path):
    lines = [
        json.dumps({"qid": "q1", "scores": [0.9, 0.5], "hit": True, "has_table_row": False}),
        json.dumps({"qid": "q2", "rerank_scores": [0.3], "table_row": True}),
        json.dumps({"scores": [1.0]}),  # no qid → skipped
    ]
    p = tmp_path / "retrieval.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    info = load_retrieval_results(p)
    assert set(info) == {"q1", "q2"}
    assert info["q1"].scores == [0.9, 0.5]
    assert info["q1"].hit is True
    assert info["q1"].has_table_row is False
    assert info["q2"].scores == [0.3]
    assert info["q2"].hit is None
    assert info["q2"].has_table_row is True


def test_load_retrieval_results_skips_malformed_scores(tmp_path, capsys):
    # one bad line must not abort the whole s2/s3 pass
    lines = [
        json.dumps({"qid": "q1", "scores": [0.9, "oops"]}),  # non-numeric entry
        json.dumps({"qid": "q2", "scores": 0.5}),  # not a list
        json.dumps({"qid": "q3", "scores": [0.3], "hit": True}),
    ]
    p = tmp_path / "retrieval.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    info = load_retrieval_results(p)
    assert set(info) == {"q3"}
    assert info["q3"].scores == [0.3]
    err = capsys.readouterr().err
    assert "2 retrieval record(s)" in err and "skipped" in err


def test_build_contexts_table_corpus_fallback_and_override():
    records = [
        _rec("a", corpus="assort_hard"),
        _rec("b", corpus="finreg"),
        _rec("c", corpus="finreg", has_table_nodes=True),
    ]
    ctxs = build_contexts(records)
    assert ctxs["a"].has_table_nodes is True  # corpus prefix
    assert ctxs["b"].has_table_nodes is False
    assert ctxs["c"].has_table_nodes is True  # record override wins


# --- signals ------------------------------------------------------


def test_s1_fires_on_structured_vocab_with_tables():
    ctx = SignalContext(has_table_nodes=True)
    assert s1_structured_lexicon("전체 상품의 평균 가격은?", ctx) > 0.0
    assert s1_structured_lexicon("how many products have price greater than 100", ctx) > 0.0


def test_s1_gated_off_without_table_nodes():
    ctx = SignalContext(has_table_nodes=False)
    assert s1_structured_lexicon("전체 상품의 평균 가격은?", ctx) == 0.0


def test_s1_plain_lookup_scores_zero():
    ctx = SignalContext(has_table_nodes=True)
    assert s1_structured_lexicon("배송 정책", ctx) == 0.0


def test_s2_s3_nan_without_retrieval_pass():
    ctx = SignalContext(has_table_nodes=True, retrieval=None)
    assert math.isnan(s2_zero_results("q", ctx))
    assert math.isnan(s2_score_flatness("q", ctx))
    assert math.isnan(s2_margin_deficit("q", ctx))
    assert math.isnan(s3_table_row_in_topk("q", ctx))


def test_s2_values_from_scores():
    ctx = SignalContext(retrieval=RetrievalInfo(scores=[0.9, 0.5, 0.1], hit=True))
    assert s2_zero_results("q", ctx) == 0.0
    assert s2_margin_deficit("q", ctx) == pytest.approx(-0.4)
    # population std of [0.9, 0.5, 0.1] = sqrt(0.32/3)
    assert s2_score_flatness("q", ctx) == pytest.approx(-math.sqrt(0.32 / 3))


def test_s2_zero_results_ignores_gold_hit():
    # zero results -> fires
    assert s2_zero_results("q", SignalContext(retrieval=RetrievalInfo(scores=[], hit=None))) == 1.0
    # gold-based hit must NOT leak into the signal: hit=False with results
    # present stays 0.0 (the old s2_no_hit oracle behaviour scored 1.0 here,
    # a tautological AUC against the GT's own hit axis)
    assert s2_zero_results("q", SignalContext(retrieval=RetrievalInfo(scores=[0.5], hit=False))) == 0.0
    assert s2_zero_results("q", SignalContext(retrieval=RetrievalInfo(scores=[0.5], hit=None))) == 0.0


def test_s2_flatness_margin_nan_with_single_score():
    ctx = SignalContext(retrieval=RetrievalInfo(scores=[0.5]))
    assert math.isnan(s2_score_flatness("q", ctx))
    assert math.isnan(s2_margin_deficit("q", ctx))


def test_s3_table_row():
    yes = SignalContext(retrieval=RetrievalInfo(has_table_row=True))
    no = SignalContext(retrieval=RetrievalInfo(has_table_row=False))
    unknown = SignalContext(retrieval=RetrievalInfo(has_table_row=None))
    assert s3_table_row_in_topk("q", yes) == 1.0
    assert s3_table_row_in_topk("q", no) == 0.0
    assert math.isnan(s3_table_row_in_topk("q", unknown))


# --- evaluate_signal -----------------------------------------------


def test_evaluate_signal_perfectly_separated_auc_one():
    records = [
        _rec("p1", label="agent_required"),
        _rec("p2", label="agent_required"),
        _rec("n1"),
        _rec("n2"),
    ]
    values = {"p1": 0.9, "p2": 0.8, "n1": 0.1, "n2": 0.2}
    ev = evaluate_signal(records, values, "sig")
    assert ev.auc == pytest.approx(1.0)
    assert (ev.n_pos, ev.n_neg, ev.n_nan) == (2, 2, 0)


def test_evaluate_signal_confirmed_tier_only():
    records = [
        _rec("p1", label="agent_required"),
        _rec("n1"),
        # provisional positive with an inverted value — must not enter
        _rec("p2", label="agent_required", tier="provisional"),
    ]
    values = {"p1": 1.0, "n1": 0.0, "p2": -99.0}
    ev = evaluate_signal(records, values, "sig")
    assert ev.auc == pytest.approx(1.0)
    assert ev.n_pos == 1


def test_evaluate_signal_unsolved_excluded():
    records = [
        _rec("p1", label="agent_required"),
        _rec("n1"),
        _rec("u1", label="unsolved"),
    ]
    values = {"p1": 1.0, "n1": 0.0, "u1": 100.0}
    ev = evaluate_signal(records, values, "sig")
    assert ev.auc == pytest.approx(1.0)
    assert ev.n_pos + ev.n_neg == 2


def test_evaluate_signal_drops_nan_and_notes():
    records = [
        _rec("p1", label="agent_required"),
        _rec("p2", label="agent_required"),
        _rec("n1"),
    ]
    values = {"p1": 1.0, "p2": math.nan, "n1": 0.0}
    ev = evaluate_signal(records, values, "sig")
    assert ev.auc == pytest.approx(1.0)
    assert ev.n_nan == 1
    assert "NaN" in ev.note


def test_evaluate_signal_all_nan_requires_retrieval_pass():
    records = [_rec("p1", label="agent_required"), _rec("n1")]
    values = {"p1": math.nan, "n1": math.nan}
    ev = evaluate_signal(records, values, "sig")
    assert ev.auc is None
    assert ev.note == "requires retrieval pass"


def test_evaluate_signal_single_class_note():
    records = [_rec("p1", label="agent_required")]
    ev = evaluate_signal(records, {"p1": 1.0}, "sig")
    assert ev.auc is None
    assert ev.note == "single-class subset"


def test_reference_tier_maps_hit_only_partial_labels():
    # T3 hit-only rows (agent axis unmeasured) carry partial labels:
    # 'single_shot_hit' is a reference-section negative (cheap path
    # succeeded); 'single_shot_miss' carries no class and is excluded
    records = [
        _rec("p1", label="agent_required", tier="provisional"),
        _rec("h1", label="single_shot_hit", tier="hit_only", corpus="autorag"),
        _rec("h2", label="single_shot_miss", tier="hit_only", corpus="autorag"),
    ]
    values = {"p1": 1.0, "h1": 0.0, "h2": 100.0}
    ev = evaluate_signal(records, values, "sig", tiers=("provisional", "hit_only"))
    assert ev.auc == pytest.approx(1.0)  # h2's inverted value never entered
    assert (ev.n_pos, ev.n_neg) == (1, 1)


def test_verdict_tier_never_maps_partial_labels():
    # the hit-only mapping is keyed on tier — a partial label on a
    # confirmed record must still fall out of both verdict classes
    records = [
        _rec("p1", label="agent_required"),
        _rec("h1", label="single_shot_hit"),  # tier=confirmed
        _rec("n1"),
    ]
    values = {"p1": 1.0, "h1": 0.0, "n1": 0.0}
    ev = evaluate_signal(records, values, "sig")
    assert (ev.n_pos, ev.n_neg) == (1, 1)


# --- precision/recall curve ----------------------------------------


def test_precision_recall_curve_and_recall_at_precision():
    pairs = [(0.9, True), (0.8, True), (0.7, False), (0.6, True), (0.1, False)]
    pts = precision_recall_curve(pairs)
    # threshold 0.9 → P=1.0 R=1/3; threshold 0.6 → P=3/4 R=1.0
    assert (0.9, 1.0, pytest.approx(1 / 3)) == pts[0]
    assert recall_at_precision(pairs, 1.0) == pytest.approx(2 / 3)  # thr 0.8
    assert recall_at_precision(pairs, 0.75) == pytest.approx(1.0)  # thr 0.6
    assert recall_at_precision(pairs, 1.01) is None


# --- budgets --------------------------------------------------------


def test_budget_key_mapping():
    assert budget_key("autorag") == "autorag"
    assert budget_key("autorag_retrieval") == "autorag"
    assert budget_key("assort") == "assort_easy"
    assert budget_key("assort_easy") == "assort_easy"
    assert budget_key("assort_hard") is None
    assert budget_key("assort_conversational") is None
    assert budget_key("finreg") is None


# --- train/heldout separation enforcement ---------------------------


def test_tune_thresholds_rejects_heldout_records():
    records = [_rec("p1", label="agent_required"), _rec("h1", split="heldout")]
    values = {"sig": {"p1": 1.0, "h1": 0.0}}
    with pytest.raises(ValueError, match="split=train"):
        tune_thresholds(records, values)


def test_evaluate_heldout_rejects_train_records():
    records = [_rec("p1", label="agent_required", split="heldout"), _rec("t1", split="train")]
    values = {"sig": {"p1": 1.0, "t1": 0.0}}
    combo = ComboThresholds(thresholds={"sig": 1.0})
    with pytest.raises(ValueError, match="split=heldout"):
        evaluate_heldout(records, values, combo)


# --- tuner + verdict end-to-end -------------------------------------


def _mk_dataset(*, autorag_noisy: bool) -> tuple[list[GTRecord], dict[str, dict[str, float]]]:
    """Synthetic GT: 5 confirmed agent_required per split at value 1.0,
    plus AutoRAG / assort-Easy cheap traffic. ``autorag_noisy`` makes
    the signal fire on 40 % of AutoRAG (blows the 15 % budget)."""
    records: list[GTRecord] = []
    values: dict[str, float] = {}
    for split in ("train", "heldout"):
        sfx = split[0]
        for i in range(5):
            qid = f"pos_{sfx}{i}"
            records.append(
                _rec(qid, label="agent_required", split=split, corpus="assort_hard")
            )
            values[qid] = 1.0
        for i in range(10):
            qid = f"ar_{sfx}{i}"
            records.append(
                _rec(qid, tier="hit_only", split=split, corpus="autorag")
            )
            values[qid] = 1.0 if (autorag_noisy and i < 4) else 0.0
        for i in range(5):
            qid = f"ae_{sfx}{i}"
            records.append(
                _rec(qid, tier="provisional", split=split, corpus="assort")
            )
            values[qid] = 0.0
    return records, {"sig": values}


def test_tune_and_heldout_go():
    records, values = _mk_dataset(autorag_noisy=False)
    train = [r for r in records if r.split == "train"]
    heldout = [r for r in records if r.split == "heldout"]
    combo = tune_thresholds(train, values)
    # threshold 1.0 separates positives from cheap traffic; lowering to
    # 0.0 would escalate 100 % of AutoRAG and is budget-blocked
    assert combo.thresholds["sig"] == 1.0
    verdict = evaluate_heldout(heldout, values, combo)
    assert verdict.recall == pytest.approx(1.0)
    assert verdict.n_pos == 5
    assert verdict.escalation["autorag"][0] == 0.0
    assert verdict.escalation["assort_easy"][0] == 0.0
    assert verdict.go is True


def test_tune_no_go_when_budget_blocks_the_only_signal():
    records, values = _mk_dataset(autorag_noisy=True)
    train = [r for r in records if r.split == "train"]
    heldout = [r for r in records if r.split == "heldout"]
    combo = tune_thresholds(train, values)
    # any firing threshold escalates 40 % of AutoRAG → tuner must keep
    # the signal disabled rather than break the budget
    assert math.isinf(combo.thresholds["sig"])
    verdict = evaluate_heldout(heldout, values, combo)
    assert verdict.recall == 0.0
    assert verdict.go is False


def test_tune_tie_break_prefers_first_signal_and_higher_threshold():
    # two identical signals: the tuner must enable the FIRST one in
    # dict order (s1 is the embedder-independent first line) at the
    # most conservative threshold that still reaches the recall
    records = [
        _rec("p1", label="agent_required"),
        _rec("p2", label="agent_required"),
        _rec("n1"),
    ]
    vals = {"p1": 1.0, "p2": 0.8, "n1": 0.1}
    values = {"first_sig": dict(vals), "second_sig": dict(vals)}
    combo = tune_thresholds(records, values)
    assert combo.thresholds["first_sig"] == 0.8  # not 0.1 — conservative
    assert math.isinf(combo.thresholds["second_sig"])


def test_combo_nan_never_fires():
    combo = ComboThresholds(thresholds={"sig": 0.5})
    assert combo.fires({"sig": {"q1": math.nan}}, "q1") is False
    assert combo.fires({"sig": {}}, "q1") is False
    assert combo.fires({"sig": {"q1": 0.5}}, "q1") is True


def test_heldout_verdict_vacuous_without_confirmed_positives():
    records = [_rec("n1", split="heldout")]
    values = {"sig": {"n1": 0.0}}
    verdict = evaluate_heldout(records, values, ComboThresholds(thresholds={"sig": 1.0}))
    assert verdict.go is False
    assert any("vacuous" in n for n in verdict.notes)


def test_heldout_verdict_no_go_when_budget_corpus_absent():
    # perfect recall but zero AutoRAG / assort-Easy rows in held-out:
    # the budget halves of the E2 gate are unverifiable → fail closed
    records = [
        _rec(f"p{i}", label="agent_required", split="heldout", corpus="assort_hard")
        for i in range(5)
    ]
    values = {"sig": {r.qid: 1.0 for r in records}}
    verdict = evaluate_heldout(records, values, ComboThresholds(thresholds={"sig": 1.0}))
    assert verdict.recall == pytest.approx(1.0)
    assert verdict.go is False
    assert any("autorag" in n and "NO-GO" in n for n in verdict.notes)
    assert any("assort_easy" in n for n in verdict.notes)


def test_heldout_verdict_no_go_when_one_budget_corpus_absent():
    # one populated budget bucket does not excuse the other
    records = [
        _rec(f"p{i}", label="agent_required", split="heldout", corpus="assort_hard")
        for i in range(5)
    ] + [
        _rec(f"ar{i}", tier="hit_only", split="heldout", corpus="autorag")
        for i in range(10)
    ]
    values = {"sig": {r.qid: (1.0 if r.qid.startswith("p") else 0.0) for r in records}}
    verdict = evaluate_heldout(records, values, ComboThresholds(thresholds={"sig": 1.0}))
    assert verdict.recall == pytest.approx(1.0)
    assert verdict.escalation["autorag"][0] == 0.0  # within budget
    assert verdict.go is False  # assort_easy still unchecked
    assert any("assort_easy" in n and "NO-GO" in n for n in verdict.notes)


# --- report + CLI ----------------------------------------------------


def _write_gt(tmp_path: Path) -> Path:
    records, values = _mk_dataset(autorag_noisy=False)
    rows = []
    for r in records:
        rows.append(
            json.dumps(
                {
                    "qid": r.qid,
                    # structured vocab so s1 fires for positives
                    "query": "전체 상품의 평균 가격은?" if r.label == "agent_required" else "배송 정책",
                    "label": r.label,
                    "tier": r.tier,
                    "split": r.split,
                    "corpus": r.corpus,
                },
                ensure_ascii=False,
            )
        )
    p = tmp_path / "gt.jsonl"
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


def test_generate_report_without_retrieval_flags_s2(tmp_path):
    gt = _write_gt(tmp_path)
    records = load_routing_gt(gt)
    contexts = build_contexts(records)
    values = compute_signal_values(records, contexts)
    report = generate_report(records, values, gt_path=gt)
    assert "requires retrieval pass" in report
    assert "VERDICT" in report
    assert "Held-out verdict" in report
    assert "train split only" in report


def test_cli_end_to_end_writes_markdown(tmp_path, capsys):
    gt = _write_gt(tmp_path)
    retrieval_rows = [
        json.dumps({"qid": f"pos_h{i}", "scores": [0.5, 0.4], "hit": False, "has_table_row": True})
        for i in range(5)
    ]
    retrieval = tmp_path / "retrieval.jsonl"
    retrieval.write_text("\n".join(retrieval_rows), encoding="utf-8")
    out = tmp_path / "report.md"
    rc = main([str(gt), "--retrieval", str(retrieval), "--out", str(out)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Tier-0 routing-signal AUC report")
    assert "s1_structured_lexicon" in text
    assert "VERDICT" in text


def test_cli_empty_gt_returns_error(tmp_path):
    gt = tmp_path / "empty.jsonl"
    gt.write_text("", encoding="utf-8")
    assert main([str(gt)]) == 2


def test_load_retrieval_specs_namespaces_corpus(tmp_path):
    from eval.routing_signal_auc import load_retrieval_specs

    a = tmp_path / "a.jsonl"
    a.write_text(
        '{"qid": "q000", "scores": [0.9, 0.2], "hit": true, "has_table_row": false}\n',
        encoding="utf-8",
    )
    b = tmp_path / "b.jsonl"
    b.write_text(
        '{"qid": "x2bee_hard:h001", "scores": [0.5], "hit": false}\n', encoding="utf-8"
    )
    out = load_retrieval_specs([f"assort={a}", str(b)])
    # CORPUS=PATH namespaces bare qids; bare PATH is already GT-keyed
    assert set(out) == {"assort:q000", "x2bee_hard:h001"}
    assert out["assort:q000"].scores == [0.9, 0.2]
    assert out["x2bee_hard:h001"].hit is False
