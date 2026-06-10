"""Tests for the routing ground-truth builder (``eval/routing_gt.py``).

v0.29 E1/T3. Locks the invariants the downstream signal-AUC harness (E2)
and router gates (E3) depend on:

1. The 2x2 label semantics — agent_required is the routing payoff class
   and must never be produced by a partial (single-axis) measurement.
2. Tier discipline — a confirmed label requires >=3 *distinct* runs per
   arm (duplicated records of one run must not mint extra runs); a
   single agent run is provisional; design annotations are provisional;
   single-shot-only sources are hit_only.
3. The agent-log join — raw qids collide across bench sections, so the
   section recovery must either attribute them correctly or drop the
   whole source (never misattribute); a zero-row join must warn.
4. The train/heldout split — deterministic per-qid hash parity, stable
   under qid-set growth, with assort (Easy) entirely held out.
5. Graceful degradation — missing input files skip the source with a
   warning instead of failing the build.
6. Axis merge — duplicate qids carrying complementary axes (xlsx agent
   axis + deterministic hits single-shot axis) combine into one full
   2x2 label instead of dropping an axis.

All tests run on synthetic fixtures under tmp_path; tests that need the
real repo data files are skipif-guarded so CI without the data still
passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.routing_gt import (
    DEFAULT_AGENT_LOG,
    DEFAULT_GT_XLSX,
    _dedupe,
    assign_splits,
    build_routing_gt,
    load_agent_log_axis,
    load_finreg_source,
    load_gt_datasets_source,
    load_hits_source,
    load_krra_graph_source,
    main,
    routing_label,
)

# --- Label semantics ------------------------------------------------


@pytest.mark.parametrize(
    ("single_shot_hit", "agent_solve", "expected"),
    [
        (True, True, "both"),
        (True, False, "cheap_sufficient"),
        (False, True, "agent_required"),
        (False, False, "unsolved"),
        (True, None, "single_shot_hit"),
        (False, None, "single_shot_miss"),
        (None, True, "agent_solved"),
        (None, False, "agent_failed"),
        (None, None, "unlabeled"),
    ],
)
def test_routing_label_2x2_and_partials(single_shot_hit, agent_solve, expected):
    assert routing_label(single_shot_hit, agent_solve) == expected


# --- Source (a): finreg per-query JSONL ------------------------------


def _perquery_record(qid: str, arm: str, run: int, correct: bool) -> dict:
    """A record in the rag_vs_agent_answer.py --out-jsonl shape."""
    return {
        "qid": qid,
        "query": f"question {qid}",
        "arm": arm,
        "run": run,
        "judge_correct": correct,
        "empty": False,
        "elapsed_s": 1.0,
        "answer": "x",
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


@pytest.fixture()
def finreg_jsonl(tmp_path: Path) -> Path:
    """3 runs x 2 arms covering all four 2x2 cells, plus a 1-run qid."""
    records = []
    # q000: rag 3/3, agent 1/3 -> cheap_sufficient
    # q001: rag 0/3, agent 2/3 -> agent_required
    # q002: rag 2/3, agent 3/3 -> both
    # q003: rag 0/3, agent 1/3 -> unsolved
    outcomes = {
        "q000": {"rag": [True, True, True], "agent": [True, False, False]},
        "q001": {"rag": [False, False, False], "agent": [True, True, False]},
        "q002": {"rag": [True, True, False], "agent": [True, True, True]},
        "q003": {"rag": [False, False, False], "agent": [False, True, False]},
    }
    for qid, arms in outcomes.items():
        for arm, runs in arms.items():
            for run, correct in enumerate(runs, start=1):
                records.append(_perquery_record(qid, arm, run, correct))
    # q004: a single run per arm -> must NOT be confirmed
    records.append(_perquery_record("q004", "rag", 1, True))
    records.append(_perquery_record("q004", "agent", 1, True))
    return _write_jsonl(tmp_path / "finreg_multihop_perquery.jsonl", records)


def test_finreg_majority_labels_and_confirmed_tier(finreg_jsonl: Path):
    rows = {r["source_qid"]: r for r in load_finreg_source([finreg_jsonl])}
    assert rows["q000"]["label"] == "cheap_sufficient"
    assert rows["q001"]["label"] == "agent_required"
    assert rows["q002"]["label"] == "both"
    assert rows["q003"]["label"] == "unsolved"
    for qid in ("q000", "q001", "q002", "q003"):
        assert rows[qid]["tier"] == "confirmed"
        assert rows[qid]["source"] == "finreg_jsonl"
    # corpus inferred from the filename, qid namespaced with it
    assert rows["q001"]["corpus"] == "finreg_multihop"
    assert rows["q001"]["qid"] == "finreg_multihop:q001"
    # majority rule boundary: 2/3 wins -> True (q001 agent), 1/3 -> False (q000 agent)
    assert rows["q001"]["agent_solve"] is True
    assert rows["q000"]["agent_solve"] is False
    # the finreg single-shot axis is judge-based, and says so
    assert rows["q000"]["single_shot_basis"] == "judge"
    assert rows["q000"]["single_shot_source"] == "finreg_jsonl"
    assert rows["q000"]["agent_source"] == "finreg_jsonl"


def test_finreg_single_run_is_provisional_not_confirmed(finreg_jsonl: Path):
    rows = {r["source_qid"]: r for r in load_finreg_source([finreg_jsonl])}
    assert rows["q004"]["tier"] == "provisional"
    # min_wins scales with the run count: a 1/1 win is that run's majority
    # (the provisional tier carries the low confidence, not the outcome)
    assert rows["q004"]["agent_solve"] is True


def test_finreg_duplicated_single_run_stays_provisional(tmp_path: Path, capsys):
    # The same run-1 record three times (concatenated/resumed JSONL) must
    # not count as 3 runs — that would mint a confirmed label from a
    # single-run outcome.
    rec_rag = _perquery_record("q000", "rag", 1, True)
    rec_agent = _perquery_record("q000", "agent", 1, True)
    path = _write_jsonl(tmp_path / "finreg_dup.jsonl", [rec_rag, rec_agent] * 3)
    (row,) = load_finreg_source([path])
    assert row["tier"] == "provisional"
    assert "duplicate (qid, arm, run)" in capsys.readouterr().err


def test_finreg_duplicate_run_keeps_last_occurrence(tmp_path: Path):
    # A resumed rerun that rewrites run 1 wins over the earlier record.
    path = _write_jsonl(
        tmp_path / "finreg_resume.jsonl",
        [
            _perquery_record("q000", "rag", 1, False),
            _perquery_record("q000", "agent", 1, False),
            _perquery_record("q000", "rag", 1, True),  # rewritten outcome
        ],
    )
    (row,) = load_finreg_source([path])
    assert row["single_shot_hit"] is True


def test_finreg_even_run_tie_is_not_a_majority(tmp_path: Path):
    # 4 distinct runs: 2/4 wins is a tie, not a majority (min_wins=3).
    records = [
        _perquery_record("q000", "rag", r, r <= 2) for r in (1, 2, 3, 4)
    ] + [_perquery_record("q000", "agent", r, r <= 3) for r in (1, 2, 3, 4)]
    path = _write_jsonl(tmp_path / "finreg_4runs.jsonl", records)
    (row,) = load_finreg_source([path])
    assert row["single_shot_hit"] is False  # 2/4 rag wins
    assert row["agent_solve"] is True  # 3/4 agent wins
    assert row["tier"] == "confirmed"


def test_finreg_rag_only_arm_gives_partial_label(tmp_path: Path):
    path = _write_jsonl(
        tmp_path / "finreg_ragonly.jsonl",
        [_perquery_record("q000", "rag", r, True) for r in (1, 2, 3)],
    )
    (row,) = load_finreg_source([path])
    assert row["agent_solve"] is None
    assert row["agent_source"] is None
    assert row["label"] == "single_shot_hit"
    assert row["tier"] == "provisional"  # agent arm unmeasured -> not confirmed


def test_finreg_missing_file_skips_with_warning(tmp_path: Path, capsys):
    rows = load_finreg_source([tmp_path / "nope.jsonl"])
    assert rows == []
    assert "skipped" in capsys.readouterr().err


# --- Source (b): agent log section recovery --------------------------

_FAKE_LOG = """\
--agent-only: skipping single-shot retrieval

Running 0 benchmarks...
  KRRA Hard (agent, max 5 turns)...       [h001] turns=4 found=23 hit=True (id)
      [h002] turns=5 found=10 hit=False | relevant={'x'} | sample_found=['y']
      [h004] turns=5 found=11 hit=True (id)
  X2BEE Hard (agent, max 5 turns)...       [h001] turns=5 found=17 hit=False (id)
      [h002] turns=2 found=3 hit=True (judge)
  assort Conv (agent, max 5 turns)...       [c001] turns=2 found=29 hit=True (id)
"""


def test_agent_log_axis_disambiguates_colliding_qids(tmp_path: Path):
    log = tmp_path / "agent.log"
    log.write_text(_FAKE_LOG, encoding="utf-8")
    axis = load_agent_log_axis(log)
    # h001/h002 appear in two sections and must not be conflated
    assert axis[("krra_hard", "h001")] is True
    assert axis[("krra_hard", "h002")] is False
    assert axis[("x2bee_hard", "h001")] is False
    assert axis[("x2bee_hard", "h002")] is True
    # "Conv" expands to the conversational sheet name
    assert axis[("assort_conversational", "c001")] is True
    # gap (h003 missing) stays within one segment
    assert axis[("krra_hard", "h004")] is True
    assert len(axis) == 6


def test_agent_log_section_mismatch_drops_source(tmp_path: Path, capsys):
    # one header but the qid stream resets -> 2 segments vs 1 header
    log = tmp_path / "broken.log"
    log.write_text(
        "  KRRA Hard (agent, max 5 turns)...       [h001] turns=1 found=1 hit=True (id)\n"
        "      [h001] turns=1 found=1 hit=False (id)\n",
        encoding="utf-8",
    )
    assert load_agent_log_axis(log) == {}
    assert "mismatch" in capsys.readouterr().err


def test_agent_log_missing_returns_empty(tmp_path: Path, capsys):
    assert load_agent_log_axis(tmp_path / "nope.log") == {}
    assert "skipped" in capsys.readouterr().err


# --- Source (b): xlsx -------------------------------------------------


def _make_xlsx(tmp_path: Path) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Synaptic Memory — Evaluation GT Datasets"])
    for sheet, qids_types in {
        "assort_hard": [("a001", "aggregation_count"), ("a002", "paraphrase")],
        "assort": [("q001", None)],
    }.items():
        ws = wb.create_sheet(sheet)
        ws.append(["Dataset", sheet])  # preamble before the header row
        ws.append([])
        ws.append(["qid", "query", "type", "level"])
        for qid, qtype in qids_types:
            ws.append([qid, f"query text {qid}", qtype, "L2"])
    path = tmp_path / "gt.xlsx"
    wb.save(path)
    return path


def test_xlsx_types_enumerated_and_agent_axis_joined(tmp_path: Path):
    xlsx = _make_xlsx(tmp_path)
    log = tmp_path / "agent.log"
    log.write_text(
        "  assort Hard (agent, max 5 turns)...       [a001] turns=2 found=5 hit=True (id)\n",
        encoding="utf-8",
    )
    rows = {r["qid"]: r for r in load_gt_datasets_source(xlsx, log)}
    assert len(rows) == 3  # Summary sheet contributes nothing
    a001 = rows["assort_hard:a001"]
    assert a001["type"] == "aggregation_count"  # read from the sheet, not hardcoded
    assert a001["agent_solve"] is True
    assert a001["tier"] == "provisional"
    assert a001["label"] == "agent_solved"  # partial: no single-shot axis
    # not in the log -> unmeasured, fully unlabeled
    a002 = rows["assort_hard:a002"]
    assert a002["agent_solve"] is None
    assert a002["tier"] == "unmeasured"
    assert a002["label"] == "unlabeled"
    assert rows["assort:q001"]["type"] is None


def test_xlsx_missing_file_skips_with_warning(tmp_path: Path, capsys):
    assert load_gt_datasets_source(tmp_path / "nope.xlsx", None) == []
    assert "skipped" in capsys.readouterr().err


def test_xlsx_zero_join_warns_instead_of_silently_degrading(tmp_path: Path, capsys):
    # If section-name normalization stops matching the sheet names, every
    # row silently falls back to unmeasured — that degradation must warn.
    xlsx = _make_xlsx(tmp_path)
    log = tmp_path / "agent.log"
    log.write_text(
        "  Renamed Bench (agent, max 5 turns)...       [a001] turns=2 found=5 hit=True (id)\n",
        encoding="utf-8",
    )
    rows = load_gt_datasets_source(xlsx, log)
    assert all(r["tier"] == "unmeasured" for r in rows)
    assert "joined 0/1" in capsys.readouterr().err


# --- Source (c): krra_graph design GT ---------------------------------


@pytest.fixture()
def krra_graph_json(tmp_path: Path) -> Path:
    data = {
        "dataset": "krra",
        "queries": [
            {
                "qid": "g001",
                "category": "A_aggregation",
                "query": "count docs by year",
                "current_answerable": False,
                "topk_inadequacy": "top-k never returns a count",
            },
            {
                "qid": "g002",
                "category": "F_constraint",
                "query": "plain lookup",
                "current_answerable": True,
                "topk_inadequacy": "",
            },
        ],
    }
    path = tmp_path / "krra_graph.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_krra_graph_design_labels(krra_graph_json: Path):
    rows = {r["source_qid"]: r for r in load_krra_graph_source(krra_graph_json)}
    g001 = rows["g001"]
    assert g001["label"] == "agent_required"  # by design, not by measurement
    assert g001["tier"] == "provisional"
    assert g001["single_shot_hit"] is False
    assert g001["single_shot_basis"] == "design"  # annotation, not a measurement
    assert g001["agent_solve"] is None
    assert g001["type"] == "A_aggregation"
    # answerable query falls back to the partial label
    assert rows["g002"]["label"] == "single_shot_hit"


def test_krra_graph_missing_skips_with_warning(tmp_path: Path, capsys):
    assert load_krra_graph_source(tmp_path / "nope.json") == []
    assert "skipped" in capsys.readouterr().err


# --- Source (d): deterministic hit-only files --------------------------


def test_hits_are_hit_only_tier(tmp_path: Path):
    path = _write_jsonl(
        tmp_path / "autorag_hits.jsonl",
        [{"qid": "ar0001", "hit": True}, {"qid": "ar0002", "hit": False}],
    )
    rows = {r["source_qid"]: r for r in load_hits_source(path)}
    assert rows["ar0001"]["tier"] == "hit_only"
    assert rows["ar0001"]["label"] == "single_shot_hit"
    assert rows["ar0001"]["single_shot_basis"] == "id_hit"  # deterministic, not judge
    assert rows["ar0002"]["label"] == "single_shot_miss"
    assert all(r["agent_solve"] is None for r in rows.values())
    assert all(r["corpus"] == "autorag" for r in rows.values())  # default corpus


def test_hits_corpus_override_namespaces_qids(tmp_path: Path):
    path = _write_jsonl(tmp_path / "ah_hits.jsonl", [{"qid": "a001", "hit": False}])
    (row,) = load_hits_source(path, corpus="assort_hard")
    assert row["qid"] == "assort_hard:a001"
    assert row["corpus"] == "assort_hard"


def test_hits_missing_skips_with_warning(tmp_path: Path, capsys):
    assert load_hits_source(tmp_path / "nope.jsonl") == []
    assert "skipped" in capsys.readouterr().err


# --- Dedupe + axis merge -------------------------------------------------


def test_dedupe_keeps_higher_tier(capsys):
    lo = {"qid": "krra_graph:g001", "tier": "unmeasured", "source": "gt_datasets_xlsx"}
    hi = {"qid": "krra_graph:g001", "tier": "provisional", "source": "krra_graph"}
    assert _dedupe([lo, hi]) == [hi]
    assert _dedupe([hi, lo]) == [hi]  # order-independent winner
    assert "duplicate qid" in capsys.readouterr().err


def _axis_row(qid: str, tier: str, source: str, **axes) -> dict:
    row = {
        "qid": qid,
        "source_qid": qid.split(":", 1)[1],
        "corpus": qid.split(":", 1)[0],
        "query": None,
        "type": None,
        "single_shot_hit": None,
        "single_shot_basis": None,
        "single_shot_source": None,
        "agent_solve": None,
        "agent_source": None,
        "label": "unlabeled",
        "tier": tier,
        "source": source,
    }
    row.update(axes)
    return row


def _agent_axis_row() -> dict:
    return _axis_row(
        "assort_hard:a001",
        "provisional",
        "gt_datasets_xlsx",
        agent_solve=True,
        agent_source="gt_datasets_xlsx",
        label="agent_solved",
        type="aggregation_count",
    )


def _hit_axis_row() -> dict:
    return _axis_row(
        "assort_hard:a001",
        "hit_only",
        "hits_jsonl",
        single_shot_hit=False,
        single_shot_basis="id_hit",
        single_shot_source="hits_jsonl",
        label="single_shot_miss",
    )


def test_dedupe_merges_complementary_axes_into_full_label(capsys):
    # The acceptance-gate path: an xlsx agent-axis row (provisional) plus
    # a deterministic single-shot miss (hit_only) must combine into ONE
    # agent_required row — not drop the single-shot axis.
    (merged,) = _dedupe([_agent_axis_row(), _hit_axis_row()])
    assert merged["label"] == "agent_required"
    assert merged["single_shot_hit"] is False
    assert merged["agent_solve"] is True
    assert merged["tier"] == "provisional"  # the agent-axis tier is kept
    assert merged["source"] == "gt_datasets_xlsx"
    # per-axis provenance survives the merge
    assert merged["single_shot_source"] == "hits_jsonl"
    assert merged["single_shot_basis"] == "id_hit"
    assert merged["agent_source"] == "gt_datasets_xlsx"
    assert merged["type"] == "aggregation_count"
    assert "merged complementary axis" in capsys.readouterr().err
    # order-independent: hits row first merges into the same full label
    (merged2,) = _dedupe([_hit_axis_row(), _agent_axis_row()])
    assert merged2["label"] == "agent_required"
    assert merged2["tier"] == "provisional"
    assert merged2["single_shot_source"] == "hits_jsonl"


def test_dedupe_same_axis_conflict_keeps_winner_value():
    # Same axis on both rows: the higher tier's value stands, no merge.
    confirmed = _axis_row(
        "finreg:q000",
        "confirmed",
        "finreg_jsonl",
        single_shot_hit=True,
        single_shot_basis="judge",
        single_shot_source="finreg_jsonl",
        agent_solve=False,
        agent_source="finreg_jsonl",
        label="cheap_sufficient",
    )
    hits = _axis_row(
        "finreg:q000",
        "hit_only",
        "hits_jsonl",
        single_shot_hit=False,
        single_shot_basis="id_hit",
        single_shot_source="hits_jsonl",
        label="single_shot_miss",
    )
    (row,) = _dedupe([confirmed, hits])
    assert row["single_shot_hit"] is True
    assert row["single_shot_basis"] == "judge"
    assert row["label"] == "cheap_sufficient"


# --- Split --------------------------------------------------------------


def _split_rows(corpus: str, n: int) -> list[dict]:
    return [{"qid": f"{corpus}:q{i:03d}", "source_qid": f"q{i:03d}", "corpus": corpus} for i in range(n)]


def test_split_balance_is_approximately_50_50():
    # Hash parity gives a binomial, not exact, balance — at 200 qids both
    # splits must hold a substantial share (deterministic given the qids).
    rows = _split_rows("finreg", 200)
    assign_splits(rows)
    n_train = sum(r["split"] == "train" for r in rows)
    assert 70 <= n_train <= 130


def test_split_deterministic_and_order_independent():
    rows_a = _split_rows("finreg", 9)
    rows_b = list(reversed(_split_rows("finreg", 9)))
    assign_splits(rows_a)
    assign_splits(rows_b)
    split_a = {r["source_qid"]: r["split"] for r in rows_a}
    split_b = {r["source_qid"]: r["split"] for r in rows_b}
    assert split_a == split_b


def test_split_membership_stable_under_qid_set_growth():
    # Parity depends only on the qid itself: adding queries to a corpus
    # (an OOM rerun, a fuller T2 JSONL) must never move an existing qid
    # between train and heldout — that would leak heldout qids into
    # tuning across GT versions.
    small = _split_rows("finreg", 10)
    grown = _split_rows("finreg", 30)  # superset of the same qids
    assign_splits(small)
    assign_splits(grown)
    grown_by_qid = {r["source_qid"]: r["split"] for r in grown}
    for r in small:
        assert grown_by_qid[r["source_qid"]] == r["split"]


def test_split_uses_both_splits_even_on_small_corpora():
    # Sanity guard: parity should not collapse a typical bench (~30q)
    # into a single split.
    rows = _split_rows("krra_conversational", 30)
    assign_splits(rows)
    splits = {r["split"] for r in rows}
    assert splits == {"train", "heldout"}


def test_assort_easy_entirely_heldout():
    rows = _split_rows("assort", 15)
    assign_splits(rows)
    assert all(r["split"] == "heldout" for r in rows)


# --- build_routing_gt end-to-end ----------------------------------------


def test_build_writes_one_row_per_qid_and_is_deterministic(
    tmp_path: Path, finreg_jsonl: Path, krra_graph_json: Path
):
    autorag = _write_jsonl(
        tmp_path / "ar.jsonl", [{"qid": "ar0001", "hit": True}, {"qid": "ar0002", "hit": False}]
    )
    out1, out2 = tmp_path / "gt1.jsonl", tmp_path / "gt2.jsonl"
    kwargs = dict(
        finreg_jsonl=[finreg_jsonl],
        gt_xlsx=None,
        agent_log=None,
        krra_graph=krra_graph_json,
        hits_jsonl=[("autorag", autorag)],
    )
    rows = build_routing_gt(out1, **kwargs)
    build_routing_gt(out2, **kwargs)
    assert out1.read_bytes() == out2.read_bytes()  # same inputs -> same bytes
    qids = [r["qid"] for r in rows]
    assert len(qids) == len(set(qids)) == 9  # 5 finreg + 2 krra_graph + 2 autorag
    parsed = [json.loads(line) for line in out1.read_text(encoding="utf-8").splitlines()]
    assert parsed == rows
    assert all(r["split"] in ("train", "heldout") for r in rows)
    assert all(r["tier"] in ("confirmed", "provisional", "hit_only", "unmeasured") for r in rows)


def test_build_axis_merge_end_to_end(tmp_path: Path):
    # xlsx agent axis (provisional, agent_solve=True) + per-corpus hits
    # file (single_shot_hit=False) -> one agent_required row. This is the
    # path that makes the plan's assort Hard acceptance gate reachable.
    xlsx = _make_xlsx(tmp_path)
    log = tmp_path / "agent.log"
    log.write_text(
        "  assort Hard (agent, max 5 turns)...       [a001] turns=2 found=5 hit=True (id)\n",
        encoding="utf-8",
    )
    hits = _write_jsonl(tmp_path / "ah_hits.jsonl", [{"qid": "a001", "hit": False}])
    rows = build_routing_gt(
        None,
        gt_xlsx=xlsx,
        agent_log=log,
        hits_jsonl=[("assort_hard", hits)],
    )
    by_qid = {r["qid"]: r for r in rows}
    merged = by_qid["assort_hard:a001"]
    assert merged["label"] == "agent_required"
    assert merged["tier"] == "provisional"
    assert merged["single_shot_basis"] == "id_hit"
    assert merged["single_shot_source"] == "hits_jsonl"
    assert merged["agent_source"] == "gt_datasets_xlsx"
    assert merged["type"] == "aggregation_count"  # xlsx metadata kept
    # the qid appears exactly once
    assert sum(1 for r in rows if r["qid"] == "assort_hard:a001") == 1


def test_build_with_no_existing_sources_yields_empty_table(tmp_path: Path, capsys):
    out = tmp_path / "gt.jsonl"
    rows = build_routing_gt(
        out,
        finreg_jsonl=[tmp_path / "nope.jsonl"],
        gt_xlsx=tmp_path / "nope.xlsx",
        agent_log=None,
        krra_graph=tmp_path / "nope.json",
        hits_jsonl=[("autorag", tmp_path / "nope2.jsonl")],
    )
    assert rows == []
    assert out.exists()
    assert capsys.readouterr().err.count("skipped") >= 3


def test_cli_main_deterministic(tmp_path: Path, finreg_jsonl: Path, krra_graph_json: Path):
    hits = _write_jsonl(tmp_path / "ah.jsonl", [{"qid": "a001", "hit": False}])
    out1, out2 = tmp_path / "cli1.jsonl", tmp_path / "cli2.jsonl"
    for out in (out1, out2):
        rc = main(
            [
                "--finreg-jsonl",
                str(finreg_jsonl),
                "--gt-xlsx",
                "",
                "--agent-log",
                "",
                "--krra-graph",
                str(krra_graph_json),
                "--hits-jsonl",
                f"assort_hard={hits}",
                "--out",
                str(out),
            ]
        )
        assert rc == 0
    assert out1.read_bytes() == out2.read_bytes()
    rows = [json.loads(line) for line in out1.read_text(encoding="utf-8").splitlines()]
    assert any(r["qid"] == "assort_hard:a001" and r["tier"] == "hit_only" for r in rows)


def test_cli_rejects_malformed_hits_spec(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["--hits-jsonl", "no-corpus-separator.jsonl", "--out", str(tmp_path / "o.jsonl")])


# --- Real repo data (extra validation, skipped when absent) -------------


@pytest.mark.skipif(not DEFAULT_AGENT_LOG.exists(), reason="v0.17.1 agent log not present")
def test_real_agent_log_parses_all_six_benches():
    axis = load_agent_log_axis(DEFAULT_AGENT_LOG)
    assert len(axis) == 172  # v0.17.1 multi-turn bench total
    corpora = {c for c, _ in axis}
    assert corpora == {
        "krra_hard",
        "assort_hard",
        "x2bee_hard",
        "krra_conversational",
        "assort_conversational",
        "x2bee_conversational",
    }
    # assort Hard ran 33/40 (the 7 OOM queries are absent -> stay unmeasured)
    assert sum(1 for c, _ in axis if c == "assort_hard") == 33


@pytest.mark.skipif(not DEFAULT_GT_XLSX.exists(), reason="gt_datasets.xlsx not present")
def test_real_xlsx_enumerates_types_from_data():
    pytest.importorskip("openpyxl")
    rows = load_gt_datasets_source(DEFAULT_GT_XLSX, None)
    assert len(rows) >= 200
    assort_hard_types = {r["type"] for r in rows if r["corpus"] == "assort_hard" and r["type"]}
    # enumerated from the sheet — the plan calls out 18 distinct types
    assert len(assort_hard_types) == 18
    assert all(r["tier"] == "unmeasured" for r in rows)  # no agent log joined


@pytest.mark.skipif(
    not (DEFAULT_GT_XLSX.exists() and DEFAULT_AGENT_LOG.exists()),
    reason="gt_datasets.xlsx or v0.17.1 agent log not present",
)
def test_real_xlsx_agent_log_join_attaches_all_entries(capsys):
    # Locks the join itself, not just the two sides: every one of the 172
    # log entries must land on an xlsx row (tier=provisional), and the
    # zero-join guard must stay silent.
    pytest.importorskip("openpyxl")
    rows = load_gt_datasets_source(DEFAULT_GT_XLSX, DEFAULT_AGENT_LOG)
    provisional = [r for r in rows if r["tier"] == "provisional"]
    assert len(provisional) == 172
    assert all(r["agent_solve"] is not None for r in provisional)
    assert "joined" not in capsys.readouterr().err
