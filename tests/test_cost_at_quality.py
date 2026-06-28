"""Tests for the E4 cost-at-quality harness (examples/ablation/cost_at_quality.py).

What this file locks (PLAN-v0.29 §E4):

1. Oracle recombination — the oracle arm is computed from per-query
   JSONL records + GT labels with ZERO execution: agent_required picks
   the agent arm's cell, everything else picks the cheap arm's.
2. tokens/query arithmetic — per-qid mean over runs first, then pool
   mean; duplicate (qid, arm, run) records must not skew either.
3. Majority discipline — an arm's outcome is the strict >=2/3 majority
   over distinct runs; duplicated records of one run mint no extra runs.
4. Gate verdicts — pass AND fail cases for all three acceptance gates
   (quality / cost / separation), plus the three-valued honesty rules:
   unmeasured ask arm and <3 ask runs report NOT EVALUABLE, never PASS.
5. The live ask() path — one mock-client case driving run_ask_arm end
   to end (graph.ask routing + judge + JSONL output), no LLM/GPU.

All fixtures are synthetic; nothing here touches eval/data or a GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.ablation.cost_at_quality import (
    compute_arms,
    evaluate_gates,
    load_gold_queries,
    load_gt_labels,
    main,
    oracle_choice,
    render_markdown,
    run_ask_arm,
    summarize_cells,
    tag_records,
)

# --- fixture builders -------------------------------------------------


def _rec(
    qid: str,
    arm: str,
    run: int,
    correct: bool,
    *,
    prompt: int = 100,
    completion: int = 10,
    corpus: str | None = None,
) -> dict:
    """One record in the rag_vs_agent_answer.py --out-jsonl shape."""
    rec = {
        "qid": qid,
        "query": f"question {qid}",
        "arm": arm,
        "run": run,
        "judge_correct": correct,
        "empty": False,
        "elapsed_s": 1.0,
        "answer": "answer text",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }
    if corpus:
        rec["corpus"] = corpus
    return rec


RAG_TOK = (100, 10)  # 110 total/record — the cheap arm
AGENT_TOK = (1000, 100)  # 1100 total/record — the agent arm


def _three_run_records(qid: str, arm: str, correct: bool, tok: tuple[int, int]) -> list[dict]:
    return [_rec(qid, arm, r, correct, prompt=tok[0], completion=tok[1]) for r in (1, 2, 3)]


def _mixed_pool_records() -> tuple[list[dict], dict[str, str]]:
    """A 30-query mixed pool in which both degenerate routers lose.

    6 agent_required (rag misses, agent solves — the assort-Hard class),
    23 cheap_sufficient (rag solves, agent misses), 1 unsolved.
    always-RAG solves 23 (loses on quality where agent is required);
    always-agent solves 6 (loses on cost everywhere). 6 one-sided
    discordant pairs are the minimum for exact-McNemar p<=0.05, which is
    exactly why the plan demands agent-required mass in the pool.
    """
    records: list[dict] = []
    labels: dict[str, str] = {}
    for i in range(6):
        qid = f"ar{i:03d}"
        records += _three_run_records(qid, "rag", False, RAG_TOK)
        records += _three_run_records(qid, "agent", True, AGENT_TOK)
        labels[f"pool:{qid}"] = "agent_required"
    for i in range(23):
        qid = f"cs{i:03d}"
        records += _three_run_records(qid, "rag", True, RAG_TOK)
        records += _three_run_records(qid, "agent", False, AGENT_TOK)
        labels[f"pool:{qid}"] = "cheap_sufficient"
    records += _three_run_records("un000", "rag", False, RAG_TOK)
    records += _three_run_records("un000", "agent", False, AGENT_TOK)
    labels["pool:un000"] = "unsolved"
    return tag_records(records, "pool"), labels


def _ask_records_routing_like(labels: dict[str, str]) -> list[dict]:
    """An ask() arm that routes perfectly: agent cost only where the GT
    says agent_required, cheap cost everywhere else; solves the union."""
    records: list[dict] = []
    for nqid, label in labels.items():
        qid = nqid.split(":", 1)[1]
        if label == "agent_required":
            correct, tok = True, AGENT_TOK
        elif label in ("cheap_sufficient", "both"):
            correct, tok = True, RAG_TOK
        else:
            correct, tok = False, RAG_TOK
        for r in (1, 2, 3):
            records.append(
                _rec(qid, "ask", r, correct, prompt=tok[0], completion=tok[1], corpus="pool")
            )
    return records


# --- oracle recombination ---------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("agent_required", "agent"),
        ("cheap_sufficient", "rag"),
        ("both", "rag"),  # both solve -> the cheaper arm is optimal
        ("unsolved", "rag"),  # nothing solves -> take the cheap loss
        ("single_shot_miss", "rag"),  # partial label: no agent evidence
        (None, "rag"),
    ],
)
def test_oracle_choice(label, expected):
    assert oracle_choice(label) == expected


def test_oracle_recombines_per_query_records_without_execution():
    records, labels = _mixed_pool_records()
    arms, meta = compute_arms(summarize_cells(records), labels)
    oracle = arms["oracle"]
    # agent_required (6, via agent) + cheap_sufficient (23, via rag); the
    # unsolved query stays unsolved on either arm.
    assert oracle.solved == 29
    assert oracle.total == 30
    # outcome per class comes from the chosen arm's records
    assert oracle.outcomes["pool:ar000"] is True  # agent cell
    assert oracle.outcomes["pool:cs000"] is True  # rag cell
    assert oracle.outcomes["pool:un000"] is False
    # tokens come from the chosen arm too: 6x1100 + 24x110 over 30 queries
    assert oracle.tokens["pool:ar000"] == 1100.0
    assert oracle.tokens["pool:cs000"] == 110.0
    assert oracle.tokens_per_query == pytest.approx((6 * 1100 + 24 * 110) / 30)
    assert meta["label_source"] == "routing_gt"


def test_oracle_without_gt_derives_labels_from_pool_majorities():
    records, labels = _mixed_pool_records()
    with_gt, _ = compute_arms(summarize_cells(records), labels)
    derived, meta = compute_arms(summarize_cells(records), None)
    # The pool's own rag/agent majorities reproduce the same 2x2 labels,
    # so the oracle must be identical (this IS routing_gt's finreg rule).
    assert derived["oracle"].outcomes == with_gt["oracle"].outcomes
    assert derived["oracle"].tokens == with_gt["oracle"].tokens
    assert meta["label_source"] == "derived_from_pool"


def test_gt_label_missing_for_qid_falls_back_to_derived():
    records, labels = _mixed_pool_records()
    del labels["pool:ar000"]  # GT row lost — derived label must fill in
    arms, meta = compute_arms(summarize_cells(records), labels)
    assert meta["labels_missing_fallback"] == 1
    assert arms["oracle"].outcomes["pool:ar000"] is True  # still agent's cell
    assert arms["oracle"].tokens["pool:ar000"] == 1100.0


# --- tokens/query arithmetic -------------------------------------------


def test_tokens_per_query_is_per_qid_mean_then_pool_mean():
    records = tag_records(
        [
            # q1 rag tokens vary across runs: 110, 220, 330 -> mean 220
            _rec("q1", "rag", 1, True, prompt=100, completion=10),
            _rec("q1", "rag", 2, True, prompt=200, completion=20),
            _rec("q1", "rag", 3, True, prompt=300, completion=30),
            # q2 rag constant 110
            *_three_run_records("q2", "rag", True, RAG_TOK),
            *_three_run_records("q1", "agent", False, AGENT_TOK),
            *_three_run_records("q2", "agent", False, AGENT_TOK),
        ],
        "c",
    )
    arms, _ = compute_arms(summarize_cells(records), None)
    assert arms["rag"].tokens["c:q1"] == pytest.approx(220.0)
    assert arms["rag"].tokens["c:q2"] == pytest.approx(110.0)
    assert arms["rag"].tokens_per_query == pytest.approx((220 + 110) / 2)
    assert arms["agent"].tokens_per_query == pytest.approx(1100.0)


def test_missing_token_fields_count_as_zero():
    rec = _rec("q1", "rag", 1, True)
    del rec["prompt_tokens"], rec["completion_tokens"]
    cells = summarize_cells(tag_records([rec], "c"))
    assert cells[("c:q1", "rag")].mean_tokens == 0.0


# --- majority + dedupe discipline ---------------------------------------


def test_outcome_is_strict_majority_over_runs():
    records = tag_records(
        [
            _rec("q1", "rag", 1, True),
            _rec("q1", "rag", 2, True),
            _rec("q1", "rag", 3, False),  # 2/3 -> solved
            _rec("q2", "rag", 1, True),
            _rec("q2", "rag", 2, False),
            _rec("q2", "rag", 3, False),  # 1/3 -> not solved
        ],
        "c",
    )
    cells = summarize_cells(records)
    assert cells[("c:q1", "rag")].outcome is True
    assert cells[("c:q2", "rag")].outcome is False
    assert cells[("c:q1", "rag")].runs == 3


def test_duplicate_records_do_not_mint_runs_and_keep_last():
    records = tag_records(
        [
            _rec("q1", "rag", 1, False),
            _rec("q1", "rag", 1, True),  # same run resubmitted -> last wins
            _rec("q1", "rag", 2, False),
        ],
        "c",
    )
    cells = summarize_cells(records)
    # 2 distinct runs, wins {run1: True, run2: False} -> majority needs 2 -> not solved
    assert cells[("c:q1", "rag")].runs == 2
    assert cells[("c:q1", "rag")].outcome is False


def test_same_raw_qid_in_two_corpora_does_not_collide():
    records = tag_records([_rec("q1", "rag", 1, True)], "a") + tag_records(
        [_rec("q1", "rag", 1, False)], "b"
    )
    cells = summarize_cells(records)
    assert cells[("a:q1", "rag")].outcome is True
    assert cells[("b:q1", "rag")].outcome is False


def test_tag_records_requires_a_corpus():
    with pytest.raises(ValueError):
        tag_records([_rec("q1", "rag", 1, True)], None)


def test_unpaired_qids_are_dropped_and_counted():
    records = tag_records(
        [
            *_three_run_records("q1", "rag", True, RAG_TOK),
            *_three_run_records("q1", "agent", False, AGENT_TOK),
            *_three_run_records("q2", "rag", True, RAG_TOK),  # no agent arm
        ],
        "c",
    )
    arms, meta = compute_arms(summarize_cells(records), None)
    assert arms["rag"].total == 1
    assert meta["unpaired_dropped"] == 1


# --- gates ---------------------------------------------------------------


def test_gates_all_pass_with_routing_like_ask_arm():
    records, labels = _mixed_pool_records()
    records += _ask_records_routing_like(labels)
    arms, _ = compute_arms(summarize_cells(records), labels)
    gates = {g.name: g for g in evaluate_gates(arms)}

    # quality: ask 29 >= max(23, 6) - 2 = 21
    assert gates["quality"].passed is True
    # cost: (6*1100 + 24*110)/30 = 308 tok/q = 0.28x always-agent 1100
    assert gates["cost"].passed is True
    assert arms["ask"].tokens_per_query == pytest.approx(308.0)
    # separation: ask 29 > rag 23, +6/-0 discordant -> exact p = 0.03125
    assert gates["separation"].passed is True
    assert "p=0.03125" in gates["separation"].detail


def test_gates_all_fail_for_always_agent_like_ask_arm():
    # ask() that escalates everything = always-agent in disguise: loses
    # quality and separation on the cheap mass, busts the cost budget.
    records, labels = _mixed_pool_records()
    for nqid, label in labels.items():
        qid = nqid.split(":", 1)[1]
        correct = label == "agent_required"
        for r in (1, 2, 3):
            records.append(
                _rec(
                    qid,
                    "ask",
                    r,
                    correct,
                    prompt=AGENT_TOK[0],
                    completion=AGENT_TOK[1],
                    corpus="pool",
                )
            )
    arms, _ = compute_arms(summarize_cells(records), labels)
    gates = {g.name: g for g in evaluate_gates(arms)}
    assert gates["quality"].passed is False  # 6 < 23 - 2
    assert gates["cost"].passed is False  # 1.00x > 0.35x
    assert gates["separation"].passed is False  # 6 < 23


def test_separation_gate_needs_significance_not_just_a_bigger_count():
    # ask beats rag by ONE discordant pair -> exact McNemar p = 1.0:
    # a +1q lead inside the noise floor must not pass the keystone gate.
    records = tag_records(
        [
            *_three_run_records("q1", "rag", False, RAG_TOK),
            *_three_run_records("q1", "agent", True, AGENT_TOK),
            *_three_run_records("q2", "rag", True, RAG_TOK),
            *_three_run_records("q2", "agent", False, AGENT_TOK),
            *[_rec("q1", "ask", r, True, prompt=1000, completion=100) for r in (1, 2, 3)],
            *[_rec("q2", "ask", r, True, prompt=100, completion=10) for r in (1, 2, 3)],
        ],
        "c",
    )
    arms, _ = compute_arms(summarize_cells(records), None)
    gates = {g.name: g for g in evaluate_gates(arms)}
    assert arms["ask"].solved == 2 and arms["rag"].solved == 1
    assert gates["separation"].passed is False
    assert "p=1" in gates["separation"].detail


def test_gates_not_evaluable_without_ask_arm():
    records, labels = _mixed_pool_records()
    arms, _ = compute_arms(summarize_cells(records), labels)
    gates = evaluate_gates(arms)
    assert [g.passed for g in gates] == [None, None, None]
    assert all(g.verdict == "NOT EVALUABLE" for g in gates)


def test_multirun_gates_not_evaluable_below_three_ask_runs():
    # 1-run ask arm: cost (deterministic) still evaluates, but quality and
    # separation must refuse a verdict — single-run conclusions are banned.
    records, labels = _mixed_pool_records()
    for nqid, label in labels.items():
        qid = nqid.split(":", 1)[1]
        records.append(
            _rec(qid, "ask", 1, label != "unsolved", prompt=100, completion=10, corpus="pool")
        )
    arms, _ = compute_arms(summarize_cells(records), labels)
    gates = {g.name: g for g in evaluate_gates(arms)}
    assert gates["quality"].passed is None
    assert gates["separation"].passed is None
    assert gates["cost"].passed is True
    assert "min runs/query = 1" in gates["quality"].detail


# --- report + CLI --------------------------------------------------------


def test_render_markdown_contains_arms_gates_and_claim_scope():
    records, labels = _mixed_pool_records()
    records += _ask_records_routing_like(labels)
    arms, meta = compute_arms(summarize_cells(records), labels)
    gates = evaluate_gates(arms)
    md = render_markdown(arms, gates, meta)
    for display in ("always-RAG", "always-agent", "oracle-router", "ask()"):
        assert f"| {display} |" in md
    assert "0.28x" in md  # cost ratio vs always-agent
    assert "**separation**: PASS" in md
    assert "oracle gap" in md
    assert "claim scope" in md  # domain generalization NOT claimed


def test_main_offline_end_to_end(tmp_path, capsys):
    records, labels = _mixed_pool_records()
    perquery = tmp_path / "perquery.jsonl"
    with perquery.open("w", encoding="utf-8") as fh:
        for rec in records:
            rec = {k: v for k, v in rec.items() if k != "corpus"}  # E1 files carry no corpus
            fh.write(json.dumps(rec) + "\n")
    gt = tmp_path / "routing_gt.jsonl"
    with gt.open("w", encoding="utf-8") as fh:
        for nqid, label in labels.items():
            fh.write(json.dumps({"qid": nqid, "label": label}) + "\n")
    out = tmp_path / "report.md"

    rc = main(["--perquery", f"pool={perquery}", "--gt", str(gt), "--out", str(out)])
    assert rc == 0
    report = out.read_text(encoding="utf-8")
    assert "| always-RAG | 23/30 |" in report
    assert "| oracle-router | 29/30 |" in report
    assert "NOT EVALUABLE" in report  # no ask arm measured -> honest gates
    assert load_gt_labels(gt)["pool:ar000"] == "agent_required"


def test_main_dry_run_makes_no_llm_calls_and_reports_plan(tmp_path, capsys):
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            {
                "queries": [
                    {"query": "with gold", "answer": "gold"},
                    {"query": "no gold answer"},
                ]
            }
        ),
        encoding="utf-8",
    )
    rc = main(
        [
            "--run-live",
            "--dry-run",
            "--live-spec",
            f"toy={tmp_path / 'g.sqlite'},{queries}",
            "--runs",
            "3",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "1 gold-answer queries x 3 runs" in out
    assert "3 ask() calls" in out  # 1 gold query x 3 runs


# --- live ask() path (mock client — no LLM, no GPU) -----------------------


class _Msg:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None

    def model_dump(self):
        return {"role": "assistant", "content": self.content or ""}


class _Usage:
    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Resp:
    def __init__(self, content, usage=None):
        self.choices = [type("Ch", (), {"message": _Msg(content)})()]
        self.usage = usage


class _FakeClient:
    """Dispatch by call shape (test_graph_ask pattern + the answer judge):
    ``tools=`` -> agent turn, "evidence auditor" -> sufficiency judge,
    "You grade" -> correctness judge, anything else -> cheap synthesis."""

    def __init__(self):
        self.judge_calls = 0
        self.agent_calls = 0
        self.chat = self
        self.completions = self

    async def create(self, *, model, messages, tools=None, max_tokens=None, temperature=None):
        if tools is not None:
            self.agent_calls += 1
            return _Resp("agent answer", _Usage(500, 50))
        system = str(messages[0].get("content", "")) if messages else ""
        if "evidence auditor" in system:
            return _Resp('{"sufficient": true}', _Usage(30, 5))
        if system.startswith("You grade"):
            self.judge_calls += 1
            return _Resp("YES", _Usage(20, 1))
        return _Resp("cheap answer", _Usage(100, 10))


async def test_run_ask_arm_live_path_with_mock_client(tmp_path):
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.models import ConsolidationLevel, Node, NodeKind

    graph_path = tmp_path / "toy.sqlite"
    backend = SqliteGraphBackend(str(graph_path))
    await backend.connect()
    await backend.save_node(
        Node(
            id="d1",
            kind=NodeKind.CHUNK,
            title="topic",
            content="evidence about the topic",
            level=ConsolidationLevel.L0_RAW,
        )
    )
    await backend.close()

    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps(
            {
                "queries": [
                    {"query": "tell me about the topic", "answer": "the topic evidence"},
                    {"query": "skipped — no gold"},  # must be filtered out
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert [q["_qid"] for q in load_gold_queries(queries_path)] == ["q000"]

    client = _FakeClient()
    out_jsonl = tmp_path / "ask.jsonl"
    records = await run_ask_arm(
        [("toy", graph_path, queries_path)],
        client=client,
        model="m",
        runs=2,
        out_jsonl=out_jsonl,
    )

    assert len(records) == 2  # 1 gold query x 2 runs
    rec = records[0]
    assert rec["arm"] == "ask" and rec["corpus"] == "toy" and rec["qid"] == "q000"
    assert rec["route"] == "single_shot" and rec["escalated"] is False
    assert rec["judge_correct"] is True
    # ask-internal usage only: synthesis (100/10) + sufficiency judge (30/5);
    # the external correctness judge is measurement cost, NOT counted.
    assert rec["prompt_tokens"] == 130
    assert rec["completion_tokens"] == 15
    assert client.judge_calls == 2 and client.agent_calls == 0
    assert {r["run"] for r in records} == {1, 2}

    # JSONL written, one line per record, recombines into an ask cell
    lines = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert len(lines) == 2
    cells = summarize_cells(tag_records(lines, None))
    assert cells[("toy:q000", "ask")].outcome is True
    assert cells[("toy:q000", "ask")].runs == 2


async def test_run_ask_arm_skips_specs_without_gold_answers(tmp_path, capsys):
    queries_path = tmp_path / "no_gold.json"
    queries_path.write_text(json.dumps({"queries": [{"query": "q"}]}), encoding="utf-8")
    records = await run_ask_arm(
        [("nogold", tmp_path / "absent.sqlite", queries_path)],
        client=_FakeClient(),
        model="m",
        runs=1,
    )
    assert records == []
    assert "no queries with a gold" in capsys.readouterr().err
