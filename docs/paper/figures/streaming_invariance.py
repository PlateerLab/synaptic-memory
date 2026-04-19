"""Figure: Empirical top-K set / order agreement between batch and
streaming ingest on Allganize RAG-ko (Section 3.4).

Reads ``examples/ablation/diagnostics/streaming_invariance.json``
produced by ``examples/ablation/streaming_experiment.py``.
Writes ``streaming_invariance.pdf`` and ``.png`` next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
INPUT = REPO_ROOT / "examples" / "ablation" / "diagnostics" / "streaming_invariance.json"


def main() -> None:
    with open(INPUT, encoding="utf-8") as f:
        r = json.load(f)

    total = r["n_queries"]
    bit_identical = r["bitwise_identical_topk"]
    same_set = r["same_set_different_order"]
    mismatched = r["set_mismatched"]
    top1 = r["top1_match"]

    fig, (ax_top, ax_mrr) = plt.subplots(
        1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )

    # Left: stacked bar showing top-10 agreement types
    shares = {
        "Bit-wise identical top-10": bit_identical / total,
        "Same set, different order": same_set / total,
        "Set mismatched (tie at rank 9-10)": mismatched / total,
    }
    left = 0.0
    colors = ["#2e7d32", "#c9a227", "#b71c1c"]
    for (label, pct), color in zip(shares.items(), colors):
        ax_top.barh(
            0,
            pct,
            left=left,
            color=color,
            edgecolor="black",
            label=f"{label} ({int(pct * total)}/{total}, {pct * 100:.1f} %)",
        )
        left += pct
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(-0.5, 0.5)
    ax_top.set_yticks([])
    ax_top.set_xlabel("Share of queries")
    ax_top.set_title(
        f"Top-10 agreement: batch vs. 10-step streaming  (n={total})",
        fontsize=10,
    )
    ax_top.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, fontsize=9
    )

    # Right: MRR side-by-side
    mrr_vals = [r["mrr_batch"], r["mrr_streaming"]]
    ax_mrr.bar(
        ["batch (Arm A)", "streaming (Arm B)"],
        mrr_vals,
        color=["#2b3a55", "#b0c4de"],
        edgecolor="black",
    )
    for i, v in enumerate(mrr_vals):
        ax_mrr.annotate(
            f"{v:.4f}",
            xy=(i, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            weight="bold",
        )
    ax_mrr.set_ylim(0, 1.0)
    ax_mrr.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax_mrr.set_axisbelow(True)
    ax_mrr.set_ylabel("MRR @ 10")
    ax_mrr.set_title(f"MRR under streaming  (Δ = {r['mrr_delta']:+.4f})", fontsize=10)

    fig.tight_layout()
    fig.savefig(HERE / "streaming_invariance.pdf")
    fig.savefig(HERE / "streaming_invariance.png", dpi=150)
    print(f"Wrote {HERE / 'streaming_invariance.pdf'}")
    print(f"Wrote {HERE / 'streaming_invariance.png'}")


if __name__ == "__main__":
    main()
