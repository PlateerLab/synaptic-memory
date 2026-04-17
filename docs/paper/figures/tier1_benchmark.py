"""Figure: Tier-1 English multi-hop benchmarks (HotPotQA / MuSiQue /
2WikiMultihopQA) — Synaptic v0.16.0 embedder-free vs. HippoRAG2's
self-reported numbers.

Note: the two sets of bars measure different quantities (retrieval
vs. answer accuracy / different recall cuts), so this figure is
contextual — it shows the rough Pareto ballpark, not a head-to-head.

Writes ``tier1_benchmark.pdf`` / ``.png`` next to this file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent

# (dataset, synaptic_metric, synaptic_value, reference_metric, reference_value)
ROWS = [
    ("HotPotQA\ndev (500q)", "MRR@10", 0.784, "HippoRAG2 string acc.", 0.567),
    ("2WikiMultihopQA\ndev (500q)", "R@5", 0.501, "HippoRAG2 R@5", 0.904),
    ("MuSiQue-Ans\ndev (500q)", "R@5", 0.379, "HippoRAG2 R@5", 0.747),
]


def main() -> None:
    names = [r[0] for r in ROWS]
    syn_vals = [r[2] for r in ROWS]
    ref_vals = [r[4] for r in ROWS]
    syn_metric = [r[1] for r in ROWS]
    ref_metric = [r[3] for r in ROWS]

    x = list(range(len(ROWS)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ax.bar(
        [i - width / 2 for i in x],
        syn_vals,
        width,
        color="#2e7d32",
        edgecolor="#1b5e20",
        label="Synaptic v0.16.0, embedder-free (0 LLM calls at index)",
    )
    ax.bar(
        [i + width / 2 for i in x],
        ref_vals,
        width,
        color="#b0c4de",
        edgecolor="#2b3a55",
        label="HippoRAG2 self-reported (LLM-extracted KG)",
    )
    for i in range(len(ROWS)):
        ax.annotate(
            f"{syn_vals[i]:.3f}\n({syn_metric[i]})",
            xy=(i - width / 2, syn_vals[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#1b5e20",
            weight="bold",
        )
        ax.annotate(
            f"{ref_vals[i]:.3f}\n({ref_metric[i].split()[-1]})",
            xy=(i + width / 2, ref_vals[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#2b3a55",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score  (metric printed above each bar)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=8.5,
    )
    ax.set_title(
        "Tier-1 English multi-hop: embedder-free Synaptic vs. LLM-extracted KG\n"
        "(metrics differ — contextual comparison, not head-to-head)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(HERE / "tier1_benchmark.pdf")
    fig.savefig(HERE / "tier1_benchmark.png", dpi=150)
    print(f"Wrote {HERE / 'tier1_benchmark.pdf'}")
    print(f"Wrote {HERE / 'tier1_benchmark.png'}")


if __name__ == "__main__":
    main()
