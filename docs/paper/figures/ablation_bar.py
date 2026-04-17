"""Figure: release-over-release MRR gain (v0.15.0 → v0.15.1 → v0.16.0)
across five public retrieval benchmarks.

Writes ``ablation_bar.pdf`` and ``.png`` next to this file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent

# (name, v0.15.0, v0.15.1, v0.16.0, lang)
DATASETS = [
    ("Allganize\nRAG-ko", 0.621, 0.743, 0.947, "ko"),
    ("Allganize\nRAG-Eval", 0.615, 0.695, 0.911, "ko"),
    ("PublicHealthQA\nKO", 0.318, 0.466, 0.546, "ko"),
    ("AutoRAG\nKO", 0.592, 0.692, 0.906, "ko"),
    ("HotPotQA-24\nEN", 0.727, 0.727, 0.875, "en"),
]


def main() -> None:
    names = [d[0] for d in DATASETS]
    v150 = [d[1] for d in DATASETS]
    v151 = [d[2] for d in DATASETS]
    v160 = [d[3] for d in DATASETS]

    x = list(range(len(DATASETS)))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(
        [i - width for i in x],
        v150,
        width,
        color="#b0c4de",
        edgecolor="#2b3a55",
        label="v0.15.0 (legacy)",
    )
    ax.bar(
        x,
        v151,
        width,
        color="#f7b32b",
        edgecolor="#8a6500",
        label="v0.15.1 (+ query-mode Kiwi)",
    )
    ax.bar(
        [i + width for i in x],
        v160,
        width,
        color="#2e7d32",
        edgecolor="#1b5e20",
        label="v0.16.0 (+ evidence engine)",
    )

    for i, v in enumerate(v160):
        delta = v - v150[i]
        if abs(delta) < 1e-3:
            continue
        ax.annotate(
            f"+{delta:.3f}",
            xy=(i + width, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#1b5e20",
            weight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("MRR @ 10  (embedder-free, no cross-encoder)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(
        "Release-over-release retrieval quality — five public benchmarks",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(HERE / "ablation_bar.pdf")
    fig.savefig(HERE / "ablation_bar.png", dpi=150)
    print(f"Wrote {HERE / 'ablation_bar.pdf'}")
    print(f"Wrote {HERE / 'ablation_bar.png'}")


if __name__ == "__main__":
    main()
