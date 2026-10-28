"""Slide-ready figure: recall and false positives per mechanism, flooding off vs on.

    uv run python -m whistle.plot results/<run>
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {False: "#2a78d6", True: "#eb6834"}  # flooding off / on
MECH_ORDER = ["unfiltered", "triaged", "structured"]


def load(run):
    rows = list(csv.DictReader(open(Path(run) / "summary.csv")))
    for r in rows:
        r["flood"] = r["flood"] == "True"
        for k in ("recall", "recall_lo", "recall_hi", "precision", "mean_fp"):
            r[k] = float(r[k])
    return rows


def plot(run):
    rows = load(run)
    for reviewer in sorted({r["reviewer"] for r in rows}):
        sub = [r for r in rows if r["reviewer"] == reviewer]
        mechs = [m for m in MECH_ORDER if any(r["mechanism"] == m for r in sub)]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
        panels = [("recall", "Violation escalated (recall)", (0, 1.05)),
                  ("mean_fp", "False-positive escalations per scenario", None)]
        for ax, (metric, title, ylim) in zip(axes, panels):
            ax.set_facecolor(SURFACE)
            w = 0.36
            for j, flood in enumerate((False, True)):
                xs, ys, errs = [], [], [[], []]
                for i, m in enumerate(mechs):
                    r = next((r for r in sub if r["mechanism"] == m and r["flood"] == flood), None)
                    if not r:
                        continue
                    xs.append(i + (j - 0.5) * (w + 0.02))
                    ys.append(r[metric])
                    if metric == "recall":
                        errs[0].append(r["recall"] - r["recall_lo"])
                        errs[1].append(r["recall_hi"] - r["recall"])
                ax.bar(xs, ys, w, color=COLORS[flood], label=f"flooding {'on' if flood else 'off'}", zorder=2)
                if metric == "recall":
                    ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor=MUTED, elinewidth=1, capsize=3, zorder=3)
                for x, y in zip(xs, ys):
                    ax.text(x, y + 0.02 * (ylim[1] if ylim else max(ys + [1])), f"{y:.2f}",
                            ha="center", va="bottom", fontsize=8, color=MUTED)
            ax.set_xticks(range(len(mechs)), mechs, color=INK)
            ax.set_title(title, color=INK, fontsize=11, loc="left")
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(MUTED)
            ax.tick_params(colors=MUTED, length=0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, labelcolor=INK, fontsize=9, loc="upper right", ncol=2)
        fig.suptitle(f"LLM reviewer: {reviewer}", color=MUTED, fontsize=9, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        name = reviewer.replace("/", "_").replace(":", "_")
        path = Path(run) / f"results_{name}.png"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f"wrote {path}")


if __name__ == "__main__":
    plot(sys.argv[1])
