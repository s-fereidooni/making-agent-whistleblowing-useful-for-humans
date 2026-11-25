"""Slide-ready figures from a run's summary files.

One flood size: bars of recall and false positives per mechanism, flooding off vs on.
Several flood sizes (--flood-n 2 4 8 ...): a dose-response figure, one line per mechanism.
If the run covers more than one difficulty tier, a second figure repeats the same panels
with one row per tier (difficulty_<reviewer>.png).

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
MECH_STYLE = {"unfiltered": ("#2a78d6", "o"), "triaged": ("#eb6834", "s"), "structured": ("#1baf7a", "^")}
TIER_ORDER = ["easy", "hard"]
PANELS = [("recall", "Violation escalated (recall, 95% CI)"), ("mean_fp", "False-positive escalations per scenario")]


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["flood"] = r["flood"] == "True"
        # runs from before the sweep existed have no flood_n column
        r["flood_n"] = int(r["flood_n"]) if r.get("flood_n") not in (None, "") else None
        r["n"] = int(r["n"])
        for k in ("recall", "recall_lo", "recall_hi", "precision", "mean_fp"):
            r[k] = float(r[k])
    return rows


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)


def _bars(ax, sub, metric, ymax):
    """Mechanisms on x, one bar each for flooding off and on."""
    mechs = [m for m in MECH_ORDER if any(r["mechanism"] == m for r in sub)]
    w = 0.36
    for j, flood in enumerate((False, True)):
        xs, ys, errs = [], [], [[], []]
        for i, m in enumerate(mechs):
            r = next((r for r in sub if r["mechanism"] == m and r["flood"] == flood), None)
            if not r:
                continue
            xs.append(i + (j - 0.5) * (w + 0.02))
            ys.append(r[metric])
            errs[0].append(r["recall"] - r["recall_lo"])
            errs[1].append(r["recall_hi"] - r["recall"])
        ax.bar(xs, ys, w, color=COLORS[flood], label=f"flooding {'on' if flood else 'off'}", zorder=2)
        if metric == "recall":
            ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor=MUTED, elinewidth=1, capsize=3, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.02 * ymax, f"{y:.2f}", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(mechs)), mechs, color=INK)


def _dose(ax, sub, metric, ymax):
    """Flood size on x (evenly spaced: sweeps are usually geometric), one line per mechanism."""
    levels = sorted({r["flood_n"] for r in sub})
    pos = {n: i for i, n in enumerate(levels)}
    mechs = [m for m in MECH_ORDER if any(r["mechanism"] == m for r in sub)]
    for j, m in enumerate(mechs):
        pts = sorted((r for r in sub if r["mechanism"] == m), key=lambda r: r["flood_n"])
        color, marker = MECH_STYLE[m]
        # small horizontal offset so overlapping lines and error bars stay readable
        xs = [pos[r["flood_n"]] + (j - (len(mechs) - 1) / 2) * 0.06 for r in pts]
        ys = [r[metric] for r in pts]
        if metric == "recall":
            err = [[r["recall"] - r["recall_lo"] for r in pts], [r["recall_hi"] - r["recall"] for r in pts]]
            ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=color, elinewidth=1, capsize=2, alpha=0.5, zorder=2)
        ax.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=2, label=m,
                markeredgecolor=SURFACE, markeredgewidth=1, clip_on=False, zorder=3)
    ax.set_xticks(range(len(levels)), [str(n) for n in levels])
    ax.set_xlabel("flood reports added by the violator (0 = no flooding)", color=MUTED, fontsize=9)


def figure(path, reviewer, groups):
    """One row of panels per group. `groups` maps a row heading ("" for none) to summary rows.

    Rows share their y-axis, so tiers can be compared directly.
    """
    every = [r for sub in groups.values() for r in sub]
    draw = _dose if len({r["flood_n"] for r in every}) > 2 else _bars
    ymax = {"recall": 1.05, "mean_fp": max([r["mean_fp"] for r in every] + [1.0]) * 1.15}
    fig, axes = plt.subplots(len(groups), 2, figsize=(10, 0.6 + 3.6 * len(groups)), facecolor=SURFACE, squeeze=False)
    for row, (heading, sub) in zip(axes, groups.items()):
        for ax, (metric, title) in zip(row, PANELS):
            _style_axis(ax)
            draw(ax, sub, metric, ymax[metric])
            ax.set_ylim(0, ymax[metric])
            n = sorted({r["n"] for r in sub})
            label = f"{heading}  (n = {n[0] if len(n) == 1 else f'{n[0]}-{n[-1]}'} per point)" if heading else ""
            ax.set_title(f"{title}\n{label}" if label else title, color=INK, fontsize=11, loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, labelcolor=INK, fontsize=9, loc="upper right", ncol=len(labels))
    fig.suptitle(f"LLM reviewer: {reviewer}", color=MUTED, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.24 / fig.get_figheight()))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def plot(run):
    run = Path(run)
    rows = load(run / "summary.csv")
    tiered = load(run / "summary_by_difficulty.csv") if (run / "summary_by_difficulty.csv").exists() else []
    for reviewer in sorted({r["reviewer"] for r in rows}):
        name = reviewer.replace("/", "_").replace(":", "_")
        sub = [r for r in rows if r["reviewer"] == reviewer]
        kind = "dose" if len({r["flood_n"] for r in sub}) > 2 else "results"
        figure(run / f"{kind}_{name}.png", reviewer, {"": sub})
        tiers = {r["difficulty"] for r in tiered if r["reviewer"] == reviewer}
        if len(tiers) > 1:
            order = [t for t in TIER_ORDER if t in tiers] + sorted(tiers - set(TIER_ORDER))
            groups = {f"{t} scenarios": [r for r in tiered if r["reviewer"] == reviewer and r["difficulty"] == t]
                      for t in order}
            figure(run / f"difficulty_{name}.png", reviewer, groups)


if __name__ == "__main__":
    plot(sys.argv[1])
