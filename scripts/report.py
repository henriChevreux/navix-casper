"""Report for scripts/experiment.py: a table and a trade-off figure.

    python scripts/report.py [--out results]    -> results/exp/report.md, results/exp/report.png (+ .pdf)

Metrics per method (mean ± SE over seeds; all methods use the same seeds):
  success     task success with the shield on (pure top-k sampling, 256 eval envs)
  violation   shield removed, greedy raw policy: share of rule-active steps where it picks the forbidden
              move. Seeds that never meet the rule are left out (n = seeds that did); on DistShift's fixed
              map most seeds never face lava, so its violation numbers rest on few seeds.
  failed / solved   seeds with success < 0.1 / >= 0.9 (outcomes are mostly bimodal per seed)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

TASKS = ["LavaGap", "DynamicObstacles", "DistShift"]
FAMILY = [("mask", "masking", "#eb6834"), ("pen", "masking + penalty", "#1baf7a"), ("casper", "CASPER-k", "#2a78d6")]


def load(path):
    d = np.load(path)
    n = d["eval_raw/n_rule_active"]
    return dict(s=d["eval_shield/success_rate"].astype(float),
                v=np.where(n > 0, d["eval_raw/raw_viol"].astype(float), np.nan))


def mse(x):
    x = x[np.isfinite(x)]
    return (x.mean(), x.std() / np.sqrt(len(x)), len(x)) if len(x) else (np.nan, np.nan, 0)


def family(m):
    return re.sub(r"[\d.]+$", "", m)  # "pen0.1" -> "pen", "casper3" -> "casper", "mask" -> "mask"


def weight(m):
    return float(m[len(family(m)):] or 0)


def collect(root):
    data = {}
    for task in TASKS:
        for p in sorted((root / task).glob("*/*.npz")):
            data.setdefault(task, {})[p.parent.name] = load(p)
    return data


def table(data):
    lines = ["# CASPER-k vs masking + penalty vs masking", "",
             "Mean ± SE over seeds (same seeds for every method). λ = penalty weight, λr = CASPER restoration "
             "weight. Violation: shield removed, greedy raw policy, share of rule-active steps; n = seeds that "
             "met the rule.", ""]
    for task, runs in data.items():
        lines += [f"## {task}", "", "| method | success | violation (n) | failed / solved |", "|---|---|---|---|"]
        order = sorted(runs, key=lambda m: (["mask", "pen", "casper"].index(family(m)), weight(m)))
        for m in order:
            r = runs[m]
            s, se_s, n_s = mse(r["s"])
            v, se_v, n_v = mse(r["v"])
            name = {"mask": "masking"}.get(m) or (f"penalty λ={weight(m):g}" if m.startswith("pen")
                                                 else f"CASPER-k λr={weight(m):g}")
            vtxt = f"{v:.3f} ± {se_v:.3f} ({n_v})" if n_v else "– (0)"
            lines.append(f"| {name} | {s:.3f} ± {se_s:.3f} | {vtxt} | "
                         f"{int((r['s'] < 0.1).sum())} / {int((r['s'] >= 0.9).sum())} of {n_s} |")
        lines.append("")
    return "\n".join(lines)


def figure(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True})
    fig, axes = plt.subplots(1, len(data), figsize=(5.6 * len(data), 5.2), squeeze=False)
    for ax, (task, runs) in zip(axes[0], data.items()):
        for fam, label, col in FAMILY:
            ms = sorted((m for m in runs if family(m) == fam), key=weight)
            pts = [(mse(runs[m]["v"]), mse(runs[m]["s"]), weight(m)) for m in ms]
            pts = [p for p in pts if p[0][2]]
            if not pts:
                continue
            xs, ys = [p[0][0] for p in pts], [p[1][0] for p in pts]
            if fam == "mask":
                ax.errorbar(xs, ys, xerr=[p[0][1] for p in pts], yerr=[p[1][1] for p in pts], fmt="s", ms=10,
                            color=col, mec="white", label=label, zorder=3)
                continue
            ax.plot(xs, ys, "-", color=col, lw=2.5, zorder=2)
            ax.errorbar(xs, ys, xerr=[p[0][1] for p in pts], yerr=[p[1][1] for p in pts], fmt="o", ms=9,
                        color=col, mec="white", label=label, zorder=3)
            sym = "λ" if fam == "pen" else "λr"
            for (vx, _, _), (sy, _, _), w in pts:
                ax.annotate(f"{sym}={w:g}", (vx, sy), xytext=(6, 6 if fam == "casper" else -15),
                            textcoords="offset points", fontsize=10, color="#52514e")
        ax.set_xlim(-0.03, 1.0)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(task, loc="left")
        ax.set_xlabel("rule violation, shield removed  (lower is better)")
    axes[0, 0].set_ylabel("task success, shield on  (higher is better)")
    axes[0, 0].legend(loc="lower left", fontsize=10.5, frameon=False)
    fig.suptitle("Turning up each method's training signal (up and left is better)", x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    root = Path(ap.parse_args().out) / "exp"
    data = collect(root)
    if not data:
        raise SystemExit(f"no results in {root}")
    text = table(data)
    (root / "report.md").write_text(text + "\n")
    print(text)
    figure(data, root / "report.png")
    print(f"\nwritten {root / 'report.md'} and {root / 'report.png'}")


if __name__ == "__main__":
    main()
