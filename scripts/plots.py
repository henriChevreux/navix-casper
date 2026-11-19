"""Figures for CASPER-k vs plain masking vs masking + penalty (probability space).

    python scripts/plots.py [--out results] [--fig-dir results/figures]

Reads results/{tune,confirm,ablation}/<cell>/<method>/*.npz and writes, as PNG and PDF:
  fig1_frontier      success (layer on) vs violation (layer removed), one panel per setting.
                     Penalty strengths are joined in lambda order; points on the confirm seeds
                     are solid, tuning-seed points (other seeds) are faint. Up-left is better.
  fig2_outcomes      per-seed success and violation for none / mask / selected penalty / CASPER.
  fig3_curves        raw-policy violation during training (rule-active steps), mean +/- SE.
DistShift is left out of the violation figures: its fixed layout means most seeds' layer-removed
rollouts never face lava (violation is vacuous); its success numbers are in verdict.md.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference palette (dataviz skill), first three categorical slots: validated all-pairs.
C = dict(casper="#2a78d6", mask="#eb6834", pen="#1baf7a", none="#8a8984")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LABEL = dict(casper="CASPER-k", mask="mask", pen="mask + penalty", none="no shield")

CELLS = [  # (cell, panel title)
    ("LavaGapS7_a3k2_eps0.05", "LavaGap · k=2 · ε=.05"),
    ("LavaGapS7_a3k2_eps0.15", "LavaGap · k=2 · ε=.15"),
    ("LavaGapS7_a3k3_eps0.05", "LavaGap · k=3 · ε=.05"),
    ("Dynamic-Obstacles-6x6_a3k2_eps0.05", "Dynamic-Obstacles · k=2 · ε=.05"),
    ("Dynamic-Obstacles-6x6_a3k2_eps0.15", "Dynamic-Obstacles · k=2 · ε=.15"),
    ("Dynamic-Obstacles-6x6_a3k3_eps0.05", "Dynamic-Obstacles · k=3 · ε=.05"),
]
BUDGET = 5_000_000


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200, "figure.facecolor": "white",
    })


def load(root: Path):
    """{cell: {method: dict(seeds, success, viol, curve_viol)}}"""
    out = defaultdict(dict)
    for npz in sorted(root.glob("*/*/*.npz")):
        cell, method = npz.parent.parent.name, npz.parent.name
        d = np.load(npz)
        n_con = d["eval_raw_greedy/n_constrained"]
        viol = np.where(n_con > 0, d["eval_raw_greedy/raw_argmax_viol"].astype(float), np.nan)
        out[cell][method] = dict(seeds=d["seeds"], success=d["eval_shield/success_rate"].astype(float),
                                 viol=viol, curve_viol=d["curve/raw_argmax_viol"].astype(float),
                                 n_con_curve=d["curve/n_constrained"].astype(float))
    return out


def family(m):
    return "pen" if m.startswith("pen") else "casper" if m.startswith("casper") else m


def lam(m):
    """The tuned hyperparameter: penalty lambda, or CASPER's restoration weight."""
    return float(re.sub(r"^pen", "", m)) if m.startswith("pen") else float(m.split("_l")[-1])


PARAM = dict(pen="λ", casper="λr")  # labels: penalty strength, CASPER restoration weight
SELECTION = {}  # cell -> {family: method}, from results/tune/selection.json


def mse(x):
    x = x[np.isfinite(x)]
    return (np.mean(x), np.std(x) / np.sqrt(len(x))) if len(x) else (np.nan, np.nan)


def save(fig, fig_dir: Path, name):
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", fig_dir / f"{name}.png")


# --------------------------------------------------------------------------- fig 1
def fig_frontier(conf, tune, fig_dir):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), sharex=True)
    for idx, (ax, (cell, title)) in enumerate(zip(axes.flat, CELLS)):
        runs = conf.get(cell, {})
        k3 = idx % 3 == 2
        ylim = (0.79, 1.025) if k3 else (-0.04, 1.06)
        placed = []  # annotation anchors already used in this panel (axes fraction)

        def frac(x, y):
            return (x + 0.03) / 1.03, (y - ylim[0]) / (ylim[1] - ylim[0])

        def label(text, x, y, dx, dy, **kw):
            fx, fy = frac(x, y)
            if any(abs(fx - a) < 0.14 and abs(fy - b) < 0.08 for a, b in placed):
                return  # would collide with a marker or another label: the legend/curve carries it
            placed.append((fx, fy))
            ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, **kw)

        # single markers first, registered so curve labels keep clear of them
        for m, marker, fam in (("none", "^", "none"), ("mask", "s", "mask"), ("casper_g0_l0", "o", "casper")):
            if m not in runs:
                continue
            (vx, vs), (sy, ss) = mse(runs[m]["viol"]), mse(runs[m]["success"])
            if not np.isfinite(vx):
                continue
            hollow = m == "casper_g0_l0"
            ax.errorbar(vx, sy, xerr=vs, yerr=ss, fmt=marker, ms=9 if fam == "casper" and not hollow else 8,
                        color=C[fam], mfc="white" if hollow else C[fam], mec="white" if not hollow else C[fam],
                        mew=1.6, elinewidth=1, capsize=0, zorder=5)
            if hollow:
                label("restoration off", vx, sy, 0, 9, ha="center", color=INK2)
            else:
                placed.append(frac(vx, sy))

        # tuned families as curves over their hyperparameter: confirm seeds (solid) first so their
        # labels win, then tuning seeds (faint). CASPER with restoration off is the ablation marker above.
        for fam in ("casper", "pen"):
            for src, alpha, ls, filled in ((runs, 1.0, "-", True), (tune.get(cell, {}), 0.45, "--", False)):
                ms_ = sorted((m for m in src if family(m) == fam and m != "casper_g0_l0"), key=lam)
                pts = [(mse(src[m]["viol"]), mse(src[m]["success"]), lam(m), m) for m in ms_]
                pts = [p for p in pts if np.isfinite(p[0][0])]
                if not pts:
                    continue
                xs, ys = [p[0][0] for p in pts], [p[1][0] for p in pts]
                ax.plot(xs, ys, ls, color=C[fam], lw=2, alpha=alpha, zorder=2 if fam == "pen" else 4)
                for (vx, vs), (sy, ss), l, m in pts:
                    big = fam == "casper" and filled
                    ax.errorbar(vx, sy, xerr=vs, yerr=ss, fmt="o", ms=9 if big else 7, color=C[fam], alpha=alpha,
                                mfc=C[fam] if filled else "white", mec="white" if filled else C[fam], mew=1.6,
                                elinewidth=1, capsize=0, zorder=3 if fam == "pen" else 5)
                    if filled and SELECTION.get(cell, {}).get(fam) == m:  # ring the selected config
                        ax.plot(vx, sy, "o", ms=15, mfc="none", mec=INK, mew=1, zorder=6)
                    label(f"{PARAM[fam]}={l:g}", vx, sy, 5, -10 if filled else 6, color=INK2,
                          alpha=1 if filled else 0.75)
        ax.set_title(title + ("  (y zoomed)" if k3 else ""), color=INK, loc="left")
        ax.set_xlim(-0.03, 1.0)
        ax.set_ylim(*ylim)
    for ax in axes[-1]:
        ax.set_xlabel("violation with the layer removed  (lower is better →←)".replace(" →←", ""))
    for ax in axes[:, 0]:
        ax.set_ylabel("task success with the layer")
    axes[0, 0].annotate("better", xy=(0.02, 1.0), xytext=(0.2, 0.82), fontsize=8, color=INK2,
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    handles = [
        plt.Line2D([], [], marker="o", ms=9, color=C["casper"], mec="white", ls="-", lw=2,
                   label="CASPER-k (confirm seeds)"),
        plt.Line2D([], [], marker="o", ms=7, color=C["casper"], mfc="white", ls="--", lw=2, alpha=0.5,
                   label="CASPER-k (tuning seeds)"),
        plt.Line2D([], [], marker="o", ms=7, color=C["pen"], ls="-", lw=2, label="mask + penalty (confirm seeds)"),
        plt.Line2D([], [], marker="o", ms=7, color=C["pen"], mfc="white", ls="--", lw=2, alpha=0.5,
                   label="mask + penalty (tuning seeds)"),
        plt.Line2D([], [], marker="s", ms=8, color=C["mask"], mec="white", ls="", label="mask"),
        plt.Line2D([], [], marker="o", ms=8, color=C["casper"], mfc="white", ls="", label="CASPER-k, restoration off"),
        plt.Line2D([], [], marker="^", ms=8, color=C["none"], mec="white", ls="", label="no shield"),
        plt.Line2D([], [], marker="o", ms=11, mfc="none", mec=INK, ls="", label="selected by the rule"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.suptitle("Success vs. rule violation: CASPER-k and mask + penalty, each tuned over 3 settings",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.935, "Mean ± SE over 32 paired seeds (tuning-seed points: 16 other seeds). "
             "Violation = share of rule-active steps where the raw greedy policy picks the forbidden move. "
             "λ = penalty strength, λr = CASPER restoration weight.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, fig_dir, "fig1_frontier")


# --------------------------------------------------------------------------- fig 2
def selected(runs, cell):
    """none, mask, and the penalty and CASPER configs picked by the selection rule (tune phase)."""
    sel = {"none": "none", "mask": "mask", "casper": "casper_g0_l1", **SELECTION.get(cell, {})}
    if "pen" not in sel:
        pens = [m for m in runs if family(m) == "pen"]
        if pens:
            sel["pen"] = pens[0]
    return sel


def fig_outcomes(conf_only, fig_dir):
    fams = ["none", "mask", "pen", "casper"]
    fig, axes = plt.subplots(2, len(CELLS), figsize=(13, 5.6), sharey="row")
    rng = np.random.default_rng(0)
    for j, (cell, title) in enumerate(CELLS):
        runs = conf_only.get(cell, {})
        sel = selected(runs, cell)
        for i, metric in enumerate(("success", "viol")):
            ax = axes[i, j]
            for x, fam in enumerate(fams):
                m = sel.get(fam)
                if m not in runs:
                    continue
                v = runs[m][metric]
                v = v[np.isfinite(v)]
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=C[fam], alpha=0.35, lw=0, zorder=2)
                mu, se = mse(v)
                ax.errorbar(x, mu, yerr=se, fmt="_", ms=22, mew=2.5, color=C[fam], elinewidth=1.5, zorder=4)
            ax.set_xticks(range(len(fams)))
            pen_lbl = f"pen\nλ={lam(sel['pen']):g}" if "pen" in sel else "pen"
            cas_lbl = f"CASPER\nλr={lam(sel['casper']):g}"
            ax.set_xticklabels(["none", "mask", pen_lbl, cas_lbl] if i == 1 else [], fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(title.replace(" · ", "\n", 1), color=INK, loc="left", fontsize=8.5)
    axes[0, 0].set_ylabel("task success\n(layer on)  ↑")
    axes[1, 0].set_ylabel("violation\n(layer removed)  ↓")
    fig.suptitle("Per-seed outcomes on 32 fresh seeds: same execution, different learned policies",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.925, "Dots = seeds, bars = mean ± SE. The penalty λ and CASPER-k's restoration weight λr are each picked by "
             "the same pre-declared rule on separate tuning seeds. No shield on Dynamic-Obstacles never moves "
             "(0 success, hence 0 violation).", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, fig_dir, "fig2_outcomes")


# --------------------------------------------------------------------------- fig 3
def fig_curves(conf_only, fig_dir):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for ax, (cell, title) in zip(axes.flat, CELLS):
        runs = conf_only.get(cell, {})
        sel = selected(runs, cell)
        for fam in ("mask", "pen", "casper"):
            m = sel.get(fam)
            if m not in runs:
                continue
            c = runs[m]["curve_viol"]
            k = max(1, c.shape[1] // 60)  # smooth over ~1/60 of training
            c = np.stack([np.convolve(row, np.ones(k) / k, mode="valid") for row in c])
            mu, se = np.nanmean(c, 0), np.nanstd(c, 0) / np.sqrt(c.shape[0])
            x = np.linspace(0, BUDGET, len(mu)) / 1e6
            ax.fill_between(x, mu - se, mu + se, color=C[fam], alpha=0.18, lw=0)
            ax.plot(x, mu, color=C[fam], lw=2)
            label = {"mask": "mask", "pen": "pen", "casper": "CASPER-k"}[fam] + \
                ("" if fam == "mask" else f" {PARAM[fam]}={lam(m):g}")
            ax.annotate(label, (x[-1], mu[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK2, annotation_clip=False)
        ax.set_title(title, color=INK, loc="left")
        ax.set_ylim(-0.02, 1.0)
        ax.set_xlim(0, BUDGET / 1e6)
    for ax in axes[-1]:
        ax.set_xlabel("environment steps (M)")
    for ax in axes[:, 0]:
        ax.set_ylabel("raw policy picks forbidden move\n(share of rule-active steps)")
    fig.suptitle("During training, masking leaves the network's preference for the forbidden move intact",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.93, "CASPER-k and the penalty both train it away; the penalty pays for it in task success "
             "(Fig. 1-2). Raw argmax before the layer, training rollouts, mean ± SE over 32 seeds; "
             "executed violations are 0 for all three.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 0.96, 0.92))
    save(fig, fig_dir, "fig3_curves")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--fig-dir", default=None)
    args = ap.parse_args()
    root = Path(args.out)
    fig_dir = Path(args.fig_dir or root / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    style()
    sel_path = root / "tune" / "selection.json"
    if sel_path.exists():
        for cell, fams in json.loads(sel_path.read_text())["selected"].items():
            SELECTION[cell] = {("pen" if f == "mask_pen" else f): m for f, m in fams.items()}
    tune, conf_only, abl = load(root / "tune"), load(root / "confirm"), load(root / "ablation")
    conf = {c: {**conf_only.get(c, {}), **abl.get(c, {})} for c in set(conf_only) | set(abl)}
    fig_frontier(conf, tune, fig_dir)
    fig_outcomes(conf_only, fig_dir)
    fig_curves(conf_only, fig_dir)


if __name__ == "__main__":
    main()
