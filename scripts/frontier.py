"""Analyse the sweep: trade-off curves, the pre-declared selection, and the paired confirmation.

    python scripts/frontier.py tune       # after `sweep.py tune`: tables, plots, selection.json
    python scripts/frontier.py confirm    # after `sweep.py confirm`: paired tests on fresh seeds

The two metrics (per seed, from the final evaluation):
  success    task success with the layer on (eval_shield/success_rate)
  violation  layer removed, greedy raw policy: fraction of rule-active steps where its
             action is forbidden (eval_raw_greedy/raw_argmax_viol). A seed whose
             layer-removed rollouts never meet the rule (n_constrained = 0) has no
             violation value: it is vacuous and excluded from violation means.

Selection rule, fixed before looking at the confirmation seeds (per cell and family):
  1. Candidates: configurations with mean violation <= TAU (default 0.05) and at most
     25% vacuous seeds.
  2. Pick the candidate with the highest mean success.
  3. If there is no candidate, pick the lowest mean violation (ties: higher success).
  Families: none, mask, mask_pen (lambda), casper (gamma, lambda).

Verdict (confirm phase, per cell): paired Wilcoxon (seeds are paired) of the selected
casper against the selected mask_pen and against mask, for both metrics, with Holm's
correction across cells. Overall:
  casper better  if it is significantly better on one metric and not significantly worse on the other
  casper worse   if the reverse holds
  tie            otherwise
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMILIES = ("none", "mask", "mask_pen", "casper")


def family(method: str) -> str:
    if method.startswith("pen"):
        return "mask_pen"
    if method.startswith("casper"):
        return "casper"
    return method


def load_phase(root: Path):
    """{cell: {method: per-seed dict}}"""
    out = defaultdict(dict)
    for npz in sorted(root.glob("*/*/*.npz")):
        cell, method = npz.parent.parent.name, npz.parent.name
        d = dict(np.load(npz))
        n_con = d.get("eval_raw_greedy/n_constrained")
        viol = d["eval_raw_greedy/raw_argmax_viol"].astype(float)
        vacuous = (n_con == 0) if n_con is not None else np.zeros_like(viol, bool)
        viol = np.where(vacuous, np.nan, viol)
        out[cell][method] = dict(seeds=d["seeds"], success=d["eval_shield/success_rate"].astype(float),
                                 viol=viol, vacuous=vacuous,
                                 lava=d["eval_raw_greedy/lava_rate"].astype(float))
    return out


def summary(r):
    s, v = r["success"], r["viol"]
    return dict(success=float(np.mean(s)), success_se=float(np.std(s) / np.sqrt(len(s))),
                viol=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                viol_se=float(np.nanstd(v) / np.sqrt(max(1, np.isfinite(v).sum()))),
                vacuous=float(np.mean(r["vacuous"])), failed=int((s < 0.1).sum()),
                solved=int((s >= 0.9).sum()), n=len(s))


def select(stats, tau):
    """{family: method} by the fixed rule."""
    chosen = {}
    for fam in FAMILIES:
        cands = {m: st for m, st in stats.items() if family(m) == fam}
        if not cands:
            continue
        ok = {m: st for m, st in cands.items()
              if np.isfinite(st["viol"]) and st["viol"] <= tau and st["vacuous"] <= 0.25}
        if ok:
            chosen[fam] = max(ok, key=lambda m: (ok[m]["success"], -ok[m]["viol"]))
        else:
            finite = {m: st for m, st in cands.items() if np.isfinite(st["viol"])} or cands
            chosen[fam] = min(finite, key=lambda m: (np.nan_to_num(finite[m]["viol"], nan=9), -finite[m]["success"]))
    return chosen


def dominates(a, b):
    """a Pareto-dominates b on (higher success, lower violation)."""
    if not (np.isfinite(a["viol"]) and np.isfinite(b["viol"])):
        return False
    return (a["success"] >= b["success"] and a["viol"] <= b["viol"]
            and (a["success"] > b["success"] or a["viol"] < b["viol"]))


def fmt(st):
    v = "vacuous" if not np.isfinite(st["viol"]) else f"{st['viol']:.3f}"
    return f"{st['success']:.3f} | {v} | {st['failed']}/{st['solved']} | {st['vacuous']:.2f}"


def plot(data, stats_by_cell, chosen_by_cell, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: no plot)")
        return
    cells = sorted(stats_by_cell)
    ncol = 3
    nrow = (len(cells) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    colors = dict(none="tab:gray", mask="tab:orange", mask_pen="tab:red", casper="tab:green")
    for ax, cell in zip(axes.ravel(), cells):
        stats = stats_by_cell[cell]
        for fam in FAMILIES:
            pts = [(m, st) for m, st in stats.items() if family(m) == fam and np.isfinite(st["viol"])]
            if not pts:
                continue
            xs = [st["viol"] for _, st in pts]
            ys = [st["success"] for _, st in pts]
            ax.scatter(xs, ys, color=colors[fam], label=fam, s=30, zorder=3)
            for m, st in pts:
                ax.annotate(m.replace("casper_", "c_"), (st["viol"], st["success"]), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
            front = sorted([p for p in pts if not any(dominates(q[1], p[1]) for q in pts)],
                           key=lambda p: p[1]["viol"])
            if len(front) > 1:
                ax.plot([p[1]["viol"] for p in front], [p[1]["success"] for p in front],
                        color=colors[fam], lw=1, alpha=0.7)
        for fam, m in chosen_by_cell.get(cell, {}).items():
            st = stats.get(m)
            if st is not None and np.isfinite(st["viol"]):
                ax.scatter([st["viol"]], [st["success"]], facecolors="none", edgecolors="k", s=120, zorder=4)
        ax.set_title(cell, fontsize=9)
        ax.set_xlabel("violation, layer removed (lower is better)")
        ax.set_ylabel("task success, layer on")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"plot: {path}")


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(np.where(np.isfinite(p), p, 2))
    adj = np.full_like(p, np.nan)
    running = 0.0
    m = np.isfinite(p).sum()
    for rank, i in enumerate(order):
        if not np.isfinite(p[i]):
            continue
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def tune(args):
    root = Path(args.out) / "tune"
    data = load_phase(root)
    if not data:
        raise SystemExit(f"no results in {root}")
    lines = [f"# Tuning phase ({root})", "",
             f"Selection rule: violation ≤ {args.tau}, ≤ 25% vacuous seeds, max success; else min violation.",
             "Columns: success | violation (layer removed) | failed/solved seeds | vacuous fraction.", ""]
    stats_by_cell, selected, dom = {}, {}, []
    for cell in sorted(data):
        stats = {m: summary(r) for m, r in data[cell].items()}
        stats_by_cell[cell] = stats
        chosen = select(stats, args.tau)
        selected[cell] = chosen
        lines += [f"## {cell}", "", "| method | success | violation | failed/solved | vacuous | selected |",
                  "|---|---|---|---|---|---|"]
        for m in sorted(stats, key=lambda m: (FAMILIES.index(family(m)), m)):
            mark = "**←**" if chosen.get(family(m)) == m else ""
            lines.append(f"| {m} | {fmt(stats[m])} | {mark} |")
        c, p = chosen.get("casper"), chosen.get("mask_pen")
        if c and p:
            pen_dom = [m for m, st in stats.items() if family(m) == "mask_pen" and dominates(st, stats[c])]
            cas_dom = [m for m, st in stats.items() if family(m) == "casper" and dominates(st, stats[p])]
            lines += ["", f"- Penalty configs that dominate the selected casper ({c}): {pen_dom or 'none'}",
                      f"- Casper configs that dominate the selected penalty ({p}): {cas_dom or 'none'}", ""]
            dom.append((cell, bool(pen_dom), bool(cas_dom)))

    # robustness: does one fixed hyperparameter work in every cell?
    lines += ["## One setting everywhere?", "",
              f"Per configuration, across cells: mean success, mean violation, cells meeting violation ≤ {args.tau}.", "",
              "| method | mean success | mean violation | cells ≤ tau |", "|---|---|---|---|"]
    per_method = defaultdict(list)
    for cell, stats in stats_by_cell.items():
        for m, st in stats.items():
            per_method[m].append(st)
    for m in sorted(per_method, key=lambda m: (FAMILIES.index(family(m)), m)):
        sts = per_method[m]
        v = [st["viol"] for st in sts if np.isfinite(st["viol"])]
        meet = sum(np.isfinite(st["viol"]) and st["viol"] <= args.tau for st in sts)
        lines.append(f"| {m} | {np.mean([st['success'] for st in sts]):.3f} | "
                     f"{np.mean(v) if v else float('nan'):.3f} | {meet}/{len(sts)} |")
    lines += ["", "## Dominance count (tuning means)", "",
              f"- cells where some penalty config dominates the selected casper: {sum(d[1] for d in dom)}/{len(dom)}",
              f"- cells where some casper config dominates the selected penalty: {sum(d[2] for d in dom)}/{len(dom)}"]
    text = "\n".join(lines)
    (root / "frontier.md").write_text(text + "\n")
    (root / "selection.json").write_text(json.dumps(dict(tau=args.tau, selected=selected), indent=2))
    print(text)
    print(f"\nwritten {root / 'frontier.md'} and {root / 'selection.json'}")
    plot(data, stats_by_cell, selected, root / "frontier.png", "Tuning seeds: success vs layer-removed violation")


def confirm(args):
    from scipy.stats import wilcoxon
    tune_root, root = Path(args.out) / "tune", Path(args.out) / "confirm"
    selected = json.loads((tune_root / "selection.json").read_text())["selected"]
    data = load_phase(root)
    if not data:
        raise SystemExit(f"no results in {root}")
    rows = []  # (cell, other, metric, mean diff, p)
    stats_by_cell, chosen_by_cell = {}, {}
    for cell in sorted(data):
        sel = dict(none="none", mask="mask", casper="casper_g0_l1", **selected.get(cell, {}))  # untuned: confirm only
        runs = data[cell]
        stats_by_cell[cell] = {m: summary(r) for m, r in runs.items()}
        chosen_by_cell[cell] = {f: m for f, m in sel.items() if m in runs}
        c = sel.get("casper")
        if c not in runs:
            continue
        for other_fam in ("mask_pen", "mask"):
            o = sel.get(other_fam)
            if o not in runs:
                continue
            a, b = runs[c], runs[o]
            common = np.intersect1d(a["seeds"], b["seeds"])
            ia, ib = np.isin(a["seeds"], common), np.isin(b["seeds"], common)
            for metric in ("success", "viol"):
                d = a[metric][ia] - b[metric][ib]
                ok = np.isfinite(d)
                p = wilcoxon(d[ok]).pvalue if ok.sum() >= 5 and np.any(d[ok] != 0) else np.nan
                rows.append((cell, other_fam, o, c, metric, float(np.nanmean(d)), p, int(ok.sum())))
    adj = np.full(len(rows), np.nan)
    for other_fam in ("mask_pen", "mask"):
        for metric in ("success", "viol"):
            idx = [i for i, r in enumerate(rows) if r[1] == other_fam and r[4] == metric]
            if idx:
                adj[idx] = holm([rows[i][6] for i in idx])
    lines = [f"# Confirmation phase ({root}): selected configs on fresh seeds", "",
             "Paired Wilcoxon, casper − other; Holm-corrected across cells per (comparison, metric).",
             f"Significance level {args.alpha}. Violation: lower is better.", ""]
    verdicts = defaultdict(list)
    for other_fam in ("mask_pen", "mask"):
        lines += [f"## casper vs {other_fam}", "",
                  "| cell | casper | other | metric | mean diff | n | p | p (Holm) |", "|---|---|---|---|---|---|---|---|"]
        for cell in sorted({r[0] for r in rows}):
            res = {}
            for i, r in enumerate(rows):
                if r[0] != cell or r[1] != other_fam:
                    continue
                _, _, o, c, metric, diff, p, n = r
                lines.append(f"| {cell} | {c} | {o} | {metric} | {diff:+.3f} | {n} | {p:.3g} | {adj[i]:.3g} |")
                sig = np.isfinite(adj[i]) and adj[i] < args.alpha
                better = (diff > 0) if metric == "success" else (diff < 0)
                res[metric] = "better" if sig and better else ("worse" if sig else "tie")
            if res:
                vals = list(res.values())
                if "better" in vals and "worse" not in vals:
                    v = "casper better"
                elif "worse" in vals and "better" not in vals:
                    v = "casper worse"
                else:
                    v = "tie / trade-off"
                verdicts[other_fam].append((cell, v, res))
        lines.append("")
        for cell, v, res in verdicts[other_fam]:
            lines.append(f"- {cell}: **{v}** (success {res.get('success')}, violation {res.get('viol')})")
        counts = defaultdict(int)
        for _, v, _ in verdicts[other_fam]:
            counts[v] += 1
        lines += ["", f"Overall vs {other_fam}: " + ", ".join(f"{k}: {n}" for k, n in sorted(counts.items())), ""]
    lines += ["## Selected configurations on fresh seeds", "",
              "| cell | method | success | violation | failed/solved | vacuous |", "|---|---|---|---|---|---|"]
    for cell in sorted(stats_by_cell):
        for m, st in sorted(stats_by_cell[cell].items(), key=lambda kv: FAMILIES.index(family(kv[0]))):
            lines.append(f"| {cell} | {m} | {fmt(st)} |")
    text = "\n".join(lines)
    (root / "verdict.md").write_text(text + "\n")
    print(text)
    print(f"\nwritten {root / 'verdict.md'}")
    plot(data, stats_by_cell, chosen_by_cell, root / "frontier.png", "Fresh seeds: selected configurations")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["tune", "confirm"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--tau", type=float, default=0.05, help="violation threshold of the selection rule")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    (tune if args.phase == "tune" else confirm)(args)


if __name__ == "__main__":
    main()
