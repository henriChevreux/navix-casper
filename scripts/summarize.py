"""Summarise one or more results/<tag> directories: table (mean ± s.e.m. over seeds),
paired tests of casper against every other shield (seeds are paired: same init and
env resets), and a plot. Several directories are merged into one table, e.g. a base
run plus extra baselines trained later with the same seeds:

    python scripts/summarize.py results/LavaGapS7_k2
    python scripts/summarize.py results/DistShift1_k2 results/DistShift1_k2_pen0.1 results/DistShift1_k2_pen1 results/DistShift1_k2_pen10

The summary and plot are written to the first directory.

Metrics:
  train success / lava   last 10% of updates; lava = lava falls per episode
  raw top-k viol         P[raw top-k(p) contains a forbidden action | rule active],
                         last 10% of training (the network with the layer removed)
  eval <mode>            final evaluation, pure top-k sampling (no exploration mix):
                         shield = with the layer; raw_topk / raw_greedy = layer removed
"""
import json
import sys
from pathlib import Path

import numpy as np

ROWS = [
    ("train success", lambda d: _late(d["curve/success_rate"])),
    ("train lava/ep", lambda d: _late(d["curve/lava_rate"])),
    ("raw top-k viol (train)", lambda d: _late(d["curve/raw_topk_viol"])),
    ("raw argmax viol (train)", lambda d: _late(d["curve/raw_argmax_viol"])),
    ("eval success (layer)", lambda d: d["eval_shield/success_rate"]),
    ("eval lava/ep (layer)", lambda d: d["eval_shield/lava_rate"]),
    ("eval success (no layer, top-k)", lambda d: d["eval_raw_topk/success_rate"]),
    ("eval lava/ep (no layer, top-k)", lambda d: d["eval_raw_topk/lava_rate"]),
    ("eval success (no layer, greedy)", lambda d: d["eval_raw_greedy/success_rate"]),
    ("eval lava/ep (no layer, greedy)", lambda d: d["eval_raw_greedy/lava_rate"]),
    ("½‖z−p‖² (train; a loss term for casper only)", lambda d: _late(d["curve/restoration"])),
]


def _late(curve, frac=0.1):
    n = max(1, int(curve.shape[1] * frac))
    return np.nanmean(curve[:, -n:], axis=1)


def _fmt(x):
    x = np.asarray(x, float)
    m, se = np.nanmean(x), np.nanstd(x) / np.sqrt(max(1, np.isfinite(x).sum()))
    return f"{m:.3f} ± {se:.3f}"


def _label(stem, cfg):
    if stem == "mask_pen":
        return f"mask_pen λ={cfg.get('penalty_lambda')}"
    if stem == "casper" and cfg.get("casper_gamma", 0.0) != 0.0:
        return f"casper γ={cfg.get('casper_gamma')}"
    return stem


def main():
    roots = [Path(a) for a in (sys.argv[1:] or ["results"])]
    root = roots[0]
    cfgs = {r: json.loads((r / "config.json").read_text()) if (r / "config.json").exists() else {} for r in roots}
    cfg = cfgs[root]
    data = {}
    for r in roots:
        for p in sorted(r.glob("*.npz")):
            label = _label(p.stem, cfgs[r])
            if label in data:
                label = f"{label} [{r.name}]"
            data[label] = dict(np.load(p))
    base = ("none", "mask", "casper")
    order = [s for s in base if s in data] + [s for s in data if s not in base]
    n_seeds = {s: len(data[s]["seeds"]) for s in order}
    lines = [f"# {' + '.join(r.name for r in roots)}",
             f"env {cfg.get('env_id')}, k = {cfg.get('k')}, budget {cfg.get('budget')} steps/seed, "
             f"seeds {n_seeds}, CASPER eps {cfg.get('casper_eps')}, gamma {cfg.get('casper_gamma')}, "
             f"lambda {cfg.get('casper_lambda')}, explore eps {cfg.get('explore_eps')}", "",
             "| metric | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for name, fn in ROWS:
        lines.append(f"| {name} | " + " | ".join(_fmt(fn(data[s])) for s in order) + " |")

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    for other in [s for s in order if s not in ("casper", "none")]:
        if "casper" not in data:
            break
        lines += ["", f"Paired casper − {other} (same seeds), Wilcoxon signed-rank:", "",
                  "| metric | mean diff | casper better in | p |", "|---|---|---|---|"]
        common = np.intersect1d(data[other]["seeds"], data["casper"]["seeds"])
        for name, fn in ROWS[:-1]:
            a = fn(data["casper"])[np.isin(data["casper"]["seeds"], common)]
            b = fn(data[other])[np.isin(data[other]["seeds"], common)]
            d = a - b
            ok = np.isfinite(d)
            lower_better = "lava" in name or "viol" in name
            better = (d[ok] < 0) if lower_better else (d[ok] > 0)
            p = np.nan
            if wilcoxon is not None and ok.sum() >= 5 and np.any(d[ok] != 0):
                p = wilcoxon(d[ok]).pvalue
            lines.append(f"| {name} | {np.nanmean(d):+.3f} | {better.sum()}/{ok.sum()} seeds | {p:.3g} |")

    text = "\n".join(lines)
    print(text)
    (root / "summary.md").write_text(text + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: no plot)")
        return
    steps_per_update = cfg.get("num_envs", 64) * cfg.get("num_steps", 128)
    panels = [("success_rate", "training success rate"), ("lava_rate", "lava falls / episode (executed)"),
              ("raw_topk_viol", "raw top-k contains forbidden\n(rule active, layer removed)"),
              ("raw_argmax_viol", "raw argmax forbidden\n(rule active, layer removed)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    for ax, (key, title) in zip(axes, panels):
        for s in order:
            c = data[s][f"curve/{key}"]
            x = np.arange(c.shape[1]) * steps_per_update / 1e6
            m = np.nanmean(c, 0)
            se = np.nanstd(c, 0) / np.sqrt(c.shape[0])
            ax.plot(x, m, label=s)
            ax.fill_between(x, m - se, m + se, alpha=0.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("env steps per seed (M)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(root / "curves.png", dpi=150)
    print(f"\nwritten {root / 'summary.md'} and {root / 'curves.png'}")


if __name__ == "__main__":
    main()
