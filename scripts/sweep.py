"""The CASPER-k vs masking + penalty comparison: a hyperparameter grid, run in two phases.

Phase 1, tune (seeds 0..15 by default): only the penalty is tuned (its lambdas, per cell).
none, mask and casper run at fixed settings in confirm only. CASPER gets no tuning,
which can only favour the penalty.
  python scripts/sweep.py tune --gpus 0,1,2

Selection (no GPU): pick each method family's configuration per cell by a fixed rule.
  python scripts/frontier.py tune

Phase 2, confirm (fresh seeds 100..131): only the selected configurations.
  python scripts/sweep.py confirm --gpus 0,1,2
  python scripts/frontier.py confirm

A cell is (environment, action setting, exploration eps). By default only the cells in
DESIGN are run (6 with two maps); --full-grid runs every setting x eps combination. Methods, all on the
output distribution (probability space):
  none                    no shield (reference)
  mask                    mask + renormalise (the lambda = 0 end of mask_pen)
  mask_pen  lambda        mask + lambda * (raw probability on forbidden actions)
  casper    gamma, lambda CASPER-k with gamma-tilted margins and restoration weight lambda

Every job is one `scripts/run.py` call (one method, all seeds of the phase,
vmapped on one GPU). Jobs are spread over one worker per GPU. The grid is
resumable: a job whose .npz already exists is skipped. Results go to
results/<phase>/<cell>/<method>/<shield>.npz.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENVS = ["Navix-LavaGapS7-v0", "Navix-DistShift1-v0", "Navix-LavaCrossingS9N1-v0"]
SETTINGS = {  # name: (full_action_set, k, budget)
    "a3k2": (False, 2, 5_000_000),
    "a3k3": (False, 3, 5_000_000),
    "a7k2": (True, 2, 20_000_000),
}
EXPLORE = [0.05, 0.15]
# Reduced design: eps = 0.15 only where seed lock-in matters (3 actions, k = 2), to test
# whether casper's k = 2 advantage is just extra exploration.
# a7k2 (7 actions, 20M) is left out by default: the task is mostly unlearned there.
# Add it with --settings a3k2 a3k3 a7k2.
DESIGN = [("a3k2", 0.05), ("a3k2", 0.15), ("a3k3", 0.05)]
PEN_LAMBDAS = [0.01, 0.1, 0.3]  # spans "too weak" to "task collapses" on DistShift
CASPER_GRID = [(0.0, 1.0)]  # (gamma, restoration lambda): the default, untuned
UNTUNED = ("none", "mask", "casper_g0_l1")  # fixed settings: confirm phase only
PHASES = {"tune": (0, 16), "confirm": (100, 32)}  # seed offset, number of seeds


def methods():
    """[(name, shield, extra flags)] for every method x hyperparameter."""
    out = [("none", "none", []), ("mask", "mask", [])]
    out += [(f"pen{l:g}", "mask_pen", ["--penalty-lambda", str(l)]) for l in PEN_LAMBDAS]
    out += [(f"casper_g{g:g}_l{l:g}", "casper", ["--casper-gamma", str(g), "--casper-lambda", str(l)])
            for g, l in CASPER_GRID]
    return out


def cell_name(env, setting, eps):
    return f"{env.replace('Navix-', '').replace('-v0', '')}_{setting}_eps{eps:g}"


def available_envs(envs):
    import navix as nx
    reg = nx.registry() if hasattr(nx, "registry") else {}
    ok = [e for e in envs if e in reg]
    for e in envs:
        if e not in reg:
            print(f"! skipping {e}: not in this NAVIX version ({getattr(nx, '__version__', '?')}); "
                  f"LavaCrossing needs NAVIX >= 0.8 (Python >= 3.12)", flush=True)
    return ok


def build_jobs(phase, envs, cells, out, selection=None, budget_scale=1.0, n_seeds=None, extra=()):
    seed_offset, default_seeds = PHASES[phase]
    n_seeds = n_seeds or default_seeds
    jobs = []
    for env, (setting, eps) in itertools.product(envs, cells):
        cell = cell_name(env, setting, eps)
        full, k, budget = SETTINGS[setting]
        budget = int(budget * budget_scale)
        chosen = None
        if phase == "confirm":
            chosen = set(selection.get(cell, {}).values())
            if not chosen:
                continue
            chosen |= set(UNTUNED)
        for name, shield, method_flags in methods():
            if chosen is not None and name not in chosen:
                continue
            if phase == "tune" and name in UNTUNED:
                continue
            tag = f"{phase}/{cell}/{name}"
            target = Path(out) / tag / f"{shield}.npz"
            cmd = [sys.executable, "scripts/run.py", "--env-id", env, "--shields", shield,
                   "--k", str(k), "--full-action-set", str(full).lower(), "--budget", str(budget),
                   "--explore-eps", str(eps), "--seeds", str(n_seeds), "--seed-offset", str(seed_offset),
                   "--tag", tag, "--out", str(out)] + method_flags + list(extra)
            jobs.append(dict(tag=tag, target=str(target), cmd=cmd, budget=budget))
    return jobs


def worker(gpu, q, logdir, lock, status):
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   PYTHONUNBUFFERED="1")
        log = Path(logdir) / (job["tag"].replace("/", "__") + ".log")
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.call(job["cmd"], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        with lock:
            status["done"] += 1
            state = "ok" if rc == 0 and Path(ROOT / job["target"]).exists() else f"FAILED (see {log})"
            if state != "ok":
                status["failed"].append(job["tag"])
            print(f"[gpu {gpu}] {status['done']}/{status['total']} {job['tag']}: {state} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        q.task_done()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=list(PHASES))
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids, one worker each")
    ap.add_argument("--envs", nargs="+", default=ENVS)
    ap.add_argument("--settings", nargs="+", default=None, choices=list(SETTINGS),
                    help="only these action settings (one not in DESIGN is added at eps 0.05)")
    ap.add_argument("--explore", nargs="+", type=float, default=None, help="only cells with these eps")
    ap.add_argument("--full-grid", action="store_true", help="every setting x eps, not just DESIGN")
    ap.add_argument("--out", default="results")
    ap.add_argument("--selection", default=None, help="confirm phase: selection.json (default results/tune/selection.json)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget-scale", type=float, default=1.0, help="multiply every budget (for quick tests)")
    ap.add_argument("--n-seeds", type=int, default=None, help="override the phase's number of seeds (for quick tests)")
    ap.add_argument("--extra", default="", help='flags appended to every run.py call, e.g. "--seed-chunk 8"')
    args = ap.parse_args()

    os.chdir(ROOT)
    envs = available_envs(args.envs)
    selection = None
    if args.phase == "confirm":
        sel_path = Path(args.selection or Path(args.out) / "tune" / "selection.json")
        if not sel_path.exists():
            sys.exit(f"{sel_path} not found: run `python scripts/frontier.py tune` first")
        selection = json.loads(sel_path.read_text())["selected"]
    cells = list(itertools.product(SETTINGS, EXPLORE)) if args.full_grid else DESIGN
    if args.settings:
        cells = [(s, e) for s, e in cells if s in args.settings]
        cells += [(s, 0.05) for s in args.settings if s not in {c[0] for c in cells}]
    if args.explore:
        cells = [(s, e) for s, e in cells if e in args.explore]
    jobs = build_jobs(args.phase, envs, cells, args.out, selection,
                      args.budget_scale, args.n_seeds, args.extra.split())
    todo = [j for j in jobs if not Path(j["target"]).exists()]
    # longest jobs first, so the 20M-step ones don't end up last on one GPU
    todo.sort(key=lambda j: -j["budget"])
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    est = sum(j["budget"] for j in todo) / 5e6 * 1.6 / 60 / max(1, len(gpus))
    print(f"{args.phase}: {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run on "
          f"GPUs {gpus} (rough estimate {est:.1f} h at ~1.6 min per 5M-step job)", flush=True)
    if args.dry_run:
        for j in todo:
            print(" ".join(j["cmd"]))
        return
    logdir = Path(args.out) / args.phase / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    q = queue.Queue()
    for j in todo:
        q.put(j)
    lock, status = threading.Lock(), dict(done=0, total=len(todo), failed=[])
    threads = [threading.Thread(target=worker, args=(g, q, logdir, lock, status)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"finished: {status['done'] - len(status['failed'])} ok, {len(status['failed'])} failed", flush=True)
    for t in status["failed"]:
        print("  failed:", t)


if __name__ == "__main__":
    main()
