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

Follow-up, ablation (same seeds as confirm, LavaGapS7 only; exploratory): casper without
restoration, and penalty lambda in {0.15, 0.2}.
  python scripts/sweep.py ablation --gpus 0,1,2
  python scripts/frontier.py ablation

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
from collections import defaultdict
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

ENVS = ["Navix-LavaGapS7-v0", "Navix-DistShift1-v0", "Navix-LavaCrossingS9N1-v0", "Navix-Dynamic-Obstacles-6x6-v0"]
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
PHASES = {"tune": (0, 16), "confirm": (100, 32), "ablation": (100, 32)}  # seed offset, number of seeds
# Follow-up, chosen after the confirm results (exploratory, same seeds as confirm so it's paired):
#   casper with restoration lambda = 0: is the gain the layer's gradient or the restoration loss?
#   penalty lambda between 0.1 and 0.3: compare with casper at matched violation.
ABLATION = [("casper_g0_l0", "casper", ["--casper-gamma", "0.0", "--casper-lambda", "0.0"]),
            ("pen0.15", "mask_pen", ["--penalty-lambda", "0.15"]),
            ("pen0.2", "mask_pen", ["--penalty-lambda", "0.2"])]
ABLATION_ENVS = ["Navix-LavaGapS7-v0"]  # DistShift's layer-removed violation is mostly vacuous


def methods(phase="confirm"):
    """[(name, shield, extra flags)] for every method x hyperparameter."""
    if phase == "ablation":
        return ABLATION
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
        if full and "Dynamic-Obstacles" in env:
            print(f"! skipping {cell}: Dynamic-Obstacles supports only the 3-action set", flush=True)
            continue
        budget = int(budget * budget_scale)
        chosen = None
        if phase == "confirm":
            chosen = set(selection.get(cell, {}).values())
            if not chosen:
                continue
            chosen |= set(UNTUNED)
        for name, shield, method_flags in methods(phase):
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


def gpu_memory(gpu):
    """(free MiB, total MiB) of one GPU from nvidia-smi, or None if it can't be read."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        free, total = (int(x) for x in out.stdout.strip().splitlines()[0].split(","))
        return free, total
    except Exception:
        return None


def wait_for_memory(gpu, need_mib, max_wait_s, poll_s=30):
    """Block until the GPU has need_mib free (other users' jobs come and go). Returns (free, total) or None."""
    t0, said = time.time(), False
    while True:
        mem = gpu_memory(gpu)
        if mem is None or mem[0] >= need_mib or time.time() - t0 > max_wait_s:
            if said:
                print(f"[gpu {gpu}] {'memory free again' if mem and mem[0] >= need_mib else 'stopped waiting'} "
                      f"after {time.time() - t0:.0f}s", flush=True)
            return mem
        if not said:
            print(f"[gpu {gpu}] waiting: {mem[0]} MiB free, need {need_mib} MiB (other users)", flush=True)
            said = True
        time.sleep(poll_s)


# Messages of a GPU that is short of memory. cuSolver/cuBLAS allocate their handles and workspaces
# outside JAX's reserved pool, so on a nearly full GPU they fail with "internal error" rather than OOM.
GPU_MEMORY_ERRORS = ("RESOURCE_EXHAUSTED", "out of memory", "cuSolver internal error",
                     "CUBLAS_STATUS_ALLOC_FAILED", "CUBLAS_STATUS_NOT_INITIALIZED",
                     "CUDA_ERROR_OUT_OF_MEMORY", "CUDNN_STATUS_ALLOC_FAILED")
HEADROOM_MIB = 3 * 1024  # left free outside the reserved pool: CUDA context, cuSolver/cuBLAS handles
LAUNCH_SETTLE_S = 45      # after starting a job, wait this long before starting another on the same GPU
_gpu_locks = defaultdict(threading.Lock)


def is_gpu_memory_error(text):
    low = text.lower()
    return any(m.lower() in low for m in GPU_MEMORY_ERRORS)


def worker(gpu, q, logdir, lock, status, mem_gb=6.0, max_wait_min=120, max_attempts=4):
    """One worker on one GPU (list a GPU twice in --gpus for two workers on it).
    Before each job: wait until mem_gb + headroom is free, then let JAX reserve mem_gb up front
    (XLA_PYTHON_CLIENT_MEM_FRACTION), so other users can't take it mid-run. Launches on the same
    GPU are serialised and spaced by LAUNCH_SETTLE_S, so two jobs never both see the same free
    memory. On a GPU memory error: retry with seeds run 8 at a time, waiting, up to max_attempts."""
    need = int(mem_gb * 1024)
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        attempt = job.get("attempt", 1)
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",  # same numbering as nvidia-smi
                   CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        log = Path(logdir) / (job["tag"].replace("/", "__") + ".log")
        with _gpu_locks[gpu]:
            mem = wait_for_memory(gpu, need + HEADROOM_MIB, max_wait_min * 60)
            if mem is not None:
                env.update(XLA_PYTHON_CLIENT_PREALLOCATE="true",
                           XLA_PYTHON_CLIENT_MEM_FRACTION=f"{min(0.9, need / mem[1] + 0.005):.3f}")
            else:  # no nvidia-smi: allocate on demand
                env.update(XLA_PYTHON_CLIENT_PREALLOCATE="false")
            t0 = time.time()
            f = open(log, "w")
            proc = subprocess.Popen(job["cmd"], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            try:  # let it create its CUDA context and reserve its pool before the next launch here
                proc.wait(timeout=LAUNCH_SETTLE_S)
            except subprocess.TimeoutExpired:
                pass
        rc = proc.wait()
        f.close()
        ok = rc == 0 and Path(ROOT / job["target"]).exists()
        text = "" if ok else log.read_text(errors="ignore")
        oom = not ok and is_gpu_memory_error(text)
        if oom and attempt < max_attempts:
            cmd = job["cmd"] if "--seed-chunk" in job["cmd"] else job["cmd"] + ["--seed-chunk", "8"]
            with lock:
                print(f"[gpu {gpu}] {job['tag']}: GPU memory error (attempt {attempt}/{max_attempts}), "
                      f"retrying with --seed-chunk 8", flush=True)
            if attempt >= 2:
                time.sleep(60)  # give the other users' memory a minute to come back
            q.put(dict(job, cmd=cmd, attempt=attempt + 1))
            q.task_done()
            continue
        with lock:
            status["done"] += 1
            state = "ok" if ok else f"FAILED{' (GPU memory error)' if oom else ''} (see {log})"
            if state != "ok":
                status["failed"].append(job["tag"])
            print(f"[gpu {gpu}] {status['done']}/{status['total']} {job['tag']}: {state} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        q.task_done()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=list(PHASES))
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids, one worker each")
    ap.add_argument("--envs", nargs="+", default=None, help="default: all (ablation: LavaGapS7 only)")
    ap.add_argument("--settings", nargs="+", default=None, choices=list(SETTINGS),
                    help="only these action settings (one not in DESIGN is added at eps 0.05)")
    ap.add_argument("--explore", nargs="+", type=float, default=None, help="only cells with these eps")
    ap.add_argument("--full-grid", action="store_true", help="every setting x eps, not just DESIGN")
    ap.add_argument("--out", default="results")
    ap.add_argument("--selection", default=None, help="confirm phase: selection.json (default results/tune/selection.json)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget-scale", type=float, default=1.0, help="multiply every budget (for quick tests)")
    ap.add_argument("--n-seeds", type=int, default=None, help="override the phase's number of seeds (for quick tests)")
    ap.add_argument("--mem-gb", type=float, default=6.0,
                    help="GPU memory each job waits for and reserves up front (default 6 GB)")
    ap.add_argument("--max-wait-min", type=float, default=120,
                    help="max minutes to wait for free GPU memory before trying anyway")
    ap.add_argument("--extra", default="", help='flags appended to every run.py call, e.g. "--seed-chunk 8"')
    args = ap.parse_args()

    os.chdir(ROOT)
    envs = available_envs(args.envs or (ABLATION_ENVS if args.phase == "ablation" else ENVS))
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
    threads = [threading.Thread(target=worker, args=(g, q, logdir, lock, status, args.mem_gb, args.max_wait_min))
               for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"finished: {status['done'] - len(status['failed'])} ok, {len(status['failed'])} failed", flush=True)
    for t in status["failed"]:
        print("  failed:", t)
    if status["failed"]:
        print("rerun the same command to retry only the failed jobs; check free memory with nvidia-smi "
              "and leave out full GPUs (e.g. --gpus 1,2)", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
