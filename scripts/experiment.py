"""The experiment: plain masking vs masking + penalty vs CASPER-k + restoration, on three NAVIX tasks.

    python scripts/experiment.py --gpus 1,2 --mem-gb 1         # run everything, then write the report
    python scripts/experiment.py --dry-run                     # list the jobs
    python scripts/report.py                                   # (re)write the report from what is done

Detached (survives closing the SSH session), see scripts/run_detached.sh:
    bash scripts/run_detached.sh 1,2 1

Grid (3 actions, top-k sampling with k = 2, 5M env steps, 64 seeds per job):
    tasks    LavaGapS7, Dynamic-Obstacles-6x6, DistShift1
    methods  mask                                  plain masking
             pen0.01, pen0.1, pen0.3               masking + lambda * (forbidden probability mass)
             casper0.3, casper1, casper3           CASPER-k + lambda_r/2 ||z - p||^2
Every job uses the same seeds, so all methods are paired seed by seed.

Each job is one `scripts/run.py` call (one method, all seeds) on one GPU. One worker per GPU in
--gpus; before each launch it waits until the GPU has --mem-gb + --headroom-gb free, then lets JAX
reserve exactly --mem-gb up front, so other users can't take it mid-run. Seeds run in chunks of
--seed-chunk to fit in that memory; a job that still runs out of memory is retried with half the
chunk. The grid is resumable: finished jobs (their .npz exists) are skipped.
Results: results/exp/<task>/<method>/<shield>.npz, logs in results/exp/logs/.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "LavaGap": "Navix-LavaGapS7-v0",
    "DynamicObstacles": "Navix-Dynamic-Obstacles-6x6-v0",
    "DistShift": "Navix-DistShift1-v0",
}
PEN_LAMBDAS = [0.01, 0.1, 0.3]
CASPER_LAMBDAS = [0.3, 1.0, 3.0]
METHODS = ([("mask", "mask", [])]
           + [(f"pen{l:g}", "mask_pen", ["--penalty-lambda", str(l)]) for l in PEN_LAMBDAS]
           + [(f"casper{l:g}", "casper", ["--casper-lambda", str(l)]) for l in CASPER_LAMBDAS])
GPU_MEMORY_ERRORS = ("resource_exhausted", "out of memory", "cusolver internal error", "cublas_status_alloc_failed",
                     "cublas_status_not_initialized", "cuda_error_out_of_memory", "cudnn_status_alloc_failed")
LAUNCH_SETTLE_S = 45  # wait between two launches on the same GPU, so each sees the other's memory


def jobs(out, seeds, budget, chunk):
    js = []
    for task, env_id in TASKS.items():
        for name, shield, flags in METHODS:
            tag = f"exp/{task}/{name}"
            cmd = [sys.executable, "scripts/run.py", "--env-id", env_id, "--shields", shield, "--k", "2",
                   "--budget", str(budget), "--seeds", str(seeds), "--seed-offset", "0", "--tag", tag,
                   "--out", str(out)] + flags
            js.append(dict(tag=tag, target=Path(out) / tag / f"{shield}.npz", cmd=cmd, chunk=chunk, attempt=1))
    return js


def gpu_memory(gpu):
    try:
        r = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        free, total = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
        return free, total
    except Exception:
        return None


def log(msg, lock):
    with lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def worker(gpu, q, args, lock, status):
    need = int((args.mem_gb + args.headroom_gb) * 1024)
    while not q.empty():
        with LOCKS[gpu]:  # one launch at a time per GPU
            t_wait, said = time.time(), False
            while True:  # wait for free memory, but don't hold a job other GPUs could take
                mem = gpu_memory(gpu)
                if mem is None or mem[0] >= need or q.empty():
                    break
                if not said:
                    log(f"[gpu {gpu}] waiting: {mem[0]} MiB free, need {need} MiB", lock)
                    said = True
                time.sleep(30)
            if said:
                log(f"[gpu {gpu}] memory available after {time.time() - t_wait:.0f}s", lock)
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                       PYTHONUNBUFFERED="1")
            if mem is not None:
                env.update(XLA_PYTHON_CLIENT_PREALLOCATE="true",
                           XLA_PYTHON_CLIENT_MEM_FRACTION=f"{min(0.9, args.mem_gb * 1024 / mem[1] + 0.002):.4f}")
            else:
                env.update(XLA_PYTHON_CLIENT_PREALLOCATE="false")
            logfile = Path(args.out) / "exp" / "logs" / (job["tag"].replace("/", "__") + ".log")
            t0 = time.time()
            f = open(logfile, "w")
            proc = subprocess.Popen(job["cmd"] + ["--seed-chunk", str(job["chunk"])], cwd=ROOT, env=env,
                                    stdout=f, stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=LAUNCH_SETTLE_S)
            except subprocess.TimeoutExpired:
                pass
        rc = proc.wait()
        f.close()
        ok = rc == 0 and (ROOT / job["target"]).exists()
        oom = not ok and any(e in logfile.read_text(errors="ignore").lower() for e in GPU_MEMORY_ERRORS)
        if oom and job["attempt"] < 4:
            job = dict(job, chunk=max(1, job["chunk"] // 2), attempt=job["attempt"] + 1)
            log(f"[gpu {gpu}] {job['tag']}: GPU memory error, retrying with --seed-chunk {job['chunk']}", lock)
            time.sleep(60 if job["attempt"] > 2 else 0)
            q.put(job)
            continue
        with lock:
            status["done"] += 1
            if not ok:
                status["failed"].append(job["tag"])
        log(f"[gpu {gpu}] {status['done']}/{status['total']} {job['tag']}: "
            f"{'ok' if ok else 'FAILED, see ' + str(logfile)} ({(time.time() - t0) / 60:.1f} min)", lock)


LOCKS = defaultdict(threading.Lock)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="1,2", help="GPU ids, one worker each (repeat an id for 2 workers)")
    ap.add_argument("--mem-gb", type=float, default=1.0, help="GPU memory each job reserves")
    ap.add_argument("--headroom-gb", type=float, default=1.0,
                    help="extra free memory required before a launch (CUDA context, cuSolver/cuBLAS handles)")
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--seed-chunk", type=int, default=8, help="seeds trained at once (fits ~1 GB)")
    ap.add_argument("--budget", type=int, default=5_000_000, help="env steps per seed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    all_jobs = jobs(args.out, args.seeds, args.budget, args.seed_chunk)
    todo = [j for j in all_jobs if not (ROOT / j["target"]).exists()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    print(f"{len(all_jobs)} jobs ({len(TASKS)} tasks x {len(METHODS)} methods, {args.seeds} seeds each), "
          f"{len(all_jobs) - len(todo)} already done, {len(todo)} to run on GPUs {gpus} "
          f"with {args.mem_gb:g} GB each", flush=True)
    if args.dry_run:
        for j in todo:
            print(" ".join(map(str, j["cmd"])), "--seed-chunk", j["chunk"])
        return
    (Path(args.out) / "exp" / "logs").mkdir(parents=True, exist_ok=True)
    q = queue.Queue()
    for j in todo:
        q.put(j)
    lock, status = threading.Lock(), dict(done=0, total=len(todo), failed=[])
    threads = [threading.Thread(target=worker, args=(g, q, args, lock, status)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"finished: {status['done'] - len(status['failed'])} ok, {len(status['failed'])} failed", flush=True)
    for t in status["failed"]:
        print("  failed:", t)
    subprocess.call([sys.executable, "scripts/report.py", "--out", args.out], cwd=ROOT)
    if status["failed"]:
        print("rerun the same command to retry only the failed jobs", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
