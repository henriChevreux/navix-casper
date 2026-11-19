# navix-casper

Does a safety shield on the policy's output teach the network the rule, and at what cost to the task?
This repo compares three ways to train a PPO agent that samples its actions from the top-k of its
policy (k = 2) under a hard rule, on three [NAVIX](https://github.com/epignatelli/navix) tasks:

| method | acting | extra training loss |
|---|---|---|
| **masking** | zero the forbidden actions, renormalise, sample from the top-k | none |
| **masking + penalty** | as masking | λ · (raw probability of the forbidden actions), λ ∈ {0.01, 0.1, 0.3} |
| **CASPER-k** | CASPER-k projection, sample from the top-k | λr/2 · ‖CASPER-k(p) − p‖², λr ∈ {0.3, 1, 3} |

All three execute 0 violations. They differ in what the network learns. With the shield removed,
does the raw policy still follow the rule (violation), and how well does the shielded agent solve
the task (success)?

**Rule:** never move forward into a hazard (lava, or a moving obstacle). **Tasks:** LavaGapS7,
Dynamic-Obstacles-6x6, DistShift1. Details are in `ncasper/env.py`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U "jax[cuda12]"          # GPU; plain `pip install jax` for CPU
pip install -r requirements.txt
pytest -q
```

## Run the experiment

The experiment has 3 tasks × 7 methods = 21 jobs, 64 seeds each, 5M env steps per seed. It runs
detached, so you can close the SSH session:

```bash
bash scripts/run_detached.sh 1,2 1     # GPUs 1 and 2, 1 GB reserved per job
tail -f logs/experiment.log            # progress (Ctrl-C only stops watching)
kill -- -$(cat logs/experiment.pid)    # stop everything
```

- **Resumable:** rerunning skips finished jobs.
- **Shared GPUs:** each launch waits until the GPU has enough free memory, then reserves its share
  up front. A job that runs out of memory is retried automatically with fewer seeds at a time.
- **Report:** when all jobs are done, it writes `results/exp/report.md` (a table) and
  `results/exp/report.png` (success vs violation per task). `python scripts/report.py` rebuilds
  them at any time from the jobs finished so far.

## Code

```
ncasper/env.py         tasks, the rule S(s), Dynamic-Obstacles collision semantics
ncasper/shields.py     masking and the CASPER-k projection (JAX)
ncasper/ppo.py         PPO with top-k sampling through a shield; the penalty / restoration losses
scripts/run.py         one job: one task, one or more shields, many seeds
scripts/experiment.py  the full grid on several GPUs
scripts/report.py      table + figure
scripts/run_detached.sh
tests/                 shields (incl. a check against the PyTorch CASPER-k reference), envs, PPO smoke test
docs/                  note: why CASPER-k's restoration is a better-aimed signal than the penalty
```
