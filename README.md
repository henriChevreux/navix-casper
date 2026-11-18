# navix-casper

CASPER-k vs masking for **safe RL with top-k sampling**, on NAVIX (MiniGrid in JAX).
Everything (environments, PPO, shields, evaluation) runs on the GPU in one jitted
function, and many paired seeds are trained in parallel with `vmap`.

## The task and the rule

- Environments: NAVIX lava grids (`Navix-LavaGapS5/6/7-v0`, `Navix-LavaCrossingS9N1…S11N5-v0`, `Navix-DistShift1/2-v0`).
- Rule: **never move forward into lava.** S(s) = every action except `forward` when the cell in front is lava.
- The action is a single categorical variable and the condition is read straight from the state, so **no constraint compilation is needed**.
- Actions: MiniGrid's three movement actions by default (turn left, turn right, forward); `--full-action-set true` uses all seven.
- Stepping into lava ends the episode with reward 0. The goal gives +1.

## Shields (`ncasper/shields.py`)

All act on the policy's output distribution p = softmax(logits). The agent samples from the top-k set T of the shield's output z, renormalised (q = q_k(z)):

| `--shields` | z | T | guarantee |
|---|---|---|---|
| `none` | p | top-k(p) | none |
| `mask` | p restricted to S, renormalised | top-k inside S | support ⊆ S |
| `mask_pen` | as `mask`; the loss adds `--penalty-lambda` × (raw probability on forbidden actions) | top-k inside S | support ⊆ S |
| `casper` | CASPER-k(p, S) | k largest of p inside S | support ⊆ S |

- **CASPER-k** is the project's surrogate (a line-by-line port of `casper_k` from `casper_k/casperk_learning_study.py`, checked against it in `tests/`):
  - T members within ε of the best non-member are pooled with the offending non-members. After that, T beats every other action by ε.
  - If |S| < k, it falls back to the face projection.
  - k = 1 is CASPER argmax. `--casper-gamma 1` gives the γ-tilted margins from the notes.
  - It adds restoration λ/2 ‖z − p‖² (`--casper-lambda`).
- **Exploration:** the behaviour policy is (1 − ε)·q + ε·Uniform(S), with ε = `--explore-eps`, default 0.05.
  - Without it, top-k sampling never samples the action ranked k+1. With 3 actions and k = 2, an agent can then lock out `forward` for good.
  - Support stays inside S, so the guarantee holds. Evaluation uses pure top-k sampling.

## PPO (`ncasper/ppo.py`)

- PureJaxRL-style, with GAE, clipped objective and value clipping. The network is an MLP on the one-hot symbolic observation.
- PPO's ratio uses the shielded behaviour probabilities.
- Seeds are **paired** across shields: the same seed gives the same initial network and the same environment resets.
- Sanity check (CPU, 2 seeds, 2M steps): unconstrained full-sampling PPO on LavaGapS7 went from 3% to 67% training success, and 72–81% evaluation success.

Logged every update: success rate, lava falls per episode, how often the rule is active, raw top-k and raw argmax violation (the network with the layer removed), entropy, KL, and ½‖z − p‖².

The final evaluation runs three modes:

| Mode | Policy |
|---|---|
| `shield` | top-k sampling through the layer |
| `raw_topk` | top-k sampling from raw p, layer removed |
| `raw_greedy` | argmax of raw p, layer removed |

## Setup (GPU machine, Python venv)

Needs Python ≥ 3.10 and an NVIDIA driver recent enough for CUDA 12.

```bash
cd navix-casper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -U "jax[cuda12]"                  # JAX with bundled CUDA 12 libraries
pip install -r requirements.txt
python -c "import jax; print(jax.devices())"  # should list CudaDevice(...), not CpuDevice
pytest -q                                     # the PyTorch reference test is skipped without torch
```

Activate the venv (`source .venv/bin/activate`) in every new shell before running anything.
`.venv/` is git-ignored.

## Run

```bash
python scripts/run.py --seeds 32 --budget 5000000                        # LavaGapS7, k = 2: none / mask / casper
python scripts/summarize.py results/LavaGapS7_k2                          # table, paired tests, curves.png
python scripts/run.py --shields mask_pen --penalty-lambda 1 --seeds 32 --tag LavaGapS7_k2_pen1   # extra baseline, same seeds
python scripts/summarize.py results/LavaGapS7_k2 results/LavaGapS7_k2_pen1  # merged table, casper vs every shield
python scripts/run.py --env-id Navix-LavaCrossingS9N1-v0 --seeds 32 --budget 5000000
python scripts/run.py --shields mask casper --casper-gamma 1.0 --tag LavaGapS7_k2_gamma1
bash scripts/run_all.sh                                                   # both envs + gamma = 1
```

- Every `Config` field is a flag (`python scripts/run.py --help`).
- Each shield is compiled once and trains all seeds in one `vmap`. If GPU memory runs out, use `--seed-chunk 8`.
- On the CPU sandbox this ran at about 0.03–0.05M env steps/s. A GPU should be orders of magnitude faster (NAVIX reports 2,048 PPO agents trained in under 50 s on an A100), but measure your own speed: `run.py` prints env steps/s per shield.

## The decisive comparison: CASPER-k vs masking + penalty (`scripts/sweep.py`, `scripts/frontier.py`)

Everything stays in probability space. Cells are map × (action setting, exploration ε):

| Grid axis | Values |
|---|---|
| maps | LavaGapS7, DistShift1, LavaCrossingS9N1 (the last needs NAVIX ≥ 0.8, i.e. Python ≥ 3.12; it's skipped otherwise) |
| (setting, ε) | `a3k2` ε = .05 and .15, `a3k3` ε = .05 (3 actions, k = 2 / 3, 5M). ε = .15 tests whether CASPER's k = 2 advantage is just exploration. `--settings a3k2 a3k3 a7k2` adds 7 actions / 20M; `--full-grid` runs every combination. |
| tuned (tune phase) | mask_pen λ ∈ {0.01, 0.1, 0.3}, selected per cell |
| fixed (confirm only) | none, mask, casper (γ = 0, restoration λ = 1). CASPER isn't tuned, which can only favour the penalty. |

Each run gives two metrics:

- **success:** task success with the layer on.
- **violation:** with the layer removed, the fraction of rule-active steps where the raw greedy policy picks the forbidden action. Seeds that never meet the rule are vacuous and are excluded and reported.

The protocol avoids tuning on the reported data:

```bash
python scripts/sweep.py tune --gpus 0,1,2        # 18 jobs (27 with LavaCrossing) × 16 seeds (0–15), resumable; ~10 min on 3 GPUs
python scripts/frontier.py tune                  # tables, trade-off plots, results/tune/selection.json
python scripts/sweep.py confirm --gpus 0,1,2     # none, mask, casper + selected penalty: 24 jobs (36), 32 fresh seeds (100–131); ~15 min
python scripts/frontier.py confirm               # paired Wilcoxon + Holm correction → results/confirm/verdict.md
```

1. **Selection rule for the penalty λ, fixed in advance, per cell:** among configs with violation ≤ 0.05 and ≤ 25% vacuous seeds, take the highest success. If none qualifies, take the lowest violation.
2. **Verdict per cell:** CASPER is *better* if it's significantly better on one metric and not significantly worse on the other; *worse* in the reverse case; otherwise it's a *tie / trade-off*.

`frontier.py tune` also reports, per λ, how many cells it satisfies, so you can see whether a single penalty strength works everywhere or needs choosing per environment (CASPER uses one fixed setting throughout).

Useful flags:

- `--dry-run` lists the jobs without running them.
- `--envs`, `--settings` and `--explore` restrict the grid.
- `--extra "--seed-chunk 8"` passes flags through to every run.
- `--budget-scale 0.003 --n-seeds 6` gives a quick smoke test.

Each job's log goes to `results/<phase>/logs/`.

## Things to know

- **What to compare:** masking and CASPER-k sample from the same support whenever |S| ≥ k (the top-k of p inside S). Their differences are within-T probabilities and gradients, so the interesting columns are the layer-removed evaluations and the raw violation rates.
- **When CASPER-k changes nothing:** with 3 actions and k = 2, T = S whenever the rule is active. With `--k 3`, CASPER falls back to its face projection.
- **Known pathology:** with γ = 0, CASPER-k is uniform on T when the raw argmax is forbidden (the flattening in the project's notes). Run `--casper-gamma 1` alongside.
- **PPO ratio:** the ratio is ~0 when the sampled action leaves the top-k under the new parameters, and the clip then takes over. The CASPER surrogate's pooling set changes discretely, so its approx-KL can be higher than masking's.
- **The unshielded agent** has to learn lava avoidance from reward alone; with top-k sampling it learns more slowly than full-sampling PPO.

## Layout

```
ncasper/shields.py   none / mask / CASPER-k on the output distribution (JAX)
ncasper/env.py       NAVIX env, lava-ahead rule, same-call autoreset
ncasper/ppo.py       vmappable PPO + final evaluation
scripts/run.py       train all shields x seeds, save results/<tag>/<shield>.npz
scripts/summarize.py summary.md (mean ± s.e.m.), paired Wilcoxon casper − mask, curves.png
scripts/run_all.sh   default comparison on one GPU
scripts/sweep.py     tune / confirm grid for CASPER-k vs masking + penalty, one worker per GPU
scripts/frontier.py  trade-off plots, fixed selection rule, paired verdict
tests/               shield guarantees, the PyTorch CASPER-k reference check, env rule
```
