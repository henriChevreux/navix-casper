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
tests/               shield guarantees, the PyTorch CASPER-k reference check, env rule
```
