"""Train PPO agents with one or more shields on one NAVIX task, many seeds in parallel.

    python scripts/run.py --env-id Navix-LavaGapS7-v0 --shields mask casper --seeds 32 --tag test

Every Config field is a flag (e.g. --casper-lambda 3 --budget 1000000). Each shield is one
jit(vmap(train)) over the seeds, in chunks of --seed-chunk to bound GPU memory. Results go to
results/<tag>/<shield>.npz with the per-update training curves (curve/*), the final evaluation with
the shield (eval_shield/*) and with the shield removed (eval_raw/*), and the seeds.
The full experiment is scripts/experiment.py.
"""
import argparse
import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncasper.ppo import Config, make_train  # noqa: E402

PARSERS = {"int": int, "float": float, "str": str, "bool": lambda s: s.lower() in ("1", "true", "yes")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shields", nargs="+", default=["mask", "mask_pen", "casper"])
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--seed-chunk", type=int, default=0, help="seeds trained at once (0 = all)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="results")
    for f in fields(Config):
        if f.name != "shield":
            ap.add_argument("--" + f.name.replace("_", "-"), type=PARSERS[str(f.type)], default=f.default)
    args = ap.parse_args()
    base = Config(**{f.name: getattr(args, f.name) for f in fields(Config) if f.name != "shield"})
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        dict(base.to_dict(), shields=args.shields, seeds=args.seeds, seed_offset=args.seed_offset), indent=2))
    print(f"devices: {jax.devices()}  ->  {out}", flush=True)

    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds)
    chunk = args.seed_chunk or len(seeds)
    for shield in args.shields:
        cfg = replace(base, shield=shield)
        train, n_updates = make_train(cfg)
        train_v = jax.jit(jax.vmap(train))
        t0 = time.time()
        parts = [jax.device_get(train_v(jax.vmap(jax.random.PRNGKey)(seeds[i:i + chunk])))
                 for i in range(0, len(seeds), chunk)]
        curves = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
        final = {f"{phase}/{k}": np.concatenate([p[1][phase][k] for p in parts])
                 for phase in parts[0][1] for k in parts[0][1][phase]}
        np.savez(out / f"{shield}.npz", seeds=seeds, **{f"curve/{k}": v for k, v in curves.items()}, **final)
        dt = time.time() - t0
        steps = len(seeds) * n_updates * cfg.num_envs * cfg.num_steps
        print(f"{shield:8s} {len(seeds)} seeds x {n_updates} updates in {dt / 60:5.1f} min "
              f"({steps / dt / 1e6:.2f}M env steps/s) | eval success {final['eval_shield/success_rate'].mean():.3f}, "
              f"raw violation {np.nanmean(final['eval_raw/raw_viol']):.3f}", flush=True)


if __name__ == "__main__":
    main()
