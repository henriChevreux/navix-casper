"""Train none / mask / casper PPO agents on a NAVIX lava task, many seeds in parallel.

    python scripts/run.py --seeds 32 --budget 1000000                      # LavaGapS7, k=2
    python scripts/run.py --env Navix-LavaCrossingS9N1-v0 --seeds 32 --k 2
    python scripts/run.py --shields mask casper --casper-gamma 1.0 --tag g1

Each shield is one jit(vmap(train)) over all seeds. Results go to
results/<tag>/<shield>.npz (+ config.json); summarise with scripts/summarize.py.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shields", nargs="+", default=["none", "mask", "casper"])
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--seed-chunk", type=int, default=0,
                    help="train seeds in chunks of this size (0 = all at once) to bound GPU memory")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--out", type=str, default="results")
    for f in fields(Config):
        if f.name == "shield":
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            ap.add_argument(flag, type={"int": int, "float": float, "str": str}[f.type], default=f.default)
    args = ap.parse_args()
    base = Config(**{f.name: getattr(args, f.name) for f in fields(Config) if f.name != "shield"})
    tag = args.tag or f"{base.env_id.replace('Navix-', '').replace('-v0', '')}_k{base.k}"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        dict(base.to_dict(), shields=args.shields, seeds=args.seeds, seed_offset=args.seed_offset), indent=2))
    print(f"devices: {jax.devices()}  ->  {out}")

    seeds = np.arange(args.seed_offset, args.seed_offset + args.seeds)
    chunk = args.seed_chunk or len(seeds)
    for shield in args.shields:
        cfg = replace(base, shield=shield)
        train, n_updates = make_train(cfg)
        train_v = jax.jit(jax.vmap(train))
        t0 = time.time()
        parts = []
        for i in range(0, len(seeds), chunk):
            keys = jax.vmap(jax.random.PRNGKey)(seeds[i:i + chunk])
            parts.append(jax.device_get(train_v(keys)))
        curves = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
        final = {f"{m}/{k}": np.concatenate([p[1][m][k] for p in parts])
                 for m in parts[0][1] for k in parts[0][1][m]}
        dt = time.time() - t0
        np.savez(out / f"{shield}.npz", seeds=seeds, **{f"curve/{k}": v for k, v in curves.items()}, **final)
        steps = len(seeds) * n_updates * cfg.num_envs * cfg.num_steps
        print(f"{shield:7s} {len(seeds)} seeds x {n_updates} updates in {dt:6.1f}s "
              f"({steps / dt / 1e6:.2f}M env steps/s incl. compile) | "
              f"final train success {curves['success_rate'][:, -5:].mean():.3f}  "
              f"lava/ep {curves['lava_rate'][:, -5:].mean():.3f}  "
              f"raw top-k viol {np.nanmean(curves['raw_topk_viol'][:, -5:]):.3f} | "
              f"eval removed-layer lava/ep {final['eval_raw_topk/lava_rate'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
