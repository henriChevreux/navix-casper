#!/usr/bin/env bash
# The default comparison on one GPU: none / mask / casper, 32 paired seeds, 5M steps each,
# on LavaGapS7 and LavaCrossingS9N1, plus a gamma-tilted CASPER-k variant.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-32}
BUDGET=${BUDGET:-5000000}
K=${K:-2}
for ENV in Navix-LavaGapS7-v0 Navix-LavaCrossingS9N1-v0; do
  name=$(echo "$ENV" | sed 's/Navix-//; s/-v0//')
  python scripts/run.py --env-id "$ENV" --k "$K" --seeds "$SEEDS" --budget "$BUDGET" --tag "${name}_k${K}"
  python scripts/run.py --env-id "$ENV" --k "$K" --seeds "$SEEDS" --budget "$BUDGET" \
      --shields casper --casper-gamma 1.0 --tag "${name}_k${K}_gamma1"
  python scripts/summarize.py "results/${name}_k${K}"
done
