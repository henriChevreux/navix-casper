#!/usr/bin/env bash
# Start the whole experiment in the background, detached from the terminal: it keeps running after you
# close the SSH session. Resumable: rerunning skips finished jobs.
#
#   bash scripts/run_detached.sh [GPUS] [MEM_GB] [extra flags for experiment.py]
#   bash scripts/run_detached.sh 1,2 1
#
# Progress:  tail -f logs/experiment.log        Stop:  kill -- -$(cat logs/experiment.pid)   (whole group)
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS="${1:-1,2}"
MEM="${2:-1}"
shift $(( $# > 2 ? 2 : $# )) || true
mkdir -p logs
if [ -f logs/experiment.pid ] && kill -0 "$(cat logs/experiment.pid)" 2>/dev/null; then
  echo "already running (pid $(cat logs/experiment.pid)); tail -f logs/experiment.log"; exit 1
fi
PY="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
nohup setsid "$PY" scripts/experiment.py --gpus "$GPUS" --mem-gb "$MEM" "$@" > logs/experiment.log 2>&1 < /dev/null &
echo $! > logs/experiment.pid
echo "started (pid $!) on GPUs $GPUS with ${MEM} GB each"
echo "progress: tail -f logs/experiment.log     stop: kill -- -\$(cat logs/experiment.pid)"
