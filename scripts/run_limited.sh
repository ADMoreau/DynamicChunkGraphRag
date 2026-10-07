#!/usr/bin/env bash
# Run an experiment with a hard resource cap so it cannot take the whole laptop down.
#   - pinned to physical cores 0-3 (logical CPUs 0-7, half the chip): less heat and power
#   - lowest CPU priority and idle IO priority: the desktop always comes first
#   - memory ceiling with no swap: if the job overruns, the kernel kills this job only
# Model thread counts in the configs are unchanged, so outputs are identical; runs just
# take longer. Usage: scripts/run_limited.sh --config configs/<name>.yaml [--force]
# Override with RAGSPLIT_CPUS (e.g. 0-3) and RAGSPLIT_MEM_MAX (e.g. 10G).
set -euo pipefail
cd "$(dirname "$0")/.."
exec systemd-run --user --scope --quiet \
  -p MemoryMax="${RAGSPLIT_MEM_MAX:-14G}" -p MemorySwapMax=0 \
  taskset -c "${RAGSPLIT_CPUS:-0-7}" nice -n 19 ionice -c 3 \
  .venv/bin/python -m ragsplit.run "$@"
