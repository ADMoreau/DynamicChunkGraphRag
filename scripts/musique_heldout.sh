#!/usr/bin/env bash
# Follow-on MuSiQue chain: sentence-unit graphs -> sentence-unit beam tuning -> pre-registered
# selection -> held-out test on all future questions. Full logs in results/q*.log.
set -u
cd "$(dirname "$0")/.."
until grep -q "^exit" results/q5_beam_tune.log 2>/dev/null; do sleep 60; done
for c in q4s_graph_tune q4s_graph q5s_beam_tune; do
  echo "started $(date -Is)" > results/$c.log
  scripts/run_limited.sh --config configs/$c.yaml >> results/$c.log 2>&1
  echo "exit $? at $(date -Is)" >> results/$c.log
done
.venv/bin/python scripts/musique_heldout.py > results/q6_selection.log 2>&1 || exit 1
for c in q6s_heldout q6_heldout; do
  echo "started $(date -Is)" > results/$c.log
  scripts/run_limited.sh --config configs/$c.yaml >> results/$c.log 2>&1
  echo "exit $? at $(date -Is)" >> results/$c.log
done
