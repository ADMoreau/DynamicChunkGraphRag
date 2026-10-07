#!/usr/bin/env bash
# Sentence-base cut-up units on MuSiQue: graphs -> beam tuning -> fixed-rule selection -> held-out.
set -u
cd "$(dirname "$0")/.."
until grep -q "^exit" results/q6_heldout.log 2>/dev/null; do sleep 60; done
run() { echo "started $(date -Is)" > results/$1.log; scripts/run_limited.sh --config configs/$1.yaml >> results/$1.log 2>&1; echo "exit $? at $(date -Is)" >> results/$1.log; }
for c in q4c111_graph_tune q4c111_graph q4c123_graph_tune q4c123_graph q5c111_beam_tune q5c123_beam_tune; do run $c; done
.venv/bin/python scripts/musique_sentbase.py > results/q6c_selection.log 2>&1 || exit 1
run q6c_heldout
