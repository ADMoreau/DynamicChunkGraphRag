#!/usr/bin/env bash
# MuSiQue pipeline after candidate generation: cost check -> Qwen 14B judging on Modal (4 x L40S)
# -> graph builds (tuning + held-out) -> beam tuning. Stops before judging if the estimate
# exceeds the approved ceiling (MAX_USD). Full logs in results/q*.log.
set -u
cd "$(dirname "$0")/.."
MAX_USD=${MAX_USD:-3.5}
until grep -q "^exit" results/q2_candidates.log 2>/dev/null; do sleep 30; done
f=$(grep -o "data/cache/merge/candidates_[0-9a-f]*\.jsonl" results/q2_candidates.log | head -1)
[ -z "$f" ] && { echo "no candidates file; stopping"; exit 1; }
est=$(.venv/bin/python - "$f" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
n = sum(1 for r in rows if r.get("cos") is None or r["source"] == "exact" or r["cos"] >= 0.85)
secs = 4 * (135 + n / 4 / 47)          # 4 containers: load + judging at ~47 pairs/s each
print(n, round(secs * 0.000542, 2))    # L40S $/s
PY
)
n=${est% *}; usd=${est#* }
echo "MuSiQue judge: $n pairs, estimated \$$usd (ceiling \$$MAX_USD)" | tee results/q3_judge.estimate
if .venv/bin/python -c "import sys; sys.exit(0 if $usd <= $MAX_USD else 1)"; then
  echo "started $(date -Is)" > results/q3_judge.log
  timeout 4000 .venv/bin/modal run modal_app/llm_judge.py --pairs "$f" --models Qwen/Qwen2.5-14B-Instruct \
    --gpus L40S --out results/q3_judge --min-cos 0.85 --chunks 4 --labels none >> results/q3_judge.log 2>&1
  echo "exit $? at $(date -Is)" >> results/q3_judge.log
else
  echo "estimate above ceiling: not judging; ask the user" | tee -a results/q3_judge.estimate; exit 2
fi
for c in q4_graph_tune q4_graph q5_beam_tune; do
  echo "started $(date -Is)" > results/$c.log
  scripts/run_limited.sh --config configs/$c.yaml >> results/$c.log 2>&1
  echo "exit $? at $(date -Is)" >> results/$c.log
done
