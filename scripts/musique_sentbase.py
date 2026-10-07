"""Pick the sentence-base cut-up beam setting for held-out by the fixed rule (highest tuning
spans_located @512 across q5c111 / q5c123) and write configs/q6c_heldout.yaml."""
import json, re
from pathlib import Path
cands = {}
for kk in ("111", "123"):
    for m, v in json.load(open(f"results/q5c{kk}_beam_tune.json")).items():
        if m.startswith("T-"):
            cands[m] = (v["by_budget"]["512"]["spans_located"], kk)
best = max(cands, key=lambda m: cands[m][0])
kk = cands[best][1]
Path("results/q6c_selection.json").write_text(json.dumps(
    {"rule": "highest tuning spans_located @512 among sentence-base cut-up beams",
     "tuning_scores": {m: v[0] for m, v in cands.items()}, "selected": best}, indent=2) + "\n")
c = open("configs/q6s_heldout.yaml").read().replace("name: q6s_heldout", "name: q6c_heldout")
c = re.sub(r"  methods: \[.*\]", f"  methods: [C-ctxs-k{'-'.join(kk)}+R, {best}]", c)
g = open(f"configs/q4c{kk}_graph.yaml").read()
c = c[:c.index("\ngraph:")] + g[g.index("\ngraph:"):]
open("configs/q6c_heldout.yaml", "w").write(c)
print(f"selected {best} ({cands[best][0]:.3f}); wrote configs/q6c_heldout.yaml")
