"""After the MuSiQue tuning runs: pick the beam setting by a rule fixed in advance (highest tuning
spans_located at 512 tokens = every hop reached the reader), record the choice, and write the
held-out config (all future questions) comparing it with the flat baselines and Provence."""
import json, re
from pathlib import Path

cands = {}
for run in ("q5_beam_tune", "q5s_beam_tune"):
    d = json.load(open(f"results/{run}.json"))
    for m, v in d.items():
        if m.startswith("T-"):
            cands[m] = v["by_budget"]["512"]["spans_located"]
# the best setting of each family goes to held-out: over the cut-up units (T-C123, the method
# under test) and over plain sentences (T-S, the comparison)
best_cut = max((m for m in cands if m.startswith("T-C123")), key=cands.get)
best_sent = max((m for m in cands if m.startswith("T-S-")), key=cands.get)
best = best_cut  # the held-out config below runs both; the graph block serves the cut-up units
graph_cfg = "configs/q4_graph.yaml"
Path("results/q6_selection.json").write_text(json.dumps(
    {"rule": "per family, highest tuning spans_located @512", "tuning_scores": cands,
     "selected_cut_up_units": best_cut, "selected_sentence_units": best_sent}, indent=2) + "\n")
c = open("configs/q1_baselines_tune.yaml").read()
c = re.sub(r"^# .*\n", "", c, flags=re.M)
c = (f"# MuSiQue held-out (all future questions): beam setting {best}, selected on tuning by a fixed rule\n"
     "# (results/q6_selection.json), vs sentences + reranker, cracked units + reranker, Provence (top 5, titles).\n") + c
c = c.replace("name: q1_baselines_tune", "name: q6_heldout").replace("  questions: tuning\n", "  questions: future\n")
c = re.sub(r"  methods: \[.*\]", f"  methods: [B3+R, C-ctx-k1-2-3+R, {best_cut}, B2-n5]", c)
g = open(graph_cfg).read()
c = c.rstrip("\n") + "\n" + g[g.index("\ngraph:"):]
open("configs/q6_heldout.yaml", "w").write(c)
# the sentence-unit beam needs the sentence graph, so it runs as its own config
s2 = c.replace("name: q6_heldout", "name: q6s_heldout")
s2 = re.sub(r"  methods: \[.*\]", f"  methods: [{best_sent}]", s2)
gs = open("configs/q4s_graph.yaml").read()
s2 = s2[:s2.index("\ngraph:")] + gs[gs.index("\ngraph:"):]
open("configs/q6s_heldout.yaml", "w").write(s2)
print(f"cut-up units: {best_cut} ({cands[best_cut]:.3f}); sentence units: {best_sent} ({cands[best_sent]:.3f}); "
      "wrote configs/q6_heldout.yaml and configs/q6s_heldout.yaml")
