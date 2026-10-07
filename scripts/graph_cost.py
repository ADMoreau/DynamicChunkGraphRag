"""Query-time and memory cost of the graph methods vs flat retrieval (HotpotQA tuning, 140 q)."""
import json, os, resource, time
from pathlib import Path
import numpy as np
from ragsplit.config import load_config
from ragsplit.run import prepare_corpus, prepare_splits, setup_method

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

cfg = load_config("configs/h6_hop_tune.yaml")
paragraphs, questions = prepare_corpus(cfg, False); sp = prepare_splits(cfg, questions, False)
q_by_id = {q.id: q for q in questions}
qs = [q_by_id[q].question for q in sp["tuning"]]
gpath = Path(cfg["graph"]["path"])
print(f"graph file {gpath.stat().st_size / 1e6:.0f} MB")
t0 = time.perf_counter(); g = json.loads(gpath.read_text()); print(f"graph JSON load {time.perf_counter() - t0:.1f} s; nodes {len(g['text_nodes']) + len(g['concepts'])}, edges {len(g['edges'])}")
del g
for m in ("C-ctx-k1-2-3", "C-ctx-k1-2-3+R", "G-a0.7-r0-q0", "G-a0.5-r0.3-q0.5", "H-s2-k20-merge"):
    r0 = rss_mb(); t0 = time.perf_counter()
    spec, units, retriever, pruner, prior, reranker = setup_method(m, cfg, sp, paragraphs, q_by_id, Path("data/cache"))
    setup_s = time.perf_counter() - t0
    for q in qs[:3]:  # warm-up
        retriever.search([q])
    times = []
    for q in qs:
        t = time.perf_counter()
        ranked = retriever.search([q])[0]
        if reranker:
            reranker.rerank(q, [units[j] for j in ranked])
        times.append(time.perf_counter() - t)
    print(f"{m:18s} setup {setup_s:5.1f} s | per question median {np.median(times)*1e3:7.1f} ms, mean {np.mean(times)*1e3:7.1f} ms | peak RSS {rss_mb():.0f} MB (+{rss_mb() - r0:.0f})")
