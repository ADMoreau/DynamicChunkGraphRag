"""Passage Recall@k on MuSiQue (fraction of a question's supporting paragraphs in the top k),
the measure HippoRAG 2 reports, for our flat retrievers. Usage: python scripts/musique_recall.py <split>"""
import json, sys
from pathlib import Path
from ragsplit.config import load_config
from ragsplit.run import prepare_corpus, prepare_splits, setup_method

cfg = load_config("configs/q1_baselines_tune.yaml")
split = sys.argv[1] if len(sys.argv) > 1 else "tuning"
paragraphs, questions = prepare_corpus(cfg, False); sp = prepare_splits(cfg, questions, False)
q_by_id = {q.id: q for q in questions}
qids = sp[split]
for method in ("B1", "B1+R"):
    spec, units, retriever, pruner, prior, reranker = setup_method(method, cfg, sp, paragraphs, q_by_id, Path("data/cache"))
    rec = {2: [], 5: [], 10: []}
    for qid in qids:
        q = q_by_id[qid]
        ranked = [units[j] for j in retriever.search([q.question])[0]]
        if reranker:
            ranked, _ = reranker.rerank(q.question, ranked)
        gold = {a.paragraph_id for a in q.answers}
        top = [u.paragraph_id for u in ranked]
        for k in rec:
            rec[k].append(len(gold & set(top[:k])) / len(gold))
    print(f"{method:5s} ({split}, {len(qids)} q): " + "  ".join(f"recall@{k} {sum(v)/len(v)*100:.1f}" for k, v in rec.items()), flush=True)
