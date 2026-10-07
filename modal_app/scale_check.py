"""Large-corpus premise check: does flat retrieval stop finding the second hop at scale?

On our 19,269-paragraph HotpotQA pool, flat retrieval (BM25 + bge-small, reciprocal rank
fusion) has both gold paragraphs in its top 30 for ~89% of bridge questions, so a graph
has nothing to fix. Here the same pool is mixed into growing random samples of the
5.2M-abstract HotpotQA Wikipedia corpus (BEIR/hotpotqa) and the same retrieval is
measured at each size (nested samples: every larger corpus contains the smaller one).

Everything heavy runs in one Modal container: download (~1 GB), sampling, embedding on an
L4 GPU, BM25 on CPU. Costs GPU time: ask before running it.
  modal run modal_app/scale_check.py --sizes 100000,300000,1000000
"""

import json
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data" / "derived" / "hotpotqa_validation_n2000_seed13"
EMBED = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

hf_cache = modal.Volume.from_name("ragsplit-hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "sentence-transformers==6.1.0", "transformers==5.18.0",
                      "bm25s", "PyStemmer", "pandas", "pyarrow", "numpy", "huggingface_hub")
         .env({"HF_HOME": "/hf"}))
app = modal.App("ragsplit-scale-check")


def rrf(lists, k=60):
    score = {}
    for ranking in lists:
        for r, idx in enumerate(ranking, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + r)
    return sorted(score, key=lambda i: (-score[i], i))


@app.function(image=image, gpu="L4", cpu=8, memory=32768, volumes={"/hf": hf_cache}, timeout=3600)
def check(pool: list[dict], questions: list[dict], sizes: list[int], seed: int = 13, depth: int = 100) -> dict:
    import random

    import bm25s
    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    path = hf_hub_download("BeIR/hotpotqa", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset")
    corpus = pd.read_parquet(path, columns=["title", "text"])
    pool_titles = {p["title"] for p in pool}
    others = corpus[~corpus["title"].isin(pool_titles)]
    order = list(range(len(others)))
    random.Random(seed).shuffle(order)
    n_extra = max(sizes) - len(pool)
    extra = others.iloc[order[:n_extra]]
    texts = [f"{p['title']}: {p['context']}" for p in pool] + \
            [f"{t}: {x}" for t, x in zip(extra["title"], extra["text"])]
    titles = [p["title"] for p in pool] + list(extra["title"])
    del corpus, others
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = SentenceTransformer(EMBED, device="cuda")
    model.half()
    emb = model.encode(texts, batch_size=1024, normalize_embeddings=True, convert_to_tensor=True,
                       show_progress_bar=False)
    q_emb = model.encode([QUERY_PREFIX + q["question"] for q in questions], batch_size=256,
                         normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    embed_s = time.perf_counter() - t0

    out = {"corpus_docs": int(len(titles)), "pool": len(pool), "load_s": load_s, "embed_s": embed_s,
           "gpu": torch.cuda.get_device_name(0), "by_size": {}}
    for size in sorted(sizes):
        t0 = time.perf_counter()
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(texts[:size], stopwords="en", show_progress=False), show_progress=False)
        docs, scores = retriever.retrieve(bm25s.tokenize([q["question"] for q in questions], stopwords="en",
                                                         show_progress=False), k=depth, show_progress=False)
        sims = q_emb @ emb[:size].T
        dense = torch.topk(sims, depth, dim=1).indices.cpu().numpy()
        stats = {"first_hop_top3": 0, "both_top10": 0, "both_top30": 0, "both_top100": 0, "bridge": 0}
        per_type = {}
        for qi, q in enumerate(questions):
            bm = [int(d) for d, s in zip(docs[qi], scores[qi]) if s > 0]
            fused = rrf([bm, [int(d) for d in dense[qi]]])[:depth]
            ranked_titles = [titles[i] for i in fused]
            gold = set(q["gold_titles"])
            rank = {t: ranked_titles.index(t) for t in gold if t in ranked_titles}
            both = lambda k: len(gold) == 2 and all(rank.get(t, depth) < k for t in gold)
            row = {"first_hop_top3": any(r < 3 for r in rank.values()), "both_top10": both(10),
                   "both_top30": both(30), "both_top100": both(100)}
            for key in row:
                stats[key] += row[key]
                per_type.setdefault(q["type"], {}).setdefault(key, 0)
                per_type[q["type"]][key] += row[key]
            per_type[q["type"]]["n"] = per_type[q["type"]].get("n", 0) + 1
        n = len(questions)
        out["by_size"][str(size)] = {"n_questions": n, "seconds": time.perf_counter() - t0,
                                     **{k: v / n for k, v in stats.items() if k != "bridge"},
                                     "by_type": {t: {k: v / d["n"] for k, v in d.items() if k != "n"} | {"n": d["n"]}
                                                 for t, d in per_type.items()}}
        print(f"size {size}: {out['by_size'][str(size)]}", flush=True)
    return out


@app.local_entrypoint()
def main(sizes: str = "100000,300000,1000000"):
    pool = [json.loads(line) for line in (DERIVED / "paragraphs.jsonl").open()]
    pool = [{"title": p["title"], "context": p["context"]} for p in pool]
    by_id = {}
    for line in (DERIVED / "paragraphs.jsonl").open():
        p = json.loads(line)
        by_id[p["id"]] = p["title"]
    questions = []
    for line in (DERIVED / "questions.jsonl").open():
        q = json.loads(line)
        gold = sorted({by_id[pid] for pid, _ in q["support"]})
        questions.append({"question": q["question"], "gold_titles": gold, "type": (q["qtype"] or "").split("/")[0]})
    res = check.remote(pool, questions, [int(s) for s in sizes.split(",")] + [len(pool)])
    path = ROOT / "results" / "h7_scale_check.json"
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\ncorpus {res['corpus_docs']} docs; load {res['load_s']:.0f} s, embed {res['embed_s']:.0f} s on {res['gpu']}")
    print("| corpus size | first hop in top 3 | both hops in top 10 | top 30 | top 100 |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for size, r in sorted(res["by_size"].items(), key=lambda kv: int(kv[0])):
        print(f"| {int(size):,} | {r['first_hop_top3']:.3f} | {r['both_top10']:.3f} | {r['both_top30']:.3f} | {r['both_top100']:.3f} |")
    print(f"\nwrote {path}")
