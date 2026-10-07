"""Controlled cost benchmark: per-question cost of each method under the same conditions.

The eval runs log latency, but they ran hours apart under different machine load,
and llama.cpp reuses the previous prompt's matching prefix (so a context repeated
across budgets is nearly free). Neither is safe for a published cost comparison.
Here every variant runs on the same questions in one process, interleaved with the
order rotated per question, after warm-up questions that are not counted. The reader
is replayed on the contexts logged by the eval runs with its state reset before every
call. Compute is also counted in a hardware-independent way: tokens each model
processes and its parameter count (FLOPs ~ 2 x non-embedding params x tokens).
"""

import json
from pathlib import Path
import time

import numpy as np


def n_params(model) -> tuple[int, int]:
    """(all parameters, parameters outside embedding tables)."""
    total = emb = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if "embeddings" in name:
            emb += p.numel()
    return total, total - emb


def ratio_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """mean(a) / mean(b) over paired questions, with a 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    r = a[idx].mean(axis=1) / b[idx].mean(axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"ratio": float(a.mean() / b.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def run_bench(cfg: dict, force: bool) -> None:
    import os

    from ragsplit.reader import load_reader
    from ragsplit.run import eval_question_ids, prepare_corpus, prepare_splits, setup_method

    bc = cfg["bench"]
    paragraphs, questions = prepare_corpus(cfg, force=False)
    sp = prepare_splits(cfg, questions, force=False)
    para_by_id = {p.id: p for p in paragraphs}
    q_by_id = {q.id: q for q in questions}
    warmup = bc["warmup"]
    qids = eval_question_ids(cfg, sp)[: warmup + bc["n_questions"]]
    cache_dir = Path(cfg["data"]["data_dir"]) / "cache"
    variants = bc["variants"]

    setups, front_models = {}, {}
    for v in variants:
        m = v["method"]
        if m not in setups:
            spec, units, retriever, pruner, prior, reranker = setup_method(
                m, cfg, sp, paragraphs, q_by_id, cache_dir)
            if prior:
                raise SystemExit("bench does not support prior methods")
            setups[m] = (units, retriever, pruner, reranker)
            if pruner:
                from transformers import AutoTokenizer

                front_models["provence"] = (pruner.model(), AutoTokenizer.from_pretrained(
                    cfg["prune"]["model_path"], trust_remote_code=True))
            if reranker:
                front_models["reranker"] = (reranker.model.model, reranker.model.tokenizer)
    params = {k: dict(zip(("total", "non_embedding"), n_params(m)))
              for k, (m, _) in front_models.items()}

    def run_front(v: dict, question: str) -> dict:
        units, retriever, pruner, reranker = setups[v["method"]]
        t0 = time.perf_counter()
        ranked = [units[j] for j in retriever.search([question])[0]]
        retrieval_s = time.perf_counter() - t0
        front_s, tokens = 0.0, 0
        if pruner:
            model, tok = front_models["provence"]
            texts = [para_by_id[u.paragraph_id].context for u in ranked[: v.get("top_n", pruner.top_n)]]
            t0 = time.perf_counter()
            model.process([question], [texts], title=None, threshold=pruner.threshold,
                          batch_size=v.get("batch_size", cfg["prune"].get("batch_size", 4)),
                          enable_warnings=False)
            front_s = time.perf_counter() - t0
            # question + passage + 3 special tokens per pair; long passages are chunked, not cut
            nq = len(tok.encode(question, add_special_tokens=False))
            tokens = sum(nq + len(tok.encode(t, add_special_tokens=False)) + 3 for t in texts)
        if reranker:
            _, front_s = reranker.rerank(question, ranked)
            _, tok = front_models["reranker"]
            tokens = sum(len(tok(question, u.text, truncation=True, max_length=512)["input_ids"])
                         for u in ranked[: reranker.top_n])
        return {"retrieval_s": retrieval_s, "front_s": front_s, "front_tokens": tokens}

    # reader replays: contexts the eval runs actually sent, keyed by (method, budget, qid)
    want = {(m, b) for m in bc["reader_methods"] for b in bc["reader_budgets"]}
    contexts = {}
    for log in bc["reader_logs"]:
        for line in open(log):
            r = json.loads(line)
            if (r["method"], r["budget"]) in want and r["qid"] in qids:
                contexts[(r["method"], r["budget"], r["qid"])] = r
    missing = [(m, b) for m, b in sorted(want) if not all((m, b, q) in contexts for q in qids)]
    if missing:
        raise SystemExit(f"reader_logs lack contexts for {missing}")
    reader = load_reader(cfg["reader"])

    out_path = Path(cfg.get("results_dir", "results")) / f"{cfg['name']}.jsonl"
    load_start = os.getloadavg()
    rows = []
    with out_path.open("w") as f:
        for i, qid in enumerate(qids):
            question = q_by_id[qid].question
            jobs = [("front", v) for v in variants] + [("reader", mb) for mb in sorted(want)]
            k = i % len(jobs)
            for kind, job in jobs[k:] + jobs[:k]:
                if kind == "front":
                    row = {"kind": "front", "variant": job["name"], **run_front(job, question)}
                else:
                    m, b = job
                    logged = contexts[(m, b, qid)]
                    reader.llm.reset()  # no prompt-prefix reuse from the previous call
                    out = reader.answer(question, logged["context"])
                    row = {"kind": "reader", "variant": m, "budget": b,
                           "reader_s": out["reader_latency_s"], "input_tokens": out["input_tokens"],
                           "output_tokens": out["output_tokens"],
                           "same_answer_as_eval": out["answer"] == logged["answer"]}
                row.update(qid=qid, warmup=i < warmup)
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
            print(f"  bench {i + 1}/{len(qids)}{' (warm-up)' if i < warmup else ''}", flush=True)
    reader.close()
    load_end = os.getloadavg()

    rows = [r for r in rows if not r["warmup"]]
    summary = {"n_questions": bc["n_questions"], "params": params,
               "loadavg_start": load_start, "loadavg_end": load_end,
               "front": {}, "reader": {}, "end_to_end": {}}
    print(f"\nload average start {load_start}, end {load_end}")
    print(f"params: {params}\n")
    print("| variant | retrieval s (median) | prune/rerank s median | mean | p95 | tokens/question | GFLOPs/question |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    front = {}
    for v in variants:
        rs = [r for r in rows if r["kind"] == "front" and r["variant"] == v["name"]]
        front[v["name"]] = {r["qid"]: r for r in rs}
        fs = np.array([r["front_s"] for r in rs])
        toks = np.mean([r["front_tokens"] for r in rs])
        model = "provence" if setups[v["method"]][2] else "reranker" if setups[v["method"]][3] else None
        gflops = 2 * params[model]["non_embedding"] * toks / 1e9 if model else 0.0
        s = {"retrieval_s_median": float(np.median([r["retrieval_s"] for r in rs])),
             "front_s_median": float(np.median(fs)), "front_s_mean": float(fs.mean()),
             "front_s_p95": float(np.percentile(fs, 95)), "front_tokens_mean": float(toks),
             "front_gflops_mean": gflops}
        summary["front"][v["name"]] = s
        print(f"| {v['name']} | {s['retrieval_s_median']:.3f} | {s['front_s_median']:.2f} | "
              f"{s['front_s_mean']:.2f} | {s['front_s_p95']:.2f} | {toks:.0f} | {gflops:.1f} |")

    print("\n| reader method | budget | reader s median | mean | input tokens | same answer as eval |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    reader_rows = {}
    for m, b in sorted(want):
        rs = [r for r in rows if r["kind"] == "reader" and r["variant"] == m and r["budget"] == b]
        reader_rows[(m, b)] = {r["qid"]: r for r in rs}
        t = np.array([r["reader_s"] for r in rs])
        s = {"reader_s_median": float(np.median(t)), "reader_s_mean": float(t.mean()),
             "input_tokens_mean": float(np.mean([r["input_tokens"] for r in rs])),
             "same_answer_rate": float(np.mean([r["same_answer_as_eval"] for r in rs]))}
        summary["reader"][f"{m}@{b}"] = s
        print(f"| {m} | {b} | {s['reader_s_median']:.2f} | {s['reader_s_mean']:.2f} | "
              f"{s['input_tokens_mean']:.0f} | {s['same_answer_rate']:.2f} |")

    # end to end per question: retrieval + prune/rerank + reader, paired against the reference
    ref = bc["reference"]
    print(f"\nend to end (retrieval + prune/rerank + reader), paired over the same questions; "
          f"ratio = {ref} / method")
    print("| method | budget | median s | mean s | mean-time ratio (95% CI) | GFLOPs/question |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    bench_qids = sorted({r["qid"] for r in rows})
    for b in bc["reader_budgets"]:
        tot = {}
        for m in bc["reader_methods"]:
            tot[m] = np.array([front[m][q]["retrieval_s"] + front[m][q]["front_s"]
                               + reader_rows[(m, b)][q]["reader_s"] for q in bench_qids])
        for m in bc["reader_methods"]:
            ci = ratio_ci(tot[ref], tot[m])
            reader_tok = np.mean([r["input_tokens"] + r["output_tokens"] for r in reader_rows[(m, b)].values()])
            gflops = (summary["front"][m]["front_gflops_mean"]
                      + 2 * bc["reader_non_embedding_params"] * reader_tok / 1e9)
            summary["end_to_end"][f"{m}@{b}"] = {"median_s": float(np.median(tot[m])),
                                                 "mean_s": float(tot[m].mean()), f"ratio_{ref}_over": ci,
                                                 "gflops_mean": float(gflops)}
            print(f"| {m} | {b} | {np.median(tot[m]):.2f} | {tot[m].mean():.2f} | "
                  f"{ci['ratio']:.1f}x ({ci['ci_low']:.1f}-{ci['ci_high']:.1f}) | {gflops:.0f} |")

    print("\nprune/rerank step alone, mean-time ratio vs each reranked method (95% CI)")
    for v in variants:
        if not setups[v["method"]][2]:
            continue
        a = np.array([front[v["name"]][q]["front_s"] for q in bench_qids])
        for m in [x for x in front if setups[next(y["method"] for y in variants if y["name"] == x)][3]]:
            ci = ratio_ci(a, np.array([front[m][q]["front_s"] for q in bench_qids]))
            summary["front"][v["name"]][f"ratio_over_{m}"] = ci
            print(f"  {v['name']} / {m}: {ci['ratio']:.1f}x ({ci['ci_low']:.1f}-{ci['ci_high']:.1f})")

    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_path} and {summary_path}")


def run_export(cfg: dict, force: bool) -> None:
    """Export the exact per-question inputs of every GPU-timed stage, so the GPU
    benchmark (modal_app/gpu_bench.py) needs no corpus, index or retriever:
    Provence's candidate paragraphs (raw text, as Provence gets them), each reranked
    method's candidates, and the reader contexts the eval runs logged."""
    from ragsplit.run import eval_question_ids, prepare_corpus, prepare_splits, setup_method

    ex = cfg["export"]
    paragraphs, questions = prepare_corpus(cfg, force=False)
    sp = prepare_splits(cfg, questions, force=False)
    para_by_id = {p.id: p for p in paragraphs}
    q_by_id = {q.id: q for q in questions}
    qids = eval_question_ids(cfg, sp)
    cache_dir = Path(cfg["data"]["data_dir"]) / "cache"

    def ranked_for(method: str) -> dict[str, list]:
        spec, units, retriever, *_ = setup_method(method, cfg, sp, paragraphs, q_by_id, cache_dir)
        return {qid: [units[j] for j in retriever.search([q_by_id[qid].question])[0]] for qid in qids}

    out = {"provence_top_n": ex["provence_top_n"], "rerank_top_n": cfg["rerank"]["top_n"],
           "questions": {}, "contexts": {}}
    # Provence's inputs exactly as the B2 run saw them (its cache); retrieval is not
    # bit-for-bit repeatable (float noise reorders near-ties), so recomputing could differ
    *_, pruner, _, _ = setup_method("B2", cfg, sp, paragraphs, q_by_id, cache_dir)
    missing = [q for q in qids if q not in pruner.cache]
    if missing:
        raise SystemExit(f"no cached B2 Provence inputs for {len(missing)} questions; run B2 first")
    cands = {m: ranked_for(m) for m in ex["rerank_methods"]}
    for qid in qids:
        q = q_by_id[qid]
        out["questions"][qid] = {
            "question": q.question,
            "gold": [a.text for a in q.answers],
            "provence_paras": [para_by_id[pid].context
                               for pid in pruner.cache[qid]["unit_ids"][: ex["provence_top_n"]]],
            "rerank_candidates": {m: [u.text for u in cands[m][qid][: cfg["rerank"]["top_n"]]]
                                  for m in ex["rerank_methods"]},
        }
    want = {(m, b) for m in ex["reader_methods"] for b in ex["reader_budgets"]}
    for log in ex["reader_logs"]:
        for line in open(log):
            r = json.loads(line)
            if (r["method"], r["budget"]) in want and r["qid"] in out["questions"]:
                out["contexts"][f"{r['method']}|{r['budget']}|{r['qid']}"] = r["context"]
    n_expected = len(want) * len(qids)
    if len(out["contexts"]) != n_expected:
        raise SystemExit(f"found {len(out['contexts'])} logged contexts, expected {n_expected}")
    path = Path(cfg.get("results_dir", "results")) / f"{cfg['name']}.json"
    path.write_text(json.dumps(out) + "\n")
    print(f"wrote {path}: {len(qids)} questions, {len(out['contexts'])} reader contexts, "
          f"{path.stat().st_size / 1e6:.1f} MB")
