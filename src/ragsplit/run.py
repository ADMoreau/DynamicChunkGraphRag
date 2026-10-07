"""Experiment entry point: python -m ragsplit.run --config configs/<name>.yaml"""

import argparse
import json
from pathlib import Path
import time

from ragsplit import data, splits
from ragsplit.config import load_config

MIN_ANSWER_MAP_RATE = 0.99


def derived_dir(cfg: dict) -> Path:
    d = cfg["data"]
    name = f"{d['hf_id'].split('/')[-1]}_{d['split']}_{d['spacy_model']}"
    if d.get("dataset") == "hotpotqa":
        name = f"hotpotqa_{d['split']}_n{d['n_questions']}_seed{d['sample_seed']}"
    if d.get("dataset") == "syllabusqa":
        name = f"syllabusqa_chunk{d['chunk_sents']}_{d['spacy_model']}"
    if d.get("dataset") == "musique":
        name = f"musique_dev_{d['spacy_model']}"
    return Path(d["data_dir"]) / "derived" / name


def splits_path(cfg: dict) -> Path:
    s = cfg["splits"]
    name = (f"splits_seed{s['seed']}_past{s['past_frac']}"
            f"_tune{s['tuning_frac']}_pilot{s['pilot_size']}.json")
    return derived_dir(cfg) / name


def prepare_corpus(cfg: dict, force: bool) -> tuple[list[data.Paragraph], list[data.Question]]:
    out = derived_dir(cfg)
    p_path, q_path = out / "paragraphs.jsonl", out / "questions.jsonl"
    if p_path.exists() and q_path.exists() and not force:
        print(f"[cache] using {out}  (pass --force to rebuild)")
        return data.read_paragraphs(p_path), data.read_questions(q_path)

    d = cfg["data"]
    if d.get("dataset") == "hotpotqa":
        from ragsplit import hotpot

        rows = hotpot.load_rows(d["hf_id"], str(Path(d["data_dir"]) / "hf_cache"))
        paragraphs, questions = hotpot.build_corpus(rows, d["n_questions"], d["sample_seed"])
        data.write_jsonl(p_path, paragraphs)
        data.write_jsonl(q_path, questions)
        stats = {"paragraphs": len(paragraphs), "questions": len(questions),
                 "sentences": sum(len(p.sents) for p in paragraphs),
                 "answer_in_text": sum(q.answers[0].start >= 0 for q in questions),
                 "support_sentences_mean": sum(len(q.support) for q in questions) / max(1, len(questions))}
        (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(f"wrote {out}: {stats}")
        return paragraphs, questions
    if d.get("dataset") == "musique":
        from ragsplit import musique

        paragraphs, questions = musique.build_corpus(
            musique.load_rows(d["hf_id"], str(Path(d["data_dir"]) / "hf_cache")), d["spacy_model"])
        data.write_jsonl(p_path, paragraphs)
        data.write_jsonl(q_path, questions)
        stats = {"paragraphs": len(paragraphs), "questions": len(questions),
                 "sentences": sum(len(p.sents) for p in paragraphs),
                 "qtypes": dict(__import__("collections").Counter(q.qtype for q in questions))}
        (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(f"wrote {out}: {stats}")
        return paragraphs, questions
    if d.get("dataset") == "syllabusqa":
        from ragsplit import syllabus

        paragraphs, questions = syllabus.build_corpus(d["raw_dir"], d["spacy_model"], d["chunk_sents"])
        data.write_jsonl(p_path, paragraphs)
        data.write_jsonl(q_path, questions)
        stats = {"paragraphs": len(paragraphs), "questions": len(questions),
                 "sentences": sum(len(p.sents) for p in paragraphs),
                 "qtypes": dict(__import__("collections").Counter(q.qtype for q in questions))}
        (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(f"wrote {out}: {stats}")
        return paragraphs, questions
    rows = data.load_squad_rows(d["hf_id"], d["split"], Path(d["data_dir"]) / "hf_cache")
    paragraphs, questions = data.build_corpus(rows)
    print(f"loaded {len(rows)} rows -> {len(paragraphs)} paragraphs; segmenting with {d['spacy_model']}")
    nlp = data.load_spacy(d["spacy_model"])
    for p, sents in zip(paragraphs, data.segment([p.context for p in paragraphs], nlp)):
        p.sents = sents
    data.map_all(paragraphs, questions)

    data.write_jsonl(p_path, paragraphs)
    data.write_jsonl(q_path, questions)
    stats = data.mapping_stats(paragraphs, questions)
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {out}")
    return paragraphs, questions


def prepare_splits(cfg: dict, questions: list[data.Question], force: bool) -> dict:
    path = splits_path(cfg)
    if path.exists() and not force:
        print(f"[cache] using {path}")
        return json.loads(path.read_text())

    s = cfg["splits"]
    gold = {q.id: (None if q.gold_sent is None else (q.paragraph_id, q.gold_sent)) for q in questions}
    past, future = splits.split_past_future(list(gold), s["past_frac"], s["seed"])
    tuning, build = splits.split_tuning(past, s["tuning_frac"], s["seed"])
    pilot = splits.pilot_slice(future, s["pilot_size"], s["seed"])
    result = {
        "config": s,
        "past": past,
        "future": future,
        "tuning": tuning,
        "build": build,
        "pilot": pilot,
        # final runs: future vs all past; tuning runs: tuning vs build
        "buckets_future": splits.overlap_buckets({q: gold[q] for q in future}, [gold[q] for q in past]),
        "buckets_tuning": splits.overlap_buckets({q: gold[q] for q in tuning}, [gold[q] for q in build]),
    }
    path.write_text(json.dumps(result) + "\n")
    print(f"wrote {path}")
    return result


def summarize(paragraphs, questions, sp: dict) -> dict:
    stats = data.mapping_stats(paragraphs, questions)
    for name in ("past", "future", "tuning", "build", "pilot"):
        stats[f"n_{name}"] = len(sp[name])
    for name in ("buckets_future", "buckets_tuning"):
        counts: dict[str, int] = {}
        for b in sp[name].values():
            counts[b] = counts.get(b, 0) + 1
        stats[name] = counts
    pilot_b = [sp["buckets_future"][q] for q in sp["pilot"]]
    stats["buckets_pilot"] = {b: pilot_b.count(b) for b in sorted(set(pilot_b))}
    return stats


def run_prepare(cfg: dict, force: bool) -> None:
    paragraphs, questions = prepare_corpus(cfg, force)
    sp = prepare_splits(cfg, questions, force)
    stats = summarize(paragraphs, questions, sp)
    print(json.dumps(stats, indent=2))
    if stats["answer_map_rate"] < MIN_ANSWER_MAP_RATE:
        raise SystemExit(
            f"FAIL: only {stats['answer_map_rate']:.2%} of answers map to a sentence "
            f"(need {MIN_ANSWER_MAP_RATE:.0%})"
        )


# --- eval ------------------------------------------------------------------

# Method id -> unit set and options. Methods in one run share reader, retriever and budgets.
METHODS = {
    "B1": {"units": "paragraphs"},
    "B2": {"units": "paragraphs", "prune": True},
    "B3": {"units": "sentences"},
    "B4": {"units": "paragraphs", "prior": True},
    # ORACLE: pieces from the evaluated questions' own gold sentences (upper bound, Gate A).
    "O-ORACLE": {"units": "paragraphs+pieces", "pieces_from": "future_gold"},
    # L+P and L-k<n>: extracting a sentence splits its node into before / sentence / after.
    "L+P": {"units": "split", "pieces_from": "past_gold", "prior": True},
    # K-*: sentences as reader text, retrieved through small machine-facing keys (keys.py).
    "K-clause": {"units": "sentences", "keys": ["clause"]},
    "K-triple": {"units": "sentences", "keys": ["triple"]},
    "K-all": {"units": "sentences", "keys": ["sentence", "clause", "triple"]},
    # learned: clause/triple keys only where past questions pointed (k = 1); sentence keys everywhere
    "K-all-k1": {"units": "sentences", "keys": ["sentence", "clause", "triple"],
                 "pieces_from": "past_gold", "k": 1},
}


def method_spec(method: str, cfg: dict) -> dict:
    """Spec for a method id. L-k<n> is L with threshold k = n. Prior weights and
    L+P's k come from the config's `prior` block (tuned on the tuning set). A "+R"
    suffix adds the light cross-encoder reranker (config block `rerank`). B2-n<k> is B2
    with Provence reading only the top k retrieved paragraphs (a cheaper setting)."""
    if method.endswith("+R"):
        return {**method_spec(method[:-2], cfg), "rerank": True}
    if method.startswith("T-"):  # beam traversal: T-<base>-b<beam>-d<depth>-k<candidates>[-x]
        parts = method.split("-")
        if parts[1].startswith("Cs"):  # sentence base + learned finer cuts: Cs<k sentence><k clause><k phrase>
            k = dict(zip(("sentence", "clause", "phrase"), map(int, parts[1][2:])))
            base_spec = {"units": "cracked", "k": k, "context_index": True, "base": "sentence"}
        elif parts[1] == "S":  # plain sentence tiles with spans (matches a graph built with units: sentences)
            base_spec = {"units": "cracked", "k": {"sentence": 10**9, "clause": 10**9, "phrase": 10**9},
                         "context_index": False, "base": "sentence"}
        else:
            base = {"B1": "B1", "B3": "B3", "E2": "E-k2", "E3": "E-k3", "C123": "C-ctx-k1-2-3"}[parts[1]]
            base_spec = method_spec(base, cfg)
        p = {x[0]: int(x[1:]) for x in parts[2:] if x[0] in "bdk" and x[1:].isdigit()}
        return {**base_spec, "beam": {"beam": p["b"], "depth": p["d"], "n_candidates": p["k"],
                                                   "cross_paragraph_adjacency": "x" in parts[2:],
                                                   "stop": "flat" if "sf" in parts[2:] else "fixed"}}
    if method.startswith("H-"):  # two-hop expansion over the graph: H-s<seeds>-k<candidates>-<merge|interleave>
        parts = method.split("-")
        k = dict(zip(("sentence", "clause", "phrase"), cfg["graph"]["unit_k"]))
        return {"units": "cracked", "k": k, "context_index": True,
                "hop": {"n_seeds": int(parts[1][1:]), "n_candidates": int(parts[2][1:]), "mode": parts[3]}}
    if method.startswith("G-"):  # graph traversal over the built graph: G-a<alpha>-r<related w>-q<concept w>
        p = {part[0]: float(part[1:]) for part in method.split("-")[1:]}
        k = dict(zip(("sentence", "clause", "phrase"), cfg["graph"]["unit_k"]))
        return {"units": "cracked", "k": k, "context_index": True,
                "graph": {"alpha": p["a"], "related_w": p["r"], "concept_w": p["q"]}}
    if method.startswith("E-k"):  # evidence-span units: E-k<questions needed>
        return {"units": "evidence", "k_evidence": int(method[3:])}
    if method.startswith("C-"):  # cracked: C-<own|ctx>-k<sentence>-<clause>-<phrase>
        _, rep, ks = method.split("-", 2)
        k = dict(zip(("sentence", "clause", "phrase"), map(int, ks[1:].split("-"))))
        # rep: own (own text), ctx (context-aware), ctxs (context-aware, sentence base)
        return {"units": "cracked", "k": k, "context_index": rep in ("ctx", "ctxs"),
                "base": "sentence" if rep == "ctxs" else "paragraph"}
    if method.startswith("B2-n"):
        return {**METHODS["B2"], "prune_top_n": int(method[4:])}
    if method.startswith("L-k"):
        return {"units": "split", "pieces_from": "past_gold", "k": int(method[3:])}
    spec = dict(METHODS[method])
    if spec.get("prior"):
        spec["w"] = cfg["prior"]["w"][method]
        spec["tau"] = cfg["prior"].get("tau", {}).get(method, 0.0)
        if method == "L+P":
            spec["k"] = cfg["prior"]["lp_k"]
    return spec


def reference_qids(cfg: dict, sp: dict) -> list[str]:
    """Past questions that may shape units and priors: the build split while
    tuning, all past questions otherwise. Never future questions."""
    return sp["build"] if cfg["eval"]["questions"] == "tuning" else sp["past"]


def piece_keys_for(spec: dict, cfg: dict, sp: dict, q_by_id: dict) -> set | None:
    source = spec.get("pieces_from")
    if source is None:
        return None
    if source == "future_gold":
        if cfg["eval"]["questions"] not in ("pilot", "future"):
            raise SystemExit("O-ORACLE is only defined on future questions")
        qs = [q_by_id[qid] for qid in sp["future"]]
        return {(q.paragraph_id, q.gold_sent) for q in qs if q.gold_sent is not None}
    if source == "past_gold":
        # a sentence becomes a piece once at least k distinct past questions needed it
        counts: dict[tuple[str, int], int] = {}
        for qid in reference_qids(cfg, sp):
            q = q_by_id[qid]
            if q.gold_sent is not None:
                key = (q.paragraph_id, q.gold_sent)
                counts[key] = counts.get(key, 0) + 1
        return {key for key, n in counts.items() if n >= spec["k"]}
    raise ValueError(f"unknown pieces_from {source!r}")


def setup_method(method: str, cfg: dict, sp: dict, paragraphs, q_by_id: dict, cache_dir: Path):
    """Units, retriever, and optional pruner / prior / reranker for one method."""
    from ragsplit.retrieve import Retriever, units_key
    from ragsplit.units import build_units

    spec = method_spec(method, cfg)
    piece_keys = piece_keys_for(spec, cfg, sp, q_by_id)
    if spec["units"] == "evidence":
        from ragsplit.crack import evidence_groups, evidence_units

        past = [q_by_id[q] for q in reference_qids(cfg, sp)]
        units = evidence_units(paragraphs, evidence_groups(paragraphs, past, spec["k_evidence"]))
    elif spec["units"] == "cracked":
        from ragsplit.crack import crack_units, extracted_spans, parse_sentences
        from ragsplit.units import sentence_units

        parses = parse_sentences(paragraphs, cfg["data"]["spacy_model"], Path(cache_dir) / "crack" /
                                 f"parses_{units_key(sentence_units(paragraphs))}.json")
        past = [q_by_id[q] for q in reference_qids(cfg, sp)]
        units = crack_units(paragraphs, extracted_spans(paragraphs, past, parses, spec["k"]),
                            spec["context_index"], spec.get("base", "paragraph"))
    else:
        units = build_units(spec["units"], paragraphs, piece_keys)
    if spec.get("keys"):
        from ragsplit.keys import KeyedRetriever, build_keys, parse_keys

        parsed = parse_keys(units, paragraphs, cfg["data"]["spacy_model"],
                            Path(cache_dir) / "keys" / f"parsed_{units_key(units)}.jsonl")
        keys, targets = build_keys(units, paragraphs, spec["keys"], parsed,
                                   piece_keys if spec.get("pieces_from") else None)
        retriever = KeyedRetriever(keys, targets, cfg["retrieval"], cache_dir)
    else:
        retriever = Retriever(units, cfg["retrieval"], cache_dir)
    if spec.get("graph"):
        from ragsplit.traverse import GraphTraverser

        graph = json.loads(Path(cfg["graph"]["path"]).read_text())
        retriever = GraphTraverser(graph, [u.id for u in units], retriever, **spec["graph"])
    if spec.get("beam"):
        from ragsplit.rerank import CrossEncoderReranker
        from ragsplit.traverse import BeamTraverser

        gpath = cfg.get("graph", {}).get("path")
        graph = None
        if gpath and Path(gpath).exists():
            graph = json.loads(Path(gpath).read_text())
            ids = {u.id for u in units}
            if sum(t["id"] in ids for t in graph["text_nodes"]) < 0.9 * len(graph["text_nodes"]):
                graph = None  # the graph was built for other units: adjacency moves only
        retriever = BeamTraverser(units, retriever, CrossEncoderReranker(cfg["rerank"]), graph, **spec["beam"])
    if spec.get("hop"):
        from ragsplit.rerank import CrossEncoderReranker
        from ragsplit.traverse import HopExpander

        graph = json.loads(Path(cfg["graph"]["path"]).read_text())
        retriever = HopExpander(graph, units, retriever, CrossEncoderReranker(cfg["rerank"]), **spec["hop"])
    retriever.encoder()  # load before timing
    pruner = prior = None
    if spec.get("prune"):
        from ragsplit.prune import ProvencePruner

        prune_cfg = {**cfg["prune"], "top_n": spec.get("prune_top_n", cfg["prune"]["top_n"])}
        pruner = ProvencePruner(prune_cfg, cache_dir, units_key(units))
    if spec.get("prior"):
        from ragsplit.prior import QueryPrior

        prior = QueryPrior(units, [q_by_id[q] for q in reference_qids(cfg, sp)],
                           retriever.encoder(), cache_dir, cfg["retrieval"]["dense_model"])
    reranker = None
    if spec.get("rerank"):
        from ragsplit.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(cfg["rerank"])
    return spec, units, retriever, pruner, prior, reranker

_DOC_ID = __import__("re").compile(r"^([a-z]+\d+)p")


def doc_of(pid: str) -> str:
    """Document a paragraph id belongs to (SQuAD a00p000 -> a00, SyllabusQA s04p021 -> s04)."""
    m = _DOC_ID.match(pid)
    return m.group(1) if m else pid


def scoped(ranked: list, q, cfg: dict) -> list:
    """With eval.scope = document, keep only units from the question's own document (e.g. a
    course assistant that knows which syllabus the student is asking about)."""
    if cfg["eval"].get("scope") != "document":
        return ranked
    d = doc_of(q.paragraph_id)
    return [u for u in ranked if doc_of(u.paragraph_id) == d]


# Config blocks that must match for runs to be compared in one table or plot.
SHARED_KEYS = ("reader", "retrieval", "data", "splits")


def comparable_rows(cfg: dict) -> list[dict]:
    """Rows from earlier runs listed in eval.compare_with, after checking they
    used the same reader, retriever, data, splits and question set. Budgets may
    differ between runs; summaries are per (method, budget)."""
    from ragsplit import report

    rows = []
    for log in cfg["eval"].get("compare_with", []):
        log = Path(log)
        other = json.loads(log.with_name(log.stem + ".config.json").read_text())
        for key in SHARED_KEYS:
            if other[key] != cfg[key]:
                raise SystemExit(f"cannot compare with {log}: '{key}' differs")
        for key in ("questions", "limit"):
            if other["eval"].get(key) != cfg["eval"].get(key):
                raise SystemExit(f"cannot compare with {log}: eval.{key} differs")
        rows += report.read_rows(log)
    return rows


def eval_question_ids(cfg: dict, sp: dict) -> list[str]:
    which = cfg["eval"]["questions"]
    if which not in ("pilot", "future", "tuning"):
        raise ValueError(f"eval.questions must be pilot, future or tuning, not {which!r}")
    limit = cfg["eval"].get("limit")  # smoke tests only
    return sp[which][:limit] if limit else sp[which]


def run_eval(cfg: dict, force: bool) -> None:
    from ragsplit import report
    from ragsplit.context import pack
    from ragsplit.metrics import answer_in_context, evidence_hit, exact_match, f1
    from ragsplit.reader import load_reader

    paragraphs, questions = prepare_corpus(cfg, force=False)
    sp = prepare_splits(cfg, questions, force=False)
    para_by_id = {p.id: p for p in paragraphs}
    q_by_id = {q.id: q for q in questions}
    qids = eval_question_ids(cfg, sp)
    buckets = sp["buckets_tuning"] if cfg["eval"]["questions"] == "tuning" else sp["buckets_future"]

    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["name"]
    log_path = results_dir / f"{name}.jsonl"
    if force and log_path.exists():
        log_path.unlink()
    done = {(r["method"], r["budget"], r["qid"]) for r in report.read_rows(log_path)}
    if done:
        print(f"[resume] {len(done)} rows already in {log_path}")
    (results_dir / f"{name}.config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    comparable_rows(cfg)  # fail now, not after hours of reader calls
    reader = load_reader(cfg["reader"])
    budgets = cfg["eval"]["budgets"]
    cache_dir = Path(cfg["data"]["data_dir"]) / "cache"
    index_sizes = {}

    with log_path.open("a") as log:
        for method in cfg["eval"]["methods"]:
            spec, units, retriever, pruner, prior, reranker = setup_method(
                method, cfg, sp, paragraphs, q_by_id, cache_dir)
            index_sizes[method] = {"n_units": len(units), "n_keys": getattr(retriever, "n_keys", len(units)),
                                   "n_pieces": sum(u.kind == "piece" for u in units),
                                   "indexed_tokens": sum(reader.count(u.text) for u in units)}
            if prior:
                index_sizes[method]["prior_units"] = prior.n_attached_units
            todo = [q for q in qids if any((method, b, q) not in done for b in budgets)]
            print(f"{method}: {len(units)} units, {len(todo)} questions to run")
            for i, qid in enumerate(todo, 1):
                q = q_by_id[qid]
                t0 = time.perf_counter()
                ranked = scoped([units[j] for j in retriever.search(
                    [q.question], prior=prior, prior_weight=spec.get("w", 0.0),
                    prior_min_sim=spec.get("tau", 0.0))[0]], q, cfg)
                retrieval_s = time.perf_counter() - t0
                prune_s = 0.0
                if pruner:
                    ranked, prune_s = pruner.prune(qid, q.question, ranked, para_by_id)
                rerank_s = 0.0
                if reranker:
                    ranked, rerank_s = reranker.rerank(q.question, ranked)
                p = para_by_id[q.paragraph_id]
                s, e = p.sents[q.gold_sent]
                gold_text = p.context[s:e]
                truths = [a.text for a in q.answers]
                for budget in budgets:
                    if (method, budget, qid) in done:
                        continue
                    t1 = time.perf_counter()
                    packed = pack(ranked, budget, reader.count, reader.truncate)
                    pack_s = time.perf_counter() - t1
                    out = reader.answer(q.question, packed.text)
                    row = {
                        "method": method, "budget": budget, "qid": qid,
                        "bucket": buckets[qid],
                        "retrieved": [u.id for u in ranked[:20]],
                        "unit_ids": packed.unit_ids,
                        "truncated": packed.truncated,
                        "context_tokens": reader.count(packed.text),
                        "input_tokens": out["input_tokens"],
                        "output_tokens": out["output_tokens"],
                        "answer": out["answer"],
                        "raw_answer": out["raw_answer"],
                        "gold": truths,
                        "em": exact_match(out["answer"], truths),
                        "f1": f1(out["answer"], truths),
                        "evidence_hit": evidence_hit(packed.text, gold_text),
                        "answer_in_context": answer_in_context(packed.text, truths),
                        "retrieval_latency_s": retrieval_s,
                        "prune_latency_s": prune_s,
                        "rerank_latency_s": rerank_s,
                        "reader_latency_s": out["reader_latency_s"],
                        "latency_s": retrieval_s + prune_s + rerank_s + pack_s + out["reader_latency_s"],
                        "context": packed.text,
                    }
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  {method} {i}/{len(todo)}", flush=True)

    reader.close()
    write_report(cfg, index_sizes)


def run_tune(cfg: dict, force: bool) -> None:
    """Pick prior weights (B4, L+P) and L+P's k on the tuning set, retrieval only.

    Units and attachments come from the build split; the score is evidence hit
    averaged over the eval budgets (no reader calls)."""
    from functools import lru_cache
    import itertools

    import numpy as np

    from ragsplit.context import pack
    from ragsplit.metrics import evidence_hit
    from ragsplit.reader import load_reader
    from ragsplit.retrieve import rrf

    if cfg["eval"]["questions"] != "tuning":
        raise SystemExit("tune stage must run on eval.questions: tuning")
    paragraphs, questions = prepare_corpus(cfg, force=False)
    sp = prepare_splits(cfg, questions, force=False)
    para_by_id = {p.id: p for p in paragraphs}
    q_by_id = {q.id: q for q in questions}
    qids = eval_question_ids(cfg, sp)
    budgets = cfg["eval"]["budgets"]
    cache_dir = Path(cfg["data"]["data_dir"]) / "cache"
    tok = load_reader({**cfg["reader"], "vocab_only": True})
    count = lru_cache(maxsize=None)(tok.count)
    grid = cfg["tune"]

    gold_text = {}
    for qid in qids:
        q = q_by_id[qid]
        p = para_by_id[q.paragraph_id]
        s, e = p.sents[q.gold_sent]
        gold_text[qid] = p.context[s:e]

    def score(units, lists_by_q, w):
        hits = {b: 0 for b in budgets}
        top1 = 0
        for qid, (bm, dn, pr) in lists_by_q.items():
            ranking = rrf([bm, dn, pr], k=cfg["retrieval"]["rrf_k"], weights=[1.0, 1.0, w]) if w else \
                rrf([bm, dn], k=cfg["retrieval"]["rrf_k"])
            ranked = [units[j] for j in ranking[: cfg["retrieval"]["depth"]]]
            q = q_by_id[qid]
            top1 += (q.paragraph_id, q.gold_sent) in ranked[0].covers()
            for b in budgets:
                hits[b] += evidence_hit(pack(ranked, b, count, tok.truncate).text, gold_text[qid])
        n = len(lists_by_q)
        by_budget = {b: hits[b] / n for b in budgets}
        return {"evidence_hit_mean": float(np.mean(list(by_budget.values()))),
                "evidence_hit": by_budget, "top1_contains_gold": top1 / n}

    results = []
    texts = [q_by_id[qid].question for qid in qids]

    for method, ks in (("B4", [None]), ("L+P", grid["lp_k"])):
        for k in ks:
            tune_cfg = {**cfg, "prior": {"w": {method: 1.0}, "lp_k": k}}
            spec, units, retriever, _, prior, _ = setup_method(
                method, tune_cfg, sp, paragraphs, q_by_id, cache_dir)
            bm = retriever.bm25_rank(texts)
            dn = retriever.dense_rank(retriever.embed_queries(texts))
            plain = retriever.encoder().encode(texts, batch_size=64, normalize_embeddings=True,
                                               convert_to_numpy=True, show_progress_bar=False)
            scored = {qid: prior.scored(pe) for qid, pe in zip(qids, plain)}
            for tau, w in itertools.product(grid["tau"], grid["w"]):
                if w == 0 and tau != grid["tau"][0]:
                    continue  # tau is irrelevant without the prior
                lists = {
                    qid: (b, d, [i for i, s in zip(*scored[qid]) if s >= tau][: cfg["retrieval"]["depth"]])
                    for qid, b, d in zip(qids, bm, dn)
                }
                r = {"method": method, "k": k, "w": w, "tau": tau, "n_units": len(units),
                     **score(units, lists, w)}
                results.append(r)
                print(f"{method} k={k} tau={tau} w={w}: evidence hit {r['evidence_hit_mean']:.4f}, "
                      f"top-1 has gold {r['top1_contains_gold']:.3f}", flush=True)

    best = {}
    for method in ("B4", "L+P"):
        rs = [r for r in results if r["method"] == method]
        # ties go to the smaller w, then the smaller k
        best[method] = max(rs, key=lambda r: (round(r["evidence_hit_mean"], 6), -r["w"], -(r["k"] or 0)))
        r0 = next(r for r in rs if r["w"] == 0 and r["k"] == best[method]["k"])
        best[method]["gain_over_no_prior"] = best[method]["evidence_hit_mean"] - r0["evidence_hit_mean"]
    out = Path(cfg.get("results_dir", "results")) / f"{cfg['name']}.json"
    out.write_text(json.dumps({"best": best, "grid": results}, indent=2) + "\n")
    tok.close()
    print(f"wrote {out}")
    for method, r in best.items():
        print(f"best {method}: k={r['k']} tau={r['tau']} w={r['w']} evidence hit "
              f"{r['evidence_hit_mean']:.4f} (gain over no prior {r['gain_over_no_prior']:+.4f})")


def run_probe(cfg: dict, force: bool) -> None:
    """Retrieval-only comparison: evidence hit, context tokens and added latency per
    method, budget and bucket, with the same packing as eval but no reader calls."""
    from collections import defaultdict
    from functools import lru_cache

    import numpy as np

    from ragsplit.context import SEP, pack
    from ragsplit.metrics import answer_located, evidence_hit
    from ragsplit.reader import load_reader
    from ragsplit.units import title_prefix

    paragraphs, questions = prepare_corpus(cfg, force=False)
    sp = prepare_splits(cfg, questions, force=False)
    para_by_id = {p.id: p for p in paragraphs}
    q_by_id = {q.id: q for q in questions}
    qids = eval_question_ids(cfg, sp)
    buckets = sp["buckets_tuning"] if cfg["eval"]["questions"] == "tuning" else sp["buckets_future"]
    budgets = cfg["eval"]["budgets"]
    cache_dir = Path(cfg["data"]["data_dir"]) / "cache"
    tok = load_reader({**cfg["reader"], "vocab_only": True})
    count = lru_cache(maxsize=None)(tok.count)

    out = {}
    per_q = []  # per-question rows, for paired comparisons between methods
    for method in cfg["eval"]["methods"]:
        spec, units, retriever, pruner, prior, reranker = setup_method(
            method, cfg, sp, paragraphs, q_by_id, cache_dir)
        acc = defaultdict(lambda: {"hit": [], "ctx": [], "loc": [], "sup": [], "spans": []})
        top1, extra_s, total_s = [], [], []
        for qid in qids:
            q = q_by_id[qid]
            p = para_by_id[q.paragraph_id]
            s, e = p.sents[q.gold_sent]
            t_q = time.perf_counter()
            ranked = scoped([units[j] for j in retriever.search(
                [q.question], prior=prior, prior_weight=spec.get("w", 0.0),
                prior_min_sim=spec.get("tau", 0.0))[0]], q, cfg)
            secs = 0.0
            if pruner:
                ranked, t = pruner.prune(qid, q.question, ranked, para_by_id)
                secs += t
            if reranker:
                ranked, t = reranker.rerank(q.question, ranked)
                secs += t
            extra_s.append(secs)
            total_s.append(time.perf_counter() - t_q)  # everything: retrieval, traversal, pruning, reranking
            top1.append(bool(ranked) and answer_located(
                [(ranked[0].paragraph_id, 0, ranked[0].text)], q.paragraph_id, title_prefix(p.title),
                [a.text for a in q.answers]))
            by_id = {u.id: u for u in ranked}
            # the question's own answer(s) in its paragraph; multi-part answers (hops, syllabus
            # spans) living in other paragraphs are checked by spans_located instead
            truths = [a.text for a in q.answers if a.paragraph_id in (None, q.paragraph_id)]
            for b in budgets:
                packed = pack(ranked, b, count, tok.truncate)
                text = packed.text
                pieces = text.split(SEP) if text else []
                if len(pieces) != len(packed.unit_ids):  # a unit's own text held SEP
                    pieces = [by_id[u].text for u in packed.unit_ids]
                parts = [(by_id[u].paragraph_id,
                          (by_id[u].span or para_by_id[by_id[u].paragraph_id].sents[by_id[u].sent_lo])[0], t)
                         for u, t in zip(packed.unit_ids, pieces)]
                located = answer_located(parts, q.paragraph_id, title_prefix(p.title), truths)
                # multi-hop: every supporting sentence (both hops) reached the reader
                support = all(answer_located(parts, spid, title_prefix(para_by_id[spid].title),
                                             [para_by_id[spid].context[slice(*para_by_id[spid].sents[si])]])
                              for spid, si in q.support) if q.support else located
                if q.answers[0].start < 0:  # yes/no answers are never in the text
                    located = support
                # conjunctive answers (every part needed, e.g. SyllabusQA multi-part answers carry
                # their own paragraph): every answer span's text reached the reader. Unlike
                # support_located this does not require the rest of the span's sentence.
                if any(a.paragraph_id for a in q.answers):
                    spans = all(answer_located(parts, a.paragraph_id or q.paragraph_id,
                                               title_prefix(para_by_id[a.paragraph_id or q.paragraph_id].title), [a.text])
                                for a in q.answers)
                else:
                    spans = located
                for key in (b, (b, buckets[qid])):
                    acc[key]["hit"].append(evidence_hit(text, p.context[s:e]))
                    acc[key]["loc"].append(located)
                    acc[key]["sup"].append(support)
                    acc[key]["spans"].append(spans)
                    acc[key]["ctx"].append(count(text) if text else 0)
                row = {"method": method, "budget": b, "qid": qid, "bucket": buckets[qid],
                       "answer_located": bool(located), "support_located": bool(support),
                       "spans_located": bool(spans)}
                if cfg["eval"].get("dump_contexts"):  # for reader / judge runs on GPU
                    row["context"] = text
                per_q.append(row)
        out[method] = {
            "n_units": len(units),
            "n_keys": getattr(retriever, "n_keys", len(units)),
            "top1_has_answer": float(np.mean(top1)),
            "added_latency_mean_s": float(np.mean(extra_s)),
            "search_latency_median_s": float(np.median(total_s)),
            "by_budget": {str(k) if isinstance(k, int) else f"{k[0]}:{k[1]}":
                          {"evidence_hit": float(np.mean(v["hit"])), "answer_located": float(np.mean(v["loc"])),
                           "support_located": float(np.mean(v["sup"])),
                           "spans_located": float(np.mean(v["spans"])),
                           "context_tokens": float(np.mean(v["ctx"])),
                           "n": len(v["hit"])} for k, v in acc.items()},
        }
        print(f"{method}: {len(units)} units, {out[method]['n_keys']} keys, top-1 has answer {out[method]['top1_has_answer']:.3f}, "
              f"added latency {out[method]['added_latency_mean_s']:.2f} s/question, "
              f"whole search median {out[method]['search_latency_median_s']:.2f} s", flush=True)
    tok.close()

    path = Path(cfg.get("results_dir", "results")) / f"{cfg['name']}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    with path.with_name(f"{cfg['name']}_rows.jsonl").open("w") as f:
        for r in per_q:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path}")
    print("\nall supporting sentences located (multi-hop) / context tokens (all)")
    print("| budget | " + " | ".join(out) + " |")
    print("| ---: |" + " ---: |" * len(out))
    for b in budgets:
        print(f"| {b} | " + " | ".join(
            f"{out[m]['by_budget'][str(b)]['support_located']:.3f} / {out[m]['by_budget'][str(b)]['context_tokens']:.0f}"
            for m in out) + " |")
    for bucket in (None, "covered", "uncovered"):
        if bucket and f"{budgets[0]}:{bucket}" not in next(iter(out.values()))["by_budget"]:
            continue  # no questions in this bucket
        print(f"\nanswer located / context tokens ({bucket or 'all'})")
        print("| budget | " + " | ".join(out) + " |")
        print("| ---: |" + " ---: |" * len(out))
        for b in budgets:
            key = str(b) if bucket is None else f"{b}:{bucket}"
            print(f"| {b} | " + " | ".join(
                f"{out[m]['by_budget'][key]['answer_located']:.3f} / {out[m]['by_budget'][key]['context_tokens']:.0f}"
                for m in out) + " |")
    for bucket in (None, "covered", "uncovered"):
        if bucket and f"{budgets[0]}:{bucket}" not in next(iter(out.values()))["by_budget"]:
            continue  # no questions in this bucket
        print(f"\nevidence hit / context tokens ({bucket or 'all'})")
        print("| budget | " + " | ".join(out) + " |")
        print("| ---: |" + " ---: |" * len(out))
        for b in budgets:
            key = str(b) if bucket is None else f"{b}:{bucket}"
            print(f"| {b} | " + " | ".join(
                f"{out[m]['by_budget'][key]['evidence_hit']:.3f} / {out[m]['by_budget'][key]['context_tokens']:.0f}"
                for m in out) + " |")


def write_report(cfg: dict, index_sizes: dict | None = None) -> None:
    """Summary JSON, table and plots from this run's log plus eval.compare_with logs."""
    from ragsplit import report

    results_dir = Path(cfg.get("results_dir", "results"))
    name = cfg["name"]
    summary_path = results_dir / f"{name}_summary.json"
    sizes = {}
    for log in [*cfg["eval"].get("compare_with", []), summary_path]:
        p = Path(log)
        p = p if p.name.endswith("_summary.json") else p.with_name(p.stem + "_summary.json")
        if p.exists():
            sizes.update(json.loads(p.read_text()).get("index_sizes", {}))
    sizes.update(index_sizes or {})

    rows = report.read_rows(results_dir / f"{name}.jsonl") + comparable_rows(cfg)
    summary = report.summarize(rows)
    summary_path.write_text(json.dumps({"index_sizes": sizes, "summary": summary}, indent=2) + "\n")
    title = f"{name} ({cfg['eval']['questions']}, reader {Path(cfg['reader']['model_path']).name})"
    report.plot_f1_vs_budget(summary, results_dir / f"{name}_f1_vs_budget.png", title)
    report.plot_costs(summary, results_dir / f"{name}_costs.png", title)
    print(report.format_table(summary))
    print(f"index sizes: {sizes}")


def run_bench(cfg: dict, force: bool) -> None:
    from ragsplit.bench import run_bench as bench

    bench(cfg, force)


def run_graph(cfg: dict, force: bool) -> None:
    from ragsplit.graph import run_graph as graph

    graph(cfg, force)


def run_candidates(cfg: dict, force: bool) -> None:
    from ragsplit.adjudicate import run_candidates as cands

    cands(cfg, force)


def run_merge(cfg: dict, force: bool) -> None:
    from ragsplit.merge import run_merge as merge

    merge(cfg, force)


def run_export(cfg: dict, force: bool) -> None:
    from ragsplit.bench import run_export as export

    export(cfg, force)


STAGES = {"prepare": run_prepare, "eval": run_eval, "tune": run_tune, "probe": run_probe,
          "bench": run_bench, "export": run_export,
          "merge": run_merge, "candidates": run_candidates,
          "graph": run_graph}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="rebuild cached outputs")
    ap.add_argument("--report-only", action="store_true",
                    help="eval stage: rebuild summary and plots from existing logs")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.report_only:
        write_report(cfg)
        return
    if cfg["stage"] not in STAGES:
        raise SystemExit(f"unknown stage {cfg['stage']!r}; known: {sorted(STAGES)}")
    STAGES[cfg["stage"]](cfg, args.force)


if __name__ == "__main__":
    main()
