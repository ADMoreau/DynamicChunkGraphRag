"""Candidate pairs for same / related / different decisions, and the LLM prompt for them.

Cheap signals propose candidates generously; an LLM, reading both mentions in their own
sentences, decides. "Same" pairs become one node, "related" pairs a typed weaker edge,
"different" pairs nothing. Candidate sources:
  exact     the same normalized string in two different paragraphs (are they the same thing?)
  alias     a within-article short form / acronym and its long form
  embed     entity strings with embedding cosine >= a floor (HippoRAG uses 0.8)
  fragment  cracked clause/phrase pieces with embedding cosine >= the floor
"""

from collections import defaultdict
import random
import re

import numpy as np

from ragsplit.data import Paragraph, sentence_of
from ragsplit.merge import TYPE_GROUP, acronym_of, normalize

LABELS = ("SAME", "RELATED", "DIFFERENT")




RULES = """Rules:
SAME - the same thing under any name or spelling ("Denver Broncos" / "The Broncos", "USSR" / "Soviet"); the same absolute year or decade, even for different events ("1994" / "1994"); the same group, religion or nationality.
RELATED - not the same, but one contains, includes or approximates the other: part and whole ("ABC News" / "ABC"); a narrower time inside a wider one ("1 July 1851" / "1851", "1967" / "the 1960s"); narrower and broader concepts ("Homo habilis" / "Homo"); a place and its adjective or people ("China" / "Chinese").
DIFFERENT - everything else: similar-looking but distinct things ("Super Bowl XXXVIII" / "Super Bowl XXVIII"); the same number counting different things ("300 patents" / "300 cities"); different numbers or dates however close ("three" / "four", "1941" / "1942"); relative times with different referents ("the previous year" in two stories); siblings ("Small Catechism" / "Larger Catechism"); overlapping periods where neither contains the other."""


# the rules sit in the system message, which is identical for every pair, so the serving
# engine can cache it once (prefix caching) instead of re-reading it per pair
SYSTEM = ("You judge whether two short mentions from an encyclopedia refer to the same thing. "
          "Read each mention in its own sentence. Answer with exactly one word: SAME, RELATED or "
          "DIFFERENT.\n\n" + RULES)

# reference-set pairs whose texts appear as examples in RULES: left out when scoring
PROMPT_EXAMPLE_PAIRS = {("Denver Broncos", "The Broncos"), ("USSR", "Soviet"), ("1994", "1994"),
                        ("ABC News", "ABC"), ("1 July 1851", "1851"), ("1967", "in the 1960s"),
                        ("Homo habilis", "Homo"), ("China", "Chinese"), ("Super Bowl XXXVIII", "Super Bowl XXVIII"),
                        ("300", "300"), ("1942", "1941"), ("the previous year", "the previous year"),
                        ("Luther's Small Catechism", "the Larger Catechism")}


def is_prompt_example(pair: dict) -> bool:
    return (pair["a"]["text"], pair["b"]["text"]) in PROMPT_EXAMPLE_PAIRS or \
           (pair["b"]["text"], pair["a"]["text"]) in PROMPT_EXAMPLE_PAIRS


def prompt(a: dict, b: dict) -> str:
    return (f'A: "{a["text"]}"\n   in ({a["article"]}): "{a["context"]}"\n'
            f'B: "{b["text"]}"\n   in ({b["article"]}): "{b["context"]}"\n\n'
            "SAME, RELATED or DIFFERENT?")


def parse_label(text: str) -> str | None:
    m = re.search(r"\b(SAME|RELATED|DIFFERENT)\b", text.upper())
    return m.group(1) if m else None


def with_context(paragraphs: dict[str, Paragraph], pid: str, s: int, e: int, text: str) -> dict:
    p = paragraphs[pid]
    si = sentence_of(s, p.sents)
    ctx = p.context[slice(*p.sents[si])] if si is not None else p.context[max(0, s - 150): e + 150]
    return {"pid": pid, "s": s, "e": e, "text": text, "article": p.title.replace("_", " "), "context": ctx}


def candidates(paragraphs: list[Paragraph], mentions: list[dict], fragments: list[dict],
               ent_vec: np.ndarray, ent_texts: list[str], ent_rep: list[int],
               frag_vec: np.ndarray, floor: float, rng: random.Random,
               max_exact_pairs_per_group: int = 3, top: int = 10) -> list[dict]:
    """All candidate pairs (each with both mentions in context and its source and cosine)."""
    para = {p.id: p for p in paragraphs}
    m_ctx = lambda i: with_context(para, mentions[i]["pid"], mentions[i]["s"], mentions[i]["e"], mentions[i]["text"])
    out = []

    # exact: same string, different paragraphs; a few pairs per group
    groups = defaultdict(list)
    for i, m in enumerate(mentions):
        groups[(normalize(m["text"]), TYPE_GROUP[m["label"]])].append(i)
    for key, idx in groups.items():
        by_pid = {}
        for i in idx:
            by_pid.setdefault(mentions[i]["pid"], i)
        reps = list(by_pid.values())
        if len(reps) < 2:
            continue
        pairs = {tuple(sorted(rng.sample(reps, 2))) for _ in range(max_exact_pairs_per_group * 3)}
        for i, j in list(pairs)[:max_exact_pairs_per_group]:
            out.append({"source": "exact", "group": key[1], "group_size": len(reps), "cos": 1.0,
                        "a": m_ctx(i), "b": m_ctx(j)})

    # alias: short form / acronym and a long form in the same article (all candidates, unique or not)
    by_article = defaultdict(list)
    for i, m in enumerate(mentions):
        by_article[m["article"]].append(i)
    for idx in by_article.values():
        longs = {}
        for i in idx:
            n = normalize(mentions[i]["text"])
            if len(n.split()) > 1:
                longs.setdefault(n, i)
        seen = set()
        for i in idx:
            m = mentions[i]
            n = normalize(m["text"])
            if n in seen or (len(n.split()) != 1 and not m["text"].isupper()):
                continue
            seen.add(n)
            for ln, j in longs.items():
                acro = m["text"].isupper() and 2 <= len(m["text"]) <= 6 and acronym_of(mentions[j]["text"].split()) == m["text"]
                short = (TYPE_GROUP[mentions[j]["label"]] == TYPE_GROUP[m["label"]] and n != ln
                         and (ln.split()[-1] == n or ln.split()[0] == n))
                if acro or short:
                    out.append({"source": "alias", "group": TYPE_GROUP[m["label"]], "cos": None,
                                "a": m_ctx(j), "b": m_ctx(i)})

    def embed_pairs(vec):
        n = len(vec)
        for b0 in range(0, n, 2048):
            sims = vec[b0:b0 + 2048] @ vec.T
            for r, row in enumerate(sims):
                i = b0 + r
                row[i] = -1
                k = min(top, n - 1)
                for j in np.argpartition(-row, k)[:k]:
                    if row[j] >= floor and i < j:
                        yield i, int(j), float(row[j])

    for i, j, c in embed_pairs(ent_vec):
        out.append({"source": "embed", "group": TYPE_GROUP[mentions[ent_rep[i]]["label"]], "cos": c,
                    "a": m_ctx(ent_rep[i]), "b": m_ctx(ent_rep[j])})
    for i, j, c in embed_pairs(frag_vec):
        fa, fb = fragments[i], fragments[j]
        out.append({"source": "fragment", "group": "fragment", "cos": c,
                    "a": {**with_context(para, fa["pid"], fa["s"], fa["e"], fa["text"]), "source": "fragment"},
                    "b": {**with_context(para, fb["pid"], fb["s"], fb["e"], fb["text"]), "source": "fragment"}})
    return out


def stratified_sample(cands: list[dict], n: int, rng: random.Random) -> list[dict]:
    """About n pairs spread over sources and similarity bands, so hard cases are covered."""
    def band(c):
        if c["cos"] is None or c["source"] == "exact":
            return c["source"]
        return f"{c['source']}:{min(int((c['cos'] - 0.8) / 0.05), 3)}"
    strata = defaultdict(list)
    for c in cands:
        strata[band(c)].append(c)
    per = max(1, n // len(strata))
    out = []
    for key in sorted(strata):
        out += rng.sample(strata[key], min(per, len(strata[key])))
    return out


def run_candidates(cfg: dict, force: bool) -> None:
    """Write every candidate pair (for the LLM) and a stratified sample (for labelling)."""
    from collections import Counter
    import json
    from pathlib import Path

    from ragsplit.crack import crack_units, extracted_spans, parse_sentences
    from ragsplit.merge import embed, extract_mentions
    from ragsplit.retrieve import units_key
    from ragsplit.run import prepare_corpus, prepare_splits
    from ragsplit.units import sentence_units

    ac = cfg["adjudicate"]
    paragraphs, questions = prepare_corpus(cfg, force=False)
    sp = prepare_splits(cfg, questions, force=False)
    q_by_id = {q.id: q for q in questions}
    data = Path(cfg["data"]["data_dir"]) / "cache"
    key = units_key(sentence_units(paragraphs))
    rng = random.Random(ac["seed"])

    mentions = extract_mentions(paragraphs, cfg["data"]["spacy_model"], data / "merge" / f"mentions_{key}.json")
    first: dict = {}
    for i, m in enumerate(mentions):
        first.setdefault((normalize(m["text"]), TYPE_GROUP[m["label"]]), i)
    ent_rep = list(first.values())
    ent_texts = [mentions[i]["text"] for i in ent_rep]
    parses = parse_sentences(paragraphs, cfg["data"]["spacy_model"], data / "crack" / f"parses_{key}.json")
    k = dict(zip(("sentence", "clause", "phrase"), ac["fragment_k"]))
    units = crack_units(paragraphs, extracted_spans(paragraphs, [q_by_id[q] for q in sp["past"]], parses, k))
    frag_first: dict = {}
    for u in units:
        if u.kind in ("piece-clause", "piece-phrase"):
            text = u.text.split(": ", 1)[1]
            frag_first.setdefault(normalize(text), {"pid": u.paragraph_id, "s": u.span[0], "e": u.span[1], "text": text})
    fragments = list(frag_first.values())
    cands = candidates(paragraphs, mentions, fragments, embed(ent_texts, cfg["retrieval"]), ent_texts, ent_rep,
                       embed([f["text"] for f in fragments], cfg["retrieval"]), ac["floor"], rng)
    for n, c in enumerate(cands):
        c["id"] = n
    out = data / "merge" / f"candidates_{key}.jsonl"
    with out.open("w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    sample = stratified_sample(cands, ac["label_sample"], rng)
    results = Path(cfg.get("results_dir", "results"))
    with (results / f"{cfg['name']}_sample.jsonl").open("w") as f:
        for c in sample:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cands)} candidates -> {out}")
    print("by source:", dict(Counter(c["source"] for c in cands)))
    print(f"wrote {len(sample)} pairs to label -> {results / (cfg['name'] + '_sample.jsonl')}")


def score(pred: dict[int, str], gold: dict[int, str]) -> dict:
    """Agreement with the reference labels. SAME precision matters most: a false SAME is a
    false merge that traversal will follow."""
    ids = [i for i in gold if i in pred]
    out = {"n": len(ids), "accuracy": sum(pred[i] == gold[i] for i in ids) / max(1, len(ids)),
           "unparsed": sum(pred[i] is None for i in ids)}
    for lab in LABELS:
        tp = sum(pred[i] == lab and gold[i] == lab for i in ids)
        pp = sum(pred[i] == lab for i in ids)
        gp = sum(gold[i] == lab for i in ids)
        out[lab] = {"precision": tp / pp if pp else None, "recall": tp / gp if gp else None,
                    "predicted": pp, "gold": gp}
    out["false_merges"] = sum(pred[i] == "SAME" and gold[i] != "SAME" for i in ids)
    return out
